# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import HistoryBuffer, print_log
from mmengine.registry import LOOPS
from mmengine.structures import BaseDataElement
from mmengine.utils import is_list_of
from .amp import autocast
from .base_loop import BaseLoop
from .utils import calc_dynamic_intervals
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image
import torch

@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
            is_raw = True, 
            quant =  4, 
            gamma_ = True,
            log = True,
            epsilon = 0,
            log_lr = False,
            data_type = 'pascalraw',
            n_gamma = 1,
            use_WB = False
            ) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval

        self.is_raw = is_raw
        self.quant = quant
        self.gamma_ = gamma_
        self.log = log
        self.epsilon = epsilon
        self.data_type = data_type
        self.n_gamma = n_gamma
        self.use_WB = use_WB
        self.log_lr = log_lr
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model
    

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.logger.info("The gamma value is " + str(self.runner.model.gamma.item()))
        self.runner.logger.info("The day gamma value is " + str(self.runner.model.gamma_day.item()))
        self.runner.logger.info("The night gamma value is " + str(self.runner.model.gamma_night.item()))

        self.runner.logger.info("The PRAW gamma value is " + str(self.runner.model.gamma_praw.item()))
        self.runner.logger.info("The Zurich gamma value is " + str(self.runner.model.gamma_zurich.item()))
        self.runner.logger.info("The Nikon gamma value is " + str(self.runner.model.gamma_nikon.item()))
        self.runner.logger.info("The Sony gamma value is " + str(self.runner.model.gamma_sony.item()))
        self.runner.logger.info("The RAOD gamma value is " + str(self.runner.model.gamma_raod.item()))

        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        l_total = []
        for idx, data_batch in enumerate(self.dataloader):
            
            outputs = self.run_iter(idx, data_batch)
            loss =  outputs['loss'].item()
            l_total.append(loss)

        self.runner.total_loss = sum(l_total)/len(l_total)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper, is_raw=self.is_raw,quant=self.quant,  gamma_=self.gamma_, log = self.log, epsilon=self.epsilon, \
                log_lr = self.log_lr, data_type = self.data_type, dataloader=self.dataloader, model_type=self.runner.model.__class__.__name__, n_gamma=self.n_gamma, use_WB = self.use_WB)


        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
        return outputs

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


class _InfiniteDataloaderIterator:
    """An infinite dataloader iterator wrapper for IterBasedTrainLoop.

    It resets the dataloader to continue iterating when the iterator has
    iterated over all the data. However, this approach is not efficient, as the
    workers need to be restarted every time the dataloader is reset. It is
    recommended to use `mmengine.dataset.InfiniteSampler` to enable the
    dataloader to iterate infinitely.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Sequence[dict]:
        try:
            data = next(self._iterator)
        except StopIteration:
            print_log(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.',
                logger='current',
                level=logging.WARNING)
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data


@LOOPS.register_module()
class IterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, \
            f'`max_iters` should be a integer number, but get {max_iters}'
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)
        # get the iterator of the dataloader
        self.dataloader_iterator = _InfiniteDataloaderIterator(self.dataloader)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            for _ in range(self._iter):
                next(self.dataloader_iterator)
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 is_raw = True, 
                 quant =  4, 
                 gamma_ = True,
                 log = True,
                 epsilon = 0,log_lr = False, data_type='pascalraw', n_gamma=1, use_WB=False) -> None:
        super().__init__(runner, dataloader)

        self.is_raw = is_raw
        self.quant = quant
        self.gamma_ = gamma_
        self.log = log
        self.epsilon = epsilon
        self.data_type = data_type
        self.n_gamma = n_gamma
        self.use_WB = use_WB
        self.log_lr = log_lr

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        self.val_loss: Dict[str, HistoryBuffer] = dict()

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        # clear val loss
        self.val_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        
        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
       
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch,  is_raw=self.is_raw,quant=self.quant,  gamma_=self.gamma_, log=self.log, epsilon=self.epsilon,\
                                                 
                                                 log_lr = self.log_lr, data_type = self.data_type, n_gamma = self.n_gamma, use_WB=self.use_WB)

        outputs, self.val_loss = _update_losses(outputs, self.val_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                dataloader: Union[DataLoader, Dict],
                evaluator: Union[Evaluator, Dict, List],
                fp16: bool = False, 
                is_raw = True, 
                quant =  4, 
                gamma_ = True,
                log = True, vis=False,
                epsilon = 0,log_lr = False, data_type='PRAW',
                n_gamma=1, use_WB=False, work_dir = ''):
        super().__init__(runner, dataloader)

        self.is_raw = is_raw
        self.quant = quant
        self.gamma_ = gamma_
        self.log = log
        self.epsilon = epsilon
        self.data_type = data_type
        self.n_gamma = n_gamma
        self.use_WB = use_WB
        self.vis = vis
        self.work_dir = work_dir
        self.log_lr = log_lr

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        self.test_loss: Dict[str, HistoryBuffer] = dict()

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.logger.info("The gamma value is " + str(self.runner.model.gamma.item()))
        self.runner.logger.info("The day gamma value is " + str(self.runner.model.gamma_day.item()))
        self.runner.logger.info("The night gamma value is " + str(self.runner.model.gamma_night.item()))
        
        self.runner.logger.info("The PRAW gamma value is " + str(self.runner.model.gamma_praw.item()))
        self.runner.logger.info("The Zurich gamma value is " + str(self.runner.model.gamma_zurich.item()))
        self.runner.logger.info("The Nikon gamma value is " + str(self.runner.model.gamma_nikon.item()))
        self.runner.logger.info("The Sony gamma value is " + str(self.runner.model.gamma_sony.item()))
        self.runner.logger.info("The RAOD gamma value is " + str(self.runner.model.gamma_raod.item()))

        self.runner.model.eval()

        # clear test loss
        # Create a text file to store the metrics

        if self.is_raw==False:
            folder_name = "sRGB"
        if self.is_raw==True:
            folder_name = str(self.quant)+"_"+'bit'
            if self.gamma_==True:
                folder_name = folder_name + "_" + str(self.n_gamma) + "_gamma"
            if self.log==True:
                folder_name = folder_name + "_log" 

        import os
        output_file= os.path.join(self.work_dir, folder_name+".txt")

        self.test_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
            if self.vis==True:
                metrics = self.evaluator.evaluate(1)
                # Open the file in append mode
                with open(output_file, 'a') as f:
                    # Iterate over the metrics for each image
                    # Write the metric to the file
                    f.write(data_batch['data_samples'][0].img_id+ " " + str(metrics['pascal_voc/mAP'])+"\n")

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics
    
    def draw_boxes_with_label_bg(self,
        image_tensor: torch.Tensor,
        boxes: torch.Tensor,
        labels: list,
        colors: list = None,
        font_size: int = 16,
        bg_color: str = "black",
        text_color: str = "white",
        folder_name = "sRGB",
    ):
        """
        Draws bounding boxes with labels and black background behind label text.

        Args:
            image_tensor (torch.Tensor): (3, H, W) uint8 tensor.
            boxes (torch.Tensor): Tensor of shape (N, 4).
            labels (list of str): List of labels for each box.
            colors (list of str): List of colors for each box.
            font_size (int): Font size for labels.
            bg_color (str): Background color behind label text.
            text_color (str): Text color.

        Returns:
            PIL.Image: Annotated image.
        """
        if torch.dtype is not torch.uint8:
            image_tensor = image_tensor.to(torch.uint8)
        assert image_tensor.dtype == torch.uint8, "Image must be uint8"
        assert image_tensor.shape[0] == 3, "Image must be in (3, H, W) format"

        image = to_pil_image(((image_tensor/image_tensor.max())*255.).to(torch.uint8))
        
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = map(int, box.tolist())
            color = colors[i % len(colors)] if colors else "green"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Get size of the label text
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # Background behind label
            text_x = x1
            text_y = max(y1 - text_h - 4, 0)
            draw.rectangle(
                [text_x, text_y, text_x + text_w + 4, text_y + text_h + 4],
                fill=bg_color
            )

            # Draw label text
            draw.text((text_x + 2, text_y + 2), label, font=font, fill=text_color)

        return image

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs,inputs = self.runner.model.test_step(data_batch,  is_raw=self.is_raw,quant=self.quant,  gamma_=self.gamma_, log = self.log, epsilon=self.epsilon,\
                                                log_lr = self.log_lr, data_type = self.data_type, n_gamma = self.n_gamma, use_WB=self.use_WB)


        if self.vis==True:

            if self.data_type == 'PRAW':

                self.METAINFO = {
                    'classes': ('car','person','bicycle'),
                    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42)]}
                
            elif self.data_type == 'RAOD':

                self.METAINFO = {'classes':
                ('Car', 'Cyclist', 'Tricycle', 'Tram', 'Truck', 'Pedestrain'),
                # palette is a list of color tuples, which is used for visualization.
                'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                            (197, 226, 255), (0, 60, 100)]}
                
            elif self.data_type == 'NEW':
                self.METAINFO  = {'classes':
                ('car', 'truck', 'tram', 'person', 'bicycle', 'motorcycle', 'bus'),
                # palette is a list of color tuples, which is used for visualization.
                'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                            (197, 226, 255), (0, 60, 100), (0, 60, 100)]}

            
            self.cat2label = {
                i: cat
                for i, cat in enumerate(self.METAINFO['classes'])
            }
            self.cat2color= {
                i: cat
                for i, cat in enumerate(self.METAINFO['palette'])
            }

            colors = [self.cat2color[cat.item()] for cat in outputs[0].pred_instances.labels]

            #import pdb; pdb.set_trace()
            #path = '/hkfs/work/workspace/scratch/ma_mfatima-mmdetection/datasets/RAW-RGB-Dataset/combined_sRGB/val/ZURICH/'+outputs[0].img_id[:-4]+'.jpg'
            #output_path= '/hkfs/work/workspace/scratch/ma_mfatima-mmdetection/mmdetection/predictions/'
            if self.is_raw==False:
                folder_name = "sRGB"
            if self.is_raw==True:
                folder_name = str(self.quant)+"_"+'bit'
                if self.gamma_==True:
                    folder_name = folder_name + "_" + str(self.n_gamma) + "_gamma"
                if self.log==True:
                    folder_name = folder_name + "_log" 
            
            import os
            output_path= os.path.join(self.work_dir, 'predictions', folder_name)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            orig_img = inputs['inputs'][0]
            orig_img = orig_img[[2, 1, 0], :, :]
            boxes = outputs[0].pred_instances.bboxes
            boxes[:,0] = (boxes[:,0]/outputs[0].ori_shape[0])*orig_img.shape[1]
            boxes[:,2] = (boxes[:,2]/outputs[0].ori_shape[0])*orig_img.shape[1]
            boxes[:,1] = (boxes[:,1]/outputs[0].ori_shape[1])*orig_img.shape[2]
            boxes[:,3] = (boxes[:,3]/outputs[0].ori_shape[1])*orig_img.shape[2]

            labels = [self.cat2label[outputs[0].pred_instances.labels[i].item()]+" : "+ str(round(outputs[0].pred_instances.scores[i].item(),3)) for i in range(len(outputs[0].pred_instances.labels))]
            img_annotated = self.draw_boxes_with_label_bg(orig_img, boxes, labels,  colors=colors, bg_color="black",text_color="white", folder_name=folder_name)


            output_p = output_path + "/" + str(outputs[0].img_id)+"_"+folder_name+'_Pred_.png'
            img_annotated.save(output_p)
            print(f"Image saved to: {output_p}")

            boxes = outputs[0].gt_instances.bboxes
            boxes[:,0] = (boxes[:,0]/outputs[0].ori_shape[0])*orig_img.shape[1]
            boxes[:,2] = (boxes[:,2]/outputs[0].ori_shape[0])*orig_img.shape[1]
            boxes[:,1] = (boxes[:,1]/outputs[0].ori_shape[1])*orig_img.shape[2]
            boxes[:,3] = (boxes[:,3]/outputs[0].ori_shape[1])*orig_img.shape[2]

            labels = [self.cat2label[outputs[0].gt_instances.labels[i].item()] for i in range(len(outputs[0].gt_instances.labels))]
            img_annotated = self.draw_boxes_with_label_bg(orig_img, boxes, labels,  colors=colors, bg_color="black",text_color="white", folder_name = folder_name)

            # Save the image
            output_p = output_path + "/" + str(outputs[0].img_id)+"_"+folder_name+'_GT_.png'
            img_annotated.save(output_p)
            print(f"Image saved to: {output_path}")

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)

        self.runner.call_hook(
                'after_test_iter',
                batch_idx=idx,
                data_batch=data_batch,
                outputs=outputs)


def _parse_losses(losses: Dict[str, HistoryBuffer],
                  stage: str) -> Dict[str, float]:
    """Parses the raw losses of the network.

    Args:
        losses (dict): raw losses of the network.
        stage (str): The stage of loss, e.g., 'val' or 'test'.

    Returns:
        dict[str, float]: The key is the loss name, and the value is the
        average loss.
    """
    all_loss = 0
    loss_dict: Dict[str, float] = dict()

    for loss_name, loss_value in losses.items():
        avg_loss = loss_value.mean()
        loss_dict[loss_name] = avg_loss
        if 'loss' in loss_name:
            all_loss += avg_loss

    loss_dict[f'{stage}_loss'] = all_loss
    return loss_dict


def _update_losses(outputs: list, losses: dict) -> Tuple[list, dict]:
    """Update and record the losses of the network.

    Args:
        outputs (list): The outputs of the network.
        losses (dict): The losses of the network.

    Returns:
        list: The updated outputs of the network.
        dict: The updated losses of the network.
    """
    if isinstance(outputs[-1],
                  BaseDataElement) and outputs[-1].keys() == ['loss']:
        loss = outputs[-1].loss  # type: ignore
        outputs = outputs[:-1]
    else:
        loss = dict()

    for loss_name, loss_value in loss.items():
        if loss_name not in losses:
            losses[loss_name] = HistoryBuffer()
        if isinstance(loss_value, torch.Tensor):
            losses[loss_name].update(loss_value.item())
        elif is_list_of(loss_value, torch.Tensor):
            for loss_value_i in loss_value:
                losses[loss_name].update(loss_value_i.item())
    return outputs, losses
