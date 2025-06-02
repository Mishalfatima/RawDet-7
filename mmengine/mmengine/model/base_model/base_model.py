# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from mmengine.utils import is_list_of
from ..base_module import BaseModule
from .data_preprocessor import BaseDataPreprocessor
import torchvision
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Union
import random

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x

class STEFunction_WB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.clip(input,0,1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class StraightThroughEstimator_WB(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_WB, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x
    

class BaseModel(BaseModule):
    """Base class for all algorithmic models.

    BaseModel implements the basic functions of the algorithmic model, such as
    weights initialize, batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.

    Subclasses inherit from BaseModel only need to implement the forward
    method, which implements the logic to calculate loss and predictions,
    then can be trained in the runner.

    Examples:
        >>> @MODELS.register_module()
        >>> class ToyModel(BaseModel):
        >>>
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.backbone = nn.Sequential()
        >>>         self.backbone.add_module('conv1', nn.Conv2d(3, 6, 5))
        >>>         self.backbone.add_module('pool', nn.MaxPool2d(2, 2))
        >>>         self.backbone.add_module('conv2', nn.Conv2d(6, 16, 5))
        >>>         self.backbone.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
        >>>         self.backbone.add_module('fc2', nn.Linear(120, 84))
        >>>         self.backbone.add_module('fc3', nn.Linear(84, 10))
        >>>
        >>>         self.criterion = nn.CrossEntropyLoss()
        >>>
        >>>     def forward(self, batch_inputs, data_samples, mode='tensor'):
        >>>         data_samples = torch.stack(data_samples)
        >>>         if mode == 'tensor':
        >>>             return self.backbone(batch_inputs)
        >>>         elif mode == 'predict':
        >>>             feats = self.backbone(batch_inputs)
        >>>             predictions = torch.argmax(feats, 1)
        >>>             return predictions
        >>>         elif mode == 'loss':
        >>>             feats = self.backbone(batch_inputs)
        >>>             loss = self.criterion(feats, data_samples)
        >>>             return dict(loss=loss)

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.

    Attributes:
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.ste = StraightThroughEstimator()
        self.ste_WB = StraightThroughEstimator_WB()

        self.gamma_night = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma_day = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma = nn.Parameter(data=torch.tensor(1.), requires_grad=True)

        self.gamma_raod = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma_praw = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma_nikon = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma_sony = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma_zurich = nn.Parameter(data=torch.tensor(1.), requires_grad=True)

        self.epsilon = nn.Parameter(data=torch.tensor(1.), requires_grad=True)

        self.mean_R = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.mean_B = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.mean_G = nn.Parameter(data=torch.tensor(1.), requires_grad=True)

        self.gamma_R = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma_B = nn.Parameter(data=torch.tensor(1.), requires_grad=True)
        self.gamma_G = nn.Parameter(data=torch.tensor(1.), requires_grad=True)

        self.gain = 1
        self.relu = nn.ReLU()

        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor

        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

    def WB(self,rgb):
        b = rgb[0,:,:].unsqueeze(dim=0).clone()
        g = rgb[1,:,:].unsqueeze(dim=0).clone()
        r = rgb[2,:,:].unsqueeze(dim=0).clone()
        
        '''mean_r = r.mean()
        mean_g = g.mean()
        mean_b = b.mean()'''

        mean_r = self.relu(self.mean_R)
        mean_g = self.relu(self.mean_G)
        mean_b = self.relu(self.mean_B)

        r *= mean_g / mean_r
        b *= mean_g / mean_b

        rgb = torch.concat([b,g,r], axis=0)
        #clip_max = (2**24) - 1
        clip_max = 1
        rgb =torch.clip(rgb, 0, clip_max)
        return rgb
    
    def extract_rgb(self,arr, data):
        
        if data == 'ZURICH':
            resized_r = arr[:, ::2,::2]
            resized_g1 = arr[:, 0::2, 1::2]
            resized_b = arr[:, 1::2, 0::2]
            resized_g2 = arr[:, 1::2, 1::2]

            green = torch.unsqueeze(torch.mean(torch.cat((resized_g1, resized_g2),0),0),0)
            green = self.ste(green)
            blue = resized_b
            red = resized_r

            img = torch.cat((blue, green, red), axis=0)
            img = torch.permute(img, (1,2,0))

        else:
            resized_r = arr[:, ::2,::2]
            resized_g1 = arr[:, 0::2, 1::2]
            resized_g2 = arr[:, 1::2, 0::2]
            resized_b = arr[:, 1::2, 1::2]

            green = torch.unsqueeze(torch.mean(torch.cat((resized_g1, resized_g2),0),0),0)
            green = self.ste(green)
            blue = resized_b
            red = resized_r

            img = torch.cat((blue, green, red), axis=0)
            img = torch.permute(img, (1,2,0))

        return img


    def downsampling(self, arr, new_h, new_w):

        blue = F.interpolate(torch.unsqueeze(torch.unsqueeze(arr[0, :,:],0),0), [new_h, new_w], mode='nearest')  
        green = F.interpolate(torch.unsqueeze(torch.unsqueeze(arr[1, :, :],0),0) ,[new_h, new_w], mode='nearest') 
        red = F.interpolate(torch.unsqueeze(torch.unsqueeze(arr[2, :, :],0),0), [new_h, new_w], mode='nearest') 

        img = torch.squeeze(torch.cat((blue, green, red), axis=0),1)

        return img
    
    def quant_module(self, data_batch, is_raw, quant, log, epsilon,log_lr, gamma_, data_type, n_gamma, use_WB):

        imgs = []

        images = data_batch['inputs']
        
        for i in range(len(images)):

            x = images[i]
            #import pdb; pdb.set_trace()
            data = data_batch['data_samples'][i].sensor
            #x = x.type(torch.cuda.FloatTensor)
            x = x.to(torch.float).cuda()
            #import pdb; pdb.set_trace()
            if is_raw:
                    if data == 'PASCAL_RAW':
                        nbits = 12
                    elif data == 'ZURICH':
                        nbits = 10
                    elif data == 'RAW_NOD_NIKON':
                        nbits = 14
                    elif data == 'RAW_NOD_SONY':
                        nbits = 14
                        x = torch.clip(x, max=((2**14)-1))
                    
                    elif data == 'RAOD':
                        nbits = 24

                    #import pdb; pdb.set_trace() 
                    norm_img =(x - 0)/((2**nbits) - 1)

                    if use_WB:
                        img_rgb = torch.permute(self.extract_rgb(norm_img, data),(2,0,1))
                        norm_img = self.WB(img_rgb)
                        norm_img = self.ste_WB(norm_img)
                        
                    if log == True and gamma_ == True:
                        norm_img = x + 1
                        norm_img = torch.log(norm_img)
                        scaled_raw = norm_img /(torch.log(torch.tensor(((2**nbits - 1) + 1))))
                        scaled_raw = self.gain * (scaled_raw ** self.relu(self.gamma))
                        scaled_raw = scaled_raw * ((2**quant)-1)
                        scaled_raw = self.ste(scaled_raw)

                    else:

                        if log:
                            if log_lr:
                                norm_img = x + 1 + self.epsilon
                                norm_img = torch.log(norm_img)
                                scaled_raw = norm_img /(torch.log(torch.tensor(((2**nbits - 1) + self.epsilon))))
                            else:
                                norm_img = x + 1
                                norm_img = torch.log(norm_img)
                                scaled_raw = norm_img /(torch.log(torch.tensor(((2**nbits - 1) + 1))))
                                scaled_raw = scaled_raw * ((2**quant)-1)
                                scaled_raw = torch.floor(scaled_raw)

                        elif gamma_:

                            if n_gamma == 5:
                                    if data_batch['data_samples'][i].sensor == 'PASCAL_RAW':
                                        scaled = self.gain * (norm_img ** self.relu(self.gamma_praw))
                                    if data_batch['data_samples'][i].sensor == 'ZURICH':
                                        scaled = self.gain * (norm_img ** self.relu(self.gamma_zurich))
                                    if data_batch['data_samples'][i].sensor == 'RAW_NOD_NIKON':
                                        scaled = self.gain * (norm_img ** self.relu(self.gamma_nikon))
                                    if data_batch['data_samples'][i].sensor == 'RAW_NOD_SONY':
                                        scaled = self.gain * (norm_img ** self.relu(self.gamma_sony))
                                    if data_batch['data_samples'][i].sensor == 'RAOD':
                                        scaled = self.gain * (norm_img ** self.relu(self.gamma_raod))

                            elif n_gamma == 2:
                                    if data_batch['data_samples'][i].time == 'day':
                                        scaled = self.gain * (norm_img ** self.relu(self.gamma_day))
                                    if data_batch['data_samples'][i].time == 'night':
                                        scaled = self.gain * (norm_img ** self.relu(self.gamma_night))
                            
                            elif n_gamma == 3:
                                if data == 'ZURICH':
                                    norm_img[:, ::2,::2]     = norm_img[:, ::2,::2].clone()     ** self.relu(self.gamma_R)  
                                    norm_img[:,  0::2, 1::2] = norm_img[:,  0::2, 1::2].clone() ** self.relu(self.gamma_G)
                                    norm_img[:,  1::2, 0::2] = norm_img[:,  1::2, 0::2].clone() ** self.relu(self.gamma_B)
                                    norm_img[:, 1::2, 1::2]  = norm_img[:, 1::2, 1::2].clone()  ** self.relu(self.gamma_G)
                                else:
                                    norm_img[:, ::2,::2]     = norm_img[:, ::2,::2].clone()    ** self.relu(self.gamma_R)  
                                    norm_img[:,  0::2, 1::2] = norm_img[:,  0::2, 1::2].clone() ** self.relu(self.gamma_G)
                                    norm_img[:,  1::2, 0::2] = norm_img[:,  1::2, 0::2].clone() ** self.relu(self.gamma_G)
                                    norm_img[:, 1::2, 1::2]  = norm_img[:, 1::2, 1::2].clone()  ** self.relu(self.gamma_B)

                                scaled = norm_img

                            elif n_gamma == 1:
                                    scaled = self.gain * (norm_img ** self.relu(self.gamma))
                            
                            #Quantization
                            scaled_raw = scaled * ((2**quant)-1)
                            scaled_raw = self.ste(scaled_raw)

                        else:
                            scaled_raw = norm_img*((2**quant)-1)
                            scaled_raw = torch.floor(scaled_raw)                    
                    
                    if not use_WB:
                        scaled_raw = torch.permute(self.extract_rgb(scaled_raw, data),(2,0,1))

                    downsampled_image = self.downsampling(scaled_raw, data_batch['data_samples'][i].img_shape[0], data_batch['data_samples'][i].img_shape[1])
                    
                    scaled_raw = (downsampled_image/ ((2**quant)-1))*255.

            else:
                scaled_raw = x
            

            imgs.append(scaled_raw)

        images = imgs
        data_batch['inputs'] = images

        return data_batch
    

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper,  is_raw,quant,gamma_,log,epsilon,log_lr, data_type, dataloader, model_type, n_gamma, use_WB) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            
            if is_raw:
                if model_type == 'YOLOX':
                    data = self.quant_module_yolox(data, is_raw, quant, log, epsilon, gamma_, data_type)
                else:
                    data = self.quant_module(data, is_raw, quant, log, epsilon, log_lr, gamma_, data_type, n_gamma, use_WB)

            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        
        parsed_losses, log_vars = self.parse_losses(losses)
          # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list], is_raw,quant,gamma_,log,epsilon,log_lr, data_type, n_gamma, use_WB) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` 
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """

        if is_raw:
                data = self.quant_module(data, is_raw, quant, log, epsilon, log_lr, gamma_, data_type, n_gamma, use_WB)
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list], is_raw,quant,gamma_,log,epsilon, log_lr, data_type, n_gamma, use_WB) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """

        if is_raw:
            data1 = self.quant_module(data, is_raw, quant, log, epsilon, log_lr, gamma_, data_type, n_gamma, use_WB)
        else:
            data1 = data
            
        data = self.data_preprocessor(data1, False)
        out = self._run_forward(data, mode='predict') 
        return out, data1

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        if args and isinstance(args[0], str) and 'npu' in args[0]:
            import torch_npu
            args = tuple([
                list(args)[0].replace(
                    'npu', torch_npu.npu.native_device if hasattr(
                        torch_npu.npu, 'native_device') else 'privateuseone')
            ])
        if kwargs and 'npu' in str(kwargs.get('device', '')):
            import torch_npu
            kwargs['device'] = kwargs['device'].replace(
                'npu', torch_npu.npu.native_device if hasattr(
                    torch_npu.npu, 'native_device') else 'privateuseone')

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._set_device(torch.device(device))
        return super().to(*args, **kwargs)

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
            device = torch.device('cuda', index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def musa(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.musa`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
            device = torch.device('musa', index=device)
        self._set_device(torch.device(device))
        return super().musa(device)

    def mlu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.mlu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        device = torch.device('mlu', torch.mlu.current_device())
        self._set_device(device)
        return super().mlu()

    def npu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.npu`
        additionally.

        Returns:
            nn.Module: The model itself.

        Note:
            This generation of NPU(Ascend910) does not support
            the use of multiple cards in a single process,
            so the index here needs to be consistent with the default device
        """
        device = torch.npu.current_device()
        self._set_device(device)
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        self._set_device(torch.device('cpu'))
        return super().cpu()

    def _set_device(self, device: torch.device) -> None:
        """Recursively set device for `BaseDataPreprocessor` instance.

        Args:
            device (torch.device): the desired device of the parameters and
                buffers in this module.
        """

        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor):
                return
            if device is not None:
                module._device = device

        self.apply(apply_fn)

    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``batch_inputs`` and ``data_sample`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.test_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (list, optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of results used for computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            dict or list:
                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of inference
                  results.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` of tensor for custom use.
        """

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
