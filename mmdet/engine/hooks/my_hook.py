# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.hooks.runtime_info_hook import RuntimeInfoHook
from collections import OrderedDict

@HOOKS.register_module()
class MyHook(Hook):
    def __init__(self, is_raw, quant, gamma_, log,log_lr, epsilon, data_type, n_gamma, use_WB):
        self.is_raw = is_raw
        self.quant=quant
        self.gamma_= gamma_
        self.log = log
        self.epsilon = epsilon
        self.data_type = data_type
        self.n_gamma = n_gamma
        self.use_WB = use_WB
        self.log_lr = log_lr
   
    def val_step(self, model, data, optim_wrapper, runner):
        
        with optim_wrapper.optim_context(model):
            if self.is_raw:
                data = runner.model.quant_module(data, self.is_raw, self.quant, self.log,
                                           self.epsilon,self.log_lr, \
                                          self.gamma_,self.data_type, \
                                            self.n_gamma, self.use_WB)
            data = model.data_preprocessor(data, False)
            losses = model(**data, mode='loss')  # type: ignore
       
        parsed_losses, log_vars = model.parse_losses(losses)  
        od = OrderedDict()
        od['test_loss'] = parsed_losses.item()

        '''od['test_loss'] = log_vars['loss']
        od['test_acc'] = log_vars['acc']
        od['test_loss_rpn_bbox'] = log_vars['loss_rpn_bbox']
        od['test_loss_rpn_cls'] = log_vars['loss_rpn_cls']
        od['test_loss_cls'] = log_vars['loss_cls']
        od['test_loss_bbox'] = log_vars['loss_bbox']
        log_vars = od'''
        return od

    def after_train_epoch(self, runner) -> None:
       
       for hook in runner._hooks:
            if isinstance(hook,(LoggerHook,)):
                    logger = hook
            elif isinstance(hook,(RuntimeInfoHook,)):
                    runtimeinfo = hook

       #import pdb; pdb.set_trace()
       od = OrderedDict()
       od['gamma'] = runner.model.gamma.item()
       od['gamma_day'] = runner.model.gamma_day.item()
       od['gamma_night'] = runner.model.gamma_night.item()
       od['gamma_raod'] = runner.model.gamma_raod.item()
       od['gamma_praw'] = runner.model.gamma_praw.item()
       od['gamma_nikon'] = runner.model.gamma_nikon.item()
       od['gamma_sony'] = runner.model.gamma_sony.item()
       od['gamma_zurich'] = runner.model.gamma_zurich.item()
       #import pdb; pdb.set_trace()
       getattr(logger, 'after_tot_train_epoch')(runner,runner.epoch+1,od)
        
       if runner.epoch % 10 == 0:
            model = runner.model
            model.eval()  #confirm that it is switched to train mode again
            optim_wrapper = runner.optim_wrapper
            dataloader = runner.test_dataloader
            od = OrderedDict()
            od['train_loss'] = runner.total_loss
            getattr(logger, 'after_tot_train_epoch')(runner,runner.epoch+1,od)

            total_val_loss = []
            for i, data in enumerate(dataloader):
                
                outputs = self.val_step(model, data, optim_wrapper, runner)
                total_val_loss.append(outputs['test_loss'])
            
            total_val_l = sum(total_val_loss)/len(total_val_loss)
            #getattr(runtimeinfo, 'after_test_iter')(runner, None, None, outputs)
            od = OrderedDict()
            od['test_loss'] = total_val_l
            getattr(logger, 'after_tot_val_epoch')(runner,runner.epoch+1,od)