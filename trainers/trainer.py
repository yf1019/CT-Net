# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import sys
import torch
import torch.nn.functional as F
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.ct_model import CTModel
import util.util as util
from collections import OrderedDict
import numpy as np


class Trainer():

    def __init__(self, opt, resume_epoch=0):
        self.opt = opt
        self.ct_model = CTModel(opt)

        if len(opt.gpu_ids) > 1:
            self.ct_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.ct_model_on_one_gpu = self.ct_model.module
        else:
            self.ct_model.to(opt.gpu_ids[0])
            self.ct_model_on_one_gpu = self.ct_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.ct_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

            if opt.continue_train and opt.which_epoch == 'latest':
                checkpoint = torch.load(os.path.join(opt.checkpoints_dir, opt.store_name, 'optimizer.pth'))
                self.optimizer_G.load_state_dict(checkpoint['G'])
                self.optimizer_D.load_state_dict(checkpoint['D'])

        self.last_data, self.last_netCorr, self.last_netG, self.last_optimizer_G = None, None, None, None

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, out = self.ct_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.losses = g_losses

        self.out = out

    def run_discriminator_one_step(self, data):
        if self.optimizer_D is None:
            self.d_losses = {}
            return

        self.optimizer_D.zero_grad()
        GforD = {}
        GforD['real_mask'] = self.out['real_mask']
        GforD['mask_refine'] = self.out['mask_refine']
        GforD['initial_fake_image'] = self.out['initial_fake_image']
        GforD['refined_clothes'] = self.out['refined_clothes']
        GforD['real_cloth_mask'] = self.out['real_cloth_mask']
        GforD['warp_out'] = self.out['warp_out']
        GforD['input_G1'] = self.out['input_G1']
        GforD['input_G2'] = self.out['input_G2']
        GforD['input_G3'] = self.out['input_G3']
        GforD['refine_gt'] = self.out['refine_gt']
        d_losses = self.ct_model(data, mode='discriminator', GforD=GforD)
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.losses.update(d_losses)

    def get_latest_losses(self):
        return {**self.losses}

    def get_latest_generated(self):
        return self.out['fake_image']

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

        # if epoch == 'latest':
        if self.optimizer_D is None:
            torch.save({'G': self.optimizer_G.state_dict(),
                        'lr': self.old_lr,
                        }, os.path.join(self.opt.checkpoints_dir, self.opt.store_name, 'optimizer.pth'))

        else:
            torch.save({'G': self.optimizer_G.state_dict(),
                    'D': self.optimizer_D.state_dict(),
                    'lr':  self.old_lr,
                    }, os.path.join(self.opt.checkpoints_dir, self.opt.store_name, 'optimizer.pth'))

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def update_fixed_params(self):
        for param in self.pix2pix_model_on_one_gpu.net['netCorr'].parameters():
            param.requires_grad = True
        G_params = [{'params': self.pix2pix_model_on_one_gpu.net['netG1'].parameters(), 'lr': self.opt.lr*0.5}]
        G_params += [{'params': self.pix2pix_model_on_one_gpu.net['netG2'].parameters(), 'lr': self.opt.lr*0.5}]
        G_params += [{'params': self.pix2pix_model_on_one_gpu.net['netG3'].parameters(), 'lr': self.opt.lr*0.5}]
        G_params += [{'params': self.pix2pix_model_on_one_gpu.net['netCorr'].parameters(), 'lr': self.opt.lr*0.5}]
        if self.opt.no_TTUR:
            beta1, beta2 = self.opt.beta1, self.opt.beta2
            G_lr = self.opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr = self.opt.lr / 2

        self.optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self.out, name)
                if isinstance(value, list):
                    # visual multi-scale ouputs
                    for i in range(len(value)):
                        visual_ret[name + str(i)] = self.convert2im(value[i], name)
                    # visual_ret[name] = util.tensor2im(value[-1].data)
                else:
                    visual_ret[name] =self.convert2im(value, name)

    def convert2im(self, value, name):
        bytes = 255.0
        imtype = np.uint8
        image_tensor = value.data
        if image_tensor.dim() == 3:
            image_numpy = image_tensor.cpu().float().numpy()
        else:
            image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

        return image_numpy.astype(imtype)

    def inference(self, data):
        out = self.pix2pix_model(data, mode='inference')
        self.out = out

