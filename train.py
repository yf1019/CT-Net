import os
import numpy as np
import torch
import sys
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from trainers.trainer import Trainer
import cv2
from util.visualizer import Visualizer


Options = TrainOptions()
opt = Options.parse()

if opt.train_corr_only:
    opt.use_mask_refine = False
    opt.use_G2 = False
    opt.warp_Mask_entropy = 0.0
    opt.mask_l1 = 0.0
    opt.Mask_entropy = 0.0
    opt.clothes_l1 = 0.0
    opt.clothes_mask_l1 = 0.0
    opt.cloth_mask_entropy = 0.0
    opt.warp_l1_loss = 0.0
    opt.weight_gan = 0.0
    opt.lambda_feat = 0.0
    opt.lambda_vgg = 0.0
    opt.reg_loss = 0.0
    opt.weight_perceptual = 0.0
    opt.ctx_w = 0.0
    opt.tps_mask_loss = 0.0
    opt.reg_occlusion = 0.0
    opt.tps_warped_cl_mask_bce = 0.0
    opt.tps_warped_cl_l1 = 0.0

# print options to help debugging
print(' '.join(sys.argv))

if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.store_name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.store_name))

#torch.manual_seed(0)
# load the dataset
dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)
testd = dataloader.dataset[0]

print("preprocessed img pixel range [{},{}]".format(torch.min(testd['image']), torch.max(testd['image'])))

visualizer = Visualizer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create trainer for our model
trainer = Trainer(opt, resume_epoch=iter_counter.first_epoch)

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output', opt.store_name)

for epoch in iter_counter.training_epochs():
    opt.epoch = epoch

    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        if not opt.train_corr_only:
            trainer.run_discriminator_one_step(data_i)

        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, \
                                            iter_counter.time_per_iter)

        if iter_counter.needs_displaying():
            seg_img = data_i['seg_img'][0:1, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0)
            seg_img = cv2.applyColorMap((seg_img * 30).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
            ref_seg = data_i['ref_seg'][0:1, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0)
            ref_seg = cv2.applyColorMap((ref_seg * 30).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]

            seg_img = torch.FloatTensor(seg_img).unsqueeze(0).permute(0, 3, 1, 2)
            ref_seg = torch.FloatTensor(ref_seg).unsqueeze(0).permute(0, 3, 1, 2)

            image = (data_i['image'][0:1, :, :, :].cpu() + 1) * 255.0 / 2
            ref = (data_i['ref'][0:1, :, :, :].cpu() + 1) * 255.0 / 2

            imgs = {}
            imgs['image'] = image.detach().numpy().transpose((0, 2, 3, 1)).squeeze(0)
            imgs['ref'] = ref.detach().numpy().transpose((0, 2, 3, 1)).squeeze(0)
            imgs['seg_img'] = seg_img.detach().numpy().transpose((0, 2, 3, 1)).squeeze(0)
            imgs['ref_seg'] = ref_seg.detach().numpy().transpose((0, 2, 3, 1)).squeeze(0)

            if not opt.train_corr_only:
                mask_refine = trainer.out['mask_refine'][0:1, :, :, :].cpu().detach()
                mask_refine = torch.max(mask_refine, dim=1, keepdim=True)[1]
                mask_refine = mask_refine.cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0)
                mask_refine = cv2.applyColorMap((mask_refine * 30.0).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
                mask_refine = torch.FloatTensor(mask_refine).unsqueeze(0).permute(0, 3, 1, 2)
                fake_result = (trainer.out['fake_result'][0:1, :, :, :].cpu() + 1) * 255.0 / 2
                imgs['mask_refine'] = mask_refine.detach().numpy().transpose((0, 2, 3, 1)).squeeze(0)
                imgs['fake_result'] = fake_result.detach().numpy().transpose((0, 2, 3, 1)).squeeze(0)
            else:
                warp_out = (trainer.out['warp_out'][0:1, :, :, :].cpu() + 1) * 255.0 / 2
                imgs['warp_out'] = warp_out.detach().numpy().transpose((0, 2, 3, 1)).squeeze(0)

            visualizer.display_current_results(imgs, epoch)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save("%d_%d" % (epoch, iter_counter.total_steps_so_far))
                iter_counter.record_current_iter()
            except OSError as err:
                print(err)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        try:
            # trainer.save('n_latest')
            trainer.save(epoch)
        except OSError as err:
            print(err)

print('Training was successfully finished.')
