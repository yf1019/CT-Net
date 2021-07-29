import os
import numpy as np
import torch
import torchvision.utils as vutils
import sys
from options.test_options import TestOptions
import data
from util.iter_counter import IterationCounter
import cv2
# from util.visualizer import Visualizer
from models.ct_model import CTModel


# parse options
Options = TestOptions()
opt = Options.parse()

opt.serial_batches = True
opt.niter = 1
opt.niter_decay = 0
opt.batchSize = 1

save_dir = '/test'
print(' '.join(sys.argv))

if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.store_name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.store_name))

#torch.manual_seed(0)
# load the dataset
dataloader = data.create_dataloader(opt)
len_dataloader = len(dataloader)
testd = dataloader.dataset[0]

print("preprocessed img pixel range [{},{}]".format(torch.min(testd['image']), torch.max(testd['image'])))

# visualizer = Visualizer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

Vt_model = CTModel(opt)
Vt_model.to(opt.gpu_ids[0])

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output', opt.store_name)
if not os.path.exists(save_root + save_dir):
    os.makedirs(save_root + save_dir)

for epoch in iter_counter.training_epochs():
    opt.epoch = epoch

    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):

        iter_counter.record_one_iteration()

        out = Vt_model(data_i, mode='inference')

        print('iter %s finished %s - %s' % (iter_counter.epoch_iter, data_i['image_path'], data_i['ref_path']))

        seg_img = data_i['seg_img'][0:1, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0)
        seg_img = cv2.applyColorMap((seg_img * 30).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
        ref_seg = data_i['ref_seg'][0:1, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0)
        ref_seg = cv2.applyColorMap((ref_seg * 30).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
        seg_img = torch.FloatTensor(seg_img).unsqueeze(0).permute(0, 3, 1, 2)
        ref_seg = torch.FloatTensor(ref_seg).unsqueeze(0).permute(0, 3, 1, 2)
        
        image = (data_i['image'][0:1, :, :, :].cpu() + 1) * 255.0 / 2
        ref = (data_i['ref'][0:1, :, :, :].cpu() + 1) * 255.0 / 2
        fake_result = (out['fake_result'][0:1, :, :, :].cpu() + 1) * 255.0 / 2

        imgs = torch.cat((image, ref, seg_img, ref_seg, fake_result), 0)

        try:
            vutils.save_image(imgs, save_root + save_dir + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '.png',
                    nrow=5, padding=0, normalize=True)

        except OSError as err:
            print(err)

print('Testing was successfully finished.')