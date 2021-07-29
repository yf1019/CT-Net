import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.util import kp2gaussian, make_coordinate_grid
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from util.util import vgg_preprocess
import util.util as util
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
import cv2
from torch.autograd import Variable
from PIL import Image, ImageDraw
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out


class VGG19_feature_color_torchversion(nn.Module):
    '''
    NOTE: there is no need to pre-process the input
    input tensor should range in [0,1]
    '''

    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        '''
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class Correspondence(BaseNetwork):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        if opt.use_coordconv:
            self.input_channel = opt.semantic_nc + 3
        else:
            self.input_channel = opt.semantic_nc

        self.fold_size = 256
        self.down = opt.down
        self.feature_channel = 64

        self.feature_img_extractor = feature_extractor(self.opt, \
                                                       self.input_channel, self.feature_channel)

        if self.opt.Corr_mode == 'double_dis':
            self.feature_ref_extractor = feature_extractor(self.opt, \
                                                  self.input_channel, self.feature_channel)
        elif self.opt.Corr_mode == 'dis_img':
            self.feature_ref_extractor = feature_extractor(self.opt, \
                                                  3, self.feature_channel)

        else:
            self.feature_ref_extractor = feature_extractor(self.opt, \
                                                  self.input_channel + 3, self.feature_channel)

        num_group = self.feature_channel * self.feature_channel
        refine_nc = self.feature_channel * self.feature_channel

        self.upsampling_bi = nn.Upsample(scale_factor=opt.down, mode='bilinear')  # for show

    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))

        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat

    def forward(self,
                ref_img,
                real_img,
                seg_map,
                ref_seg_map,
                temperature=0.01,
                detach_flag=False,
                return_corr=False,
                test_mask=None):

        coor_out = {}
        batch_size = ref_img.shape[0]
        image_height = ref_img.shape[2]
        image_width = ref_img.shape[3]
        feature_height = int(image_height / self.opt.down)
        feature_width = int(image_width / self.opt.down)

        # seg_img = test_mask['seg_img']
        ref_seg = test_mask['ref_seg'].detach()

        # real_mask = self.mask_convert(seg_img)
        # lower_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 5).astype(np.float))
        # upper_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 4).astype(np.float))
        t_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 5).astype(np.float))
        ref_mask = ref_seg * (1 - t_mask) + t_mask * 4
        ref_mask[ref_mask != 4] = 0
        ref_mask[ref_mask == 4] = 1
        coor_out['cl_mask'] = ref_mask
        # c_ref_cl_mask = self.mask_convert(ref_mask, 2)

        masked_ref = ref_img * ref_mask
        coor_out['masked_ref'] = masked_ref
        if self.opt.predict_mode == 'k' or self.opt.predict_mode == 'k+dp' or self.opt.predict_mode == 'cloth+k':
            t_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 1).astype(np.float))
            no_head_mask = ref_seg * (1 - t_mask) + t_mask * 0

            t_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 6).astype(np.float))
            no_head_and_shoes_mask = no_head_mask * (1 - t_mask) + t_mask * 0

            c_no_head_and_shoes_mask = self.mask_convert(no_head_and_shoes_mask)
            coor_out['no_head_and_shoes_mask'] = no_head_and_shoes_mask

            ref_mask = c_no_head_and_shoes_mask
        elif self.opt.predict_mode == 'cloth+k+dp':
            ref_mask = ref_mask

        img_input = seg_map

        if self.opt.Corr_mode == 'double_dis':
            ref_input = ref_seg_map
        elif self.opt.Corr_mode == 'dis_img':
            ref_input = ref_img
        else:
            ref_input = torch.cat([ref_seg_map, ref_img], dim=1)
        if self.opt.use_coordconv:
            img_input = self.addcoords(img_input)
            ref_input = self.addcoords(ref_input)

        feature_img = self.feature_img_extractor(img_input)
        # feature_img = None
        # for feature in features_img:
        #     feature = F.interpolate(feature, (feature_height, feature_width), mode='bilinear')
        #     feature_img = feature if feature_img is None else torch.cat((feature_img, feature), \
        #                                                                   dim=1)
        feature_ref = self.feature_ref_extractor(ref_input)
        # feature_ref = self.feature_img_extractor(ref_seg_map)

        # if self.opt.down_sample_f:
        #     feature_img = F.max_pool2d(feature_img, 2)
        #     feature_ref = F.max_pool2d(feature_ref, 2)
        #     self.opt.down = 8

        adaptive_feature_img = util.feature_normalize(feature_img)
        adf_img = F.unfold(adaptive_feature_img, \
                kernel_size = self.opt.match_kernel, padding = int(self.opt.match_kernel // 2))
        adf_img_t = adf_img.permute(0, 2, 1)

        adaptive_feature_ref = util.feature_normalize(feature_ref)

        adf_ref = F.unfold(adaptive_feature_ref, \
                               kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))

        f = torch.matmul(adf_img_t, adf_ref)

        if self.opt.freeze_warp:
            f = f.detach()
            f.requires_grad_()

        f_WTA = f / temperature
        f_div_C = F.softmax(f_WTA, dim=-1)

        # c_ref_seg = self.mask_convert(ref_seg)
        # coor_out['c_ref_seg'] = c_ref_seg

        # if self.opt.use_grid_warp or self.opt.use_grid_loss:
        #     grid = make_coordinate_grid((image_height, image_width), real_img.type())
        #     grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2)
        #     grid = F.unfold(grid, self.opt.down, stride=self.opt.down)
        #     grid = grid.permute(0, 2, 1)
        #
        #     g = torch.matmul(f_div_C, grid)
        #     g = g.permute(0, 2, 1)
        #     g = F.fold(g, self.fold_size, self.opt.down, stride=self.opt.down)
        #
        #     coor_out['flow'] = g

        if self.opt.use_grid_warp:
            g = g.permute(0, 2, 3, 1)
            y = F.grid_sample(ref_img, g)
            warp_masked_ref = F.grid_sample(masked_ref, g)
            warp_ref_seg = F.grid_sample(ref_mask, g)
        else:
            ref = F.unfold(ref_img, self.opt.down, stride=self.opt.down)
            ref = ref.permute(0, 2, 1)

            c_ref_seg = F.unfold(ref_mask, self.opt.down, stride=self.opt.down)
            c_ref_seg = c_ref_seg.permute(0, 2, 1)

            masked_ref = F.unfold(masked_ref, self.opt.down, stride=self.opt.down)
            masked_ref = masked_ref.permute(0, 2, 1)

            if self.opt.use_max_Corr:
                assert f_div_C.size()[0] == 1
                max_ind = torch.max(f_div_C.squeeze(), dim=1)[1]
                y = ref[:, max_ind, :]
                warp_masked_ref = masked_ref[:, max_ind, :]
                warp_ref_seg = c_ref_seg[:, max_ind, :]
            else:
                y = torch.matmul(f_div_C, ref)
                warp_masked_ref = torch.matmul(f_div_C, masked_ref)
                warp_ref_seg = torch.matmul(f_div_C, c_ref_seg)

            y = y.permute(0, 2, 1)
            y = F.fold(y, self.fold_size, self.opt.down, stride=self.opt.down)

            warp_masked_ref = warp_masked_ref.permute(0, 2, 1)
            warp_masked_ref = F.fold(warp_masked_ref, self.fold_size, self.opt.down, stride=self.opt.down)

            warp_ref_seg = warp_ref_seg.permute(0, 2, 1)
            warp_ref_seg = F.fold(warp_ref_seg, self.fold_size, self.opt.down, stride=self.opt.down)

        coor_out['warp_out'] = y
        coor_out['warp_masked_ref'] = warp_masked_ref
        coor_out['warp_ref_seg'] = warp_ref_seg

        # masked_ref_lower = ref_img * lower_mask
        # masked_ref_lower = F.unfold(masked_ref_lower, self.opt.down, stride=self.opt.down)
        # masked_ref_lower = masked_ref_lower.permute(0, 2, 1)
        #
        # warp_masked_ref_lower = torch.matmul(f_div_C, masked_ref_lower)
        # warp_masked_ref_lower = warp_masked_ref_lower.permute(0, 2, 1)
        # warp_masked_ref_lower = F.fold(warp_masked_ref_lower, 256, self.opt.down, stride=self.opt.down)
        #
        # coor_out['warp_masked_ref_lower'] = warp_masked_ref_lower
        #
        # masked_ref_upper = ref_img * upper_mask
        # masked_ref_upper = F.unfold(masked_ref_upper, self.opt.down, stride=self.opt.down)
        # masked_ref_upper = masked_ref_upper.permute(0, 2, 1)
        #
        # warp_masked_ref_upper = torch.matmul(f_div_C, masked_ref_upper)
        # warp_masked_ref_upper = warp_masked_ref_upper.permute(0, 2, 1)
        # warp_masked_ref_upper = F.fold(warp_masked_ref_upper, 256, self.opt.down, stride=self.opt.down)
        #
        # coor_out['warp_masked_ref_upper'] = warp_masked_ref_upper

        if self.opt.use_cl_refine and self.opt.refine_type == 'single_warp':
            i_feature_img = self.feature_img_extractor(seg_map)
            i_feature_ref = self.feature_img_extractor(ref_seg_map)
            adaptive_feature_i_img = util.feature_normalize(i_feature_img)
            adaptive_feature_i_ref = util.feature_normalize(i_feature_ref)
            adf_i_img = F.unfold(adaptive_feature_i_img, \
                                 kernel_size = self.opt.match_kernel, padding = int(self.opt.match_kernel // 2))
            adf_i_ref = F.unfold(adaptive_feature_i_ref, \
                kernel_size = self.opt.match_kernel, padding = int(self.opt.match_kernel // 2))
            adf_i_img_t = adf_i_img.permute(0, 2, 1)

            id_f = torch.matmul(adf_i_img_t, adf_i_ref)

            if self.opt.freeze_id_warp:
                id_f = id_f.detach()
                id_f.requires_grad_()

            id_f_WTA = id_f / temperature
            id_f_div_C = F.softmax(id_f_WTA.squeeze(), dim=-1)
            id_y = torch.matmul(id_f_div_C, ref)
            id_y = id_y.permute(0, 2, 1)
            id_y = F.fold(id_y, 256, self.opt.down, stride=self.opt.down)
            coor_out['id_warp_out'] = id_y
            warp_i_masked_ref = torch.matmul(id_f_div_C, masked_ref)
            warp_i_masked_ref = warp_i_masked_ref.permute(0, 2, 1)
            warp_i_masked_ref = F.fold(warp_i_masked_ref, 256, self.opt.down, stride=self.opt.down)

            coor_out['warp_i_masked_ref'] = warp_i_masked_ref

        if self.opt.draw_corr:
            index = (feature_height // 2) * feature_width + feature_width // 2
            hot_map = f_div_C[index, :]
            hot_map = hot_map.view(feature_height, feature_width).unsqueeze(0).unsqueeze(0)
            hot_map = F.interpolate(hot_map, size=real_img.size()[2:], mode='nearest')
            coor_out['hot_map'] = hot_map

        if self.opt.warp_cycle_w > 0:
            f_div_C_v = F.softmax(f_WTA.transpose(1, 2), dim=-1)
            if self.opt.warp_patch:
                o_warp_mased_ref = F.unfold(warp_masked_ref, self.opt.down, stride=self.opt.down)
                o_warp_mased_ref = o_warp_mased_ref.permute(0, 2, 1)
                cycle_masked_ref = torch.matmul(f_div_C_v, o_warp_mased_ref)
                cycle_masked_ref = cycle_masked_ref.permute(0, 2, 1)
                cycle_masked_ref = F.fold(cycle_masked_ref, 256, self.opt.down, stride=self.opt.down)
                coor_out['cycle_masked_ref'] = cycle_masked_ref

        return coor_out

    def mask_convert(self, seg, c=7):
        size = seg.size()
        oneHot_size = (size[0], c, size[2], size[3])
        c_seg = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_seg_tensor = c_seg.scatter_(1, seg.data.long().cuda(), 1.0)

        return c_seg_tensor

    def create_optimizer(self, opt):
        Corr_params = list()
        Corr_params += [{'params': self.parameters(), 'lr': opt.lr * 0.5}]

        beta1, beta2 = opt.beta1, opt.beta2
        Corr_lr = opt.lr

        optimizer_Corr = torch.optim.Adam(Corr_params, lr=Corr_lr, betas=(beta1, beta2))
        return optimizer_Corr

    def fuse_mask(self, seg, items, target):
        c_seg = seg
        for item in items:
            t_mask = torch.cuda.FloatTensor((seg.cpu().numpy() == item).astype(np.float))
            c_seg = c_seg * (1 - t_mask) + t_mask * target
        return c_seg


class feature_extractor(BaseNetwork):
    def __init__(self, opt, input_channel, feature_channel):
        super().__init__()
        self.opt = opt
        self.input_channel = input_channel
        self.feature_channel = feature_channel

        self.padding_1 = nn.ReflectionPad2d(1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(self.input_channel, self.feature_channel \
                                           , 3, stride=1, padding=0))
        self.layer2 = norm_layer(nn.Conv2d(self.feature_channel * 1, self.feature_channel * 2, \
                                           4, stride=2, padding=0))
        self.layer3 = norm_layer(nn.Conv2d(self.feature_channel * 2, self.feature_channel * 4, \
                                           3, stride=1, padding=0))
        self.layer4 = norm_layer(nn.Conv2d(self.feature_channel * 4, \
                                           self.feature_channel * 4, \
                                           4, stride=2, padding=0))

        self.layer5 = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                      kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1))

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, input):
        x = self.layer1(self.padding_1(input))
        x = self.layer2(self.padding_1(self.actvn(x)))
        x = self.layer3(self.padding_1(self.actvn(x)))
        x = self.layer4(self.padding_1(self.actvn(x)))
        x = self.layer5(self.actvn(x))

        return x


class tps_feature_extractor(BaseNetwork):
    def __init__(self, opt, input_channel, feature_channel):
        super().__init__()
        self.opt = opt
        self.input_channel = input_channel
        self.feature_channel = feature_channel

        self.padding_1 = nn.ReflectionPad2d(1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(self.input_channel, self.feature_channel \
                                           , 3, stride=1, padding=0))
        self.layer2 = norm_layer(nn.Conv2d(self.feature_channel * 1, self.feature_channel * 2, \
                                           4, stride=2, padding=0))
        self.layer3 = norm_layer(nn.Conv2d(self.feature_channel * 2, self.feature_channel * 4, \
                                           3, stride=1, padding=0))
        self.layer4 = norm_layer(nn.Conv2d(self.feature_channel * 4, \
                                           self.feature_channel * 4, \
                                           4, stride=2, padding=0))
        self.layer5 = norm_layer(nn.Conv2d(self.feature_channel * 4, self.feature_channel * 4, \
                                           3, stride=1, padding=0))
        self.layer6 = norm_layer(nn.Conv2d(self.feature_channel * 4, \
                                           self.feature_channel * 4, \
                                           4, stride=2, padding=0))
        self.layer7 = norm_layer(nn.Conv2d(self.feature_channel * 4, self.feature_channel * 4, \
                                           3, stride=1, padding=0))
        self.layer8 = norm_layer(nn.Conv2d(self.feature_channel * 4, \
                                           self.feature_channel * 4, \
                                           4, stride=2, padding=0))

        self.layer_for_corr = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1)
        )

        if not self.opt.use_ds_tps_corr:
            self.layer_for_tps = nn.Sequential(
                ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                              kernel_size=3, padding=1, stride=1),
                ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                              kernel_size=3, padding=1, stride=1)
            )

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, input):
        x = self.layer1(self.padding_1(input))
        x = self.layer2(self.padding_1(self.actvn(x)))
        x = self.layer3(self.padding_1(self.actvn(x)))
        x = self.layer4(self.padding_1(self.actvn(x)))
        corr_f = self.layer_for_corr(self.actvn(x))

        tps_f = None
        if not self.opt.use_ds_tps_corr:
            x = self.layer5(self.padding_1(self.actvn(x.detach())))
            x = self.layer6(self.padding_1(self.actvn(x)))
            x = self.layer7(self.padding_1(self.actvn(x)))
            x = self.layer8(self.padding_1(self.actvn(x)))
            tps_f = self.layer_for_tps(self.actvn(x))

        return corr_f, tps_f


class feature_encoder(BaseNetwork):
    def __init__(self, opt, input_nc):
        super(feature_encoder, self).__init__()

        self.opt = opt
        self.layers = 3
        ngf = 64

        norm_layer = nn.InstanceNorm2d
        nonlinearity = get_nonlinearity_layer(activation_type= 'LeakyReLU')

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                   nonlinearity)
        mult = 1
        for i in range(self.layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), 1024 // ngf)

            if i != self.layers - 2:
                block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                     nonlinearity)
            else:
                block = ExpandBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                     nonlinearity)

            setattr(self, 'encoder' + str(i), block)

    def forward(self, input):
        out = self.block0(input)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out

def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlock, self).__init__()


        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = conv(input_nc,  output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc),  nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2,)

    def forward(self, x):
        out = self.model(x)
        return out


class ExpandBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ExpandBlock, self).__init__()


        kwargs_down = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = conv(input_nc,  output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc),  nonlinearity, conv1,
                                       norm_layer(output_nc), nonlinearity, conv2,)

    def forward(self, x):
        out = self.model(x)
        return out


def conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    return nn.Conv2d(input_nc, output_nc, **kwargs)


class UnetFeatureExtractor(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc):
        super(UnetFeatureExtractor, self).__init__()

        nl = nn.InstanceNorm2d

        self.conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024),
                                     nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
              nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
              nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
              nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
              nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)
                                     ])

    def forward(self, input_G):
        results = []
        conv1 = self.conv1(input_G)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([drop4, up6], 1))

        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], 1))


        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], 1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], 1))

        return conv9


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResBlock(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.InstanceNorm2d):
        super(ResBlock, self).__init__()
        self.activation = nn.PReLU()
        self.conv = get_conv(input_nc, input_nc, kernel_size=3, padding=1, norm_layer=norm_layer)

    def forward(self, x):
        out = self.conv(self.activation(x))
        out = x + out
        return out


def get_conv(input_nc, output_nc, kernel_size=3, padding=1, stride=1, norm_layer=None):
    if norm_layer is not None:
        model = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding),
            norm_layer(output_nc),
        )
    else:
        model = nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding)
    return model


class DownBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, start=False):
        super(DownBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.conv = get_conv(input_nc, output_nc, kernel_size=3, padding=1, stride=2, norm_layer=norm_layer)
        self.start = start

    def forward(self, x):
        if self.start:
            out = self.conv(x)
        else:
            out = self.conv(self.activation(x))

        return out


class ClothFlow(BaseNetwork):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        self.kp_variance = 0.01
        self.num_kp = 18
        num_channels = 3
        self.hourglass = Hourglass(block_expansion=64, in_features=(self.num_kp + 1) * (num_channels + 1),
                                   max_features=1024, num_blocks=5)
        self.mask = nn.Conv2d(self.hourglass.out_filters, self.num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        n_res_blocks = 2
        nf = 64
        max_nf = 256

        self.num_scales = 6
        self.n_res_blocks = n_res_blocks
        self.label_nc = 20

        self.src_0_down = DownBlock(self.label_nc, nf * 1, start=True)
        for i in range(n_res_blocks):
            self.__setattr__('src_%d_res_%d'%(0, i), ResBlock(nf * 1))

        for l in range(1, self.num_scales):
            c_in = min(nf * 2**(l-1), max_nf)
            c_out = min(nf * 2**l, max_nf)

            self.__setattr__('src_%d_down'%l, DownBlock(c_in, c_out))

            for i in range(n_res_blocks):
                self.__setattr__('src_%d_res_%d'%(l, i), ResBlock(c_out))

        self.tar_0_down = DownBlock(self.label_nc + 3, nf * 1, start=True)
        for i in range(n_res_blocks):
            self.__setattr__('tar_%d_res_%d' % (0, i), ResBlock(nf * 1))

        for l in range(1, self.num_scales):
            c_in = min(nf * 2**(l-1), max_nf)
            c_out = min(nf * 2**l, max_nf)

            self.__setattr__('tar_%d_down' % l, DownBlock(c_in, c_out))

            for i in range(n_res_blocks):
                self.__setattr__('tar_%d_res_%d' % (l, i), ResBlock(c_out))

        for l in range(self.num_scales, 0, -1):
            c_in = 256
            c_out = 2

            if not self.opt.use_coordconv:
                conv = get_conv(c_in * 2, c_out, kernel_size=3, stride=1, padding=1)
            else:
                conv = get_conv(c_in * 2 + 3 * 2, c_out, kernel_size=3, stride=1, padding=1)

            pred = nn.Sequential(
                nn.LeakyReLU(0.1),
                conv,
            )
            self.__setattr__('flow_predict_%d'%(l-1), pred)

        for l in range(self.num_scales):
            c_in = min(nf * 2**l, max_nf)
            c_out = 256

            conv1_1 = get_conv(c_in, c_out, kernel_size=1, stride=1, padding=0)
            conv3_3 = get_conv(c_out, c_out, kernel_size=3, stride=1, padding=1)

            self.__setattr__('src_rpn_%d_conv1'%l, conv1_1)
            self.__setattr__('src_rpn_%d_conv3'%l, conv3_3)

        for l in range(self.num_scales):
            c_in = min(nf * 2**l, max_nf)
            c_out = 256

            conv1_1 = get_conv(c_in, c_out, kernel_size=1, stride=1, padding=0)
            conv3_3 = get_conv(c_out, c_out, kernel_size=3, stride=1, padding=1)

            self.__setattr__('tar_rpn_%d_conv1'%l, conv1_1)
            self.__setattr__('tar_rpn_%d_conv3'%l, conv3_3)

        self.upsampling_bi = nn.Upsample(scale_factor=2, mode='bilinear')

    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

#         xx_channel = xx_channel.float() / (w - 1)
#         yy_channel = yy_channel.float() / (h - 1)
#         xx_channel = 2 * xx_channel - 1
#         yy_channel = 2 * yy_channel - 1
        xx_channel = xx_channel.float()
        yy_channel = yy_channel.float()

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))

        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        num_kp = self.num_kp

        d_flag = kp_driving['flag'] == -1
        s_flag = kp_source['flag'] == -1
        flag = d_flag | s_flag

        ref_lds = kp_source['value'].detach()
        image_lds = kp_driving['value'].detach()

        ref_lds[flag, :] = torch.cuda.FloatTensor([0, 0])
        image_lds[flag, :] = torch.cuda.FloatTensor([0, 0])

        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - image_lds.view(bs, num_kp, 1, 1, 2)

        driving_to_source = coordinate_grid + ref_lds.view(bs, num_kp, 1, 1, 2)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions, flag

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def mask_convert(self, seg, c=7):
        size = seg.size()
        oneHot_size = (size[0], c, size[2], size[3])
        c_seg = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_seg_tensor = c_seg.scatter_(1, seg.data.long().cuda(), 1.0)

        return c_seg_tensor

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def forward(self,
                ref_img,
                real_img,
                seg_map,
                ref_seg_map,
                detach_flag=False,
                return_corr=False,
                test_mask=None):
        corr_out = {}
        bs, _, h, w = real_img.shape

        image_lds = test_mask['image_lds']
        ref_lds = test_mask['ref_lds']
        heatmap_representation = self.create_heatmap_representations(ref_img, image_lds, ref_lds)
        sparse_motions, occlusion_flag = self.create_sparse_motions(ref_img, image_lds, ref_lds)
        heatmap_representation[:, 1:, :, :][occlusion_flag, :] = torch.zeros(heatmap_representation.size()[2:]).cuda()

        deformed_ref_img = self.create_deformed_source_image(ref_img, sparse_motions)
        corr_out['deformed_ref_img'] = deformed_ref_img

        input = torch.cat([heatmap_representation, deformed_ref_img], dim=2)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)
        mask = self.mask(prediction)
        mask[:, 1:, :, :][occlusion_flag, :] = torch.zeros(mask.size()[2:]).cuda()
        mask = F.softmax(mask, dim=1)
        corr_out['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motions = sparse_motions.permute(0, 1, 4, 2, 3)

        deformation = (sparse_motions * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        corr_out['deformation'] = deformation

        if self.opt.debug:
            def draw_lds(img, lds):
                img = img.copy()
                colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
                          [0, 255, 0], \
                          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
                          [85, 0, 255], \
                          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

                for i in range(7,8):
                    if lds[0, i, :] == (-1, -1):
                        continue

                    x, y = lds[0, i, :]
                    cv2.circle(img, (int((x + 1) * 256 / 2), int((y + 1) * 256 / 2)), 4, colors[i], thickness=-1)

                return img

            debug = {}
            image = (real_img[0:1, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['image'] = image
            ref = (ref_img[0:1, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['ref'] = ref

            debug['image_lds'] = draw_lds(image, image_lds['value'])
            debug['ref_lds'] = draw_lds(ref, ref_lds['value'])
            debug['deformation0'] = (deformed_ref_img[0:1, 0, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation1'] = (deformed_ref_img[0:1, 1, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation2'] = (deformed_ref_img[0:1, 2, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation3'] = (deformed_ref_img[0:1, 3, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation4'] = (deformed_ref_img[0:1, 4, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation5'] = (deformed_ref_img[0:1, 5, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation6'] = (deformed_ref_img[0:1, 6, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation7'] = (deformed_ref_img[0:1, 7, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation8'] = (deformed_ref_img[0:1, 8, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation9'] = (deformed_ref_img[0:1, 9, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation10'] = (deformed_ref_img[0:1, 10, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation11'] = (deformed_ref_img[0:1, 11, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation12'] = (deformed_ref_img[0:1, 12, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation13'] = (deformed_ref_img[0:1, 13, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0

            debug['deformation14'] = (deformed_ref_img[0:1, 14, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation15'] = (deformed_ref_img[0:1, 15, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation16'] = (deformed_ref_img[0:1, 16, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation17'] = (deformed_ref_img[0:1, 17, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
            debug['deformation18'] = (deformed_ref_img[0:1, 18, :, :, :].cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) + 1.0) * 255.0 / 2.0
          # dp_seg_img = mask.cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0)
            # debug['seg_img'] = cv2.applyColorMap((dp_seg_img * 30).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :] * 0.7 + image * 0.3 * 255.0
            self.opt.v.display_current_results(debug, 0)

        warped_ref = F.grid_sample(ref_img, deformation)
        corr_out['warped_ref_stage1'] = warped_ref

        warped_semantics = F.grid_sample(ref_seg_map, deformation)
        corr_out['warped_semantics'] = warped_semantics

        src_out = []
        tar_out = []

        src_o = seg_map
        tar_o = torch.cat((warped_semantics, warped_ref), dim=1)

        for l in range(self.num_scales):
            src_o = self.__getattr__('src_%d_down'%l)(src_o)
            for i in range(self.n_res_blocks):
                src_o = self.__getattr__('src_%d_res_%d'%(l, i))(src_o)

            src_out.append(src_o)

        for l in range(self.num_scales):
            tar_o = self.__getattr__('tar_%d_down' % l)(tar_o)
            for i in range(self.n_res_blocks):
                tar_o = self.__getattr__('tar_%d_res_%d' % (l, i))(tar_o)

            tar_out.append(tar_o)

        rpn_src_out = []
        rpn_tar_out = []
        src_pre = None
        tar_pre = None

        for l in range(self.num_scales, 0, -1):
            src_o = src_out[l-1]

            src_o = self.__getattr__('src_rpn_%d_conv1'%(l-1))(src_o)

            if l == self.num_scales:
                src_o = src_o
            else:
                src_o = src_o + src_pre

            src_pre = self.upsampling_bi(src_o)

            rpn_src_o = self.__getattr__('src_rpn_%d_conv3'%(l-1))(src_o)
            rpn_src_o = util.feature_normalize(rpn_src_o)

            if self.opt.use_coordconv:
                rpn_src_o = self.addcoords(rpn_src_o)

            rpn_src_out.append(rpn_src_o)

        for l in range(self.num_scales, 0, -1):
            tar_o = tar_out[l - 1]

            tar_o = self.__getattr__('tar_rpn_%d_conv1' % (l - 1))(tar_o)

            if l == self.num_scales:
                tar_o = tar_o
            else:
                tar_o = tar_o + tar_pre

            tar_pre = self.upsampling_bi(tar_o)

            rpn_tar_o = self.__getattr__('tar_rpn_%d_conv3' % (l - 1))(tar_o)
            rpn_tar_o = util.feature_normalize(rpn_tar_o)

            if self.opt.use_coordconv:
                rpn_tar_o = self.addcoords(rpn_tar_o)

            rpn_tar_out.append(rpn_tar_o)

        rpn_src_out = rpn_src_out[::-1]
        rpn_tar_out = rpn_tar_out[::-1]

        record_flows = []
        pred_flow = None
        for l in range(self.num_scales, 0, -1):
            src_f = rpn_src_out[l - 1]
            tar_f = rpn_tar_out[l - 1]

            if l == self.num_scales:
                input_f = torch.cat((tar_f, src_f), 1)
                pred_flow = self.__getattr__('flow_predict_%d' % (l - 1))(input_f)

            else:
                b, c, h, w = tar_f.size()
                grid = make_coordinate_grid((h, w), tar_f.type())
                grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
                pred_flow = F.interpolate(pred_flow, (h, w), mode='bilinear').permute(0, 2, 3, 1)

                grid = grid + pred_flow
                tar_f = F.grid_sample(tar_f, grid)
                input_f = torch.cat((tar_f, src_f), 1)

                pred_flow = self.__getattr__('flow_predict_%d'%(l-1))(input_f)

            record_flows.append(pred_flow)

        tb, tc, th, tw = ref_img.size()

        final_flow = torch.zeros((tb, 2, th, tw)).cuda()
        for flow in record_flows:
            pred_flow = F.interpolate(flow, (th, tw), mode='bilinear')
            final_flow += pred_flow

        flow_x = (2*final_flow[:, 0, :, :] / (tw - 1)).view(tb, 1, th, tw)
        flow_y = (2*final_flow[:, 1, :, :] / (th - 1)).view(tb, 1, th, tw)
        flow = torch.cat((flow_x, flow_y), 1)

        flow = flow.permute(0, 2, 3, 1)
        flow += deformation
#         flow = pred_flow
        
        corr_out['flow'] = flow
        corr_out['record_flow'] = record_flows

        # grid = make_coordinate_grid((th, tw), ref_img.type())
        # grid = grid.unsqueeze(0).repeat(tb, 1, 1, 1)
        # grid = grid + flow.permute(0, 2, 3, 1)

        warped_ref = F.grid_sample(ref_img, flow)
        corr_out['warp_out'] = warped_ref

        ref_mask = test_mask['ref_seg'].detach()
        t_ref_mask = self.mask_convert(ref_mask)
        warped_ref_mask = F.grid_sample(t_ref_mask, flow)
        corr_out['warp_ref_seg'] = warped_ref_mask

        return corr_out


class C_DownBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d, start=False):
        super(C_DownBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.padding = nn.ReflectionPad2d(1)
        self.conv = get_conv(input_nc, output_nc, kernel_size=4, padding=0, stride=2, norm_layer=norm_layer)

    def forward(self, x):
        out = self.padding(x)
        out = self.conv(out)
        out = self.activation(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.InstanceNorm2d):
        super(ConvBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.padding = nn.ReflectionPad2d(1)
        self.conv = get_conv(input_nc, output_nc, kernel_size=3, padding=0, norm_layer=norm_layer)

    def forward(self, x):
        out = self.padding(x)
        out = self.conv(out)
        out = self.activation(out)
        return out


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_dim=6, use_cuda=True, input_s=(16, 16)):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Linear(64 * int(input_s[0]/4) * int(input_s[1]/4), output_dim) # for warping clothes
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)            # torch.Size([1, 64, 16, 12])
        x = x.reshape(x.size()[0], -1)   # torch.Size([1, 12288])
        # print("x.view:", x.shape)
        x = self.linear(x)          # torch.Size([1, 32])
        x = self.tanh(x)            # torch.Size([1, 32])

        return x


class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = Variable(self.grid_X.cuda())
            self.grid_Y = Variable(self.grid_Y.cuda())

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            P_X = Variable(P_X.cuda())
            P_Y = Variable(P_Y.cuda())

            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()
                # self.P_X_base = Variable(self.P_X_base)
                # self.P_Y_base = Variable(self.P_Y_base)

    def forward(self, theta):
        # gpu_id = theta.get_device()
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = Variable(torch.FloatTensor(N, 1).fill_(1).cuda())
        Z = Variable(torch.FloatTensor(3, 3).fill_(0).cuda())
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        # print(Q_X.get_device(), self.P_X_base.get_device())
        # print("*" * 10)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        # tmp = dist_squared == 0
        # print(dist_squared.get_device(), tmp.get_device())
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


class Warpnet(BaseNetwork):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()
        if opt.use_coordconv:
            self.input_channel = opt.semantic_nc + 3
        else:
            self.input_channel = opt.semantic_nc

        self.feature_channel = 64
        n_res_blocks = 2
        n_conv_block = 4
        nf = 64
        max_nf = 256

        self.temperature = 0.001
        self.num_scales = 4
        self.n_res_blocks = n_res_blocks
        self.label_nc = 20
        self.opt.down = 4

        if self.opt.dataset_mode == 'posetemporal':
            height = 256
            width = 192
            input_Rgr = 192
            self.fold_size = (256, 192)
            self.input_s = (16, 12)
        else:
            height = 256
            width = 256
            input_Rgr = 256
            self.fold_size = 256
            self.input_s = (16, 16)

        self.feature_img_extractor = tps_feature_extractor(self.opt, \
                                                       self.input_channel, self.feature_channel)

        if self.opt.Corr_mode == 'double_dis':
            self.feature_ref_extractor = tps_feature_extractor(self.opt, \
                                                  self.input_channel, self.feature_channel)
        elif self.opt.Corr_mode == 'dis_img':
            self.feature_ref_extractor = tps_feature_extractor(self.opt, \
                                                  3, self.feature_channel)

        else:
            self.feature_ref_extractor = tps_feature_extractor(self.opt, \
                                                  self.input_channel + 3, self.feature_channel)

        # self.src_0_down = C_DownBlock(self.label_nc, nf * 1, start=True)
        # self.src_0_conv = ConvBlock(nf * 1, nf * 2)
        #
        # for i in range(self.n_res_blocks):
        #     self.__setattr__('src_%d_res_%d' % (0, i), ResidualBlock(nf * 2, nf * 2))
        #
        # for l in range(1, self.num_scales):
        #     c_in = min(nf * 2 ** (l), max_nf)
        #     c_out = min(nf * 2 ** (l + 1), max_nf)
        #     c_out1 = min(nf * 2 ** (l + 2), max_nf)
        #
        #     self.__setattr__('src_%d_down' % l, C_DownBlock(c_in, c_out))
        #     self.__setattr__('src_%d_conv' % l, ConvBlock(c_out, c_out1))
        #
        #     for i in range(n_res_blocks):
        #         self.__setattr__('src_%d_res_%d' % (l, i), ResidualBlock(c_out1, c_out1))
        #
        # self.tar_0_down = C_DownBlock(self.label_nc + 3, nf * 1, start=True)
        # self.tar_0_conv = ConvBlock(nf * 1, nf * 2)
        #
        # for i in range(self.n_res_blocks):
        #     self.__setattr__('tar_%d_res_%d' % (0, i), ResidualBlock(nf * 2, nf * 2))
        #
        # for l in range(1, self.num_scales):
        #     c_in = min(nf * 2 ** (l), max_nf)
        #     c_out = min(nf * 2 ** (l + 1), max_nf)
        #     c_out1 = min(nf * 2 ** (l + 2), max_nf)
        #
        #     self.__setattr__('tar_%d_down' % l, C_DownBlock(c_in, c_out))
        #     self.__setattr__('tar_%d_conv' % l, ConvBlock(c_out, c_out1))
        #
        #     for i in range(n_res_blocks):
        #         self.__setattr__('tar_%d_res_%d' % (l, i), ResidualBlock(c_out1, c_out1))

        # for l in range(1, self.num_scales):
        #     c_in = min(nf * 2 ** l, max_nf)
        #     c_out = 256
        #
        #     conv1_1 = get_conv(c_in, c_out, kernel_size=1, stride=1, padding=0)
        #     conv3_3 = get_conv(c_out, c_out, kernel_size=3, stride=1, padding=1)
        #
        #     self.__setattr__('src_rpn_%d_conv1' % l, conv1_1)
        #     self.__setattr__('src_rpn_%d_conv3' % l, conv3_3)
        #
        # for l in range(1, self.num_scales):
        #     c_in = min(nf * 2 ** l, max_nf)
        #     c_out = 256
        #
        #     conv1_1 = get_conv(c_in, c_out, kernel_size=1, stride=1, padding=0)
        #     conv3_3 = get_conv(c_out, c_out, kernel_size=3, stride=1, padding=1)
        #
        #     self.__setattr__('tar_rpn_%d_conv1' % l, conv1_1)
        #     self.__setattr__('tar_rpn_%d_conv3' % l, conv3_3)

        self.upsampling_bi = nn.Upsample(scale_factor=2, mode='bilinear')

        grid_size = self.opt.grid_size

        self.grid_size = grid_size
        self.regression = FeatureRegression(input_nc=input_Rgr, output_dim=2 * grid_size ** 2, use_cuda=True, input_s=self.input_s)
        self.gridGen = TpsGridGen(height, width, use_cuda=True, grid_size=grid_size)
        self.gridGen = nn.DataParallel(self.gridGen, device_ids=self.opt.gpu_ids)  ## fixed bug for multiGPU
        # (会报错RuntimeError: arguments are located on different GPUs at /home/donghaoye/pytorc)
        self.P_X_base = self.gridGen.module.P_X_base.data.squeeze().cpu()
        self.P_Y_base = self.gridGen.module.P_Y_base.data.squeeze().cpu()


    def make_coordinate_grid(self, spatial_size, type):
        """
        Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
        """
        h, w = spatial_size
        x = torch.arange(w).type(type)
        y = torch.arange(h).type(type)

        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)

        yy = y.view(-1, 1).repeat(1, w)
        xx = x.view(1, -1).repeat(h, 1)

        meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

        return meshed

    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1
        xx_channel = xx_channel.float()
        yy_channel = yy_channel.float()

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))

        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat

    def mask_convert(self, seg, c=7):
        size = seg.size()
        oneHot_size = (size[0], c, size[2], size[3])
        c_seg = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_seg_tensor = c_seg.scatter_(1, seg.data.long().cuda(), 1.0)

        return c_seg_tensor

    def forward(self,
                ref_img,
                real_img,
                seg_map,
                ref_seg_map,
                detach_flag=False,
                return_corr=False,
                test_mask=None,
                mode='clothflow'):

        corr_out = {}
        src_out = []
        tar_out = []

        input_img = seg_map
        if self.opt.Corr_mode == 'double_dis':
            input_ref = ref_seg_map
        elif self.opt.Corr_mode == 'dis_img':
            input_ref = ref_img
        else:
            input_ref = torch.cat([ref_seg_map, ref_img], dim=1)

        batch_size = ref_img.shape[0]
        image_height = ref_img.shape[2]
        image_width = ref_img.shape[3]
        feature_height = int(image_height / self.opt.down)
        feature_width = int(image_width / self.opt.down)

        # seg_img = test_mask['seg_img']
        ref_seg = test_mask['ref_seg'].detach()

        # real_mask = self.mask_convert(seg_img)
        # lower_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 5).astype(np.float))
        # upper_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 4).astype(np.float))

        if self.opt.dataset_mode == 'deepfashion' or self.opt.dataset_mode == 'posetemporal':
            t_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 5).astype(np.float))
            ref_cl_mask = ref_seg * (1 - t_mask) + t_mask * 4
            ref_cl_mask[ref_cl_mask != 4] = 0
            ref_cl_mask[ref_cl_mask == 4] = 1
        elif self.opt.dataset_mode == 'viton':
            ref_cl_mask = ref_seg

        corr_out['cl_mask'] = ref_cl_mask
        # c_ref_cl_mask = self.mask_convert(ref_mask, 2)

        masked_ref = ref_img * ref_cl_mask
        corr_out['masked_ref'] = masked_ref
        if self.opt.predict_mode == 'k' or self.opt.predict_mode == 'k+dp' \
                or self.opt.predict_mode == 'k+dp+tps' or self.opt.predict_mode == 'dual_U':
            t_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 1).astype(np.float))
            no_head_mask = ref_seg * (1 - t_mask) + t_mask * 0

            t_mask = torch.cuda.FloatTensor((ref_seg.cpu().numpy() == 6).astype(np.float))
            no_head_and_shoes_mask = no_head_mask * (1 - t_mask) + t_mask * 0
            c_no_head_and_shoes_mask = self.mask_convert(no_head_and_shoes_mask)
            corr_out['no_head_and_shoes_mask'] = no_head_and_shoes_mask

            ref_mask = c_no_head_and_shoes_mask
        elif self.opt.predict_mode == 'cloth+k':
            ref_mask = ref_cl_mask

        feature_img, tps_img = self.feature_img_extractor(input_img)
        feature_ref, tps_ref = self.feature_ref_extractor(input_ref)

        adaptive_feature_img = util.feature_normalize(feature_img)
        adf_img = F.unfold(adaptive_feature_img, \
                           kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
        adf_img_t = adf_img.permute(0, 2, 1)

        adaptive_feature_ref = util.feature_normalize(feature_ref)

        adf_ref = F.unfold(adaptive_feature_ref, \
                           kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))

        f = torch.matmul(adf_img_t, adf_ref)

        corr_out['M_corr'] = f.detach()

        if self.opt.freeze_warp:
            f = f.detach()
            f.requires_grad_()

        f_WTA = f / self.temperature
        f_div_C = F.softmax(f_WTA, dim=-1)

        # c_ref_seg = self.mask_convert(ref_seg)
        # coor_out['c_ref_seg'] = c_ref_seg

        # if self.opt.use_grid_warp or self.opt.use_grid_loss:
        #     # input_f = f_div_C.unsqueeze(1)
        #     fb, fh, fw = f.size()
        #     input_f = f.reshape(fb, fh, 16, 16)
        #     if self.opt.use_coordconv:
        #         input_f = self.addcoords(input_f)
        #
        #     pred_flow = self.__getattr__('flow_predict_3')(input_f)
        #
        #     pred_flow = F.interpolate(pred_flow, 256, mode='bilinear')
        #     g = pred_flow
        #     corr_out['flow'] = pred_flow

        if self.opt.use_grid_warp:
            g = g.permute(0, 2, 3, 1)
            y = F.grid_sample(ref_img, g)
            warp_masked_ref = F.grid_sample(masked_ref, g)
            warp_ref_seg = F.grid_sample(ref_mask, g)
        else:
            ref = F.unfold(ref_img, self.opt.down, stride=self.opt.down)
            ref = ref.permute(0, 2, 1)

            c_ref_seg = F.unfold(ref_mask, self.opt.down, stride=self.opt.down)
            c_ref_seg = c_ref_seg.permute(0, 2, 1)

            masked_ref = F.unfold(masked_ref, self.opt.down, stride=self.opt.down)
            masked_ref = masked_ref.permute(0, 2, 1)

            if self.opt.use_max_Corr:
                assert f_div_C.size()[0] == 1
                max_ind = torch.max(f_div_C.squeeze(), dim=1)[1]
                y = ref[:, max_ind, :]
                warp_masked_ref = masked_ref[:, max_ind, :]
                warp_ref_seg = c_ref_seg[:, max_ind, :]
            else:
                y = torch.matmul(f_div_C, ref)
                warp_masked_ref = torch.matmul(f_div_C, masked_ref)
                warp_ref_seg = torch.matmul(f_div_C, c_ref_seg)

            y = y.permute(0, 2, 1)
            y = F.fold(y, self.fold_size, self.opt.down, stride=self.opt.down)

            warp_masked_ref = warp_masked_ref.permute(0, 2, 1)
            warp_masked_ref = F.fold(warp_masked_ref, self.fold_size, self.opt.down, stride=self.opt.down)

            warp_ref_seg = warp_ref_seg.permute(0, 2, 1)
            warp_ref_seg = F.fold(warp_ref_seg, self.fold_size, self.opt.down, stride=self.opt.down)

        corr_out['warp_out'] = y
        corr_out['warp_masked_ref'] = warp_masked_ref
        corr_out['warp_ref_seg'] = warp_ref_seg

        if self.opt.isTrain and self.opt.warp_cycle_w > 0:
            f_div_C_v = F.softmax(f_WTA.transpose(1, 2), dim=-1)
            o_warp_mased_ref = F.unfold(warp_masked_ref, self.opt.down, stride=self.opt.down)
            o_warp_mased_ref = o_warp_mased_ref.permute(0, 2, 1)
            cycle_masked_ref = torch.matmul(f_div_C_v, o_warp_mased_ref)
            cycle_masked_ref = cycle_masked_ref.permute(0, 2, 1)
            cycle_masked_ref = F.fold(cycle_masked_ref, self.fold_size, self.opt.down, stride=self.opt.down)
            corr_out['cycle_masked_ref'] = cycle_masked_ref

        if self.opt.use_ds_tps_corr:
            adf_geo_src_f = F.unfold(adaptive_feature_img, \
                                     kernel_size=4, stride=4)
            adf_geo_src_f_t = adf_geo_src_f.permute(0, 2, 1)

            adf_geo_tar_f = F.unfold(adaptive_feature_ref, \
                                     kernel_size=4, stride=4)

        else:
            geo_src_f = tps_img
            geo_tar_f = tps_ref

            adaptive_geo_src_f = util.feature_normalize(geo_src_f)
            adf_geo_src_f = F.unfold(adaptive_geo_src_f, \
                               kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
            adf_geo_src_f_t = adf_geo_src_f.permute(0, 2, 1)

            adaptive_geo_tar_f = util.feature_normalize(geo_tar_f)
            adf_geo_tar_f = F.unfold(adaptive_geo_tar_f, \
                               kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))

        geo_corr = torch.matmul(adf_geo_src_f_t, adf_geo_tar_f)

        if self.opt.freeze_geo_corr:
            geo_corr = geo_corr.detach()

        geo_corr = geo_corr / self.temperature
        geo_corr = F.softmax(geo_corr, dim=-1)

        # geo_corr_WTA = geo_corr / self.temperature
        # ref_geo = F.unfold(ref_img, 16, stride=16)
        # ref_geo = ref_geo.permute(0, 2, 1)

        gcorr_b, gcorr_h, gcorr_w = geo_corr.size()

        geo_corr = geo_corr.reshape(gcorr_b, gcorr_h, self.input_s[0], self.input_s[1])

        theta = self.regression(geo_corr)
        warped_grid = self.gridGen(theta)

        Q_X = theta[:, :(self.opt.grid_size * self.opt.grid_size)].unsqueeze(-1)
        Q_Y = theta[:, (self.opt.grid_size * self.opt.grid_size):].unsqueeze(-1)
        reg_grid = torch.cat([Q_X, Q_Y], dim=-1)
        corr_out['warped_grid'] = reg_grid

        vis_grid = drawConvas(theta, self.P_X_base, self.P_Y_base, grid_size=self.grid_size)
        corr_out['vis_grid'] = vis_grid.type(torch.cuda.FloatTensor)

        TPS_warped_ref = F.grid_sample(ref_img, warped_grid)
        corr_out['TPS_warp_out'] = TPS_warped_ref

        TPS_warped_cl_mask = F.grid_sample(ref_cl_mask, warped_grid)
        corr_out['TPS_warp_cl_mask'] = TPS_warped_cl_mask

        TPS_warped_masked_ref = F.grid_sample(corr_out['masked_ref'], warped_grid)
        corr_out['TPS_warp_masked_ref'] = TPS_warped_masked_ref

        return corr_out


def drawConvas(theta, P_X, P_Y, grid_size=4):
    width = 256
    height = 256
    # grid_size = opt.grid_size
    N = grid_size * grid_size
    convas = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(convas)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    Q_X = theta[0, :N]
    Q_Y = theta[0, N:]
    r = 3
    draw.rectangle(((width - 1) / 4.0, (height - 1) / 4.0, (width - 1) * 3.0 / 4.0, (height - 1) * 3.0 / 4.0))
    for i in range(grid_size):
        for j in range(grid_size):
            qx = (Q_X[i * grid_size + j] + P_X[i * grid_size + j]) * (width - 1) / 4.0 + (width - 1) / 2.0
            qy = (Q_Y[i * grid_size + j] + P_Y[i * grid_size + j]) * (height - 1) / 4.0 + (height - 1) / 2.0

            if j > 0:
                qx_prev = (Q_X[i * grid_size + j - 1] + P_X[i * grid_size + j - 1]) * (width - 1) / 4.0 + (width - 1) / 2.0
                qy_prev = (Q_Y[i * grid_size + j - 1] + P_Y[i * grid_size + j - 1]) * (height - 1) / 4.0 + ( height - 1) / 2.0
                draw.line((qx_prev, qy_prev, qx, qy), 'red', width=r)
            if i > 0:
                qx_prev = (Q_X[(i - 1) * grid_size + j] + P_X[(i - 1) * grid_size + j]) * (width - 1) / 4.0 + (width - 1) / 2.0
                qy_prev = (Q_Y[(i - 1) * grid_size + j] + P_Y[(i - 1) * grid_size + j]) * (height - 1) / 4.0 + (height - 1) / 2.0
                draw.line((qx_prev, qy_prev, qx, qy), 'red', width=r)
            draw.ellipse((qx - r, qy - r, qx + r, qy + r), 'white', 'white')

    convas = trans(convas)

    return convas