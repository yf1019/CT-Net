import torch.nn as nn
import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util
import numpy as np
import cv2
from util.util import make_coordinate_grid


class CTModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        parser.add_argument('--down', default=4, help='down sample in correspondence layer')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # set loss functions
        if opt.isTrain:
            if opt.vggloss:
                self.vggnet_fix = networks.correspondence.VGG19_feature_color_torchversion(
                    vgg_normal_correct=opt.vgg_normal_correct)
                self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
                self.vggnet_fix.eval()
                for param in self.vggnet_fix.parameters():
                    param.requires_grad = False

                self.vggnet_fix.to(self.opt.gpu_ids[0])
                self.contextual_forward_loss = networks.ContextualLoss_forward(opt)

                if opt.which_perceptual == '5_2':
                    self.perceptual_layer = -1
                elif opt.which_perceptual == '4_2':
                    self.perceptual_layer = -2

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=torch.cuda.FloatTensor , opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.MSE_loss = torch.nn.MSELoss()
            self.BCE = torch.nn.BCEWithLogitsLoss()
            self.No_D = False
            self.criterionMaskL1 = lambda x: 1 - x.mean()

    def forward(self, data, mode, GforD=None, alpha=1):
        input_label, input_semantics, real_image, \
            self_ref, ref_image, ref_label, ref_semantics, seg = self.preprocess_input(data)

        generated_out = {}
        if mode == 'generator':
            g_loss, generated_out = self.compute_generator_loss(input_label,
                                                                    input_semantics, real_image, ref_label,
                                                                    ref_semantics, ref_image, self_ref, seg)

            out = {}
            data['label'] = input_semantics
            data['label_ref'] = ref_semantics
            out['mask_refine'] = None if 'mask_refine' not in generated_out else generated_out['mask_refine']
            out['fake_result'] = None if 'fake_result' not in generated_out else generated_out['fake_result']
            out['initial_fake_image'] = None if 'initial_fake_image' not in generated_out else generated_out['initial_fake_image']
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            out['warp_masked_ref'] = None if 'warp_masked_ref' not in generated_out else generated_out['warp_masked_ref']
            out['warp_ref_seg'] = None if 'warp_ref_seg' not in generated_out else generated_out[
                'warp_ref_seg']
            out['mask_refine'] = None if 'mask_refine' not in generated_out else generated_out[
                'mask_refine']
            out['warp_cycle'] = None if 'warp_cycle' not in generated_out else generated_out['warp_cycle']
            out['input_cl'] = None if 'input_cl' not in generated_out else generated_out['input_cl']
            out['input_G1'] = None if 'input_G1' not in generated_out else generated_out['input_G1']

            out['input_G2'] = None if 'input_G2' not in generated_out else generated_out['input_G2']
            out['input_G3'] = None if 'input_G3' not in generated_out else generated_out['input_G3']
            out['real_mask'] = None if 'real_mask' not in generated_out else generated_out['real_mask']
            out['clothes_mask'] = None if 'clothes_mask' not in generated_out else generated_out['clothes_mask']
            out['real_cloth_mask'] = None if 'real_cloth_mask' not in generated_out else generated_out['real_cloth_mask']
            data['label'] = input_semantics
            data['label_ref'] = ref_semantics
            out['image_hole'] = None if 'image_hole' not in generated_out else generated_out['image_hole']
            out['refined_clothes'] = None if 'refined_clothes' not in generated_out else generated_out['refined_clothes']
            out['masked_warped_cl'] = None if 'masked_warped_cl' not in generated_out else generated_out['masked_warped_cl']
            out['id_warp_out'] = None if 'id_warp_out' not in generated_out else generated_out['id_warp_out']
            out['warp_i_masked_ref'] = None if 'warp_i_masked_ref' not in generated_out else generated_out['warp_i_masked_ref']
            out['skin_color'] = None if 'skin_color' not in generated_out else generated_out['skin_color']
            out['warped_ref_stage1'] = None if 'warped_ref_stage1' not in generated_out else generated_out['warped_ref_stage1']
            out['flow'] = None if 'flow' not in generated_out else generated_out['flow']
            out['deformation'] = None if 'deformation' not in generated_out else generated_out['deformation']
            out['no_head_seg_img'] = None if 'no_head_seg_img' not in generated_out else generated_out['no_head_seg_img']
            out['masked_ref'] = None if 'masked_ref' not in generated_out else generated_out['masked_ref']
            out['refine_gt'] = None if 'refine_gt' not in generated_out else generated_out['refine_gt']
            out['cycle_masked_ref'] = None if 'cycle_masked_ref' not in generated_out else generated_out['cycle_masked_ref']
            out['TPS_warp_out'] = None if 'TPS_warp_out' not in generated_out else generated_out['TPS_warp_out']
            out['TPS_warp_cl_mask'] = None if 'TPS_warp_cl_mask' not in generated_out else generated_out['TPS_warp_cl_mask']
            out['TPS_warp_masked_ref'] = None if 'TPS_warp_masked_ref' not in generated_out else generated_out['TPS_warp_masked_ref']
            out['occlusion_tps'] = None if 'occlusion_tps' not in generated_out else generated_out['occlusion_tps']
            out['TPS_refine'] = None if 'TPS_refine' not in generated_out else generated_out['TPS_refine']
            out['vis_grid'] = None if 'vis_grid' not in generated_out else generated_out['vis_grid']
            out['warp_flow'] = None if 'warp_flow' not in generated_out else generated_out['warp_flow']

            return g_loss, out

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, GforD, label=input_label)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                out = self.inference(input_semantics, ref_semantics, ref_image, self_ref, real_image, seg)

            generated_out = {}
            data['label'] = input_semantics
            data['label_ref'] = ref_semantics
            generated_out['fake_result'] = None if 'fake_result' not in out else out['fake_result']

            return generated_out
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params, D_params = list(), list()

        if opt.use_mask_refine:
            G_params += [{'params': self.net['netG1'].parameters(), 'lr': opt.lr * 0.5}]
        if opt.max_attn:
            G_params += [{'params': self.net['netG3'].parameters(), 'lr': opt.lr * 0.5}]

        if opt.use_G2:
            G_params += [{'params': self.net['netG2'].parameters(), 'lr': opt.lr * 0.5}]
        G_params += [{'params': self.net['netCorr'].parameters(), 'lr': opt.lr * 0.5}]

        if opt.isTrain:
            if opt.use_mask_refine and self.opt.use_D1:
                D_params += list(self.net['netD1'].parameters())

            if opt.use_G2:
                D_params += list(self.net['netD2'].parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)

        if len(D_params) == 0:
            self.no_D = True
            optimizer_D = None
        else:
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        if self.opt.use_mask_refine:
            util.save_network(self.net['netG1'], 'G1', epoch, self.opt)

            if self.opt.use_D1:
                util.save_network(self.net['netD1'], 'D1', epoch, self.opt)
        if self.opt.max_attn:
            util.save_network(self.net['netG3'], 'G3', epoch, self.opt)

        if self.opt.use_G2:
            util.save_network(self.net['netG2'], 'G2', epoch, self.opt)
            util.save_network(self.net['netD2'], 'D2', epoch, self.opt)

        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        net = {}
        if self.opt.use_mask_refine:
            net['netG1'] = networks.define_G1(opt, opt.G1_inc, opt.G1_oc)
            if self.opt.use_D1:
                net['netD1'] = networks.define_D1(opt, opt.D1_inc, 1) if opt.isTrain else None

        if self.opt.max_attn:
            net['netG3'] = networks.define_G3(opt, 42, 2)

        if self.opt.use_G2:
            if self.opt.G2_mode == 'tps':
                net['netG2'] = networks.define_G2(opt, opt.G2_inc, 3, occlusion_mask=True)
            else:
                net['netG2'] = networks.define_G2(opt, opt.G2_inc, 3)

            net['netD2'] = networks.define_D2(opt, opt.D2_inc, 1) if opt.isTrain else None
        net['netCorr'] = networks.define_Corr(opt)

        lastest_epoch = opt.restore_epoch

        if not opt.isTrain or opt.continue_train:
            if self.opt.use_mask_refine:
                net['netG1'] = util.load_network(net['netG1'], 'G1', lastest_epoch, opt)

            if self.opt.use_G2:
                net['netG2'] = util.load_network(net['netG2'], 'G2', lastest_epoch, opt)
            if opt.isTrain:
                if self.opt.use_mask_refine and self.opt.use_D1:
                    net['netD1'] = util.load_network(net['netD1'], 'D1', lastest_epoch, opt)
                if self.opt.use_G2:
                    net['netD2'] = util.load_network(net['netD2'], 'D2', lastest_epoch, opt)

            if self.opt.max_attn:
                net['netG3'] = util.load_network(net['netG3'], 'G3', lastest_epoch, opt)

            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', lastest_epoch, opt)
        return net

    def preprocess_input(self, data):
        input_semantics = data['label'].clone().cuda().float()
        data['label'] = data['label'][:, :3, :, :]
        ref_semantics = data['label_ref'].clone().cuda().float()
        data['label_ref'] = data['label_ref'][:, :3, :, :]

        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['ref'] = data['ref'].cuda()
            data['label_ref'] = data['label_ref'].cuda()
            if self.opt.dataset_mode != 'deepfashion':
                data['label_ref'] = data['label_ref'].long()
            data['self_ref'] = data['self_ref'].cuda()

        seg = {}
        seg['seg_img'] = data['seg_img'].cuda()
        seg['ref_seg'] = data['ref_seg'].cuda()

        if self.opt.use_mask:
            seg['M_tensor'] = data['M_tensor'].cuda()

        if self.opt.use_dp:
            seg['img_dp_mask'] = data['img_dp_mask'].cuda()
            seg['ref_dp_mask'] = data['ref_dp_mask'].cuda()

            if self.opt.use_cl_refine and self.opt.refine_type == 'densepose':
                seg['warped_cloth'] = data['warped_cloth'].cuda()

        if self.opt.load_lds:
            seg['image_lds'] = {}
            seg['image_lds']['value'] = data['image_lds']['value'].cuda()
            seg['image_lds']['flag'] = data['image_lds']['flag'].cuda()

            seg['ref_lds'] = {}
            seg['ref_lds']['value'] = data['ref_lds']['value'].cuda()
            seg['ref_lds']['flag'] = data['ref_lds']['flag'].cuda()

        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data[
            'label_ref'], ref_semantics, seg

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(
            self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        if self.opt.use_22ctx:
            contextual_style2_1 = torch.mean(
                self.contextual_forward_loss(F.avg_pool2d(source[-4], 4), F.avg_pool2d(target[-4].detach(), 4))) * 1
            return contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.permute(0, 2, 3, 1).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, target, weight=weight)

        return loss

    def compute_flow_loss(self, flow):
        n, c, h, w = flow.size()

        flow_x = flow[:, 0, :, :]
        flow_y = flow[:, 1, :, :]

        Rx_00 = flow_x[:, :-1, :-1]
        Rx_11 = flow_x[:, 1:, 1:]

        Ry_01 = flow_y[:, :-1, 1:]
        Ry_10 = flow_y[:, 1:, :-1]

        x_loss = torch.abs(Rx_00 - Rx_11)
        y_loss = torch.abs(Ry_01 - Ry_10)

        t_loss = torch.mean(x_loss + y_loss)
        return t_loss

    def compute_generator_loss(self, input_label, input_semantics, real_image, ref_label=None, ref_semantics=None,
                               ref_image=None, self_ref=None, seg=None):
        G_losses = {}
        generate_out = self.generate_fake(
            input_semantics, real_image, ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref, seg=seg)

        seg_img = seg['seg_img'].clone()
        ref_seg = seg['ref_seg'].clone()

        if self.opt.dataset_mode == 'deepfashion' or self.opt.dataset_mode == 'posetemporal':
            real_cloth_mask = self.fuse_mask(seg_img, [5], 4)
            real_cloth_mask[real_cloth_mask != 4] = 0
            real_cloth_mask[real_cloth_mask == 4] = 1

            ref_cloth_mask = self.fuse_mask(ref_seg, [5], 4)
            ref_cloth_mask[ref_cloth_mask != 4] = 0
            ref_cloth_mask[ref_cloth_mask == 4] = 1
        elif self.opt.dataset_mode == 'viton':
            real_cloth_mask = seg_img.clone()
            real_cloth_mask[seg_img != 4] = 0
            real_cloth_mask[seg_img == 4] = 1

            ref_cloth_mask = ref_seg.clone()

        generate_out['real_cloth_mask'] = real_cloth_mask

        real_mask = self.mask_convert(seg_img)
        generate_out['real_mask'] = real_mask

        if self.opt.predict_mode == 'k' or self.opt.predict_mode == 'k+dp' \
                or self.opt.predict_mode == 'k+dp+tps' or self.opt.predict_mode == 'dual_U':
            t_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() == 1).astype(np.float))
            no_head_seg_img = seg_img * (1 - t_mask) + t_mask * 0

            t_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() == 6).astype(np.float))
            no_head_and_shoes_seg_img = no_head_seg_img * (1 - t_mask) + t_mask * 0
            c_no_head_and_shoes_seg_img = self.mask_convert(no_head_and_shoes_seg_img)

            refine_gt = no_head_and_shoes_seg_img
            c_refine_gt = c_no_head_and_shoes_seg_img

        elif self.opt.predict_mode == 'cloth+k':
            refine_gt = seg_img
            c_refine_gt = self.mask_convert(seg_img)

        elif self.opt.predict_mode == 'cloth+k+dp':
            refine_gt = real_cloth_mask
            c_refine_gt = refine_gt

        generate_out['refine_gt'] = c_refine_gt

        if self.opt.reg_loss > 0:
            row = self.get_row(generate_out['warped_grid'], self.opt.grid_size)
            col = self.get_col(generate_out['warped_grid'], self.opt.grid_size)
            rg_loss = sum(self.grad_row(generate_out['warped_grid'], self.opt.grid_size))
            cg_loss = sum(self.grad_col(generate_out['warped_grid'], self.opt.grid_size))
            rg_loss = torch.max(rg_loss, torch.tensor(0.02).cuda())
            cg_loss = torch.max(cg_loss, torch.tensor(0.02).cuda())

            rx, ry, cx, cy = torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda() \
                , torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda()
            row_x, row_y = row[:, :, 0], row[:, :, 1]
            col_x, col_y = col[:, :, 0], col[:, :, 1]
            rx_loss = torch.max(rx, row_x).mean()
            ry_loss = torch.max(ry, row_y).mean()
            cx_loss = torch.max(cx, col_x).mean()
            cy_loss = torch.max(cy, col_y).mean()
            reg_loss = torch.mean(rg_loss + cg_loss + rx_loss + ry_loss + cx_loss + cy_loss)
            G_losses['reg_loss'] = reg_loss * self.opt.reg_loss

        if self.opt.tps_mask_loss > 0:
            G_losses['tps_mask_loss'] = F.l1_loss(generate_out['TPS_refine_mask'].detach(), generate_out['occlusion_mask']) * self.opt.tps_mask_loss

        if self.opt.reg_occlusion > 0:
            G_losses['reg_occlusion'] = self.criterionMaskL1(generate_out['occlusion_mask']) * self.opt.reg_occlusion

        if self.opt.tps_warped_cl_mask_bce > 0:
            G_losses['tps_warped_cl_mask_bce'] = self.BCE(generate_out['TPS_warp_cl_mask'], real_cloth_mask) * self.opt.tps_warped_cl_mask_bce

        if self.opt.tps_warped_cl_l1 > 0:
            G_losses['tps_warped_cl_l1'] = F.l1_loss(generate_out['TPS_warp_masked_ref'], real_cloth_mask * real_image) * self.opt.tps_warped_cl_l1

        if self.opt.warp_cycle_w > 0:
            G_losses['l1_warp_cycle'] = F.l1_loss(generate_out['cycle_masked_ref'], ref_image * ref_cloth_mask) * self.opt.warp_cycle_w

        if self.opt.warp_self_w > 0:
            sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(
                -1).unsqueeze(-1)
            G_losses['l1_warp_self'] = torch.mean(F.l1_loss(generate_out['warp_out'], real_image,
                                                           reduce=False) * sample_weights) * self.opt.warp_self_w

        if self.opt.warp_Mask_entropy > 0:
            #             t_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() == 1).astype(np.float))
            #             no_head_seg_img = seg_img * (1 - t_mask) + t_mask * 0
            if generate_out['warp_ref_seg'].size()[1] == 1:
                G_losses['G1_warp_mask_entropy'] = self.BCE(generate_out['warp_ref_seg'], \
                                                            real_cloth_mask) * self.opt.warp_Mask_entropy
            else:
                C_size = generate_out['warp_ref_seg'].size()[1]
                Crossentropy_weight = torch.ones(C_size).cuda().float()
                Crossentropy_weight[2:4] = 10.0
                G_losses['G1_warp_mask_entropy'] = self.cross_entropy2d(generate_out['warp_ref_seg'], \
                                                                        refine_gt.transpose(0, 1)[
                                                                            0].long(), weight=Crossentropy_weight) * self.opt.warp_Mask_entropy

        if self.opt.use_mask_refine:
            if self.opt.Mask_entropy > 0:
                #                 t_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() == 1).astype(np.float))
                #                 no_head_seg_img = seg_img * (1 - t_mask) + t_mask * 0

                if generate_out['mask_refine'].size()[1] == 1:
                    G_losses['G1_mask_entropy'] = self.BCE(generate_out['mask_refine'], \
                                                                   refine_gt) * self.opt.Mask_entropy
                else:
                    C_size = generate_out['mask_refine'].size()[1]
                    Crossentropy_weight = torch.ones(C_size).cuda().float()

                    # gt = self.fuse_mask(refine_gt, [3, 5, 6], 0)
                    G_losses['G1_mask_entropy'] = self.cross_entropy2d(generate_out['mask_refine'], \
                                                                   refine_gt.transpose(0, 1)[
                                                                       0].long(), weight=Crossentropy_weight) * self.opt.Mask_entropy

        if self.opt.clothes_l1 > 0:
            G_losses['cloth_l1'] = torch.mean(F.l1_loss(generate_out['warp_masked_ref'], \
                                                        real_cloth_mask * real_image,
                                                        reduce=False)) * self.opt.clothes_l1

        if self.opt.warp_l1_loss > 0:
            G_losses['G2_l1'] = torch.mean(F.l1_loss(generate_out['initial_fake_image'], \
                                                     real_image,
                                                     reduce=False)) * self.opt.warp_l1_loss

        if self.opt.use_mask_refine and self.opt.use_D1:
            input_D1_fake = torch.cat([generate_out['input_G1'], generate_out['mask_refine']], dim=1)
            input_D1_real = torch.cat([generate_out['input_G1'], c_refine_gt], dim=1)
            pred_D1_fake = self.net['netD1'](input_D1_fake)
            pred_D1_real = self.net['netD1'](input_D1_real)

            G_losses['G1'] = self.criterionGAN(pred_D1_fake, True,
                                               for_discriminator=False) * self.opt.weight_gan
        if self.opt.use_G2:
            input_D2_fake = torch.cat([generate_out['input_G2'], generate_out['initial_fake_image']], dim=1)
            input_D2_real = torch.cat([generate_out['input_G2'], real_image], dim=1)
            pred_D2_fake = self.net['netD2'](input_D2_fake)
            pred_D2_real = self.net['netD2'](input_D2_real)

            G_losses['G2'] = self.criterionGAN(pred_D2_fake, True,
                                           for_discriminator=False) * self.opt.weight_gan

        if not self.opt.no_ganFeat_loss:
            if self.opt.use_mask_refine and self.opt.use_D1:
                num_D1 = len(pred_D1_fake)
                G1_Feat_loss = torch.cuda.FloatTensor(1).fill_(0)
                for i in range(num_D1):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_D1_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(
                            pred_D1_fake[i][j], pred_D1_real[i][j].detach())
                        G1_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D1
                G_losses['G1_Feat'] = G1_Feat_loss

            if self.opt.use_G2:
                num_D2 = len(pred_D2_fake)
                G2_Feat_loss = torch.cuda.FloatTensor(1).fill_(0)
                for i in range(num_D2):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_D2_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(
                            pred_D2_fake[i][j], pred_D2_real[i][j].detach())
                        G2_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D2
                G_losses['G2_Feat'] = G2_Feat_loss

        if self.opt.use_G2:
            if self.opt.vggloss:
                generate_out['ref_features'] = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                               preprocess=True)
                generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                                preprocess=True)
                fake_features = self.vggnet_fix(generate_out['initial_fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                preprocess=True)

                sample_weights = (self_ref[:, 0, 0, 0] / (sum(self_ref[:, 0, 0, 0]) + 1e-5)).unsqueeze(-1).unsqueeze(
                    -1).unsqueeze(-1)
                weights = self.opt.vgg_weights

                loss = 0
                for i in range(len(generate_out['real_features'])):
                    loss += weights[i] * util.weighted_l1_loss(fake_features[i], generate_out['real_features'][i].detach(),
                                                               sample_weights)
                G_losses['i_fm'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio

                if self.opt.G2_mode == 'tps':
                    fake_result_features = self.vggnet_fix(generate_out['fake_result'], ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                    preprocess=True)

                    loss = 0
                    for i in range(len(generate_out['real_features'])):
                        loss += weights[i] * util.weighted_l1_loss(fake_result_features[i], generate_out['real_features'][i].detach(),
                                                                   sample_weights)
                    G_losses['fm'] = loss * self.opt.lambda_vgg * self.opt.fm_ratio

                if self.opt.netCorr == 'clothflow':
                    warp_loss = 0
                    warp_s1_loss = 0

                    warp_features = self.vggnet_fix(generate_out['warp_out'], ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                    preprocess=True)
                    warp_s1_features = self.vggnet_fix(generate_out['warped_ref_stage1'],
                                                       ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)

                    for i in range(len(generate_out['real_features'])):
                        warp_loss += weights[i] * util.weighted_l1_loss(warp_features[i],
                                                                        generate_out['real_features'][i].detach(),
                                                                        sample_weights)
                    G_losses['warp_fm'] = warp_loss * self.opt.lambda_vgg * self.opt.fm_ratio

                    for i in range(len(generate_out['real_features'])):
                        warp_s1_loss += weights[i] * util.weighted_l1_loss(warp_s1_features[i],
                                                                           generate_out['real_features'][i].detach(),
                                                                           sample_weights)
                    G_losses['warp_s1_fm'] = warp_s1_loss * self.opt.lambda_vgg * self.opt.fm_ratio

                if self.opt.cl_perc_w:
                    generate_out['i_fake_cloth_f'] = self.vggnet_fix(generate_out['initial_fake_image'] * real_cloth_mask,
                                                                   ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
                    generate_out['real_cloth_f'] = self.vggnet_fix(real_image * real_cloth_mask,
                                                                   ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)

                    cl_feat_loss = util.mse_loss(generate_out['i_fake_cloth_f'][self.perceptual_layer],
                                                 generate_out['real_cloth_f'][self.perceptual_layer].detach()) * 0.001
                    cl_feat_loss += self.get_ctx_loss(generate_out['i_fake_cloth_f'], generate_out['ref_features']) * self.opt.cl_perc_w
                    G_losses['cl_perc_f'] = cl_feat_loss

                    if self.opt.warp_cycle_w > 0:
                        generate_out['cycle_cloth_f'] = self.vggnet_fix(generate_out['cycle_masked_ref'],
                                                                   ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
                        generate_out['ref_cloth_f'] = self.vggnet_fix(ref_image * ref_cloth_mask,
                                                                       ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                                       preprocess=True)

                        cl_feat_loss = util.mse_loss(generate_out['i_fake_cloth_f'][self.perceptual_layer],
                                                     generate_out['real_cloth_f'][
                                                         self.perceptual_layer].detach())
                        # cl_feat_loss += self.get_ctx_loss(generate_out['i_fake_cloth_f'],
                        #                                   generate_out['ref_features']) * self.opt.cl_perc_w
                        G_losses['cycle_perc_f'] = cl_feat_loss * self.opt.cl_cycle_perc_w

                if self.opt.w_mask_perc:
                    generate_out['mask_refine_f'] = self.vggnet_fix(generate_out['mask_refine'],
                                                                    ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                                    preprocess=True)
                    generate_out['real_mask_f'] = self.vggnet_fix(refine_gt,
                                                                  ['r12', 'r22', 'r32', 'r42', 'r52'],
                                                                  preprocess=True)
                    mask_feat_loss = util.mse_loss(generate_out['mask_refine_f'][self.perceptual_layer],
                                                 generate_out['real_mask_f'][self.perceptual_layer].detach())
                    G_losses['mask_perc_f'] = mask_feat_loss * self.opt.w_mask_perc

                if self.opt.style_loss > 0:
                    s_loss = 0
                    for i in range(len(fake_features)):
                        N, C, H, W = fake_features[i].shape
                        for n in range(N):
                            phi_x = fake_features[i][n]
                            phi_y = generate_out['real_features'][i][n]
                            phi_x = phi_x.reshape(C, H * W)
                            phi_y = phi_y.reshape(C, H * W)
                            G_x = torch.matmul(phi_x, phi_x.t()) / (C * H * W)
                            G_y = torch.matmul(phi_y, phi_y.t()) / (C * H * W)
                            s_loss += torch.sqrt(torch.mean((G_x - G_y) ** 2)) * weights[i]
                    G_losses['i_style_loss'] = s_loss * self.opt.style_loss

                G_losses['i_contextual'] = self.get_ctx_loss(fake_features, generate_out[
                    'ref_features']) * self.opt.lambda_vgg * self.opt.ctx_w
                    
                if self.opt.weight_perceptual > 0:
                    feat_loss = util.mse_loss(fake_features[self.perceptual_layer],
                                              generate_out['real_features'][self.perceptual_layer].detach())
                    G_losses['i_perc'] = feat_loss * self.opt.weight_perceptual

                    if self.opt.G2_mode == 'tps':
                        o_s_loss = 0
                        for i in range(len(fake_result_features)):
                            N, C, H, W = fake_result_features[i].shape
                            for n in range(N):
                                phi_x = fake_result_features[i][n]
                                phi_y = generate_out['real_features'][i][n]
                                phi_x = phi_x.reshape(C, H * W)
                                phi_y = phi_y.reshape(C, H * W)
                                G_x = torch.matmul(phi_x, phi_x.t()) / (C * H * W)
                                G_y = torch.matmul(phi_y, phi_y.t()) / (C * H * W)
                                o_s_loss += torch.sqrt(torch.mean((G_x - G_y) ** 2)) * weights[i]

                        feat_loss = util.mse_loss(fake_result_features[self.perceptual_layer],
                                                  generate_out['real_features'][
                                                      self.perceptual_layer].detach()) * 0.001
                        feat_loss += self.get_ctx_loss(fake_result_features, generate_out['ref_features'])  * self.opt.w_origin_perc
                        G_losses['fake_perc'] = feat_loss + o_s_loss

                    if self.opt.netCorr == 'clothflow':
                        feat_loss = util.mse_loss(warp_features[self.perceptual_layer],
                                                  generate_out['real_features'][self.perceptual_layer].detach())
                        G_losses['warp_perc'] = feat_loss * self.opt.weight_perceptual

                        feat_loss = util.mse_loss(warp_s1_features[self.perceptual_layer],
                                                  generate_out['real_features'][self.perceptual_layer].detach())
                        G_losses['warp_s1_perc'] = feat_loss * self.opt.weight_perceptual

        return G_losses, generate_out

    def compute_discriminator_loss(self, input_semantics, real_image, GforD, label=None):
        D_losses = {}
        with torch.no_grad():
            if self.opt.use_mask_refine and self.opt.use_D1:
                input_G1 = GforD['input_G1'].detach()
                input_G1.requires_grad_()
                mask_refine = GforD['mask_refine'].detach()
                mask_refine.requires_grad_()

                refine_gt = GforD['refine_gt'].detach()
                refine_gt.requires_grad_()

            fake_image = GforD['initial_fake_image'].detach()
            fake_image.requires_grad_()
            input_G2 = GforD['input_G2'].detach()
            input_G2.requires_grad_()

            # real_cloth_mask = GforD['real_cloth_mask'].detach()
            # real_cloth_mask.requires_grad_()

        if self.opt.use_mask_refine and self.opt.use_D1:
            input_D1_fake = torch.cat([input_G1, mask_refine], dim=1)
            input_D1_real = torch.cat([input_G1, refine_gt], dim=1)
            pred_D1_fake = self.net['netD1'](input_D1_fake)
            pred_D1_real = self.net['netD1'](input_D1_real)

            D_losses['D1_Fake'] = self.criterionGAN(pred_D1_fake, False,
                                                   for_discriminator=True) * self.opt.weight_gan
            D_losses['D1_real'] = self.criterionGAN(pred_D1_real, True,
                                               for_discriminator=True) * self.opt.weight_gan

        input_D2_fake = torch.cat([input_G2, fake_image], dim=1)
        input_D2_real = torch.cat([input_G2, real_image], dim=1)
        pred_D2_fake = self.net['netD2'](input_D2_fake)
        pred_D2_real = self.net['netD2'](input_D2_real)

        D_losses['D2_Fake'] = self.criterionGAN(pred_D2_fake, False,
                                               for_discriminator=True) * self.opt.weight_gan
        D_losses['D2_real'] = self.criterionGAN(pred_D2_real, True,
                                               for_discriminator=True) * self.opt.weight_gan

        return D_losses

    def mask_convert(self, seg, c=7):
        size = seg.size()
        oneHot_size = (size[0], c, size[2], size[3])
        c_seg = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_seg_tensor = c_seg.scatter_(1, seg.data.long().cuda(), 1.0)

        return c_seg_tensor

    def fuse_mask(self, seg, items, target):
        c_seg = seg
        for item in items:
            t_mask = torch.cuda.FloatTensor((seg.cpu().numpy() == item).astype(np.float))
            c_seg = c_seg * (1 - t_mask) + t_mask * target
        return c_seg

    def morpho(self, mask, iter, kernel_s, mode=0, size=(256, 256), bs=1):
        new = []
        for i in range(bs):
            if kernel_s[i][0] != 0 and kernel_s[i][1] != 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_s[i])

                for j in range(mask.size()[1]):
                    tem = mask[i, j, ...].cpu().detach().numpy().squeeze().reshape(size[0], size[1], 1) * 255
                    tem = tem.astype(np.uint8)
                    if mode == 0:
                        tem = cv2.dilate(tem, kernel, iterations=iter)
                    elif mode == 1:
                        tem = cv2.morphologyEx(tem, cv2.MORPH_CLOSE, kernel=kernel, iterations=iter)
                    elif mode == 2:
                        tem = cv2.erode(tem, kernel, iterations=iter)
                    tem = tem.astype(np.float64)
                    tem = tem.reshape(1, size[0], size[1])
                    new.append(tem.astype(np.float64) / 255.0)
            else:
                for j in range(mask.size()[1]):
                    tem = mask[i, j, ...].cpu().detach().numpy().squeeze().reshape(size[0], size[1], 1) * 255
                    tem = tem.astype(np.float64)
                    tem = tem.reshape(1, size[0], size[1])
                    new.append(tem.astype(np.float64) / 255.0)

        new = np.stack(new)
            # new = torch.FloatTensor(new).cuda()
        res = torch.FloatTensor(new).cuda()

        return res

    def generate_fake(self, input_semantics, real_image, ref_semantics=None, ref_image=None, self_ref=None, seg=None):
        generate_out = {}

        coor_out = self.net['netCorr'](ref_image, real_image, \
                                      input_semantics, ref_semantics, test_mask=seg)

        seg_img = seg['seg_img'].detach()
        ref_seg = seg['ref_seg'].detach()
        t_seg_img = self.mask_convert(seg_img)
        warped_mask = coor_out['warp_ref_seg']

        if self.opt.max_attn:
            b, _, _, _ = real_image.size()
            f = coor_out['M_corr']
            max_f = torch.max(f, dim=-1)[1].float()
            max_f = max_f.reshape(b, 64, 64)
            # d_ref_img = F.interpolate(ref_image, (64, 64), mode='bilinear').detach()
            max_grid = torch.zeros((b, 64, 64, 2)).cuda()
            max_grid[:, :, :, 1] = max_f // 64 / 64 * 2.0 - 1.0
            max_grid[:, :, :, 0] = max_f % 64 / 64 * 2.0 - 1.0
            Z_max_f = F.interpolate(max_grid.permute(0, 3, 1, 2), (256, 256), mode='bilinear').detach()

            # warped_ref_semantics = F.grid_sample(ref_semantics, max_grid)
            uf_ref_semantics = F.unfold(ref_semantics, 4, stride=4)
            uf_ref_semantics = uf_ref_semantics.permute(0, 2, 1)
            uf_ref_semantics = uf_ref_semantics.reshape(b, 64, 64, -1).permute(0, 3, 1, 2)

            warped_ref_semantics = F.grid_sample(uf_ref_semantics, max_grid, mode='nearest')

            warped_ref_semantics = warped_ref_semantics.reshape(b, -1, 64 * 64)
            warped_ref_semantics = F.fold(warped_ref_semantics, (256, 256), 4, stride=4)

            input_G3 = torch.cat([Z_max_f.detach(), input_semantics, warped_ref_semantics], dim=1)
            generate_out['finer_flow'] = self.net['netG3'](input_G3)
            generate_out['warp_flow'] = F.grid_sample(ref_image, generate_out['finer_flow'].permute(0, 2, 3, 1))
            generate_out['wf_warp_masked_ref'] = F.grid_sample(coor_out['masked_ref'], generate_out['finer_flow'].permute(0, 2, 3, 1))
            generate_out['wf_cl_mask'] = F.grid_sample(coor_out['cl_mask'], generate_out['finer_flow'].permute(0, 2, 3, 1))

        if self.opt.netCorr == 'Correspondence' or self.opt.netCorr == 'warpnet':
            warp_c = coor_out['warp_masked_ref']

            if self.opt.dataset_mode == 'deepfashion' or self.opt.dataset_mode == 'posetemporal':
                real_cloth_mask = self.fuse_mask(seg_img, [5], 4)
                real_cloth_mask[real_cloth_mask != 4] = 0
                real_cloth_mask[real_cloth_mask == 4] = 1
            elif self.opt.dataset_mode == 'viton':
                real_cloth_mask = seg_img.clone()
                real_cloth_mask[real_cloth_mask != 4] = 0
                real_cloth_mask[real_cloth_mask == 4] = 1

            if self.opt.use_mask_refine:

                if self.opt.predict_mode == 'k':
                    input_G1 = torch.cat((warped_mask, \
                                          input_semantics), dim=1)
                elif self.opt.predict_mode == 'cloth+k':
                    c_fuse_seg_img = self.mask_convert(self.fuse_mask(seg_img, [4], 2))

                    input_G1 = torch.cat((warped_mask, c_fuse_seg_img, input_semantics), dim=1)
                elif self.opt.predict_mode == 'k+dp' or self.opt.predict_mode == 'cloth+k+dp' or \
                        self.opt.predict_mode == 'k+dp+tps':
                    img_dp_mask = seg['img_dp_mask'].detach()
                    t_mask = torch.cuda.FloatTensor((img_dp_mask.cpu().numpy() == 1).astype(np.float))
                    no_head_mask = img_dp_mask * (1 - t_mask) + t_mask * 0

                    t_mask = torch.cuda.FloatTensor((img_dp_mask.cpu().numpy() == 6).astype(np.float))
                    no_head_and_foot_mask = no_head_mask * (1 - t_mask) + t_mask * 0
                    c_no_head_and_foot_mask = self.mask_convert(no_head_and_foot_mask)

                    input_G1 = torch.cat((warped_mask, c_no_head_and_foot_mask, input_semantics), dim=1)
                elif self.opt.predict_mode == 'k+tps':
                    tps_warped_mask = coor_out['TPS_warp_cl_mask']

                    input_G1 = torch.cat((warped_mask, tps_warped_mask, \
                                          input_semantics), dim=1)
                elif self.opt.predict_mode == 'dual_U':
                    c_ref_seg = self.mask_convert(ref_seg)
                    input_G1 = torch.cat((c_ref_seg, ref_semantics, warped_mask.detach(), input_semantics), dim=1)

                generate_out['mask_refine'] = self.sigmoid(self.net['netG1'](input_G1))
                generate_out['input_G1'] = input_G1

        if self.opt.use_G2:
            if self.opt.isTrain:
                if self.opt.netCorr == 'clothflow':
                    aligned_image = coor_out['warp_out'].detach()
                    input_G2 = torch.cat((aligned_image, t_seg_img, input_semantics), dim=1)
                    generate_out['initial_fake_image'] = self.tanh(self.net['netG2'](input_G2))
                    generate_out['input_G2'] = input_G2

                else:
                    if self.opt.predict_mode == 'cloth+k':
                        input_cl = warp_c
                    else:
                        input_cl = warp_c * real_cloth_mask


                    masked_label = self.mask_convert(seg_img * (1 - real_cloth_mask))

                    al_mask = self.fuse_mask(seg_img, [3], 2)
                    al_mask[al_mask != 2] = 0
                    al_mask[al_mask == 2] = 1
                    fore_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() > 0).astype(np.int))

                    M_mask = seg['M_tensor']
                    image_hole = real_image * (1 - real_cloth_mask) * al_mask * M_mask \
                                 + real_image * (1 - real_cloth_mask) * (1 - al_mask) * fore_mask
                    generate_out['image_hole'] = image_hole

                    skin_color = self.get_average_color(al_mask, real_image * (1 - real_cloth_mask) * al_mask)
                    if self.opt.G2_mode == 'origin':
                        input_G2 = torch.cat((input_cl, masked_label, image_hole, skin_color), dim=1)
                    elif self.opt.G2_mode == 'k+refine':
                        input_G2 = torch.cat((input_cl, masked_label, image_hole, skin_color, input_semantics), dim=1)
                    elif self.opt.G2_mode == 'no_vt':
                        warp_out = coor_out['warp_out']
                        input_G2 = torch.cat((warp_out, input_semantics), dim=1)
                    elif self.opt.G2_mode == 'tps':
                        tps_warp_out = coor_out['TPS_warp_masked_ref']

                        TPS_warp_cl_mask = coor_out['TPS_warp_cl_mask']
                        b, c, h, w = TPS_warp_cl_mask.shape
                        grid = make_coordinate_grid((h, w), TPS_warp_cl_mask.type())
                        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
                        #
                        # ref_cloth_mask = self.fuse_mask(ref_seg, [], 4)
                        # ref_cloth_mask[ref_cloth_mask != 4] = 0
                        # ref_cloth_mask[ref_cloth_mask == 4] = 1

                        c_grid = (grid + 1.0) / 2 * 255.0
                        masked_grid = TPS_warp_cl_mask.permute(0, 2, 3, 1) * c_grid
                        masked_grid =masked_grid.detach().cpu().numpy()

                        x1 = np.min(masked_grid[:, :, :, 0], axis=(1, 2))
                        x2 = np.max(masked_grid[:, :, :, 0], axis=(1, 2))
                        y1 = np.min(masked_grid[:, :, :, 1], axis=(1, 2))
                        y2 = np.max(masked_grid[:, :, :, 1], axis=(1, 2))

                        kernel_x = (x2 - x1) / 25.0
                        kernel_y = (y2 - y1) / 25.0

                        kernel = []
                        for i in range(len(kernel_x)):
                            kernel.append((int(kernel_x[i]), int(kernel_y[i])))

                        TPS_refine_mask = self.morpho(TPS_warp_cl_mask, 3, kernel,
                                                 mode=2, bs=b, size=(int(h), int(w))) * real_cloth_mask
#                         TPS_refine = TPS_refine_mask * tps_warp_out + (1 - TPS_refine_mask) * input_cl

                        generate_out['TPS_refine_mask'] = real_cloth_mask * TPS_warp_cl_mask

                        TPS_refine = TPS_warp_cl_mask * tps_warp_out * real_cloth_mask
                        generate_out['TPS_refine'] = TPS_refine
                        generate_out['input_cl'] = input_cl

#                         input_cl = TPS_refine

                        if self.opt.input_tps:
                            input_G2 = torch.cat((input_cl, TPS_refine.detach(), masked_label, image_hole, skin_color, input_semantics), dim=1)
                        else:
                            input_G2 = torch.cat((input_cl, masked_label, image_hole, skin_color, input_semantics), dim=1)

                    if self.opt.G2_mode == 'tps':
                        if not self.opt.use_max_f:
                            initial_fake_image, occlusion_mask = self.net['netG2'](input_G2)
                        else:
                            f = coor_out['M_corr']
                            max_f = torch.max(f, dim=-1)[0]
                            max_f = max_f.reshape(b, 1, 64, 64)
                            max_f = F.interpolate(max_f, (256, 256), mode='nearest').detach()
                            initial_fake_image, occlusion_mask = self.net['netG2'](input_G2, max_f=max_f)

                        initial_fake_image = self.tanh(initial_fake_image)
                        occlusion_mask = self.sigmoid(occlusion_mask)

                        generate_out['occlusion_mask'] = occlusion_mask
                        generate_out['initial_fake_image'] = initial_fake_image
                        generate_out['occlusion_tps'] = generate_out['occlusion_mask'] * coor_out['TPS_warp_masked_ref']

                        generate_out['fake_result'] = initial_fake_image.detach() * (1 - generate_out['occlusion_mask']) + coor_out['TPS_warp_masked_ref'].detach() \
                                                    * generate_out['occlusion_mask']

                    else:
                        generate_out['initial_fake_image'] = self.tanh(self.net['netG2'](input_G2))

                    generate_out['input_G2'] = input_G2

        generate_out = {**generate_out, **coor_out}

        return generate_out

    def get_average_color(self, mask, arms):
        color = torch.zeros(arms.shape).cuda()
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 1:
                color[i, 0, :, :]= 0
                color[i, 1, :, :]= 0
                color[i, 2, :, :]= 0
            else:
                color[i, 0, :, :]= arms[i, 0, :, :].sum()/count
                color[i, 1, :, :]= arms[i, 1, :, :].sum()/count
                color[i, 2, :, :]= arms[i, 2, :, :].sum()/count
        return color

    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None, real_image=None, seg=None):
        generate_out = {}

        coor_out = self.net['netCorr'](ref_image, real_image, \
                                       input_semantics, ref_semantics, test_mask=seg)

        seg_img = seg['seg_img'].detach()
        ref_seg = seg['ref_seg'].detach()
        t_seg_img = self.mask_convert(seg_img)
        warped_mask = coor_out['warp_ref_seg']

        # warped_mask = self.mask_convert(torch.max(warped_mask, dim=1, keepdim=True)[1])

        if self.opt.netCorr == 'Correspondence' or self.opt.netCorr == 'warpnet':
            warp_c = coor_out['warp_masked_ref']

            if self.opt.dataset_mode == 'deepfashion' or self.opt.dataset_mode == 'posetemporal':
                real_cloth_mask = self.fuse_mask(seg_img, [5], 4)
                real_cloth_mask[real_cloth_mask != 4] = 0
                real_cloth_mask[real_cloth_mask == 4] = 1

                generate_out['real_cloth'] = real_cloth_mask * real_image

                ref_cloth_mask = self.fuse_mask(ref_seg, [5], 4)
                ref_cloth_mask[ref_cloth_mask != 4] = 0
                ref_cloth_mask[ref_cloth_mask == 4] = 1

                generate_out['ref_cloth'] = ref_cloth_mask * ref_image
            elif self.opt.dataset_mode == 'viton':
                real_cloth_mask = seg_img.clone()
                real_cloth_mask[real_cloth_mask != 4] = 0
                real_cloth_mask[real_cloth_mask == 4] = 1

            if self.opt.use_mask_refine:

                if self.opt.predict_mode == 'k':
                    input_G1 = torch.cat((warped_mask, \
                                          input_semantics), dim=1)
                elif self.opt.predict_mode == 'cloth+k':
                    c_fuse_seg_img = self.mask_convert(self.fuse_mask(seg_img, [4], 2))

                    input_G1 = torch.cat((warped_mask, c_fuse_seg_img, input_semantics), dim=1)
                elif self.opt.predict_mode == 'k+dp' or self.opt.predict_mode == 'cloth+k+dp' or \
                        self.opt.predict_mode == 'k+dp+tps':
                    img_dp_mask = seg['img_dp_mask'].detach()
                    c_img_dp_mask = self.mask_convert(img_dp_mask)

                    t_mask = torch.cuda.FloatTensor((img_dp_mask.cpu().numpy() == 1).astype(np.float))
                    no_head_mask = img_dp_mask * (1 - t_mask) + t_mask * 0

                    t_mask = torch.cuda.FloatTensor((img_dp_mask.cpu().numpy() == 6).astype(np.float))
                    no_head_and_foot_mask = no_head_mask * (1 - t_mask) + t_mask * 0
                    c_no_head_and_foot_mask = self.mask_convert(no_head_and_foot_mask)

                    input_G1 = torch.cat((warped_mask, c_no_head_and_foot_mask, input_semantics), dim=1)
                elif self.opt.predict_mode == 'k+tps':
                    tps_warped_mask = coor_out['TPS_warp_cl_mask']

                    input_G1 = torch.cat((warped_mask, tps_warped_mask, \
                                          input_semantics), dim=1)
                elif self.opt.predict_mode == 'dual_U':
                    c_ref_seg = self.mask_convert(ref_seg)
                    input_G1 = torch.cat((c_ref_seg, ref_semantics, warped_mask.detach(), input_semantics), dim=1)

                generate_out['mask_refine'] = self.sigmoid(self.net['netG1'](input_G1))
                generate_out['input_G1'] = input_G1
            else:
                generate_out['mask_refine'] = coor_out['warp_ref_seg']

            if self.opt.use_cl_refine:
                if self.opt.refine_type == 'densepose':
                    warped_i_c = seg['warped_cloth']
                elif self.opt.refine_type == 'single_warp':
                    warped_i_c = coor_out['warp_i_masked_ref']

                input_G3 = torch.cat((warped_i_c, warp_c), dim=1)
                generate_out['refined_clothes_mask'] = self.sigmoid(self.net['netG3'](input_G3))
                generate_out['input_G3'] = input_G3

                a = generate_out['refined_clothes_mask']
                input_cl = a * warped_i_c + (1 - a) * warp_c
                generate_out['refined_clothes'] = input_cl
                generate_out['masked_warped_cl'] = a * warped_i_c

                input_cl = input_cl * real_cloth_mask
            else:
                input_cl = warp_c * real_cloth_mask

        warp_c = coor_out['warp_masked_ref']

        refine_seg = generate_out['mask_refine'].detach()

        masked_label = self.mask_convert(seg_img * (1 - real_cloth_mask))
        c_refine_label = self.mask_convert(self.fuse_mask(refine_seg, [5], 4))

        arm_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() == 2).astype(np.int))
        leg_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() == 3).astype(np.int))

        fore_mask = torch.cuda.FloatTensor((seg_img.cpu().numpy() > 0).astype(np.int))

        if self.opt.predict_mode == 'cloth+k+dp':
            dp_arm_mask = c_img_dp_mask[:, 2, :, :]
            dp_leg_mask = c_img_dp_mask[:, 3, :, :] + c_img_dp_mask[:, 6, :, :]

            refine_seg[refine_seg > 0.5] = 1
            refine_seg[refine_seg != 1] = 0

            refine_cloth_mask = refine_seg

            refine_arm_mask = dp_arm_mask * (1 - refine_seg)
            refine_leg_mask = dp_leg_mask * (1 - refine_seg)
        else:
            refine_seg = torch.max(refine_seg, dim=1, keepdim=True)[1]

            refine_cloth_mask = self.fuse_mask(refine_seg, [5], 4)
            refine_cloth_mask[refine_cloth_mask != 4] = 0
            refine_cloth_mask[refine_cloth_mask == 4] = 1
            #
            warped_cloth = coor_out['warp_masked_ref']
            # input_G3 = torch.cat((warped_cloth, masked_label, input_semantics), dim=1)
            #
            # generate_out['clothes_mask'] = self.sigmoid(self.net['netG3'](input_G3))
            # generate_out['input_G3'] = input_G3

            # real_masked_cloth = self.mask_convert(seg_img * (1 - real_cloth_mask))
            # real_masked_cloth[:, 0, :, :] = (masked_label[:, 0, :, :] + masked_label[:, 1, :, :]) * (1 - real_masked_cloth[:, 1, :, :])
            # real_masked_cloth[:, 2, :, :] = masked_label[:, 2, :, :]
            # real_masked_cloth[:, 3, :, :] = masked_label[:, 3, :, :]

            refine_arm_mask = torch.cuda.FloatTensor((refine_seg.cpu().numpy() == 2).astype(np.int))
            refine_leg_mask = torch.cuda.FloatTensor((refine_seg.cpu().numpy() == 3).astype(np.int))
            refine_fore_mask = torch.cuda.FloatTensor((refine_seg.cpu().numpy() > 0).astype(np.int))

            # warp_c = torch.cat((coor_out['warp_masked_ref_upper'], coor_out['warp_masked_ref_lower']), \
            #                          1)  # test G1 output

            warped_cloth_mask = self.fuse_mask(torch.max(coor_out['warp_ref_seg'], dim=1, keepdim=True)[1], [5], 4)
            warped_cloth_mask[warped_cloth_mask != 4] = 0
            warped_cloth_mask[warped_cloth_mask == 4] = 1

            # warped_arm_mask = torch.cuda.FloatTensor((warped_seg.cpu().numpy() == 2).astype(np.int))
            # warped_leg_mask = torch.cuda.FloatTensor((warped_seg.cpu().numpy() == 3).astype(np.int))
            # refine_arm_mask = refine_arm_mask + warped_arm_mask - refine_arm_mask * warped_arm_mask
            # refine_leg_mask = refine_leg_mask + warped_leg_mask - refine_leg_mask * warped_leg_mask
            # refine_leg_mask = refine_leg_mask - refine_arm_mask * refine_leg_mask
            # refine_leg_mask = refine_leg_mask * (1 - refine_cloth_mask)

            # new_upper_cloth_mask = c_refine_label[:, 4:5, :, :]
            # new_lower_cloth_mask = c_refine_label[:, 5:6, :, :]

            # smaller_arm_mask = self.morpho(real_cloth_mask, 3, False)
            # smaller_leg_mask = self.morpho(real_cloth_mask, 3, False)

        if self.opt.use_layout_prediction:
            occlude_al_mask = refine_arm_mask * real_cloth_mask
            occlude_leg_mask = refine_leg_mask * real_cloth_mask
            new_arm_mask = occlude_al_mask + arm_mask * (1 - refine_cloth_mask)
            new_leg_mask = occlude_leg_mask + leg_mask * (1 - refine_cloth_mask)

            if self.opt.predict_mode == 'cloth+k+dp' or self.opt.predict_mode == 'k+dp':
                dp_arm_mask = c_img_dp_mask[:, 2, :, :]
                dp_leg_mask = c_img_dp_mask[:, 3, :, :] + c_img_dp_mask[:, 6, :, :]

                new_arm_mask = new_arm_mask * dp_arm_mask
                new_leg_mask = dp_leg_mask * new_leg_mask

                new_leg_mask = new_leg_mask * (1 - new_arm_mask)
        else:
            new_arm_mask = refine_arm_mask * (1 - masked_label[:, 6, :, :]) * (1 - masked_label[:, 1, :, :])
            new_leg_mask = refine_leg_mask * (1 - masked_label[:, 6, :, :]) * (1 - masked_label[:, 1, :, :])

            # if self.opt.predict_mode == 'k+dp':
            #     dp_arm_mask = c_img_dp_mask[:, 2, :, :]
            #     dp_leg_mask = c_img_dp_mask[:, 3, :, :] + c_img_dp_mask[:, 6, :, :]
            #
            #     new_arm_mask = new_arm_mask * dp_arm_mask
            #     new_leg_mask = dp_leg_mask * new_leg_mask
            #
            #     new_leg_mask = new_leg_mask * (1 - new_arm_mask)

        _, _, h, w = real_image.size()
        bigger_real_cloth_mask = self.morpho(real_cloth_mask, 5, kernel_s=[(3,3)], mode=0, size=(h, w))
        # smaller_real_head_mask = self.morpho(real_head_mask, 2, mode=2, size=(h, w))
        image_hole = real_image * (1 - bigger_real_cloth_mask) * new_arm_mask \
                     + real_image * (1 - bigger_real_cloth_mask) * new_leg_mask \
                     + real_image * (1 - real_cloth_mask) * (1 - arm_mask) \
                     * (1 - leg_mask) * (1 - new_arm_mask) * (1 - new_leg_mask) * fore_mask

        generate_out['image_hole'] = image_hole

        masked_label[:, 2, :, :] = new_arm_mask * (1 - masked_label[:, 1, :, :]) * (1 - masked_label[:, 6, :, :])
        masked_label[:, 3, :, :] = new_leg_mask * (1 - masked_label[:, 1, :, :]) * (1 - masked_label[:, 6, :, :])
        # masked_label[:, 4, :, :] = c_refine_label[:, 4, :, :] * (1 - new_arm_mask) * (1 - new_leg_mask) * (1 - masked_label[:, 1, :, :])
        # masked_label[:, 5, :, :] = c_refine_label[:, 5, :, :] * (1 - new_arm_mask) * (1 - new_leg_mask) * (1 - masked_label[:, 1, :, :])
        # masked_label[:, 4, :, :] = c_refine_label[:, 4, :, :] * (1 - new_arm_mask) * (1 - new_leg_mask) * (1 - masked_label[:, 1, :, :])
        # masked_label[:, 6, :, :] *= (1 - masked_label[:, 4, :, :])
        # masked_label[:, 6, :, :] *= (1 - masked_label[:, 5, :, :])
        b, c, w, h = masked_label.size()
        masked_label[:, 0, :, :] = torch.ones(torch.Size((w, h)))
        masked_label[:, 0, :, :] *= (1 - masked_label[:, 1, :, :])[0]
        masked_label[:, 0, :, :] *= (1 - new_arm_mask)[0]
        masked_label[:, 0, :, :] *= (1 - new_leg_mask)[0]
        masked_label[:, 0, :, :] *= (1 - masked_label[:, 6, :, :])[0]
        masked_label[:, 0, :, :] *= (1 - masked_label[:, 4, :, :])[0]
        masked_label[:, 0, :, :] *= (1 - masked_label[:, 5, :, :])[0]
        generate_out['test_mask_refine'] = masked_label

        # input_mask = masked_label.detach()
        # input_mask0[:, 4, :, :] = generate_out['clothes_mask'] * (1 - refine_al_mask)
        # input_mask = self.mask_convert( self.fuse_mask(refine_seg, [5], 4) )
        limb_mask = new_arm_mask + new_leg_mask
        skin_color = self.get_average_color((1 - bigger_real_cloth_mask) * limb_mask,
                                            real_image * (1 - bigger_real_cloth_mask) * limb_mask)
        generate_out['skin_color'] = skin_color
        #
        # warped_cloth = seg['warped_cloth']
        # a = generate_out['refined_clothes_mask']
        # generate_out['masked_warped_cl'] = a * warped_cloth
        # input_cl = a * warped_cloth + (1 - a) * warp_c
        # generate_out['refined_clothes'] = input_cl
        # generate_out['masked_warped_cl'] = a * warped_cloth

        # b_refine_cloth_mask = refine_cloth_mask + warped_cloth_mask - refine_cloth_mask * warped_cloth_mask
        # bigger_refine_cloth_mask = self.morpho(b_refine_cloth_mask, 1, kernel_s=[(3,3)], mode=1, size=(w, h))
        # bigger_refine_cloth_mask = refine_cloth_mask + warped_cloth_mask - refine_cloth_mask * warped_cloth_mask

        input_cl = warp_c
        input_cl = input_cl * (1 - new_arm_mask) * (1 - new_leg_mask) * (1 - masked_label[:, 1, :, :]) * (1 - masked_label[:, 6, :, :])
        input_cl = input_cl * refine_cloth_mask

        # generate_out['bigger_refine_cloth_mask'] = bigger_refine_cloth_mask
            # b1, c1, w1, h1 = skin_color.size()
            # G2_noise = self.gen_noise(torch.Size((b1, 1, w1, h1)))
            # input_G2 = torch.cat((warp_c, masked_label, image_hole, skin_color, G2_noise), dim=1)

        if self.opt.G2_mode == 'origin':
            input_G2 = torch.cat((input_cl, masked_label, image_hole, skin_color), dim=1)
        elif self.opt.G2_mode == 'k+refine':
            input_G2 = torch.cat((input_cl, masked_label, image_hole, skin_color, input_semantics), dim=1)
        elif self.opt.G2_mode == 'no_vt':
            warp_out = coor_out['warp_out']
            input_G2 = torch.cat((warp_out, input_semantics), dim=1)
        elif self.opt.G2_mode == 'tps':
            tps_warp_out = coor_out['TPS_warp_masked_ref']

            TPS_warp_cl_mask = coor_out['TPS_warp_cl_mask']
            b, c, w, h = TPS_warp_cl_mask.shape
            grid = make_coordinate_grid((w, h), TPS_warp_cl_mask.type())
            grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
            #
            # ref_cloth_mask = self.fuse_mask(ref_seg, [], 4)
            # ref_cloth_mask[ref_cloth_mask != 4] = 0
            # ref_cloth_mask[ref_cloth_mask == 4] = 1

            c_grid = (grid + 1.0) / 2 * 255.0
            masked_grid = TPS_warp_cl_mask.permute(0, 2, 3, 1) * c_grid
            masked_grid = masked_grid.detach().cpu().numpy()

            x1 = np.min(masked_grid[:, :, :, 0], axis=(1, 2))
            x2 = np.max(masked_grid[:, :, :, 0], axis=(1, 2))
            y1 = np.min(masked_grid[:, :, :, 1], axis=(1, 2))
            y2 = np.max(masked_grid[:, :, :, 1], axis=(1, 2))

            kernel_x = (x2 - x1) / 25.0
            kernel_y = (y2 - y1) / 25.0

            kernel = []
            for i in range(len(kernel_x)):
                kernel.append((int(kernel_x[i]), int(kernel_y[i])))

            TPS_refine_mask = self.morpho(TPS_warp_cl_mask, 3, kernel,
                                          mode=2, bs=b, size=(w, h)) * real_cloth_mask
            #                         TPS_refine = TPS_refine_mask * tps_warp_out + (1 - TPS_refine_mask) * input_cl

            generate_out['TPS_refine_mask'] = TPS_refine_mask

            TPS_refine = TPS_warp_cl_mask * tps_warp_out * real_cloth_mask
            generate_out['TPS_refine'] = TPS_refine

            if self.opt.paste_tps:
                generate_out['input_cl'] = input_cl * (1 - TPS_refine_mask) + TPS_refine_mask * tps_warp_out
            else:
                generate_out['input_cl'] = input_cl

            #                         input_cl = TPS_refine

            input_G2 = torch.cat((input_cl, TPS_refine.detach(), masked_label, image_hole, skin_color, input_semantics),
                                 dim=1)

        if self.opt.G2_mode == 'tps':
            if not self.opt.use_max_f:
                initial_fake_image, occlusion_mask = self.net['netG2'](input_G2)
            else:
                f = coor_out['M_corr']
                max_f = torch.max(f, dim=-1)[0]
                max_f = max_f.reshape(b, 1, 64, 64)
                max_f = F.interpolate(max_f, (256, 256), mode='nearest').detach()
                initial_fake_image, occlusion_mask = self.net['netG2'](input_G2, max_f=max_f)

            initial_fake_image = self.tanh(initial_fake_image)
            occlusion_mask = self.sigmoid(occlusion_mask)

            generate_out['occlusion_mask'] = occlusion_mask * refine_cloth_mask
            generate_out['initial_fake_image'] = initial_fake_image
            generate_out['occlusion_tps'] = generate_out['occlusion_mask'] * coor_out['TPS_warp_masked_ref']

            generate_out['fake_result'] = initial_fake_image.detach() * (1 - generate_out['occlusion_mask']) + coor_out[
                'TPS_warp_masked_ref'].detach() \
                                         * generate_out['occlusion_mask']
            generate_out['ori_fake_cloth'] = generate_out['fake_result'] * real_cloth_mask
            generate_out['fake_cloth'] = generate_out['initial_fake_image'] * real_cloth_mask
        else:
            generate_out['initial_fake_image'] = self.tanh(self.net['netG2'](input_G2))
            generate_out['i_fake_cloth'] = generate_out['initial_fake_image'] * real_cloth_mask

        generate_out['input_G2'] = input_G2

        generate_out = {**generate_out, **coor_out}

        return generate_out

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def compute_D_seg_loss(self, out, gt):
        fake_seg, real_seg = self.divide_pred([out])
        fake_seg_loss = F.cross_entropy(fake_seg[0][0], gt)
        real_seg_loss = F.cross_entropy(real_seg[0][0], gt)

        down_gt = F.interpolate(gt.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze().long()
        fake_seg_loss_down = F.cross_entropy(fake_seg[0][1], down_gt)
        real_seg_loss_down = F.cross_entropy(real_seg[0][1], down_gt)

        seg_loss = fake_seg_loss + real_seg_loss + fake_seg_loss_down + real_seg_loss_down
        return seg_loss

    def gen_noise(self, shape):
        noise = np.zeros(shape, dtype=np.uint8)
        ### noise
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()

    def get_row(self,coor,num):
        sec_dic=[]
        for j in range(num):
            sum=0
            buffer=0
            flag=False
            max=-1
            for i in range(num-1):
                differ=(coor[:,j*num+i+1,:]-coor[:,j*num+i,:])**2
                if not flag:
                    second_dif=0
                    flag=True
                else:
                    second_dif=torch.abs(differ-buffer)
                    sec_dic.append(second_dif)

                buffer=differ
                sum+=second_dif
        return torch.stack(sec_dic,dim=1)

    def get_col(self,coor,num):
        sec_dic=[]
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for j in range(num - 1):
                differ = (coor[:, (j+1) * num + i , :] - coor[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)
                    sec_dic.append(second_dif)
                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic,dim=1)

    def grad_row(self, coor, num):
        sec_term = []
        for j in range(num):
            for i in range(1, num - 1):
                x0, y0 = coor[:, j * num + i - 1, :][0]
                x1, y1 = coor[:, j * num + i + 0, :][0]
                x2, y2 = coor[:, j * num + i + 1, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term

    def grad_col(self, coor, num):
        sec_term = []
        for i in range(num):
            for j in range(1, num - 1):
                x0, y0 = coor[:, (j - 1) * num + i, :][0]
                x1, y1 = coor[:, j * num + i, :][0]
                x2, y2 = coor[:, (j + 1) * num + i, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term