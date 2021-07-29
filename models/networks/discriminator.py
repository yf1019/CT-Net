import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc, output_nc=1):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc, output_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, input_nc, output_nc):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, input_nc, output_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input):
        result = []
        segs = []
        cam_logits = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, input_nc, output_nc=1):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int((kw - 1.0) / 2)
        nf = opt.ndf
        input_nc = input_nc

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for name, submodel in self.named_children():
            if 'model' not in name:
                continue
            x = results[-1]
            intermediate_output = submodel(x)
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            retu = results[1:]
        else:
            retu = results[-1]

        return retu


def get_skipped_frames(B_all, t_scales, tD):
    B_skipped = [None] * t_scales
    if B_all is None:
        return B_all, B_skipped

    for s in range(t_scales):
        tDs = tD ** s  # number of skipped frames between neighboring frames (e.g. 1, 3, 9, ...)
        span = tDs * (tD - 1)  # number of frames the final triplet frames span before skipping (e.g., 2, 6, 18, ...)
        n_groups = min(B_all.size()[1] - span, 1)
        if n_groups > 0:
            skip =  B_all[:,-span - 1::tDs].contiguous()
            B_skipped[s] = torch.cat([B_skipped[s], skip]) if B_skipped[s] is not None else skip
    max_prev_frames = tD ** (t_scales - 1) * (tD - 1)
    if B_all.size()[1] > max_prev_frames:
        B_all = B_all[:, -max_prev_frames:]
    return B_all, B_skipped


def get_skipped_flows(flowNet, flow_ref_all, conf_ref_all, real_B, t_scales, tD):
    flow_ref_skipped, conf_ref_skipped = [None] * t_scales, [None] * t_scales
    flow_ref_all, flow = get_skipped_frames(flow_ref_all, 1, tD-1)
    conf_ref_all, conf = get_skipped_frames(conf_ref_all, 1, tD-1)
    if flow[0] is not None:
        flow_ref_skipped[0], conf_ref_skipped[0] = flow[0], conf[0]

    for s in range(1, t_scales):
        if real_B[s] is not None and real_B[s].size()[1] == tD:
            flow_ref_skipped[s], conf_ref_skipped[s] = flowNet(real_B[s][:, 1:], real_B[s][:, :-1])
    return flow_ref_all, conf_ref_all, flow_ref_skipped, conf_ref_skipped