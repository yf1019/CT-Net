import torch
import os
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
import torch
import itertools
import numpy as np
import torch.nn as nn
from models.networks.base_network import BaseNetwork
import torch.nn.functional as F


class UnetGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc, occlusion_mask=False):
        super(UnetGenerator, self).__init__()
        self.opt = opt

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

        self.occlusion_mask = occlusion_mask
        if occlusion_mask:
            self.attn_in = 128
            if opt.use_max_f:
                self.attn_in += 1

            self.predict_occlusion = nn.Sequential(*[nn.Conv2d(self.attn_in, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
                                     ])

    def forward(self, input_G, max_f=None):
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

        if self.occlusion_mask:
            if not self.opt.use_max_f:
                occlusion_mask = self.predict_occlusion(torch.cat([conv1, up9], 1))
            else:
                occlusion_mask = self.predict_occlusion(torch.cat([max_f, conv1, up9], 1))
            return conv9, occlusion_mask
        else:
            return conv9


class DualUnetGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc):
        super(DualUnetGenerator, self).__init__()

        nl = nn.InstanceNorm2d

        self.a_conv1 = nn.Sequential(*[nn.Conv2d(input_nc['a'], 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.a_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.a_conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.a_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.a_conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.a_pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.a_conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.a_drop4 = nn.Dropout(0.5)
        self.a_pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv5 = nn.Sequential(*[nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nl(1024),
                                     nn.ReLU()])
        self.drop5 = nn.Dropout(0.5)

        self.b_conv1 = nn.Sequential(*[nn.Conv2d(input_nc['b'], 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU()])
        self.b_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.b_conv2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])
        self.b_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.b_conv3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])
        self.b_pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.b_conv4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.b_drop4 = nn.Dropout(0.5)
        self.b_pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.up6 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
              nl(512),
              nn.ReLU()])

        self.conv6 = nn.Sequential(*[nn.Conv2d(1536, 1024, kernel_size=3, stride=1, padding=1), nl(1024), nn.ReLU(),
                                     nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU()])
        self.up7 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
              nl(256),
              nn.ReLU()])
        self.conv7 = nn.Sequential(*[nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1), nl(512), nn.ReLU(),
                                     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), nl(256), nn.ReLU()])

        self.up8 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
              nl(128),
              nn.ReLU()])

        self.conv8 = nn.Sequential(*[nn.Conv2d(384, 192, kernel_size=3, stride=1, padding=1), nl(192), nn.ReLU(),
                                     nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1), nl(128), nn.ReLU()])

        self.up9 = nn.Sequential(
            *[nn.UpsamplingNearest2d(scale_factor=2), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
              nl(64),
              nn.ReLU()])

        self.conv9 = nn.Sequential(*[nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1), nl(96), nn.ReLU(),
                                     nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1), nl(64), nn.ReLU(),
                                     nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1)
                                     ])


    def forward(self, input_G):
        input_a = input_G[:, :27, ...]
        input_b = input_G[:, 27:, ...]

        a_conv1 = self.a_conv1(input_a)
        a_pool1 = self.a_pool1(a_conv1)

        a_conv2 = self.a_conv2(a_pool1)
        a_pool2 = self.a_pool2(a_conv2)

        a_conv3 = self.a_conv3(a_pool2)
        a_pool3 = self.a_pool3(a_conv3)

        a_conv4 = self.a_conv4(a_pool3)
        a_drop4 = self.a_drop4(a_conv4)
        a_pool4 = self.a_pool4(a_drop4)

        b_conv1 = self.b_conv1(input_b)
        b_pool1 = self.b_pool1(b_conv1)

        b_conv2 = self.b_conv2(b_pool1)
        b_pool2 = self.b_pool2(b_conv2)

        b_conv3 = self.b_conv3(b_pool2)
        b_pool3 = self.b_pool3(b_conv3)

        b_conv4 = self.b_conv4(b_pool3)
        b_drop4 = self.b_drop4(b_conv4)
        b_pool4 = self.b_pool4(b_drop4)

        conv5 = self.conv5(torch.cat([a_pool4, b_pool4], 1))
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        conv6 = self.conv6(torch.cat([a_drop4, b_drop4, up6], 1))

        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([a_conv3, b_conv3, up7], 1))

        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([a_conv2, b_conv2, up8], 1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([a_conv1, b_conv1, up9], 1))

        return conv9


class fusenet(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc):
        super(fusenet, self).__init__()

        nl = nn.InstanceNorm2d
        self.feature_channel = 32
        self.layers = 2
        ngf = 64

        norm_layer = nn.InstanceNorm2d
        nonlinearity = get_nonlinearity_layer(activation_type='LeakyReLU')

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                   nonlinearity)
        mult = 1
        for i in range(self.layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), 1024 // ngf)

            # if i != self.layers - 2:
            block = EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                 nonlinearity)
            # else:
            #     block = ExpandBlock(ngf * mult_prev, ngf * mult, norm_layer,
            #                         nonlinearity)

            setattr(self, 'encoder' + str(i), block)

        self.ResBlock = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4,
                          kernel_size=3, padding=1, stride=1))

        mult = min(self.layers, 1024 // ngf)
        for i in range(self.layers):
            mult_prev = mult
            mult = min((self.layers - i - 1), 1024 // ngf) if i != self.layers - 1 else 1
            up = nn.Sequential(ResidualBlock(ngf * mult_prev, ngf * mult_prev),
                                ResBlockDecoder(ngf * mult_prev, ngf * mult))

            setattr(self, 'decoder' + str(i), up)

        self.outconv = Output(ngf, output_nc, 3)

    def forward(self, input_G):
        out = self.block0(input_G)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)

        out = self.ResBlock(out)
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return out_image


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
        out = self.bn1(x)
        out = self.prelu(out)
        out = self.padding1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out += residual
        return out


class ResBlockDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, \
                norm_layer=nn.InstanceNorm2d, nonlinearity=nn.LeakyReLU()):
        super(ResBlockDecoder, self).__init__()
        norm_layer = nn.InstanceNorm2d
        nonlinearity = nn.LeakyReLU()

        conv1 = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)
        conv2 = nn.ConvTranspose2d(output_nc, output_nc, kernel_size=3, stride=2, padding=1,
                                   output_padding=1)
        bypass = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1,
                                    output_padding=1)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(output_nc),
                                       nonlinearity, conv2, )

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out


class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.InstanceNorm2d, nonlinearity= nn.LeakyReLU()):
        super(Output, self).__init__()

        norm_layer = nn.InstanceNorm2d
        nonlinearity = nn.LeakyReLU()

        kernel_size = 3
        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = conv(input_nc, output_nc, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out


def conv(input_nc, output_nc, **kwargs):
    """use coord convolution layer to add position information"""
    return nn.Conv2d(input_nc, output_nc, **kwargs)


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


class ResBlock(nn.Module):
    def __init__(self, input_nc, output_nc=None, norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()

        self.f_short_cut = False
        if output_nc == None:
            output_nc = input_nc
        elif output_nc != input_nc:
            self.f_short_cut = True
            self.short_cut = get_conv(input_nc, output_nc, kernel_size=3, padding=1)

        self.activation = nn.PReLU()
        self.conv = get_conv(input_nc, output_nc, kernel_size=3, padding=1, norm_layer=norm_layer)

    def forward(self, x):
        out = self.conv(self.activation(x))
        if self.f_short_cut:
            out = self.short_cut(x) + out
        else:
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
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, start=False):
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


class render(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc=3):
        super(render, self).__init__()

        n_res_blocks = 2
        nf = 64
        max_nf = 512

        self.input_nc = input_nc
        self.num_scales = 6
        self.n_res_blocks = n_res_blocks

        self.upsampling_nn = nn.Upsample(scale_factor=2, mode='nearest')

        self.src_0_down = DownBlock(input_nc, nf * 1, start=True)
        for i in range(n_res_blocks):
            self.__setattr__('src_%d_res_%d' % (0, i), ResBlock(nf * 1))

        for l in range(1, self.num_scales):
            c_in = min(nf * 2 ** (l - 1), max_nf)
            c_out = min(nf * 2 ** l, max_nf)

            self.__setattr__('src_%d_down' % l, DownBlock(c_in, c_out))

            for i in range(n_res_blocks):
                self.__setattr__('src_%d_res_%d' % (l, i), ResBlock(c_out))

        for l in range(self.num_scales, 0, -1):
            if l == self.num_scales:
                c_in = min(nf * 2 ** (l-1), max_nf)
                c_out = min(nf * 2 ** (l-1), max_nf)
            elif l == 1:
                c_in = min(nf * 2 ** l, 2 * max_nf)
                c_out = 3
            else:
                c_in = min(nf * 2 ** l, 2 * max_nf)
                c_out = min(nf * 2 ** (l-2), 2 * max_nf)

            conv = get_conv(c_in, c_out, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d)
            self.__setattr__('src_%d_up_conv' % (l - 1), conv)
            self.__setattr__('src_%d_up_res_%d' % (l - 1, 0), ResBlock(c_out, c_out))
            self.__setattr__('src_%d_up_res_%d' % (l - 1, 1), ResBlock(c_out, c_out))

    def forward(self, x):
        outs = []

        out = x
        for l in range(self.num_scales):
            out = self.__getattr__('src_%d_down'%l)(out)
            for i in range(self.n_res_blocks):
                out = self.__getattr__('src_%d_res_%d'%(l, i))(out)

            outs.append(out)

        for l in range(self.num_scales, 0, -1):
            if l == self.num_scales:
                out = out
            else:
                out = torch.cat((out, outs[l-1]), dim=1)

            out = self.upsampling_nn(out)
            out = self.__getattr__('src_%d_up_conv' %(l-1))(out)

            for i in range(self.n_res_blocks):
                out = self.__getattr__('src_%d_up_res_%d'%(l-1, i))(out)

        return out
