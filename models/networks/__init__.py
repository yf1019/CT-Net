import torch
from models.networks.base_network import BaseNetwork
import util.util as util
from models.networks.ContextualLoss import *
from models.networks.loss import *


def find_network_using_name(target_network_name, filename, add=True):
    target_class_name = target_network_name + filename if add else target_network_name
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
       "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    # netG_cls = find_network_using_name(opt.netG, 'generator')
    # parser = netG_cls.modify_commandline_options(parser, is_train)
    # if is_train:
    #     netD_cls = find_network_using_name(opt.netD, 'discriminator')
    #     parser = netD_cls.modify_commandline_options(parser, is_train)
    # netE_cls = find_network_using_name('conv', 'encoder')
    # parser = netE_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt, input_nc=0, output_nc=0, occlusion_mask=False):
    if input_nc != 0 and output_nc != 0:
        if occlusion_mask:
            net = cls(opt, input_nc, output_nc, occlusion_mask)
        else:
            net = cls(opt, input_nc, output_nc)

    else:
        net = cls(opt)

    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_G1(opt, input_nc, output_nc):
    netG_cls = find_network_using_name(opt.netG1, 'generator', add=False)
    return create_network(netG_cls, opt, input_nc, output_nc)


def define_G2(opt, input_nc, output_nc, occlusion_mask=False):
    netG_cls = find_network_using_name(opt.netG2, 'generator', add=False)
    return create_network(netG_cls, opt, input_nc, output_nc, occlusion_mask)

def define_G3(opt, input_nc, output_nc):
    netG_cls = find_network_using_name(opt.netG3, 'generator', add=False)
    return create_network(netG_cls, opt, input_nc, output_nc)

def define_D1(opt, input_nc, output_nc):
    netD_cls = find_network_using_name(opt.netD1, 'discriminator', add=False)
    return create_network(netD_cls, opt, input_nc, output_nc)

def define_D2(opt, input_nc, output_nc):
    netD_cls = find_network_using_name(opt.netD2, 'discriminator', add=False)
    return create_network(netD_cls, opt, input_nc, output_nc)

def define_D3(opt, input_nc, output_nc):
    netD_cls = find_network_using_name(opt.netD3, 'discriminator', add=False)
    return create_network(netD_cls, opt, input_nc, output_nc)


def define_Corr(opt):
    netCoor_cls = find_network_using_name(opt.netCorr, 'correspondence', False)
    return create_network(netCoor_cls, opt)
