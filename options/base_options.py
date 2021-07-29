import sys
import argparse
import os
from typing import DefaultDict
from util import util
import torch
import models
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='ct_net', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='ct', help='which model to use')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--store_name', type=str, default='one_corr_tps', help='dir to save the checkpoints')
        
        # input/output sizes
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        # parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=20, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        # parser.add_argument('--clothes', type=list, default=[5, 6, 7, 9, 10, 12], help='seg class')

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='/home/mist/CT_Net/dataset')
        parser.add_argument('--dataset_mode', type=str, default='deepfashion')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', default=True, action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')
        parser.add_argument('--use_lds_dis_map', default=False, type=bool, help='use lds map or joints map')
        parser.add_argument('--load_lds', default=False, type=bool, help='load lds or not')
        parser.add_argument('--canvs', default=True, type=bool, help='use canvs or not (17/20)')
        parser.add_argument('--draw_corr', default=False, type=bool, help='draw heatmap')
        parser.add_argument('--use_dp', default=True, type=bool, help='use densepose or not')
        parser.add_argument('--use_mask', default=True, type=bool, help='use layout prediction module or not')
        parser.add_argument('--finer_dist', default=True, type=bool, help='use finer distance map or not')
        parser.add_argument('--stick', default=1, type=int, help='width of the stick')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

        # display parameter define
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='display id of the web')
        parser.add_argument('--display_port', type=int, default=61504, help='visidom port of the web display')
        parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                            help='if positive, display all images in a single visidom web panel')
        parser.add_argument('--display_env', type=str, default=parser.parse_known_args()[0].name.replace('_', ''),
                            help='the environment of visidom display')
        parser.add_argument('--no_html', default=True, help='do not save intermediate training results')

        # for correspondence layer
        parser.add_argument('--netCorr', type=str, default='warpnet', help='type of the correspondence layer')
        parser.add_argument('--double_dis', type=bool, default=False, help='only input distance map')
        parser.add_argument('--freeze_id_warp', type=bool, default=False, help='detach the grad to the feature extractor')
        parser.add_argument('--freeze_warp', type=bool, default=False, help='detach the grad to the feature extractor')
        parser.add_argument('--freeze_geo_corr', type=bool, default=True, help='detach the grad from tps warping to the feature extractor')             
        parser.add_argument('--use_max_Corr', type=bool, default=False, help='use torch.max before warping')
        parser.add_argument('--predict_mode', type=str, default='k+dp', help='mode to predict the target layout')
        parser.add_argument('--use_layout_prediction', type=bool, default=False, help='use layout prediction or not')   
        parser.add_argument('--use_ds_tps_corr', type=bool, default=True, help='use additional res layers for the estimation of TPS warping')

        parser.add_argument('--Corr_mode', type=str, default='origin', help='type of inputs into the feature extractors')
        parser.add_argument('--G2_mode', type=str, default='tps', help='mode to generate the garment transfer results')
        parser.add_argument('--input_tps', type=bool, default=True, help='whether inputing the tps warping result into the generator or not')
        parser.add_argument('--use_max_f', type=bool, default=False, help='using the corresponding matrix for attention estimation or not')
        parser.add_argument('--grid_size', type=int, default=5, help='Size of grid used in the estimation of TPS warping')
        parser.add_argument('--paste_tps', type=bool, default=False, help='paste the warping result of TPS onto the warping result of DF-guided warping')
        parser.add_argument('--max_attn', type=bool, default=False, help='extra network will be used for the estimation of attention mask')
        parser.add_argument('--use_grid_warp', type=bool, default=False, help='estimate the warping based on the grid')

        # for generator
        parser.add_argument('--netG1', type=str, default='Unetgenerator', help='selects model to use for netG1')
        parser.add_argument('--G1_inc', type=int, default=34, help='input channels for G1')
        parser.add_argument('--G1_oc', type=int, default=7, help='output channels for G1')
        parser.add_argument('--use_mask_refine', type=bool, default=True, help='whether use layout prediction module or not')
        parser.add_argument('--use_G2', type=bool, default=True, help='use G2 or not')
        parser.add_argument('--netG2', type=str, default='Unetgenerator')
        parser.add_argument('--G2_inc', type=int, default=39, help='input channels for G2')
        parser.add_argument('--use_cl_refine', type=bool, default=False, help='refine the warping results utilizing additional network')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--eqlr_sn', action='store_true', help='if true, use equlr, else use sn')
        parser.add_argument('--match_kernel', type=int, default=3, help='correspondence matrix match kernel size')
        parser.add_argument('--vgg_normal_correct', default=True, action='store_true', help='if true, correct vgg normalization and replace vgg FM model with ctx model')
        parser.add_argument('--apex', action='store_true', help='if true, use apex')
        parser.add_argument('--use_coordconv', default=False, help='if true, use coordconv in CorrNet')
        parser.add_argument('--warp_patch', default=True, action='store_true', help='use corr matrix to warp 4*4 patch')
        parser.add_argument('--warp_stride', type=int, default=4, help='corr matrix 256 / warp_stride')
        # parser.add_argument('--video_like', default=True, action='store_true', help='useful in deepfashion')
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to be used in multiscale')
        parser.add_argument('--netD_subarch', type=str, default='n_layer', help='architecture of each discriminator')
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.store_name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()  #gather options from base, train, dataset, model
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
