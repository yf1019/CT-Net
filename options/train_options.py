from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        # parser.add_argument('--load_size', type=int, default=286, help='Scale images to this size. The final image will be cropped to --crop_size.')
        
        # for displays
        parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=15000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--restore_epoch', type=str, default=1, help='restore epoch')

        # for training
        parser.add_argument('--continue_train', default=False, help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--train_corr_only', default=False, action='store_true', help='to train the complementary warping module only')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--no_ganFeat_loss', default=False, help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='ls', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', default=True, help='Use TTUR training scheme')

        parser.add_argument('--which_perceptual', type=str, default='4_2', help='relu5_2 or relu4_2')
        parser.add_argument('--weight_mask', type=float, default=0.0, help='weight of warped mask loss, used in direct/cycle')
        parser.add_argument('--novgg_featpair', type=float, default=10.0, help='in no vgg setting, use pair feat loss in domain adaptation')
        parser.add_argument('--D_cam', type=float, default=0.0, help='weight of CAM loss in D')
        parser.add_argument('--fm_ratio', type=float, default=1.0, help='vgg fm loss weight comp with ctx loss')
        parser.add_argument('--use_22ctx', action='store_true', help='if true, also use 2-2 in ctx loss')
        parser.add_argument('--mask_epoch', type=int, default=-1, help='useful when noise_for_mask is true, first train mask_epoch with mask, the rest epoch with noise')

        parser.add_argument('--use_D1', type=bool, default=False, help='use discriminator1 or not')
        parser.add_argument('--netD1', type=str, default='multiscalediscriminator', help='whih discriminator to use')
        parser.add_argument('--D1_inc', type=int, default=41, help='input channels for D1')
        parser.add_argument('--netD2', type=str, default='multiscalediscriminator', help='whih discriminator to use')
        parser.add_argument('--D2_inc', type=int, default=42, help='input channels for D2')
        parser.add_argument('--netD3', type=str, default='multiscalediscriminator', help='whih discriminator to use')
        
        parser.add_argument('--vgg_weights', type=list, default=[1/32.0, 1/16.0, 1/8.0, 1/4.0, 1], help='weights for the perceptual loss')
        parser.add_argument('--warp_cycle_w', type=float, default=10.0, help='weights for the cycle loss')
        parser.add_argument('--warp_self_w', type=float, default=100.0, help='weights for the l1 losss')
        parser.add_argument('--warp_Mask_entropy', type=float, default=100.0, help='weights for the warp mask entropy loss')
        parser.add_argument('--mask_l1', type=float, default=0.0, help='weights for the mask l1 loss')
        parser.add_argument('--Mask_entropy', type=float, default=100.0, help='weights for the mask entropy loss')
        parser.add_argument('--weight_gan', type=float, default=10.0, help='weights for gan')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weights for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weights for vgg')
        parser.add_argument('--vggloss', type=bool, default=True, help='use vgg loss or not')
        parser.add_argument('--weight_perceptual', type=float, default=0.001, help='weights for perceptual loss')
        parser.add_argument('--w_mask_perc', type=float, default=0.0, help='weights for perceptual loss on the mask')
        parser.add_argument('--ctx_w', type=float, default=1.0, help='weights for contextural loss')
        parser.add_argument('--PONO', default=True, action='store_true', help='use positional normalization ')
        parser.add_argument('--PONO_C', default=True, action='store_true', help='use C normalization in corr module')
        parser.add_argument('--style_loss', type=float, default=0.0, help='weights for style loss')
        parser.add_argument('--cl_perc_w', type=float, default=0.001, help='weights for contextual loss on the clothing region')
        parser.add_argument('--cl_cycle_perc_w', type=float, default=0.0001, help='weights for the perceptual loss on the cycle clothing')
        parser.add_argument('--w_origin_perc', type=float, default=2.0, help='weights for contextual loss')
        parser.add_argument('--clothes_l1', type=float, default=0.0, help='weights for the l1 loss on the DF-guided warping result')
        parser.add_argument('--warp_l1_loss', type=float, default=0.0, help='weights for the l1 loss on the initial generation result')
        parser.add_argument('--tps_warped_cl_mask_bce', type=float, default=100.0, help='weights for the BCE loss on the TPS warping result')
        parser.add_argument('--tps_warped_cl_l1', type=float, default=100.0, help='weights for the l1 loss on the TPS warping result')
        parser.add_argument('--tps_mask_loss', type=float, default=100.0, help='restrict the attention mask on the clothing area' )
        parser.add_argument('--reg_occlusion', type=float, default=100.0, help='regularization on the attention mask')
        parser.add_argument('--reg_loss', type=float, default=1.0, help='regularization on the estimation of TPS warping')

        self.isTrain = True
        parser.set_defaults(phase='train')
        return parser
