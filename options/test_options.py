from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--save_per_img', action='store_true', help='if specified, save per image')
        parser.add_argument('--show_corr', default='true', action='store_true', help='if specified, save bilinear upsample correspondence')
        # parser.add_argument('--load_size', type=int, default=256, help='Scale images to this size. The final image will be cropped to --crop_size.')

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.set_defaults(no_flip=True, help='if specified, do not flip the images for data argumentation')
        parser.set_defaults(phase='test', help='train, val, test, etc')

        parser.add_argument('--use_D1', type=bool, default=False, help='use Discriminator1 or not')
        parser.add_argument('--restore_epoch', type=str, default='latest', help='name of the checkpoint')
        parser.add_argument('--test_txt', type=str, default='demo', help='name of the test list')
        parser.add_argument('--random_show', type=bool, default=False, help='use random image pairs for test')
        self.isTrain = False
        return parser
