from data.base_dataset import BaseDataset, get_params, get_transform
import torch
import torchvision.transforms as transforms
from PIL import Image
import util.util as util
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from scipy.ndimage.filters import gaussian_filter


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, paths = self.get_paths(opt)
        image_paths = paths['image_paths']

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        
        if opt.phase == 'test':
            ref_paths = paths['ref_paths']
            ref_paths = ref_paths[:opt.max_dataset_size]
            self.ref_paths = ref_paths

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

        self.real_reference_probability = opt.real_reference_probability
        self.hard_reference_probability = opt.hard_reference_probability
        self.ref_dict, self.train_test_folder = self.get_ref(opt)

        head = [1, 2, 4, 11, 13]
        arms = [3, 14, 15]
        legs = [16, 17]
        upper_clothes = [5, 6, 7, 10]
        lower_clothes = [9, 12]
        shoes = [8, 18, 19]

        self.mask = [head, arms, legs, upper_clothes, lower_clothes, shoes]

        dir_M = '_mask'
        self.dir_M = os.path.join(opt.dataroot, 'train' + dir_M)
        self.M_paths = sorted(self.make_dataset(self.dir_M))
        print('Mask num : {}'.format(len(self.M_paths)))

        dp_head = [23, 24]
        dp_arms = [3, 4, 15, 16, 17, 18, 19, 20, 21, 22]
        dp_legs = [7, 8, 9, 10, 11, 12, 13, 14]
        dp_torso = [1, 2]
        dp_shoes = [5, 6]
        dp_lower_cloth = []
        self.dp_mask = [dp_head, dp_arms, dp_legs, dp_torso, dp_lower_cloth, dp_shoes]

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_label_tensor(self, path):
        label = Image.open(path)
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        return label_tensor, params1

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        label_tensor, params1, image_lds = self.get_label_tensor(label_path)

        # input image (real images)
        image_path = self.image_paths[index]

        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params1)
        image_tensor = transform_image(image)

        ref_tensor = 0
        label_ref_tensor = 0

        if self.opt.phase == 'test':

            if self.opt.random_show:
                path_ref = random.sample(self.image_paths, 1)[0]
            else:
                path_ref = self.ref_paths[index]

            image_ref = Image.open(path_ref).convert('RGB')
            if self.opt.dataset_mode != 'deepfashion':
                path_ref_label = path_ref.replace('.jpg', '.png')
                path_ref_label = self.imgpath_to_labelpath(path_ref_label)
            else: 
                path_ref_label = self.imgpath_to_labelpath(path_ref)

            label_ref_tensor, params, ref_lds = self.get_label_tensor(path_ref_label)
            transform_image = get_transform(self.opt, params)
            ref_tensor = transform_image(image_ref)
            #ref_tensor = self.reference_transform(image_ref)

            self_ref_flag = torch.ones_like(ref_tensor)
        else:
            pair = False
            if self.opt.dataset_mode == 'deepfashion':
                key = image_path.replace('\\', '/').split(self.opt.dataroot + '/')[-1]
                val = self.ref_dict[key]
                ref_name = val[0]
                key_name = key
                if os.path.dirname(ref_name) == os.path.dirname(key_name) and os.path.basename(ref_name).split('_')[0] == os.path.basename(key_name).split('_')[0]:
                    path_ref = os.path.join(self.opt.dataroot, ref_name)
                    image_ref = Image.open(path_ref).convert('RGB')
                    label_ref_path = self.imgpath_to_labelpath(path_ref)
                    label_ref_tensor, params, ref_lds = self.get_label_tensor(label_ref_path)
                    transform_image = get_transform(self.opt, params)
                    ref_tensor = transform_image(image_ref) 
                    pair = True

            if not pair:
                image_ref = image.copy()
                label_ref_tensor, params, ref_lds = self.get_label_tensor(label_path)
                transform_image = get_transform(self.opt, params)
                ref_tensor = transform_image(image)

                path_ref = image_path
                #
            #ref_tensor = self.reference_transform(image)
            self_ref_flag = torch.ones_like(ref_tensor)

        seg_img_path = image_path.replace('img', 'seg_and_dp').strip('\n')[:-4] + '_seg.png'
        seg_img = Image.open(seg_img_path)
        seg_img = seg_img.convert('RGB')
        transform_seg_img = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        seg_img_tensor = transform_seg_img(seg_img) * 255.0
        seg_img_tensor = self.convert_seg(seg_img_tensor)

        ref_seg_path = path_ref.replace('img', 'seg_and_dp').strip('\n')[:-4] + '_seg.png'
        ref_seg = Image.open(ref_seg_path)
        ref_seg = ref_seg.convert('RGB')

        ref_seg_array = np.array(ref_seg)

        # ref_cloth = np.array(image_ref)

        transform_ref_seg = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        ref_seg_tensor = transform_ref_seg(ref_seg) * 255.0
        ref_seg_tensor = self.convert_seg(ref_seg_tensor)

        if self.opt.use_dp:
            img_dp_path = image_path.replace('img', 'seg_and_dp').strip('\n')[:-4] + '_IUV.png'
            img_dp = Image.open(img_dp_path)
            img_dp = img_dp.convert('RGB')
            transform_dp_img = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
            img_dp_tensor = transform_dp_img(img_dp) * 255.0
            img_dp_tensor = self.convert_dp_mask(img_dp_tensor)

            ref_dp_path = path_ref.replace('img', 'seg_and_dp').strip('\n')[:-4] + '_IUV.png'
            ref_dp = Image.open(ref_dp_path)
            ref_dp = ref_dp.convert('RGB')
            transform_dp_img = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            ref_dp_tensor = transform_dp_img(ref_dp) * 255.0
            ref_dp_tensor = self.convert_dp_mask(ref_dp_tensor)

            if self.opt.use_cl_refine and self.opt.refine_type == 'densepose':
                cloth = self.mask[3] + self.mask[4]
                ref_cloth_mask = ref_seg_array
                for item in cloth:
                    t_mask = ref_seg_array == item
                    ref_cloth_mask = ref_cloth_mask * (1 - t_mask) + t_mask * 100
                ref_cloth_mask[ref_cloth_mask != 100] = 0
                ref_cloth_mask[ref_cloth_mask == 100] = 1
                ref_cloth = np.array(image_ref) * ref_cloth_mask

                texture_ref = get_texture(ref_cloth, np.array(ref_dp)[:,:,::-1])[:,:,::-1]
                texture_ref_c = np.zeros([24, 200, 200, 3])

                for i in range(4):
                    for j in range(6):
                        texture_ref_c[(6 * i + j), :, :, :] = texture_ref[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200), :]

                bg_im = np.zeros(ref_cloth.shape)
                warped_cloth = TransferTexture(texture_ref_c, bg_im, np.array(img_dp)[:,:,::-1])
                warped_cloth = Image.fromarray(np.uint8(warped_cloth))
                transform_image = get_transform(self.opt, params1)
                warped_cloth_tensor = transform_image(warped_cloth)
            else:
                warped_cloth_tensor = torch.zeros_like(image_tensor)
        else:
            img_dp_tensor = torch.zeros_like(image_tensor)
            ref_dp_tensor = torch.zeros_like(image_tensor)
            warped_cloth_tensor = torch.zeros_like(image_tensor)

        if self.opt.use_mask:
            M_path = self.M_paths[np.random.randint(12000)]
            M = Image.open(M_path).convert('L').convert('RGB')
            M = M.resize((256, 256))
            transform_M = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
            M_tensor = transform_M(M)
        else:
            M_tensor = torch.zeros_like(image_tensor)

        if self.opt.load_lds:
            input_dict = {'label': label_tensor,
                          'image': image_tensor,
                          'image_path': image_path,
                          'ref_path': path_ref,
                          'self_ref': self_ref_flag,
                          'ref': ref_tensor,
                          'label_ref': label_ref_tensor,
                          'seg_img': seg_img_tensor,
                          'ref_seg': ref_seg_tensor,
                          'M_tensor': M_tensor,
                          'img_dp_mask': img_dp_tensor,
                          'ref_dp_mask': ref_dp_tensor,
                          'warped_cloth': warped_cloth_tensor,
                          'image_lds': image_lds,
                          'ref_lds': ref_lds
                          }
        else:
            input_dict = {'label': label_tensor,
                      'image': image_tensor,
                    'image_path': image_path,
                    'ref_path': path_ref,
                      'self_ref': self_ref_flag,
                      'ref': ref_tensor,
                      'label_ref': label_ref_tensor,
                      'seg_img': seg_img_tensor,
                      'ref_seg': ref_seg_tensor,
                          'M_tensor': M_tensor,
                      'img_dp_mask': img_dp_tensor,
                          'ref_dp_mask': ref_dp_tensor,
                          'warped_cloth': warped_cloth_tensor
                      }

        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_ref(self, opt):
        pass

    def imgpath_to_labelpath(self, path):
        return path

    def convert_seg(self, seg):
        t_seg = seg[0, :, :]
        size = t_seg.size()
        # oneHot_size = (len(self.mask) + 1, size[1], size[2])
        # c_seg = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_seg = torch.FloatTensor(torch.Size(size)).zero_()

        for i, items in enumerate(self.mask, start=1):
            for item in items:
                # t_mask = torch.FloatTensor((t_seg.long().cpu().numpy() == item))
                c_seg[t_seg.long().cpu().numpy() == item] = i

        return c_seg.unsqueeze(0)

    def color_mask(self, seg):
        Mask = np.array(seg)

        Mask_vis = cv2.applyColorMap((Mask * 15).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
        return Image.fromarray(Mask_vis.astype('uint8')).convert('RGB')

    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        f = dir.split('/')[-1].split('_')[-1]
        print(dir, f)
        dirs = os.listdir(dir)
        for img in dirs:
            path = os.path.join(dir, img)
            # print(path)
            images.append(path)
        return images

    def convert_dp_mask(self, dp):
        dp_mask = dp[2, :, :]
        size = dp_mask.size()
        # oneHot_size = (len(self.mask) + 1, size[1], size[2])
        # c_seg = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_dp_mask = torch.FloatTensor(torch.Size(size)).zero_()

        for i, items in enumerate(self.dp_mask, start=1):
            for item in items:
                # t_mask = torch.FloatTensor((t_seg.long().cpu().numpy() == item))
                c_dp_mask[dp_mask.long().cpu().numpy() == item] = i

        return c_dp_mask.unsqueeze(0)


def get_texture(im, IUV, solution=200):
    #
    # inputs:
    #   solution is the size of generated texture, in notebook provided by facebookresearch the solution is 200
    #   If use lager solution, the texture will be sparser and smaller solution result in denser texture.
    #   im is original image
    #   IUV is densepose result of im
    # output:
    #   TextureIm, the 24 part texture of im according to IUV
    solution_float = float(solution) - 1

    U = IUV[:, :, 1]
    V = IUV[:, :, 2]
    parts = list()
    for PartInd in range(1, 25):  ## Set to xrange(1,23) to ignore the face part.
        actual_part = np.zeros((solution, solution, 3))
        x, y = np.where(IUV[:, :, 0] == PartInd)
        if len(x) == 0:
            parts.append(actual_part)
            continue

        u_current_points = U[x, y]  # Pixels that belong to this specific part.
        v_current_points = V[x, y]
        ##
        tex_map_coords = ((255 - v_current_points) * solution_float / 255.).astype(int), (
                    u_current_points * solution_float / 255.).astype(int)
        for c in range(3):
            actual_part[tex_map_coords[0], tex_map_coords[1], c] = im[x, y, c]

        valid_mask = np.array((actual_part.sum(-1) != 0) * 1, dtype='uint8')
        radius_increase = 10
        kernel = np.ones((radius_increase, radius_increase), np.uint8)
        dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
        region_to_fill = dilated_mask - valid_mask
        invalid_region = 1 - valid_mask
        actual_part_max = actual_part.max()
        actual_part_min = actual_part.min()
        actual_part_uint = np.array((actual_part - actual_part_min) / (actual_part_max - actual_part_min) * 255,
                                    dtype='uint8')

        #         invalid_region = np.stack((invalid_region, invalid_region, invalid_region), -1)
        #         print(actual_part_uint.shape)
        #         print(invalid_region.shape)
        #         print(valid_mask.shape)
        actual_part_uint = cv2.inpaint(actual_part_uint, invalid_region, 1, cv2.INPAINT_TELEA).transpose((2, 0, 1))

        actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
        # only use dilated part
        actual_part = actual_part * dilated_mask
        #         print(actual_part_uint.shape)
        actual_part = actual_part.transpose((1, 2, 0))
        parts.append(actual_part)

    TextureIm = np.zeros([solution * 6, solution * 4, 3]);

    for i in range(4):
        for j in range(6):
            TextureIm[(solution * j):(solution * j + solution), (solution * i):(solution * i + solution), :] = parts[
                i * 6 + j]

    return TextureIm


def TransferTexture(TextureIm,im,IUV):
    U = IUV[:,:,1]
    V = IUV[:,:,2]
    #
    R_im = np.zeros(U.shape)
    G_im = np.zeros(U.shape)
    B_im = np.zeros(U.shape)
    ###
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
        tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
        #####
        R = tex[:,:,0]
        G = tex[:,:,1]
        B = tex[:,:,2]
        ###############
        x,y = np.where(IUV[:,:,0]==PartInd)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        ##
        r_current_points = R[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]
        g_current_points = G[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]
        b_current_points = B[((255-v_current_points)*199./255.).astype(int),(u_current_points*199./255.).astype(int)]
        ##  Get the RGB values from the texture images.
        R_im[IUV[:,:,0]==PartInd] = r_current_points
        G_im[IUV[:,:,0]==PartInd] = g_current_points
        B_im[IUV[:,:,0]==PartInd] = b_current_points
    generated_image = np.concatenate((B_im[:,:,np.newaxis],G_im[:,:,np.newaxis],R_im[:,:,np.newaxis]), axis =2 ).astype(np.uint8)
    BG_MASK = generated_image==0
    generated_image[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
    return generated_image