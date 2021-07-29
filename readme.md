# CT-Net: Complementary Transfering Network for Garment Transfer with Arbitrary Geometric Changes，CVPR'21.

Code for CT-Net: Complementary Transfering Network for Garment Transfer with Arbitrary Geometric Changes，CVPR'21. 

\[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_CT-Net_Complementary_Transfering_Network_for_Garment_Transfer_With_Arbitrary_Geometric_CVPR_2021_paper.pdf)] \[[Supp](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Yang_CT-Net_Complementary_Transfering_CVPR_2021_supplemental.pdf)]

![image](https://github.com/yf1019/CT-Net/blob/master/img/result.png)

## Abstract

>Garment transfer shows great potential in realistic applications with the goal of transfering outfits across different people images. However, garment transfer between
>images with heavy misalignments or severe occlusions still remains as a challenge. In this work, we propose Complementary Transfering Network (CT-Net) to adaptively model
>different levels of geometric changes and transfer outfits between different people. In specific, CT-Net consists of three modules: i) A complementary warping module first
>estimates two complementary warpings to transfer the desired clothes in different granularities. ii) A layout prediction module is proposed to predict the target layout, which guides the preservation or generation of the body parts in the synthesized images. iii) A dynamic fusion module adaptively combines the advantages of the complementary warpings to render the garment transfer results. Extensive experiments conducted on DeepFashion dataset demonstrate that our network synthesizes high-quality garment transfer images and significantly outperforms the state-of-art methods both qualitatively and quantitatively. 

## Installation

Clone the Synchronized-BatchNorm-PyTorch repository.

```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

Install dependencies:

```
pip install -r requirements.txt
```

## Dataset.

* Download the In-shop Clothes Retrieval Benchmark of [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).We use images of resolution 256 * 256. Save it as `dataset/img`.
* We use [OpenPose](https://github.com/Hzzone/pytorch-openpose) to estimate pose of DeepFashion. Download the [openpose results](https://drive.google.com/drive/folders/1j_27swc9cKqH6R35vvNOBi1lLUFuNjVL?usp=sharing) and save it as `dataset/pose`. 
* We use [LIP_JPPNet](https://github.com/Engineering-Course/LIP_JPPNet) and [Densepose](https://github.com/facebookresearch/DensePose) to estimate the segmentation and densepose descriptor of DeepFashion. Download the [seg and densepose results](https://drive.google.com/drive/folders/1w801EchmCWnSxuZf0WEklwr9KOO68c1d?usp=sharing) and save it as `dataset/seg_dp`.
* We use [Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting) to randomly remove part of the limbs in the body images in the dynamic generation module for mimicking image inpainting. Downloads the dataset and save it as `dataset/train_mask`.

## Inference

Download the pretrained model from [here](https://drive.google.com/drive/folders/1xZlKeOIuxsO58AAmsFrDSYUqdVxfqse_?usp=sharing) and save them in 'checkpoints/one_corr_tps'

Then run the command.

```
python test.py 
```

## Training

Download the pretrained VGG weights for vgg-based loss from [here](https://drive.google.com/file/d/1hU2wBEB2KrMZ8F8FKGoTT95IXf6UBOQl/view?usp=sharing) and save it at `model/`. Our model is trained in two stages. 

* First, the Complementary Warping Module is trained for 20 epoches to estimate reasonable warpings. Run the command.

  ```
  python -m visdom.server -port 8097
  python train.py --niter 20 --niter_decay 0 --display_port 8097 --train_corr_only
  ```

* Then our model is jointly trained in an end-to-end manner for another 80 epoches. Learning rate is fixed at 0.0002 for the first 40 epoches and then decays to zero linearly in the remaining
  steps. Run the command.

  ```
  python train.py --niter 40 --niter_decay 40 --display_port 8097
  ```

## Citation

If you use this code for your research, please cite our papers.

```
@inproceedings{yang2021ct,
  title={CT-Net: Complementary Transfering Network for Garment Transfer with Arbitrary Geometric Changes},
  author={Yang, Fan and Lin, Guosheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9899--9908},
  year={2021}
}
```

## License

The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

## Acknowledgments

This code borrows heavily from [CoCosNet](https://github.com/microsoft/CoCosNet). We also thanks Jiayuan Mao for his [Synchronized Batch Normalization code](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).









