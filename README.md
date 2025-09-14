
# Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022 -- Oral)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.09881)

## Network Architecture

<img src = "https://i.imgur.com/ulLoEig.png"> 

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.

## Demo

To test the pre-trained Restormer models of [Deraining](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u), [Motion Deblurring](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK), [Defocus Deblurring](https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4?usp=sharing), and [Denoising](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0) on your own images, you can either use Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C2818h7KnjNv4R1sabe14_AYL7lWhmu6?usp=sharing), or command line as following
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```
Example usage to perform Defocus Deblurring on a directory of images:
```
python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
```
Example usage to perform Defocus Deblurring on an image directly:
```
python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'
```

## Training and Evaluation

Training and Testing instructions for Deraining, Motion Deblurring, Defocus Deblurring, and Denoising are provided in their respective directories. Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="left">Task</th>
    <th align="center">Training Instructions</th>
    <th align="center">Testing Instructions</th>
  </tr>
  <tr>
    <td>Motion Deblurring</td>
    <td align="center"><a href="Motion_Deblurring/README.md#training">Link</a></td>
    <td align="center"><a href="Motion_Deblurring/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1qla3HEOuGapv1hqBwXEMi2USFPB2qmx_?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Gaussian Denoising</td>
    <td align="center"><a href="Denoising/README.md#training">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation">Link</a></td>
    <td align="center"><a href="https://drive.google.com/drive/folders/1rEAHUBkA9uCe9Q0AzI5zkYxePSgxYDEG?usp=sharing">Download</a></td>
  </tr>
  <tr>
    <td>Real Denoising</td>
    <td align="center"><a href="Denoising/README.md#training-1">Link</a></td>
    <td align="center"><a href="Denoising/README.md#evaluation-1">Link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/1CsEiN6R0hlmEoSTyy48nnhfF06P5aRR7/view?usp=sharing">Download</a></td>
  </tr>
</table>

## Results
Experiments are performed for different image processing tasks including, image deraining, single-image motion deblurring, defocus deblurring (both on single image and dual pixel data), and image denoising (both on Gaussian and real data). 


<details>
<summary><strong>Single-Image Motion Deblurring</strong> (click to expand) </summary>

<p align="center"><img src = "https://i.imgur.com/htagDSl.png" width="400"></p>
</details>



<details>
<summary><strong>Gaussian Image Denoising</strong> (click to expand) </summary>

Top super-row: learning a single model to handle various noise levels.
Bottom super-row: training a separate model for each noise level.

<table>
  <tr>
    <td> <img src = "https://i.imgur.com/4vzV8Qy.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/Sx986Xs.png" width="500"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Grayscale</b></p></td>
    <td><p align="center"><b>Color</b></p></td>
  </tr>
</table>
</details>

<details>
<summary><strong>Real Image Denoising</strong> (click to expand) </summary>

<img src = "https://i.imgur.com/6v5PRxj.png">
</details>

## Citation
If you use Restormer, please consider citing:

    @inproceedings{Zamir2021Restormer,
        title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
        author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
                and Fahad Shahbaz Khan and Ming-Hsuan Yang},
        booktitle={CVPR},
        year={2022}
    }
