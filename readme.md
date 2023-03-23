nnUNet is the first deep learning segmentation pipline that is designed to deal with the dataset diversity found in the domain. It is also implemented for Linux environments.

This repo contains my modifications to the original code to enable training, validation and testing on a Windows machine.

If you are completely new to nnUNet, you can first find more introduction in the [original nnUNet repo](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

[2023.03.23] Update: nnUNet authors stated that [nnUNet V2](https://github.com/MIC-DKFZ/nnUNet) is now available. The authors improved code structures and made other trivial edits. This repo will stay on V1 by now since V2 does not likely to improve performance (or only slightly) in most cases.

# How to use

nnUNet is designed to have a interface that is as simple as possible. However, as a developer and python user, I find it most comfortable to run a python file directly (so we can make sure what's going on and what's going wrong (hopefully not)). So instead of install nnUNet directly, let's just download the python code and I will let you know which file do which job.

## Set up enviornment

1. Install python, Pytorch and cuda: 

    Here I use anaconda to manage my environments on Windows: [Download anaconda here](https://www.anaconda.com/).
    After installing anaconda, open Anaconda prompt and creat an enviroment (Python 3.8 or later is preferred).
    ```bash
    conda create -n py39 python=3.9
    ```
    Activate the enviroment.
    ```bash
    activate py39
    ```
    Now install Pytorch, cuda, cudatoolkit, etc, in anaconda
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
    If you don't use anaconda, or you don't have a GPU or you just prefer CPU-only training, simply copy the command from [here](https://pytorch.org/) instead.

2. Install required packages:

    Now we install all the packages used in nnUNet pipline. Copy and paste to the anaconda prompt:
    ```bash
    pip install tqdm dicom2nifti scikit-image medpy scipy batchgenerators==0.21 numpy sklearn SimpleITK pandas requests nibabel tifffile
    ```
    Except the batchgenerators, all other packages can be as latest as possible. Pytorch version need to be torch>=1.6.0a and scikit-image need to be scikit-image>=0.14.
    
## Change the path file

Since Linux and Windows use different path separators, you need to manually change the path file here: \nnUNet_for_Windows\nnunet\paths.py. Change **line 33** to **line 35** to your local path like this.

    
# nnU-Net

In 3D biomedical image segmentation, dataset properties like imaging modality, image sizes, voxel spacings, class 
ratios etc vary drastically.
For example, images in the [Liver and Liver Tumor Segmentation Challenge dataset](https://competitions.codalab.org/competitions/17094) 
are computed tomography (CT) scans, about 512x512x512 voxels large, have isotropic voxel spacings and their 
intensity values are quantitative (Hounsfield Units).
The [Automated Cardiac Diagnosis Challenge dataset](https://acdc.creatis.insa-lyon.fr/) on the other hand shows cardiac 
structures in cine MRI with a typical image shape of 10x320x320 voxels, highly anisotropic voxel spacings and 
qualitative intensity values. In addition, the ACDC dataset suffers from slice misalignments and a heterogeneity of 
out-of-plane spacings which can cause severe interpolation artifacts if not handled properly. 

In current research practice, segmentation pipelines are designed manually and with one specific dataset in mind. 
Hereby, many pipeline settings depend directly or indirectly on the properties of the dataset 
and display a complex co-dependence: image size, for example, affects the patch size, which in 
turn affects the required receptive field of the network, a factor that itself influences several other 
hyperparameters in the pipeline. As a result, pipelines that were developed on one (type of) dataset are inherently 
incomaptible with other datasets in the domain.

**nnU-Net is the first segmentation method that is designed to deal with the dataset diversity found in the domain. It 
condenses and automates the keys decisions for designing a successful segmentation pipeline for any given dataset.**

nnU-Net makes the following contributions to the field:
