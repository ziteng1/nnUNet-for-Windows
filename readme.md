nnUNet is the first deep learning segmentation pipline that is designed to deal with the dataset diversity found in the domain. It is also implemented for Linux environments.

This repo contains my modifications to the original code to enable training, validation and testing on a Windows machine.

If you are completely new to nnUNet, you can first find more introduction in the [original nnUNet repo](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

[2023.03.23] Update: nnUNet authors stated that [nnUNet V2](https://github.com/MIC-DKFZ/nnUNet) is now available. The authors improved code structures and made other trivial edits. This repo will stay on V1 by now since V2 does not likely to improve performance (or only slightly) in most cases.

# How to use

nnUNet is designed to have a interface that is as simple as possible. However, as a developer and python user, I find it most comfortable to run a python file directly (so we can make sure what's going on and what's going wrong (hopefully not)). So instead of install nnUNet as a framework, let's just download the python code and I will let you know which file do which job. **Note: If you need multi-GPU training, you still have to install nnUNet as a framework**

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

<p align="center">
    <img src="https://github.com/ziteng1/nnUNet_for_Windows/blob/master/documentation/paths.PNG" width="500">
</p>

## Process Dataset

You may have more understanding of what kind of dataset need to be pre-processed by reading [this documentation](https://github.com/ziteng1/nnUNet-for-Windows/blob/master/documentation/dataset_conversion.md).

1. Now let's assume we have a [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/) format dataset. So we need to convert it to a nnUNet format dataset. The code you need to do this is \nnUNet_for_Windows\nnunet\experiment_planning\nnUNet_convert_decathlon_task.py. If you prefer prompt, you can cd to that file's directory and run this:
```bash
python nnUNet_convert_decathlon_task -i F:\amos22\Task01_AMOS_CT
```
where "F:\amos22\Task01_AMOS_CT" is the path I store the raw MSD dataset that I wish to convert.

Or, if you use Pycharm or similar IDE like me, you can do this by edit configurations and then hit run:
<p align="center">
    <img src="https://github.com/ziteng1/nnUNet_for_Windows/blob/master/documentation/dataset_conversion.PNG" width="500">
</p>

The results will be stored in the "base" path you just modified in the previous step.

2. MSD dataset contains a json file that contains some information about the whole dataset. If you are preparing your own dataset, except rename your files [as required by nnUNet](https://github.com/ziteng1/nnUNet-for-Windows/blob/master/documentation/data_format_inference.md), you also need to generate this json file using generate_dataset_json() function located in \nnUNet_for_Windows\nnunet\dataset_conversion\utils.py. Here is an example from me:

<p align="center">
    <img src="https://github.com/ziteng1/nnUNet_for_Windows/blob/master/documentation/generate_json.PNG" width="500">
</p>

3. Then nnUNet need to walk through the dataset, determine training plans and preprocessing the dataset (normalizing, resampling, concat input and label to a single npz, etc.). This is done by \nnUNet_for_Windows\nnunet\experiment_planning\nnUNet_plan_and_preprocess.py. So run this:
```bash
python nnUNet_convert_decathlon_task -t 1
```
where '1' is the task id (the number in \nnUNet_for_Windows\nnunet\DATASET\nnUNet_raw\nnUNet_raw_data\Task**001**_AMOS_CT)

<p align="center">
    <img src="https://github.com/ziteng1/nnUNet_for_Windows/blob/master/documentation/dataset_plan_and_process.PNG" width="500">
</p>

## Run training

Now we can start the training. Default is 1000 epoch. 1000 epoch is a lot, let alone 5-fold cross validation. Maybe first train and test on the first fold. The training interface is actually this file: \nnUNet_for_Windows\nnunet\run\run_training.py

So cd to that folder and use this line to run:
```bash
python run_training 3d_fullres nnUNetTrainerV2 1 0
```
where '1' still refers to the task id.

<p align="center">
    <img src="https://github.com/ziteng1/nnUNet_for_Windows/blob/master/documentation/run_training.PNG" width="500">
</p>
