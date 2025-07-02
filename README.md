# VMPicker: A novel cryo-EM particle picker integrating Vision Mamba with the Segment Anything Model 

VMPicker is a novel cryo-EM particle picking method that integrates the high-order Vision Mamba-based segmentation network (CryoVMUNet) with the Segment Anything Model (SAM) for precise and efficient identification of protein particles in cryo-EM micrographs. It leverages a cascaded pipeline combining Topaz denoising, high-fidelity CryoVMUNet segmentation, and SAM’s automatic mask generation to robustly detect particles under low SNR and complex backgrounds. Trained and tested on 10 diverse cryo-EM datasets, VMPicker achieves superior performance in terms of precision, F1 score, and Dice score, while maintaining high computational efficiency. It outputs standard .star files compatible with tools such as RELION and CryoSPARC, making it a powerful and practical solution for high-resolution structural analysis in cryo-EM.

-----

## Overview

![Alt text](<figures/overview.jpg>)

## Installation

#### Clone project
```
git clone https://github.com/zb01234/VMPicker.git
cd VMPicker/
```
#### Download SAM model
```
curl https://calla.rnet.missouri.edu/CryoSegNet/pretrained_models.tar.gz --output pretrained_models.tar.gz
tar -xvf pretrained_models.tar.gz
rm pretrained_models.tar.gz
```
#### Dataset
```
All dataset used in this paper are publicly available can be accessed here:
- cryoPPP: https://github.com/BioinfoMachineLearning/cryoppp

```
#### Create conda environment
```
conda env create -f environment.yml
conda activate VMPicker
```

## Step-by-step running

#### Preprocessing
```
Topaz denoise：
cd utils/topaz/
python topaz_denoise.py -o ./data/10947/denoised/ ./data/10947/micrographs/*.jpg

contrast enhance：
cd utils/topaz//utils
python enhance_contrast.py
```

#### Train CryoVMUNet
```
    python train.py
```

#### Prediction
```
    python predict.py --empiar_id 10081

    python generate_starfile.py --empiar_id 10081 --file_name 10081.star
```

