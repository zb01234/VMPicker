# 🧊 VMPicker: A Novel Cryo-EM Particle Picker Leveraging Vision Mamba and the Segment Anything Model

VMPicker is a novel cryo-EM particle picking method that integrates the Vision Mamba-based segmentation network (CryoVMUNet) with the Segment Anything Model (SAM) for precise and efficient identification of protein particles in cryo-EM micrographs. It leverages a cascaded pipeline combining Topaz denoising, high-fidelity CryoVMUNet segmentation, and SAM’s automatic mask generation to robustly detect particles under low SNR and complex backgrounds. Trained and tested on 10 diverse cryo-EM datasets, VMPicker achieves superior performance in terms of precision, F1 score, and Dice score, while maintaining high computational efficiency. It outputs standard .star files compatible with tools such as RELION and CryoSPARC, making it a powerful and practical solution for high-resolution structural analysis in cryo-EM.

-----


## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/zb01234/VMPicker.git
cd VMPicker/
```

### 2. Download pretrained SAM models
```bash
curl -L https://calla.rnet.missouri.edu/CryoSegNet/pretrained_models.tar.gz -o pretrained_models.tar.gz
tar -xvf pretrained_models.tar.gz
rm pretrained_models.tar.gz
```

### 3. Download datasets
All datasets used in this paper are publicly available:

- **CryoPPP**: [https://github.com/BioinfoMachineLearning/cryoppp](https://github.com/BioinfoMachineLearning/cryoppp)

### 4. Create and activate Conda environment
```bash
conda env create -f environment.yml
conda activate VMPicker
```

---

## 🧩 Step-by-Step Usage

### 1. Preprocessing

#### a. Topaz Denoising
```bash
cd utils/topaz/
python topaz_denoise.py -o ./data/10947/denoised/ ./data/10947/micrographs/*.jpg
```

#### b. Contrast Enhancement
```bash
cd utils/topaz/utils/
python enhance_contrast.py
```

---

### 2. Train CryoVMUNet
```bash
python train.py
```

---

### 3. Particle Prediction
```bash
python predict.py --empiar_id 10081
```

---

### 4. Generate .star File
```bash
python generate_starfile.py --empiar_id 10081 --file_name 10081.star
```

---

## 📁 Output
- **Particle coordinates** saved as `.star` files  
- Fully compatible with:
  - [RELION](https://www3.mrc-lmb.cam.ac.uk/relion/)
  - [CryoSPARC](https://cryosparc.com/)


