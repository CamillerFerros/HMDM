# HMDM-Net
The paper titled "High-resolution Fusion Mamba and Deep-feature Memory for Medical Image Segmentation" 



## 1. Prepare data

- [Synapse multi-organ segmentation] The Synapse datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). 
- [ACDC cardiac segmentation]
- [Decathlon brain tumor segmentation]

Put pretrained weights into folder **"data/"** under the main "HMDM-Net" directory, e.g., **"data/Synapse"**, **"data/ACDC"**.

## 2. Environment
- We recommend an evironment with python >= 3.8, and then install the following dependencies:
```
pip install -r requirements.txt

## 3. Train 

### a. Run the training script
For example, for training ACDC model, run the following command:
```
python train_ACDC.py

## Acknowledgements

This code is built on the top of [Swin UNet](https://github.com/HuCaoFighting/Swin-Unet) and [Agileformer](https://github.com/sotiraslab/AgileFormer), we thank to their efficient and neat codebase. 

