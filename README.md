# Pulmonary Nodule Detection (Undergraduate project)
Using U-net, 3D CNN, cGAN to accomplish Pulmonary Nodule Detection. Also it's my undergraduate research program. ([Chinese Blog series](https://blog.csdn.net/liuz_notes/article/details/106438215))


# Content
A pulmonary nodule detection model is established based on deep convolutional neural networks to achieve lung nodule extraction and false-positive filtering.The model uses a U-shaped network[1] structure for candidate nodule extraction, and a multi-scale 3D convolutional neural network[2] to filter false positives of lung nodules, retain the true nodules and improve the specificity of the model.To solve the problem of insufficient positive samples in the data set, the CT-GAN model[3] based on the conditional generated adversarial network is used to expand the image data of the lung nodules.<br><br>

**Specifically, it includes three part codes corresponding to the three folders.**

# How to start
## Software
python 3.6, CUDA 10.0, cudnn 7.6.5, Keras 2.2.5, tensorflow-gpu 1.14.0, scipy 1.4.1, Pillow 7.0.0, SimpleITK 1.2.4, opencv-python 4.2.0.32,
pydicom.

## Usage
Detailed description in each section folder. I just reproduced their work. Please refer to their papers for details. You can test their performance separately, and finally integrate the three parts to complete the whole process of pulmonary nodule detection

## Dataset
LIDC's subset: **LUNA16 datast** 


# Notes
Due to hardware limitations, I only use a small part of the data, so the performance of the model is not very good.And there is a certain overfitting phenomenon.

# Results
<div align=center><img src="https://github.com/liuzwin98/Pulmonary-Nodule-Detection-Based-on-Deep-Learning/blob/master/Results/Unet_loss.jpg" width = "400" height = "300"> <br>
Figure 1. Unet_loss
</div><br>

<div align=center><img src="https://github.com/liuzwin98/Pulmonary-Nodule-Detection-Based-on-Deep-Learning/blob/master/Results/Unet_results.png" width = "300" height = "300"> 
  <img src="https://github.com/liuzwin98/Pulmonary-Nodule-Detection-Based-on-Deep-Learning/blob/master/Results/Unet_results2.png" width = "300" height = "300">
  <br>
Figure 2. Unet_results
</div><br>

<div align=center><img src="https://github.com/liuzwin98/Pulmonary-Nodule-Detection-Based-on-Deep-Learning/blob/master/Results/3DCNN_acc.jpg" width = "350" height = "300"> <br>
Figure 3. 3DCNN_acc
</div><br>

<div align=center><img src="https://github.com/liuzwin98/Pulmonary-Nodule-Detection-Based-on-Deep-Learning/blob/master/Results/Roc.png" width = "350" height = "300"> <br>
Figure 4. Roc_curve
</div><br>

<div align=center><img src="https://github.com/liuzwin98/Pulmonary-Nodule-Detection-Based-on-Deep-Learning/blob/master/Results/GAN_loss_curve.jpg" width = "350" height = "300"> <br>
Figure 5. GAN_loss
</div><br>

<div align=center><img src="https://github.com/liuzwin98/Pulmonary-Nodule-Detection-Based-on-Deep-Learning/blob/master/Results/GAN_results.png" width = "700" height = "300"> <br>
Figure 6. GAN_results
</div><br>


# Reference
[1]Ronneberger O , Fischer P , Brox T . U-Net: Convolutional Networks for Biomedical Image Segmentation[J]. 2015.<br>
[2]Kim B C , Yoon J S , Choi J S , et al. Multi-scale gradual integration CNN for false positive reduction in pulmonary nodule detection[J]. Neural Networks, 2019, 115:1-10.<br>
[3]Mirsky Y, Mahler T, Shelef I, et al. CT-GAN: Malicious tampering of 3D medical imagery using deep learning[C]//28th {USENIX} Security Symposium ({USENIX} Security 19). 2019: 461-478.<br>
