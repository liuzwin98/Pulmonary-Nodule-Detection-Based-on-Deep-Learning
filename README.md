# Pulmonary-Nodule-Detection-Undergraduate-project-
Using U-net, 3D CNN, cGAN to accomplish Pulmonary Nodule Detection. Also it's my undergraduate research program.


# Content
A pulmonary nodule detection model is established based on deep convolutional neural networks to achieve lung nodule extraction and false-positive filtering.The model uses a U-shaped network[1] structure for candidate nodule extraction, and a multi-scale 3D convolutional neural network[2] to filter false positives of lung nodules, retain the true nodules and improve the specificity of the model.To solve the problem of insufficient positive samples in the data set, the CT-GAN model[3] based on the conditional generated adversarial network is used to expand the image data of the lung nodules.\<br>
Specifically, it includes three part codes corresponding to the three folders.

# How to start
## Software
python 3.6, CUDA 10.0, cudnn 7.6.5, Keras 2.2.5, tensorflow-gpu 1.14.0, scipy 1.4.1, Pillow 7.0.0, SimpleITK 1.2.4, opencv-python 4.2.0.32,
pydicom.

## Usage
Detailed description in each section folder. I just reproduced their work. Please refer to their papers for details. You can test their performance separately, and finally integrate the three parts to complete the whole process of pulmonary nodule detection

## Dataset
LIDC's subset: **LUAN16 datast** 


# Notes
Due to hardware limitations, I only use a small part of the data, so the performance of the model is not very good.And there is a certain overfitting phenomenon.

# Results

