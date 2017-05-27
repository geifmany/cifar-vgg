# cifar-vgg
VGG16 models for CIFAR-10 and CIFAR-100 using Keras.
This package contains 2 classes one for each datasets, the architecture is based on the VGG-16 [1] with adaptation to CIFAR datasets based on [2].
The CIFAR-10 reaches a validation accuracy of 93.56%
CIFAR-100 reaches validation accuracy of 70.48%.
On instantiation the model can either be trained or loaded from previous save weight file.


References:

[1] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[2] Shuying Liu and Weihong Deng. Very deep convolutional neural network based image classifi- cation using small training sample size. In Pattern Recognition (ACPR), 2015 3rd IAPR Asian Conference on, pages 730–734. IEEE, 2015.



