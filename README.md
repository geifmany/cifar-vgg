# cifar-vgg
This is a Keras model based on VGG16 architecture for CIFAR-10 and CIFAR-100. it can be used either with pretrained weights file or trained from scratch.

This package contains 2 classes one for each datasets, the architecture is based on the VGG-16 [1] with adaptation to CIFAR datasets based on [2]. By running the py files you can get a sample of a trining and estimation of validation error.

The CIFAR-10 reaches a validation accuracy of 93.56%
CIFAR-100 reaches validation accuracy of 70.48%.
On instantiation the model can either be trained or loaded from previous saved weight file.

[cifar-100 weights](https://drive.google.com/file/d/0B4odNGNGJ56qTEdnT1RjTU44Zms/view?usp=sharing&resourcekey=0-1cQT1h4rx1QVuYsnZNvtKw)
[cifar-10 weights](https://drive.google.com/file/d/0B4odNGNGJ56qVW9JdkthbzBsX28/view?usp=sharing&resourcekey=0-4S027Hj5jKjZluUe4rt8IA)


References:

[1] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[2] Shuying Liu and Weihong Deng. Very deep convolutional neural network based image classifi- cation using small training sample size. In Pattern Recognition (ACPR), 2015 3rd IAPR Asian Conference on, pages 730–734. IEEE, 2015.



