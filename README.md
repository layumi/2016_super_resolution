# 2016_super_resolution
ICCV2015 Image Super-Resolution Using Deep Convolutional Networks
I include train and test code in master branch.

# Training data
I random selected about 60,000 pic from 2014 ILSVR2014_train (only academic) You can download from https://pan.baidu.com/s/1c0TvFyw

# Result
This code get the better performance than 'bicubic' for enlarging a 2x pic. It can be trained and tested now. 

original pic -> super resolution pic (trained by matconvnet)

![](https://github.com/layumi/2016_super_resolution/blob/master/2_small.JPEG) ========= ![](https://github.com/layumi/2016_super_resolution/blob/master/2_product.jpg)

# Improvmet
1.I add rmsprop to matconvnet(You can learn more from /matlab/cnn_daga.m)

2.I fix the scale factor 2(than 2+2*rand). It seems to be easy for net to learn more information.

3.How to initial net? (You can learn more from /matlab/+dagnn/@DagNN/initParam.m) In this work, the initial weight is important
