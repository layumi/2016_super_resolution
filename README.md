# 2016_super_resolution
ICCV2015 Image Super-Resolution Using Deep Convolutional Networks
I include train and test code in master branch.

# Training data
I random selected about 60,000 pic from 2014 ILSVR2014_train (only academic) You can download from https://pan.baidu.com/s/1c0TvFyw

# Result
This code get the better performance than 'bicubic' for enlarging a 2x pic. It can be trained and tested now. 

original pic -> super resolution pic (trained by matconvnet)

![](https://github.com/layumi/2016_super_resolution/blob/master/3_bicubic.jpg) 
![](https://github.com/layumi/2016_super_resolution/blob/master/3_srnet.jpg) 

![](https://github.com/layumi/2016_super_resolution/blob/master/4_bicubic.jpg) 
![](https://github.com/layumi/2016_super_resolution/blob/master/4_srnet.jpg) 

# How to train & test
1.You may compile matconvnet first by running gpu_compile.m  (you need to change some setting in it)

For more compile information, you can learn it from www.vlfeat.org/matconvnet/install/#compiling

2.run testSRnet_result.m for test result.

3.If you want to train it by yourself, you may download my data and use prepare_ur_data.m to produce imdb.mat which include every picture path.

4.Use train_SRnet.m to have fun~
 
# Improvement
1.I add rmsprop to matconvnet(You can learn more from /matlab/cnn_daga.m)

2.I fix the scale factor 2(than 2+2*rand). It seems to be easy for net to learn more information.

3.How to initial net? (You can learn more from /matlab/+dagnn/@DagNN/initParam.m) In this work, the initial weight is important
