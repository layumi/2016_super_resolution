function train_SRnet(varargin)
addpath('../MATLAB');
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('./2014_data_url_onlycolor.mat') ;
imdb = imdb.imdb;
imdb.meta.sets=['train','val'];
ss = size(imdb.images.data);
imdb.images.set = ones(1,ss(2));
imdb.images.set(ceil(rand(1,ceil(ss(2)/20))*ss(2))) = 2;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = SRnet();
net.conserveMemory = true;
%net = load('data/48net-cifar-v3-custom-dither0.1-128-3hard/f48net-cpu.mat');


% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------

opts.train.batchSize = 128;
%opts.train.numSubBatches = 1 ;
opts.train.continue = true; 
opts.train.gpus = 2;
opts.train.prefetch = false ;
%opts.train.sync = false ;
%opts.train.errorFunction = 'multiclass' ;
opts.train.expDir = '/home/zzd/super-resolution/data/SRnet-v1-ycbcr-128' ; 
opts.train.learningRate = [1e-5*ones(1,3) 1e-6*ones(1,1)];
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.derOutputs = {'objective',1} ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

%record
if(~isdir(opts.expDir))
    mkdir(opts.expDir);
    copyfile('SRnet.m',opts.expDir);
end

% Call training function in MatConvNet
[net,info] = cnn_train_daga(net, imdb, @getBatch,opts) ;


% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
% --------------------------------------------------------------------
imlist = imdb.images.data(:,batch) ;
batch_size = numel(batch);
img = zeros(128,128,3,batch_size,'single');
training = opts.learningRate>0;
for i=1:batch_size
    p1 = imlist{i};
    im_1 = imread(p1);
    %im_1 = rgb2ycbcr(im_1);
    %im_1 = rgb2gray(im_1);
    im_1 = im2single(im_1);
    im_1 = single(random_cut128(im_1));
    img(:,:,:,i) = im_1;
end
if(rand>0.5)
    img = fliplr(img);
end
label = img(7:end-6,7:end-6,:,:);
[w,h,~,~] = size(img);
r = 2 + 2*rand;
input = imresize(imresize(img,1/r),[w,h]);
inputs = {'input',gpuArray(input),'label',gpuArray(label)};
