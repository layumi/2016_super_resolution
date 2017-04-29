function net = SRnet()
% this code is written by zhedong zheng
% matconvnet model
net = dagnn.DagNN();
reluBlock = dagnn.ReLU('leak',0);
%conv  -8 -4
conv2_1Block = dagnn.Conv('size',[9 9 1 64],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv2_1',conv2_1Block,{'input'},{'conv2_1'},{'c2_1f','c2_1b'});
net.addLayer('relu2_1',reluBlock,{'conv2_1'},{'conv2_1x'},{});
conv2_2Block = dagnn.Conv('size',[1 1 64 32],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv2_2',conv2_2Block,{'conv2_1x'},{'conv2_2'},{'c2_2f','c2_2b'});
net.addLayer('relu2_2',reluBlock,{'conv2_2'},{'conv2_2x'},{});
conv2_3Block = dagnn.Conv('size',[5 5 32 1],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv2_3',conv2_3Block,{'conv2_2x'},{'prediction'},{'c2_3f','c2_3b'});

% You can try some other loss here.  Like EPELoss().
net.addLayer('loss',HuberLoss(),{'prediction','label'},'objective');
net.initParams();