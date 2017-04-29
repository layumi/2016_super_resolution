clear;
%--------resize
img = imread('3.jpg');
img = im2single(img);  %original img as groundtruth file
[w,h,~] = size(img);  
input = imresize(img,1/2);  % compress the img as input
[w_s,h_s,~] = size(input);
%--------bicubic
tic;
result_bi = imresize(input,2);
toc;
truth = img(7:end-6,7:end-6,:);
result_bi = result_bi(7:end-6,7:end-6,:);
loss_bi = bsxfun(@minus,truth,result_bi).^2;
fprintf('bicubic_loss:%f\n',mean(reshape(sum(loss_bi,3),1,[])));
figure(2);
imshow(result_bi);
%--------srcnn

%netstruct = load('./data/SRnet-128-test/net-epoch-15.mat');   %use your model

netstruct = load('./data/SRnet-color-128/net-epoch-15.mat'); %use layumi's model

net = dagnn.DagNN.loadobj(netstruct.net);
net.mode = 'test' ;
net.conserveMemory = false;
net.move('gpu');
input_big = imresize(input, [w, h]);
tic;
net.eval({'input',gpuArray(input_big)});
toc;
index = net.getVarIndex('prediction');
result_srcnn = gather(net.vars(index).value);
loss_srcnn = bsxfun(@minus,truth,result_srcnn).^2;
fprintf('bicubic_loss:%f\n',mean(reshape(sum(loss_srcnn,3),1,[])));
imwrite(result_srcnn,'product.jpg');
figure(3);
imshow(result_srcnn);