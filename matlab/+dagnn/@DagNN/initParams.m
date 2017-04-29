function initParams(obj)
% INITPARAM  Initialize the paramers of the DagNN
%   OBJ.INITPARAM() uses the INIT() method of each layer to initialize
%   the corresponding parameters (usually randomly).

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

for l = 1:numel(obj.layers)
    p = obj.getParamIndex(obj.layers(l).params) ;
    %params = obj.layers(l).block.initParams() ;
    params=[];
    if(isequal(class(obj.layers(l).block),'dagnn.Conv'))
        size=obj.layers(l).block.size;
        h =size(1); w =size(2); in = size(3); out=size(4);
        %sc = sqrt(2/(h*w*out)) ;
        sc = sqrt(3/(h*w*in)) ;
        params{1,1} = (rand(h, w, in, out, 'single')*2 - 1)*sc;
        params{1,2} = zeros(out, 1, 'single') ;
    elseif(isequal(class(obj.layers(l).block),'dagnn.ConvTranspose'))
        size=obj.layers(l).block.size;
        h =size(1); w =size(2); in = size(3); out=size(4);
        sc = sqrt(2/(h*w*in)) ;
        params{1,1} = randn(h, w, in, out, 'single')*sc ;
        params{1,2} = zeros(in, 1, 'single') ;
    elseif(isequal(class(obj.layers(l).block),'dagnn.BatchNorm'))
        params{1,1} = ones(out,1, 'single') ;
        params{1,2} = zeros(out, 1, 'single') ;
        params{1,3} = zeros(out, 2, 'single') ;
    else
        params = obj.layers(l).block.initParams() ;
    end
    switch obj.device
        case 'cpu'
            params = cellfun(@gather, params, 'UniformOutput', false) ;
        case 'gpu'
            params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
    end
    [obj.params(p).value] = deal(params{:}) ;
    if(isequal(class(obj.layers(l).block),'dagnn.Conv'))
        [obj.params(p(1)).learningRate]=.1;
        [obj.params(p(2)).learningRate]=2;
        %[obj.params(p(1)).trainMethod] = 'rmsprop';
        %[obj.params(p(2)).trainMethod] = 'rmsprop';
        if(~isempty(strfind(obj.layers(l+1).name,'loss')))
            [obj.params(p(1)).learningRate]= 0.01;
            [obj.params(p(2)).learningRate]= 0.2;
        end
    end
    if(isequal(class(obj.layers(l).block),'dagnn.BatchNorm'))
        [obj.params(p(1)).learningRate]=2;
        [obj.params(p(2)).learningRate]=1;
        [obj.params(p(3)).learningRate]=0.5;
        [obj.params(p(1)).weightDecay]=0;
        [obj.params(p(2)).weightDecay]=0;
        [obj.params(p(3)).weightDecay]=0;
    end
end
