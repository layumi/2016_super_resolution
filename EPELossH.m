classdef EPELossH < dagnn.Loss
%EPE
  methods
    function outputs = forward(obj, inputs, params)
      [w,h,~,~] = size(inputs{1});
      %0.5*(c-x)^2
      t = bsxfun(@minus,inputs{2},inputs{1});
      t = gather(t);
      t = reshape(t,1,[]);
      t(abs(t)>1) = abs(t(abs(t)>1));
      t(abs(t)<1) = 0.5*(t(abs(t)<1)).^2;
      outputs{1} = sum(t)/(w*h); 
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [w,h,~,~] = size(inputs{2});
      %x -y ;
      Y = gather(bsxfun(@minus,inputs{1},inputs{2}));
      %Y(Y>1)= 1;  % x-y>1 
      %Y(Y<-1) = -1; % y-x<1
      derInputs{1} = gpuArray(bsxfun(@times, derOutputs{1},Y));
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = EPELossH(varargin)
      obj.load(varargin) ;
    end
  end
end
