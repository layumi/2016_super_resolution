addpath matlab;
addpath examples;
run matlab/vl_setupnn ;

vl_compilenn('enableGpu', true, ...
'cudaRoot', '/usr/local/cuda', ...  %change it 
'cudaMethod', 'nvcc',...
'enableCudnn',false,... 
'cudnnroot','local');
%}
warning('off');