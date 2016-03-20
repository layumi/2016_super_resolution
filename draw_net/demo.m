clear;
%simplenn net
net = cnn_cifar_init();
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'error') ;
draw_net(net,'cifar_net');

%dagnn net
net =test_net(); % 2_stream_inception_net
draw_net(net,'test_net');
