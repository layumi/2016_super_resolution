function imt = random_cut33(im)
    w = size(im,1);
    h = size(im,2);
    rw = randperm(w-32);
    rh = randperm(h-32);
    imt = im(rw(1):rw(1)+32,rh(1):rh(1)+32,:);
