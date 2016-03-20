function imt = random_cut128(im)
    w = size(im,1);
    h = size(im,2);
    rw = randperm(w-127);
    rh = randperm(h-127);
    imt = im(rw(1):rw(1)+127,rh(1):rh(1)+127,:);
