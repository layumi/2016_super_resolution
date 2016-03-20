clear;
p = './2014data/';
imdb.meta.sets=['train','test'];
subdir = dir(p);
counter_data=1;
for i=3:length(subdir)
    subdirname = subdir(i).name;
    filename = dir(strcat(p,subdirname));
    for j=3:length(filename)
        if(~isempty(strfind(filename(j).name,'.JPEG')))
            url = strcat(p,subdirname,'/',filename(j).name);
            im = imread(url);
            if(size(im,3)==1)
                continue;
            end
            if(size(im,1)<128||size(im,2)<128)
                continue;
            end
            imdb.images.data(:,counter_data) = cellstr(url);
            counter_data = counter_data+1;
            %disp(counter_data);
        end
    end
end

save('./2014_data_url_onlycolor.mat','imdb','-v7.3');