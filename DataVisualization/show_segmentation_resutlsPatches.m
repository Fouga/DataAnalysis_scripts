function rgbIm = show_segmentation_resutlsPatches(source_dir,frame,optical,segmentation_dir)

% threshold = 2000;
if frame < 10 
  counter = strcat('00',int2str(frame)); 
elseif frame < 100 
  counter = strcat('0',int2str(frame));   
else
  counter = int2str(frame);   
end
prefix = 'section_';
name = strcat(prefix, counter);
mask_name = [segmentation_dir, name, '_', int2str(optical) ,'.pbm'];
BW = imread(mask_name);
if sum(BW(:))>0
    [pth filter ext] = fileparts(source_dir);
    ext = '.tif';
            green = imread([pth '/1/', name, '_0',  int2str(optical), ext]);
            red = imread([pth '/2/', name, '_0',  int2str(optical), ext]);
            blue = imread([pth '/3/', name, '_0',  int2str(optical), ext]);
            % convert to 8 bit
%             l = red;
%             l(l>threshold)=threshold;
%             im1_8 = uint8(double(l)./double(max(l(:)))*2^8);
%             m=green;
%             m(m>threshold)=threshold;
%             im2_8 = uint8(double(m)./double(max(m(:)))*2^8);
%             n = blue;
%             n(n>threshold)=threshold;
%             im3_8 = uint8(double(n)./double(max(n(:)))*2^8);
%             rgbIm = cat(3, im1_8,im2_8,im3_8);
rgbIm =  cat(3, green,red,blue);
    cc = bwconncomp(BW,8);
    s = regionprops(cc,'basic');
    centroids = cat(1, s.Centroid);
% figure, imshow(rgbIm)
% hold on
% plot(centroids(:,1),centroids(:,2), 'y*')
% hold off

% pause(2)

% bw4_perim = bwperim(BW);
% overlay = imoverlay(rgbIm, bw4_perim);
% figure, imshow(overlay,[])

    % save patches
    save_dir = [segmentation_dir 'Patches/'];
    if ~exist(save_dir)
        mkdir(save_dir)
    end

    rect = round(cat(1,s.BoundingBox));
    shift =200;% 2*max(max(rect(:,3:4)))+2;
    for i =1:size(centroids,1)
        rect2 = rect(i,:);
        if shift > rect2(1) || shift > rect2(2) || rect2(1)+shift>size(rgbIm,2) || rect2(2)+shift>size(rgbIm,1)
            shift = 0;
            Im = imcrop(rgbIm,[rect2(1)-shift/2 rect2(2)-shift/2 rect2(3)+shift rect2(4)+shift]);
        else
            Im = imcrop(rgbIm,[rect2(1)-shift/2 rect2(2)-shift/2 rect2(3)+shift rect2(4)+shift]);
        end
        name  = sprintf('Frame_%i_optical_%i_Bacterium_%i.tif',frame,optical,i);
    %     figure, imshow(Im,[])
        imwrite(Im,[save_dir name]);

    end
end
