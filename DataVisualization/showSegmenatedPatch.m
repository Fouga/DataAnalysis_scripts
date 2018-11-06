function  Patch = showSegmenatedPatch(M, red, green, blue)

thresh1 = 2000;
thresh2 = 1600;
thresh3 = 800;


l = red;
l(l>thresh1)=thresh1;
im1_8 = uint8(double(l)./double(max(l(:)))*2^8);
m=green;
m(m>thresh2)=thresh2;
im2_8 = uint8(double(m)./double(max(m(:)))*2^8);
n = blue;
n(n>thresh3)=thresh3;
im3_8 = uint8(double(n)./double(max(n(:)))*2^8);

rgbIm = cat(3, im1_8,im2_8,im3_8);
bw4_perim = bwperim(M);
overlay = imoverlay(rgbIm, bw4_perim);


cc = bwconncomp(M,8);
s = regionprops(cc,'basic');
% centroids = cat(1, s.Centroid);
rect = round(cat(1,s.BoundingBox));  
    
shift = 50;
if shift > rect(1) || shift > rect(2) || rect(1)+shift>size(red,2) || rect(2)+shift>size(red,1)
    shift = 0;
end  
cropVec = [rect(1)-shift/2 rect(2)-shift/2 rect(3)+shift rect(4)+shift];
Patch = imcrop(overlay,cropVec);        



