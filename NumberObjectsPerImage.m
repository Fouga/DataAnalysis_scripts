function T = NumberObjectsPerImage(read_dir)
A = readtable(fullfile(read_dir,'Allpositions_filter3D.txt'));
frames = max(A.Frame);
opt = max(A.Optical);

k=1;
NumObjects = zeros(1,frames*opt);
for i=1:frames
    for j=1:opt
        NumObjects(k) = sum(A.Frame==i & A.Optical==j);
        k = k+1;
    end
end
Frame = reshape(repmat(1:frames,opt,1),1,frames*opt)';
Opt = repmat(1:opt, 1,frames)';
T = table(Frame,Opt,NumObjects');     