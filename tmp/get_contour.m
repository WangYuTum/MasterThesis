function bin_contour = get_contour(file_name)
%% Return uint8 binary contour map given .mat file path
% load gt in mat format
gt_mat = load(file_name);
gt_mat = gt_mat.LabelMap;

% compute gradient and dilate
[Gmag, ~] = imgradient(gt_mat);
dilate_struct = [0 1 0; 1 1 1; 0 1 0];
boundary = imdilate(Gmag,dilate_struct);
L_boundary = logical(boundary);
bin_contour = uint8(L_boundary);
