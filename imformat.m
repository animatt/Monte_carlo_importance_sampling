function image = imformat(str, arrsize)
% Read the file whose name is given in 'str', reduce its dimensions to
% 'arrsize', and return the result.
I = double(imread(str));
I = imresize(rgb2gray(I), arrsize);
GR = imquantize(I, multithresh(I, 1));

GR(GR(:, end) == 2, end) = 1.5;
GR(end, GR(end, :) == 2) = 1.5;

image = GR;
end