function [image, finish_bound1, finish_bound2] = imformat(str, arrsize)
% Read the file whose name is given in 'str', reduce its dimensions to
% 'arrsize', and return the result.
I = double(imread(str));
I = imresize(rgb2gray(I), arrsize);
GR = imquantize(I, multithresh(I, 1));

GR(GR(:, end) == 2, end) = 1.5;
GR(end, GR(end, :) == 2) = 1.5;

finish_bound1 = [find(GR(:, end) == 1.5, 1, 'first'); arrsize(2)];
finish_bound2 = [find(GR(:, end) == 1.5, 1, 'last'); arrsize(2)];

image = GR;
end