clear
addpath(genpath('/extra'));
ref = imread("clean.bmp");
noisy = imread("noisy.bmp");
clean = imread("clean.bmp");
figure,imshow(noisy)
denoised = BM3D("noisy.bmp",10);
figure,imshow(denoised)
