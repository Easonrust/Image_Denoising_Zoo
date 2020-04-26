addpath(genpath('WNNM'));
addpath(genpath('PSNR'));
addpath(genpath('MSE'));
addpath(genpath('SSIM'));
pathRoot='images/';
imgDir=dir([pathRoot '*.png']);

PSNR = [];
SSIM = [];

sigma=15;

for i=1:92
    randn('seed',0);
    readPath=[pathRoot,imgDir(i).name]
    cleanImg=im2double(imread(readPath));
    

    noisyImg=cleanImg + sigma* randn(size(cleanImg));

    Par   = ParSet(sigma);
    denoisedImg = WNNM_DeNoising( noisyImg, cleanImg, Par );
    PSNR = [PSNR csnr( cleanImg, denoisedImg, 0, 0 )];
    SSIM = [SSIM cal_ssim( denoisedImg, cleanImg, 0, 0 )];
    fprintf( 'Denoised Image: sigma = %2.3f, PSNR = %2.2f, SSIM = %2.4f \n\n\n', sigma, csnr( cleanImg, denoisedImg, 0, 0 ),cal_ssim( denoisedImg, cleanImg, 0, 0 ) );
end
mPSNR=mean(PSNR,2);
mSSIM=mean(SSIM,2);
fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR,mSSIM);
