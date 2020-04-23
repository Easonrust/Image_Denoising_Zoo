addpath(genpath('BM3D'));
addpath(genpath('WNNM'));
addpath(genpath('PSNR'));
addpath(genpath('MSE'));
addpath(genpath('SSIM'));
pathRoot='images/';
imgDir=dir([pathRoot '*.png']);

bm3d_sumMSE=0;
bm3d_sumPSNR=0;
bm3d_sumSSIM=0;

wnnm_sumMSE=0;
wnnm_sumPSNR=0;
wnnm_sumSSIM=0;

sigma=15;

for i=1:92
    readPath=[pathRoot,imgDir(i).name]
    cleanImg=imread(readPath);
    cleanImg = repmat(cleanImg,[1,1,3]);

    noisyImg=imnoise(cleanImg,'gaussian',0, sigma^2/255^2);

    referenceImg=imread(readPath);
    referenceImg = repmat(referenceImg,[1,1,3]);
    denoisedImg = BM3D(noisyImg,sigma);

    mse  =  MSE(referenceImg,cleanImg,denoisedImg);
    psnr =  PSNR(referenceImg,cleanImg,denoisedImg);
    ssim =  SSIM(referenceImg,cleanImg,denoisedImg);
    
    bm3d_sumMSE=mse+bm3d_sumMSE;
    bm3d_sumPSNR=psnr+bm3d_sumPSNR;
    bm3d_sumSSIM=ssim+bm3d_sumSSIM;
    
    denoisedImg = WNNM_WRAP(noisyImg,sigma);

    mse  =  MSE(referenceImg,cleanImg,denoisedImg);
    psnr =  PSNR(referenceImg,cleanImg,denoisedImg);
    ssim =  SSIM(referenceImg,cleanImg,denoisedImg);
    
    wnnm_sumMSE=mse+wnnm_sumMSE;
    wnnm_sumPSNR=psnr+wnnm_sumPSNR;
    wnnm_sumSSIM=ssim+wnnm_sumSSIM;
end
bm3d_average_mse=bm3d_sumMSE/92;
bm3d_average_psnr=bm3d_sumPSNR/92;
bm3d_average_ssim=bm3d_sumSSIM/92;

wnnm_average_mse=wnnm_sumMSE/92;
wnnm_average_psnr=wnnm_sumPSNR/92;
wnnm_average_ssim=wnnm_sumSSIM/92;
