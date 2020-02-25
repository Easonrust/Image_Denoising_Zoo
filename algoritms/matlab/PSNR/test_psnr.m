clear
ref = imread("clean.bmp");
noisy = imread("noisy.bmp");
clean = imread("clean.bmp")
diff = ref - clean;
psnr1  = PSNR(ref,ref,noisy)
psnr2  = psnr(ref,noisy)