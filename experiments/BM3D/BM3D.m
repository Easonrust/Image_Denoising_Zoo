function  denoised = BM3D(image, sigma)
    yRGB = im2double(image);
    zRGB = yRGB + (sigma/255)*randn(size(yRGB));
    [~, yRGB_est] = CBM3D(1, zRGB, sigma); 
    
    denoised = uint8(255 * mat2gray(yRGB_est));
end