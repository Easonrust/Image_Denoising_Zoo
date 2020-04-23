function denoised = WNNM_WRAP(image, sigma)

    N_Img = double(rgb2gray(image));

    Par   = ParSet(sigma);   
    Denoised_Image = WNNM_DeNoising(N_Img, Par);
    
    recombinedRGBImage = cat(3, Denoised_Image, Denoised_Image, Denoised_Image);
        
    denoised = uint8(recombinedRGBImage);
   
end



