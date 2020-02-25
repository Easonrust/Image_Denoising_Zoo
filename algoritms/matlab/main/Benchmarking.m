addpath(genpath('../BM3D'));
addpath(genpath('../FOE'));
addpath(genpath('../KSVD'));
addpath(genpath('../NCSR'));
addpath(genpath('../FOE'));
addpath(genpath('../WNNM'));
addpath(genpath('../PSNR'));
addpath(genpath('../MSE'));
addpath(genpath('../SSIM'));

IS_INCLUDED_BM3D = 1;
IS_INCLUDED_KSVD = 0;
IS_INCLUDED_WNNM = 0;
IS_INCLUDED_FOE  = 0;
IS_INCLUDED_NCSR = 0;
IS_INCLUDED_EPLL = 0;

% SIGMA [ 10, 15, 20, 25, 50 ]
SIGMA =[ 25];

for s = SIGMA
    
    original_mse  = [];
    original_psnr = [];
    original_ssim = [];
    
    bm3d_mse   = [];
    bm3d_psnr  = [];
    bm3d_ssim  = [];
    bm3d_time  = [];
    
    ksvd_mse   = [];
    ksvd_psnr  = [];
    ksvd_ssim  = [];
    ksvd_time  = [];

    wnnm_mse   = [];
    wnnm_psnr  = [];
    wnnm_ssim  = [];
    wnnm_time  = [];
    
    foe_mse   = [];
    foe_psnr  = [];
    foe_ssim  = [];
    foe_time  = [];
    
    ncsr_mse   = [];
    ncsr_psnr  = [];
    ncsr_ssim  = [];
    ncsr_time  = [];
    
    epll_mse   = [];
    epll_psnr  = [];
    epll_ssim  = [];
    epll_time  = [];
    
    %images
    for i = 1:5
        
        BatchPath = strcat('../../../dataset1/batch',int2str(i));

        NOISY = strcat(BatchPath ,'/noisy.jpg');
        REFERENCE = strcat(BatchPath ,'/reference.jpg');
        CLEAN = strcat(BatchPath ,'/clean.jpg');

        % Read images
        Noisy_Image     = imread(NOISY); 
        Reference_Image = imread(REFERENCE);
        Clean_Image     = imread(CLEAN);
             
        % Init table
        tableResults = table();
        
        % Evaluation before denoising
        mse  =  MSE(Reference_Image,Clean_Image,Noisy_Image);
        psnr =  PSNR(Reference_Image,Clean_Image,Noisy_Image);
        ssim =  SSIM(Reference_Image,Clean_Image,Noisy_Image);
        sprintf('Before denoising: MSE= %g, PSNR= %g, SSIM= %g',mse, psnr, ssim);
        
        original_mse = [original_mse mse];
        original_psnr = [original_psnr psnr];
        original_ssim = [original_ssim ssim];

        row = cell2table({'before-denoising',mse,psnr,ssim,0});
        tableResults = [tableResults ; row];

        %--------------------------------------------------------------------------------------------------------
        
        % BM3D
        if IS_INCLUDED_BM3D == 1
            tic
            Denoised_Image_BM3D = BM3D(NOISY,s);
            time = toc;
            mse  =  MSE(Reference_Image,Clean_Image,Denoised_Image_BM3D);
            psnr =  PSNR(Reference_Image,Clean_Image,Denoised_Image_BM3D);
            ssim =  SSIM(Reference_Image,Clean_Image,Denoised_Image_BM3D);
            sprintf('After BM3D denoising: MSE= %g, PSNR= %g, SSIM= %g',mse, psnr, ssim);

            bm3d_mse = [bm3d_mse mse];
            bm3d_psnr = [bm3d_psnr psnr];
            bm3d_ssim = [bm3d_ssim ssim];
            bm3d_time = [bm3d_time time];

            row = cell2table({'BM3D', mse, psnr, ssim, time});
            tableResults = [tableResults ; row];
        end

        %--------------------------------------------------------------------------------------------------------

        % KSVD
        if IS_INCLUDED_KSVD == 1
            tic
            Denoised_Image_KSVD = KSVD_WRAP(NOISY,s);
            time = toc;
            mse  =   MSE(Reference_Image,Clean_Image,Denoised_Image_KSVD);
            psnr =  PSNR(Reference_Image,Clean_Image,Denoised_Image_KSVD);
            ssim =  SSIM(Reference_Image,Clean_Image,Denoised_Image_KSVD);
            sprintf('After KSVD denoising: MSE= %g, PSNR= %g, SSIM= %g',mse, psnr, ssim);
            
            ksvd_mse  = [ksvd_mse mse];
            ksvd_psnr = [ksvd_psnr psnr];
            ksvd_ssim = [ksvd_ssim ssim];
            ksvd_time = [ksvd_time time];

            row = cell2table({'KSVD',mse,psnr,ssim,time});
            tableResults = [tableResults ; row];
        end
        %--------------------------------------------------------------------------------------------------------

        % WNNM
        if IS_INCLUDED_WNNM == 1
            tic
            Denoised_Image_WNNM = WNNM_WRAP(NOISY,s);
            time = toc;
            mse  =   MSE(Reference_Image,Clean_Image,Denoised_Image_WNNM);
            psnr =  PSNR(Reference_Image,Clean_Image,Denoised_Image_WNNM);
            ssim =  SSIM(Reference_Image,Clean_Image,Denoised_Image_WNNM);
            sprintf('After WNNM denoising: MSE= %g, PSNR= %g, SSIM= %g',mse, psnr, ssim);
            
            wnnm_mse  = [wnnm_mse mse];
            wnnm_psnr = [wnnm_psnr psnr];
            wnnm_ssim = [wnnm_ssim ssim];
            wnnm_time = [wnnm_time time];

            row = cell2table({'WNNM',mse,psnr,ssim,time});
            tableResults = [tableResults ; row];
        end
        %--------------------------------------------------------------------------------------------------------

        % FOE
        if IS_INCLUDED_FOE == 1
            tic
            Denoised_Image_FOE = FOE_WRAP(NOISY, s);
            time = toc;
            mse  =   MSE(Reference_Image,Clean_Image,Denoised_Image_FOE);
            psnr =  PSNR(Reference_Image,Clean_Image,Denoised_Image_FOE);
            ssim =  SSIM(Reference_Image,Clean_Image,Denoised_Image_FOE);
            sprintf('After FOE denoising: MSE= %g, PSNR= %g, SSIM= %g',mse, psnr, ssim);
            
            foe_mse  = [foe_mse mse];
            foe_psnr = [foe_psnr psnr];
            foe_ssim = [foe_ssim ssim];
            foe_time = [foe_time time];

            row = cell2table({'FOE',mse,psnr,ssim,time});
            tableResults = [tableResults ; row];
        end

        %--------------------------------------------------------------------------------------------------------

        % NCSR
        if IS_INCLUDED_NCSR == 1
            tic
            Denoised_Image_NCSR = NCSR_WRAP(NOISY,s);
            time = toc;
            mse  =   MSE(Reference_Image,Clean_Image,Denoised_Image_NCSR);
            psnr =  PSNR(Reference_Image,Clean_Image,Denoised_Image_NCSR);
            ssim =  SSIM(Reference_Image,Clean_Image,Denoised_Image_NCSR);
            sprintf('After NCSR denoising: MSE= %g, PSNR= %g, SSIM= %g',mse, psnr, ssim);
            
            ncsr_mse  = [ncsr_mse mse];
            ncsr_psnr = [ncsr_psnr psnr];
            ncsr_ssim = [ncsr_ssim ssim];
            ncsr_time = [ncsr_time time];

            row = cell2table({'NCSR',mse,psnr,ssim,time});
            tableResults = [tableResults ; row];
        end
        
        %--------------------------------------------------------------------------------------------------------

               
        % EPLL
        if IS_INCLUDED_EPLL == 1
            tic
            Denoised_Image_EPLL = EPLL_WRAP(NOISY, s);
            time = toc;
            mse  =   MSE(Reference_Image,Clean_Image,Denoised_Image_EPLL);
            psnr =  PSNR(Reference_Image,Clean_Image,Denoised_Image_EPLL);
            ssim =  SSIM(Reference_Image,Clean_Image,Denoised_Image_EPLL);
            sprintf('After EPLL denoising: MSE= %g, PSNR= %g, SSIM= %g',mse, psnr, ssim);
            
            epll_mse  = [epll_mse mse];
            epll_psnr = [epll_psnr psnr];
            epll_ssim = [epll_ssim ssim];
            epll_time = [epll_time time];

            row = cell2table({'EPLL',mse,psnr,ssim,time});
            tableResults = [tableResults ; row];
        end

        %--------------------------------------------------------------------------------------------------------

        filename = strcat('../../../results/','results_sigma',int2str(s),'_image', int2str(i),'.csv');
        tableResults.Properties.VariableNames = {'Algorithm' 'MSE' 'PSNR' 'SSIM', 'TIME'};
        writetable(tableResults,filename, 'Delimiter',',','QuoteStrings',true)
        type filename
    
    end
    
    finalResults = table();
    
    % MSE
    row = cell2table({'MSE',mean(original_mse),mean(bm3d_mse),mean(ksvd_mse),mean(wnnm_mse),mean(foe_mse),mean(ncsr_mse),mean(epll_mse)});
    finalResults = [finalResults ; row];
    
    % PSNR
    row = cell2table({'PSNR',mean(original_psnr),mean(bm3d_psnr),mean(ksvd_psnr),mean(wnnm_psnr),mean(foe_psnr),mean(ncsr_psnr),mean(epll_psnr)});
    finalResults = [finalResults ; row];
   
    % SSIM
    row = cell2table({'SSIM',mean(original_ssim),mean(bm3d_ssim),mean(ksvd_ssim),mean(wnnm_ssim),mean(foe_ssim),mean(ncsr_ssim),mean(epll_ssim)});
    finalResults = [finalResults ; row];
    
    row = cell2table({'TIME',0,mean(bm3d_time),mean(ksvd_time),mean(wnnm_time),mean(foe_time),mean(ncsr_time),mean(epll_time)});
    finalResults = [finalResults ; row];
    
    filename = strcat('../../../results/', 'final_results_sigma',int2str(s),'.csv');
    finalResults.Properties.VariableNames = {'QM' 'ORIGINAL' 'BM3D' 'KSVD' 'WNNM' 'FOE' 'NCSR' 'EPLL'};
    
    writetable(finalResults,filename, 'Delimiter',',','QuoteStrings',true)
    type filename

end   

% Show images
% figure; imshow(Noisy_Image);   
% figure; imshow(Denoised_Image);



