clc;
clear all;
close all;

% Choose whether normalization to a reference dataset is needed
imgNorm=false;

% Initialize metrics storage for each modality
allMetrics = struct('SSIM', {[]}, 'MSE', {[]}, 'PSNR', {[]}, 'L1', {[]}, 'GSSIM', {[]});

% Load the appropriate network based on modality
loadedData = load('*\**.mat', '-mat'); % Type in the full path of the generator network. *-full path of the directory, **-name of the network
net = loadedData.netGenerator; % Assuming the variable name in the .mat file is 'net'

dirx = '*'; % * - full path to the input image directory
dirxref = []; %Full path to the reference image directory (can be left empty)

% Get lists of all input image files in the directory
files = dir(fullfile(dirx, '*.png'));  % Assuming the images are in PNG format

% Filter input image files based on the patterns 'MPEF', 'THG'
grayImages1Files = {files(contains({files.name}, 'MPEF')).name};
grayImages2Files = {files(contains({files.name}, 'THG')).name};

if isempty(files)
    error('No PNG files in the input directory');
end
if isempty(grayImages1Files)
    error('No MPEF files in the input directory');
end
if isempty(grayImages2Files)
    error('No THG files in the input directory');
end

% Read images into arrays
grayImages1 = cellfun(@(f) im2single(imread(fullfile(dirx, f))), grayImages1Files, 'UniformOutput', false);
grayImages2 = cellfun(@(f) im2single(imread(fullfile(dirx, f))), grayImages2Files, 'UniformOutput', false);
% If normalization is enabled, the same is done for the reference dataset. Data value ranges are subsequently calculated for both datasets
if imgNorm
    try
        if ~exist(dirxref, 'dir')
            error('DIR_NOT_FOUND');
        end
        filesref = dir(fullfile(dirxref, '*.png'));  % Assuming the images are in PNG format
        % Get lists of all reference image files in the directory
        if isempty(filesref)
            error('NO_PNG_FILES');
        end
        grayImages1FilesRef = {filesref(contains({filesref.name}, 'MPEF')).name};
        if isempty(grayImages1FilesRef)
            error('NO_MPEF_FILES');
        end
        grayImages2FilesRef = {filesref(contains({filesref.name}, 'THG')).name};
        if isempty(grayImages2FilesRef)
            error('NO_THG_FILES');
        end
        grayImagesRef1 = cellfun(@(f) im2single(imread(fullfile(dirxref, f))), grayImages1FilesRef, 'UniformOutput', false);
        grayImagesRef2 = cellfun(@(f) im2single(imread(fullfile(dirxref, f))), grayImages2FilesRef, 'UniformOutput', false);
        %Determine the value range of the reference dataset (MPEF and THG modalities)
        Ref1_max=-inf;
        Ref1_min=inf;
        Ref2_max=-inf;
        Ref2_min=inf;
    
        for jk=1:length(grayImagesRef1)
            current_min1=min(grayImagesRef1{jk}(:));
            current_min2=min(grayImagesRef2{jk}(:));
            current_max1=max(grayImagesRef1{jk}(:));
            current_max2=max(grayImagesRef2{jk}(:));
                if current_min1<Ref1_min
                    Ref1_min=current_min1;
                end
                if current_min2<Ref2_min
                    Ref2_min=current_min2;
                end
                if current_max1>Ref1_max
                    Ref1_max=current_max1;
                end
                if current_max2>Ref2_max
                    Ref2_max=current_max2;
                end
        end
        %Determine the value range of the external dataset (MPEF and THG modalities)
        Ext1_max=-inf;
        Ext1_min=inf;
        Ext2_max=-inf;
        Ext2_min=inf;
        
        for jk=1:length(grayImages1)
            current_min1=min(grayImages1{jk}(:));
            current_min2=min(grayImages2{jk}(:));
            current_max1=max(grayImages1{jk}(:));
            current_max2=max(grayImages2{jk}(:));
            if current_min1<Ext1_min
                Ext1_min=current_min1;
            end
            if current_min2<Ext2_min
                Ext2_min=current_min2;
            end
            if current_max1>Ext1_max
                Ext1_max=current_max1;
            end
            if current_max2>Ext2_max
                Ext2_max=current_max2;
            end
        end
    catch ME
        imgNorm=false;
        switch ME.message
            case 'DIR_NOT_FOUND'
                fprintf('Reference dataset directory does not exist: %s. Proceeding without normalization.\n',dirxref);
    
            case 'NO_PNG_FILES'
                fprintf('No PNG files found in reference dataset directory: %s. Proceeding without normalization.\n', dirxref);
    
            case 'NO_MPEF_FILES'
                fprintf('No MPEF files found in reference dataset directory: %s. Proceeding without normalization.\n', dirxref);
    
            case 'NO_THG_FILES'
                fprintf('No THG files found in reference dataset directory: %s. Proceeding without normalization.\n', dirxref);
    
            otherwise
                fprintf(['Unexpected error: ' ME.message]);
        end
    end
end

clearvars grayImagesRef1 grayImagesRef2 grayImages1 grayImages2

% Define patch size for reconstruction
patchSize = 128;

% Define stride
stride = 32;

% Preallocate per-image metric storage
numImages = numel(grayImages1Files);

ssim_all   = cell(numImages, 1);
mse_all    = cell(numImages, 1);
psnr_all   = cell(numImages, 1);
l1_all     = cell(numImages, 1);
gssim_all  = cell(numImages, 1);

% Colorize the New Image
parfor i = 1:numel(grayImages1Files) % Parallel for loop. If no Parallel Computing Toolbox exists or the available resources are scarce, can be substituted with a simple for loop
    % Load the new grayscale images to be colorized
    newGrayImage1 = single(imread(fullfile(dirx, grayImages1Files{i}))) / 255;
    newGrayImage2 = single(imread(fullfile(dirx, grayImages2Files{i}))) / 255;
    %Normalization
    if imgNorm
        ext1_min=min(newGrayImage1(:));
        ext1_max=max(newGrayImage1(:));
        ext2_min=min(newGrayImage2(:));
        ext2_max=max(newGrayImage2(:));
        newGrayImage1 = (newGrayImage1 - ext1_min) / (ext1_max - ext1_min) * (Ref1_max - Ref1_min) + Ref1_min;
        newGrayImage2 = ((newGrayImage2 - ext2_min) / (ext2_max - ext2_min) * (Ref2_max - Ref2_min) + Ref2_min);
    end

    % Extract patches from the new grayscale images
    [X_new, patchPositions] = extractPatchesForPrediction(newGrayImage1, newGrayImage2, patchSize, stride);
    
    % Convert to dlarray
    dlX_new = dlarray(X_new, 'SSCB');
    
    % Predict the colors for the new grayscale image
    Y_new = forward(net, dlX_new);
    
    % Reconstruct the colorized image from patches using Gaussian weighting
    colorizedImage = reconstructImage(Y_new, size(newGrayImage1), patchPositions, patchSize);
    
    % Apply a bilateral filter to preserve edges
    colorizedImage = bilateralFilter(colorizedImage);

    try
        % Load the corresponding ground truth color image
        groundTruthImage = im2single(imread(fullfile(dirx, [erase(grayImages1Files{i}, 'MPEF.png') 'HE.png'])));
        groundTruthImage = im2single(groundTruthImage);
        % Extract ground truth patches
        groundTruthPatches = extractGroundTruthPatches(groundTruthImage, patchPositions, patchSize);
        
        %Temporary arrays
        patchSSIM = [];
        patchMSE = [];
        patchPSNR = [];
        patchL1 = [];
        patchGSSIM = [];
        
        % Compute metrics for each patch
        for patchIdx = 1:size(Y_new, 4)
            predictedPatch = Y_new(:, :, :, patchIdx);
            groundTruthPatch = groundTruthPatches(:, :, :, patchIdx);
        
            % MSE
            mse_value = immse(single(extractdata(predictedPatch)), groundTruthPatch);
        
            % PSNR
            psnr_value = 10 * log10(1 / mse_value);  % Max pixel value is 1 after normalization
        
            % SSIM
            dummyPredictedPatch=im2single(extractdata(predictedPatch));
            dummyGroundTruthPatch=groundTruthPatch;
            ssim_value=zeros(3,1);
            % Per channel SSIM
            for ii=1:3
                ssim_value(ii) = ssim(dummyPredictedPatch(:,:,ii), dummyGroundTruthPatch(:,:,ii));
            end
        
            % Average SSIM
            ssim_value=mean(ssim_value);
        
            % L1 Loss
            l1_value = mean(abs(single(extractdata(predictedPatch)) - groundTruthPatch), 'all');
        
            % GSSIM (Gradient SSIM)
            gssim_value = gssim(single(extractdata(predictedPatch)), groundTruthPatch);
        
            % Store the metrics
            patchSSIM(end+1)  = ssim_value;
            patchMSE(end+1)   = mse_value;
            patchPSNR(end+1)  = psnr_value;
            patchL1(end+1)    = l1_value;
            patchGSSIM(end+1) = gssim_value;
        end
        % Save per-image metrics
        ssim_all{i} = patchSSIM(:);
        mse_all{i} = patchMSE(:);
        psnr_all{i} = patchPSNR(:);
        l1_all{i} = patchL1(:);
        gssim_all{i} = patchGSSIM(:);
    catch
        fprintf('No ground truth image found in %s. Image quality metrics will not be calculated.\n', fullfile(dirx, [erase(grayImages1Files{i}, 'MPEF.png') 'HE.png']));
    end
    
    % Save the colorized image; *-full path to the directory where the images need to be saved
    imwrite(colorizedImage, ['*\', erase(erase(grayImages1Files{i}, 'MPEF'),'.png'), 'result_MPEF_THG.png'], 'png');
end

    % Concatenate metrics
    allMetrics.SSIM = vertcat(ssim_all{:});
    allMetrics.MSE = vertcat(mse_all{:});
    allMetrics.PSNR = vertcat(psnr_all{:});
    allMetrics.L1 = vertcat(l1_all{:});
    allMetrics.GSSIM = vertcat(gssim_all{:});

if ~isempty(allMetrics.SSIM)&&~isempty(allMetrics.MSE)&&~isempty(allMetrics.PSNR)&&~isempty(allMetrics.L1)&&~isempty(allMetrics.GSSIM)
    % Compute mean, standard deviation, and error range for each metric
    meanSSIM = mean(allMetrics.SSIM);
    stdSSIM = std(allMetrics.SSIM);
    meanMSE = mean(allMetrics.MSE);
    stdMSE = std(allMetrics.MSE);
    meanPSNR = mean(allMetrics.PSNR);
    stdPSNR = std(allMetrics.PSNR);
    meanL1 = mean(allMetrics.L1);
    stdL1 = std(allMetrics.L1);
    meanGSSIM = mean(allMetrics.GSSIM);
    stdGSSIM = std(allMetrics.GSSIM);
    medianSSIM = median(allMetrics.SSIM);
    iqrSSIM = iqr(allMetrics.SSIM);
    medianMSE = median(allMetrics.MSE);
    iqrMSE = iqr(allMetrics.MSE);
    medianPSNR = median(allMetrics.PSNR);
    iqrPSNR = iqr(allMetrics.PSNR);
    medianL1 = median(allMetrics.L1);
    iqrL1 = iqr(allMetrics.L1);
    medianGSSIM = median(allMetrics.GSSIM);
    iqrGSSIM = iqr(allMetrics.GSSIM);
    
    % Display the statistics
    fprintf('Metrics');
    fprintf('Mean SSIM: %.4f ± %.4f\n', meanSSIM, stdSSIM);
    fprintf('Mean MSE: %.4f ± %.4f\n', meanMSE, stdMSE);
    fprintf('Mean PSNR: %.2f ± %.2f dB\n', meanPSNR, stdPSNR);
    fprintf('Mean L1: %.4f ± %.4f\n', meanL1, stdL1);
    fprintf('Mean GSSIM: %.4f ± %.4f\n', meanGSSIM, stdGSSIM);
    fprintf('Median SSIM: %.4f ± %.4f\n', medianSSIM, iqrSSIM);
    fprintf('Median MSE: %.4f ± %.4f\n', medianMSE, iqrMSE);
    fprintf('Median PSNR: %.2f ± %.2f dB\n', medianPSNR, iqrPSNR);
    fprintf('Median L1: %.4f ± %.4f\n', medianL1, iqrL1);
    fprintf('Median GSSIM: %.4f ± %.4f\n', medianGSSIM, iqrGSSIM);
    fprintf('\n');
    
    % Generate box plots for each metric
    figure;
    subplot(2, 3, 1);
    boxplot(allMetrics.SSIM);
    title('SSIM');
    
    subplot(2, 3, 2);
    boxplot(allMetrics.MSE);
    title('MSE');
    
    subplot(2, 3, 3);
    boxplot(allMetrics.PSNR);
    title('PSNR');
    
    subplot(2, 3, 4);
    boxplot(allMetrics.L1);
    title('L1');
    
    subplot(2, 3, 5);
    boxplot(allMetrics.GSSIM);
    title('GSSIM');
    
    % Save the box plots to the directory '*'
    saveas(gcf, '*\boxplot.png');
end

% Function to extract patches for prediction
function [X, patchPositions] = extractPatchesForPrediction(grayImage1, grayImage2, patchSize, stride)
    [rows, cols, ~] = size(grayImage1);
    numPatches = 0;
    for i = 1:stride:rows - patchSize + 1
        for j = 1:stride:cols - patchSize + 1
            numPatches = numPatches + 1;
        end
    end
    X = zeros(patchSize, patchSize, 2, numPatches);
    patchPositions = zeros(numPatches, 2);
    
    patchIdx = 1;
    for i = 1:stride:rows - patchSize + 1
        for j = 1:stride:cols - patchSize + 1
            grayPatch1 = grayImage1(i:i+patchSize-1, j:j+patchSize-1);
            grayPatch2 = grayImage2(i:i+patchSize-1, j:j+patchSize-1);
            X(:, :, 1, patchIdx) = grayPatch1;
            X(:, :, 2, patchIdx) = grayPatch2;
            patchPositions(patchIdx, :) = [i, j];
            patchIdx = patchIdx + 1;
        end
    end
end

% Function to extract ground truth patches
function patches = extractGroundTruthPatches(groundTruthImage, patchPositions, patchSize)
    numPatches = size(patchPositions, 1);
    patches = single(zeros(patchSize, patchSize, 3, numPatches));
    for patchIdx = 1:numPatches
        i = patchPositions(patchIdx, 1);
        j = patchPositions(patchIdx, 2);
        patches(:, :, :, patchIdx) = groundTruthImage(i:i+patchSize-1, j:j+patchSize-1, :);
    end
end

% Function to reconstruct the image from patches using Gaussian weighting
function colorizedImage = reconstructImage(patches, imageSize, patchPositions, patchSize)
    colorizedImage = zeros(imageSize(1), imageSize(2), 3);
    countImage = zeros(imageSize(1), imageSize(2));
    
    % Create Gaussian weighting mask
    sigma = patchSize / 4; % Reduced sigma for sharper mask
    [X, Y] = meshgrid(1:patchSize, 1:patchSize);
    center = patchSize / 2;
    gaussMask = single(exp(-((X - center).^2 + (Y - center).^2) / (2 * sigma^2)));
    
    for patchIdx = 1:size(patches, 4)
        i = patchPositions(patchIdx, 1);
        j = patchPositions(patchIdx, 2);
        for k = 1:3
            colorizedImage(i:i+patchSize-1, j:j+patchSize-1, k) = ...
                colorizedImage(i:i+patchSize-1, j:j+patchSize-1, k) + ...
                patches(:, :, k, patchIdx) .* gaussMask;
        end
        countImage(i:i+patchSize-1, j:j+patchSize-1) = ...
            countImage(i:i+patchSize-1, j:j+patchSize-1) + gaussMask;
    end
    
    % Normalize the reconstructed image by the count image
    colorizedImage = colorizedImage ./ countImage;
    colorizedImage = min(max(colorizedImage, 0), 1); % Clip values to [0, 1] range
end

% Function to apply bilateral filter to an image
function filteredImage = bilateralFilter(image)
    % Convert to LAB color space
    labImage = rgb2lab(image);
    filteredLabImage = labImage;
    
    % Apply bilateral filter to each channel
    for i = 1:3
        filteredLabImage(:, :, i) = imbilatfilt(labImage(:, :, i), 0.1, 10);
    end
    
    % Convert back to RGB color space
    filteredImage = lab2rgb(filteredLabImage);
end

% GSSIM (Gradient Structural Similarity) function
function gssimValue = gssim(predictedPatch, groundTruthPatch)
    
    [h, w, c] = size(predictedPatch);
    gradxPredicted = zeros(h, w, c);
    gradxTarget = zeros(h, w, c);
    gradyPredicted = zeros(h, w, c);
    gradyTarget = zeros(h, w, c);
    gradPredicted = zeros(h, w, c);
    gradTarget = zeros(h, w, c);

    % Vectorized gradient computation
    for ch = 1:c
        % Compute gradients
        [gradxPredicted(:,:,ch), gradyPredicted(:,:,ch)] = imgradientxy(predictedPatch(:,:,ch));
        [gradxTarget(:,:,ch), gradyTarget(:,:,ch)] = imgradientxy(groundTruthPatch(:,:,ch));
        gradPredicted(:,:,ch)=abs(gradxPredicted(:,:,ch))+abs(gradyPredicted(:,:,ch));
        gradTarget(:,:,ch)=abs(gradxTarget(:,:,ch))+abs(gradyTarget(:,:,ch));
    end
    
    % Compute per channel GSSIM
    gssimValues = zeros(c,1);
    for ch = 1:c
        gssimValues(ch) = ssim(gradPredicted(:,:,ch), gradTarget(:,:,ch), 'Exponents',[0,1,1])*ssim(predictedPatch(:,:,ch),groundTruthPatch(:,:,ch), 'Exponents', [1,0,0]);
    end
    
    % Average SSIM values
    gssimValue = mean(gssimValues(:));
end