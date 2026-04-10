% % Close the progress bar
% Delete waitbar (even if h is lost)
waitbars = findall(0, 'Type', 'figure', 'Name', 'Training Progress');
delete(waitbars);
clc;
clear all;
close all;

% Choose whether to include SSIM in the content loss function (1-true, 0-false)
ssim_loss=1;

% Training and validation dataset directories; '*', '**' - full path to the datasets
trainDir = '*';
valDir = '**';

% Load training image file list
trainFiles = dir(fullfile(trainDir, '*.png'));  % Assuming the images are in PNG format
grayImages1Files = {trainFiles(contains({trainFiles.name}, 'MPEF')).name};
grayImages2Files = {trainFiles(contains({trainFiles.name}, 'THG')).name};
colorImagesFiles = {trainFiles(contains({trainFiles.name}, 'HE')).name};

% Load validation image file list
valFiles = dir(fullfile(valDir, '*.png'));  % Assuming the images are in PNG format
valGrayImages1Files = {valFiles(contains({valFiles.name}, 'MPEF')).name};
valGrayImages2Files = {valFiles(contains({valFiles.name}, 'THG')).name};
valColorImagesFiles = {valFiles(contains({valFiles.name}, 'HE')).name};

% Read images into arrays
grayImages1 = cellfun(@(f) single(imread(fullfile(trainDir, f)))/255, grayImages1Files, 'UniformOutput', false);
grayImages2 = cellfun(@(f) single(imread(fullfile(trainDir, f)))/255, grayImages2Files, 'UniformOutput', false);
colorImages = cellfun(@(f) single(imread(fullfile(trainDir, f)))/255, colorImagesFiles, 'UniformOutput', false);

valGrayImages1 = cellfun(@(f) single(imread(fullfile(valDir, f)))/255, valGrayImages1Files, 'UniformOutput', false);
valGrayImages2 = cellfun(@(f) single(imread(fullfile(valDir, f)))/255, valGrayImages2Files, 'UniformOutput', false);
valColorImages = cellfun(@(f) single(imread(fullfile(valDir, f)))/255, valColorImagesFiles, 'UniformOutput', false);

% Define patch size and stride
patchSize = 128;
stride = 128;

% Initialize data containers for training
X_train = [];
Y_train = [];

% Extract patches from all grayscale and color images for training
for i = 1:numel(grayImages1)
    [X_train_i, Y_train_i] = extractPatchesForTraining(grayImages1{i}, grayImages2{i}, colorImages{i}, patchSize, stride);
    X_train = cat(4, X_train, X_train_i);
    Y_train = cat(4, Y_train, Y_train_i);
end

% Initialize data containers for validation
X_val = [];
Y_val = [];

% Extract patches from all grayscale and color images for validation
for i = 1:numel(valGrayImages1)
    [X_val_i, Y_val_i] = extractPatchesForTraining(valGrayImages1{i}, valGrayImages2{i}, valColorImages{i}, patchSize, stride);
    X_val = cat(4, X_val, X_val_i);
    Y_val = cat(4, Y_val, Y_val_i);
end

% Define image augmentation options
imageAugmenter = imageDataAugmenter(...
    'RandRotation', [-20 20], ...         % Random rotation between -20 and 20 degrees
    'RandXReflection', true, ...          % Randomly flip images horizontally
    'RandYReflection', true, ...          % Randomly flip images vertically
    'RandXTranslation', [-10 10], ...     % Random horizontal translation
    'RandYTranslation', [-10 10], ...     % Random vertical translation
    'RandScale', [0.25 1.0]);              % Random scaling

% Neural Network Input Size
inputSizeGenerator = [patchSize patchSize 2];
inputSizeDiscriminator = [patchSize patchSize 3];

% Define GAN Architecture
%Generator
%netGenerator=pix2pixHDGlobalGenerator(inputSizeGenerator, "ConvolutionWeightsInitializer", "glorot", "FinalActivationLayer", "sigmoid", "Dropout", 0.35);
% Define CNN Architecture with Skip Connections for 128x128 patches
layers = [
    imageInputLayer([patchSize patchSize 2], 'Name', 'input', 'Normalization', 'none')
    % Encoder
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu1_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu1_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu2_1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu2_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu3_1')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu3_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu4_1')
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu4_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')

    % Decoder with Skip Connections
    transposedConv2dLayer(3, 512, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv4', 'WeightsInitializer', 'he')
    reluLayer('Name', 'uprelu4')
    concatenationLayer(3, 2, 'Name', 'concat4')
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'deconv4_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu4_1')
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'deconv4_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu4_2')

    transposedConv2dLayer(3, 256, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv3', 'WeightsInitializer', 'he')
    reluLayer('Name', 'uprelu3')
    concatenationLayer(3, 2, 'Name', 'concat3')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'deconv3_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu3_1')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'deconv3_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu3_2')

    transposedConv2dLayer(3, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'uprelu2')
    concatenationLayer(3, 2, 'Name', 'concat2')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'deconv2_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu2_1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'deconv2_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu2_2')

    transposedConv2dLayer(3, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'uprelu1')
    concatenationLayer(3, 2, 'Name', 'concat1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'deconv1_1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu1_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'deconv1_2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'derelu1_2')

    convolution2dLayer(3, 3, 'Padding', 'same', 'Name', 'finalconv', 'WeightsInitializer', 'he')
];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu4_2', 'concat4/in2');
lgraph = connectLayers(lgraph, 'relu3_2', 'concat3/in2');
lgraph = connectLayers(lgraph, 'relu2_2', 'concat2/in2');
lgraph = connectLayers(lgraph, 'relu1_2', 'concat1/in2');

% Convert to dlnetwork
netGenerator = dlnetwork(lgraph);

%Discriminator
netDiscriminator=patchGANDiscriminator(inputSizeDiscriminator, "FinalActivationLayer", "sigmoid");
% Training options
numEpochs = 48;
miniBatchSize = 32;
learnRate = 1e-3;
validationInterval = 100;  % Validate every 100 batches

% Prepare data for training
numObservations = size(X_train, 4);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);
numValObservations = size(X_val, 4);

% Initialize progress plot and progress bar
figure;
lineLossGenTrain = animatedline('Color','b');
lineLossDisTrain = animatedline('Color','r');
lineLossGenVal = animatedline('Color','g');
ylim([0 inf]);
xlabel("Iteration");
ylabel("Loss");
grid on;

% Initialize image display figure
figure('Name', 'Reconstructed vs Ground Truth Patches');

% Maximum number of iterations
maxIterations = numEpochs * numIterationsPerEpoch;

% Initialize the progress bar
hWaitbar = waitbar(0, 'Training Progress', 'Name', 'Training Progress', 'CreateCancelBtn', @(~,~) disp('Training Cancelled'));

% Initialize training state
averageGradGen = [];
averageSqGradGen = [];
averageGradDis = [];
averageSqGradDis = [];

% Best model tracking
bestValLoss = inf;
bestModel = netGenerator;

% Set the display interval for patches
displayInterval = 20;

% Loss weights
weightContentMin = 50.0;  % Adjust as necessary
weightAdversarial = 1.0;
weightContent = 650.0;        % Adjust as necessary
contShift=0;
counter=0;

% A stopwatch timer
startTime = tic;

% Training loop
for epoch = 1:numEpochs
    % Shuffle training data
    idx = randperm(numObservations);
    X_train = X_train(:,:,:,idx);
    Y_train = Y_train(:,:,:,idx);

    % Use the augmented data store for batch training
    for i = 1:numIterationsPerEpoch
        counter=counter+1;
        % Extract mini-batch
        idxBatch = (i-1)*miniBatchSize + 1:i*miniBatchSize;
        XBatch = X_train(:,:,:,idxBatch);  % Extract input grayscale images
        YBatch = Y_train(:,:,:,idxBatch);  % Extract target RGB images
        
        % Apply synchronized augmentation
        [augmentedXBatch, augmentedYBatch] = synchronizedAugmentation(XBatch, YBatch, imageAugmenter);
        
        % Convert mini-batch to dlarray
        dlX = dlarray(augmentedXBatch, 'SSCB');  % Input should be 2 channels
        dlY = dlarray(augmentedYBatch, 'SSCB');  % Output should be 3 channels

        if canUseGPU
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end
        
        % GAN training steps: alternate between discriminator and generator
        % 1. Train discriminator (3 updates for each generator update)
        if counter>20
            for discCount=1:3
                [gradientsDis, lossDis] = dlfeval(@modelGradientsDiscriminator, netGenerator, netDiscriminator, dlX, dlY);
                [netDiscriminator, averageGradDis, averageSqGradDis] = adamupdate(netDiscriminator, gradientsDis, averageGradDis, averageSqGradDis, i, learnRate);
            end
        end

        % 2. Train generator
        [gradientsGen, lossGen] = dlfeval(@modelGradientsGenerator, netGenerator, netDiscriminator, dlX, dlY, weightContentMin, weightContent, weightAdversarial, contShift, counter, ssim_loss);
        [netGenerator, averageGradGen, averageSqGradGen] = adamupdate(netGenerator, gradientsGen, averageGradGen, averageSqGradGen, i, learnRate);
        
        %Update adversarial loss multiplier
        if mod(counter, 15)==0
            contShift=contShift+1;
        end

        % Elapsed time
        elapsed = toc(startTime);

        % Update progress plot
        D = seconds(elapsed);
        D.Format = 'hh:mm:ss';
        if counter>10
        addpoints(lineLossGenTrain, (epoch-1) * numIterationsPerEpoch + i, double(gather(lossGen)));
        end

        if counter>23
        addpoints(lineLossDisTrain, (epoch-1) * numIterationsPerEpoch + i, 650*double(gather(lossDis)));
        title("Epoch: " + epoch + ", Iteration: " + i + ", Elapsed: " + string(D) + ", Gen Loss: " + gather(extractdata(lossGen)) + ", Dis Loss: " + 650*gather(extractdata(lossDis)));
        else
        title("Epoch: " + epoch + ", Iteration: " + i + ", Elapsed: " + string(D) + ", Gen Loss: " + gather(extractdata(lossGen)) + ", Dis Loss: ");
        drawnow;
        end

         % Perform validation at specified intervals
        if mod(i, validationInterval) == 0
            % Evaluate validation loss
            valLoss = evaluateValidationLoss(netGenerator, X_val, Y_val, ssim_loss);
            addpoints(lineLossGenVal, (epoch-1) * numIterationsPerEpoch + i, 650*valLoss);
            title("Epoch: " + epoch + ", Iteration: " + i + ", Gen Loss: " + gather(extractdata(lossGen)) + ", Dis Loss: " + 650*gather(extractdata(lossDis)) + ", Val Loss: " + 650*gather(extractdata(valLoss)));
            drawnow;
            
            % Save the best model based on validation loss
            if valLoss < bestValLoss
                bestValLoss = valLoss;
                bestModel = netGenerator;
                % Save the best model
                save('*\bestModel.mat', 'bestModel');
            end
        end
    
        % Update waitbar
        waitbar((epoch-1) * numIterationsPerEpoch + i / maxIterations, hWaitbar, sprintf('Epoch: %d, Iteration: %d/%d', epoch, i, numIterationsPerEpoch));

        % Display patches at intervals
        if mod(i, displayInterval) == 0
            dlYPredPatches = forward(netGenerator, dlX);
            displayReconstructedPatches(dlX, dlYPredPatches, dlY);
        end
    end
end

% Close the progress bar
delete(hWaitbar);

% Function to calculate model gradients and loss
function [gradientsGen, lossGen] = modelGradientsGenerator(netGenerator, netDiscriminator, dlX, dlY, weightContentMin, weightContent, weightAdversarial, contShift, counter, ssim_loss)
    % Forward pass through the generator
    dlYGenerated = forward(netGenerator, dlX);
    
    % Forward pass through the discriminator with generated images
    dlYPredGenerated = forward(netDiscriminator, dlYGenerated);
    
    % Compute the content loss using the customLoss function
    contentLoss = customLoss(dlYGenerated, dlY, ssim_loss);
    epsilon = eps(single(1));  % Small constant to prevent log(0)
    dlYPredGenerated = max(dlYPredGenerated, epsilon);  % Clip to avoid exactly 0
    % Compute the adversarial loss for the generator
    adversarialLoss = -mean(log(dlYPredGenerated), 'all');

% Ensure losses are non-zero to avoid division by zero
if adversarialLoss == 0
   adversarialLoss = 1e-8; % Small epsilon value
end
if contentLoss == 0
    contentLoss = 1e-8; % Small epsilon value
end

    weightContent=max(weightContent-contShift, weightContentMin);
    
    % Combine the adversarial and content loss with specified weights
    if counter > 20
    lossGen = (weightContent * contentLoss + weightAdversarial * adversarialLoss);
    else
    lossGen = (weightContent * contentLoss);
    end
    
    % Compute gradients for the generator
    gradientsGen = dlgradient(lossGen, netGenerator.Learnables);
end

function [gradientsDis, lossDis] = modelGradientsDiscriminator(netGenerator, netDiscriminator, dlX, dlY)
    % Forward pass through the generator
    dlYGenerated = forward(netGenerator, dlX);
    
    % Forward pass through the discriminator with generated images
    dlYPredGenerated = forward(netDiscriminator, dlYGenerated);

    % Forward pass through the discriminator with real images
    dlYPredReal = forward(netDiscriminator, dlY);
    
    % Compute loss for the discriminator
    lossDis = ganLossDiscriminator(dlYPredReal, dlYPredGenerated);
    
    % Compute gradients for the discriminator
    gradientsDis = dlgradient(lossDis, netDiscriminator.Learnables);
end

% GAN loss function for discriminator
function lossDis = ganLossDiscriminator(dlYPredReal, dlYPredGenerated)
    epsilon = eps(single(1));  % Small constant to prevent log(0)    
    % Calculate the loss for the discriminator
    dlYPredReal = max(dlYPredReal, epsilon);  % Clip to avoid exactly 0
    lossReal = -mean(log(dlYPredReal), 'all');
    dlYPredGenerated = min(dlYPredGenerated, 1 - epsilon);  % Clip to avoid exactly 1
    lossGenerated = -mean(log(1 - dlYPredGenerated), 'all');
    
    % Combine the losses for real and generated images
    lossDis = lossReal + lossGenerated;
end

% Function to evaluate validation loss
function valLoss = evaluateValidationLoss(netGenerator, X_val, Y_val, ssim_loss)
    dlX=dlarray(X_val, 'SSCB');
    dlYVal=dlarray(Y_val, 'SSCB');

    if canUseGPU
        dlX = gpuArray(dlX);
        dlYVal = gpuArray(dlYVal);
    end

    % Forward pass
    dlYPredVal = forward(netGenerator, dlX);
    % Compute the custom loss
    valLoss = double(gather(customLoss(dlYPredVal, dlYVal, ssim_loss)));
end

% Function to extract patches for training
function [X, Y] = extractPatchesForTraining(grayImage1, grayImage2, colorImage, patchSize, stride)
    [rows, cols, ~] = size(grayImage1);
    numPatches = floor((rows - patchSize) / stride + 1) * floor((cols - patchSize) / stride + 1);
    X = zeros(patchSize, patchSize, 2, numPatches, 'single');
    Y = zeros(patchSize, patchSize, 3, numPatches, 'single');
    
    % Vectorized patch extraction
    patchIdx = 1;
    for i = 1:stride:rows - patchSize + 1
        for j = 1:stride:cols - patchSize + 1
            X(:,:,:,patchIdx) = cat(3, grayImage1(i:i+patchSize-1, j:j+patchSize-1), grayImage2(i:i+patchSize-1, j:j+patchSize-1));
            Y(:,:,:,patchIdx) = colorImage(i:i+patchSize-1, j:j+patchSize-1, :);
            patchIdx = patchIdx + 1;
        end
    end
end

function loss = customLoss(predicted, target, ssim_loss)
    % Calculate L1-norm (mean absolute error)
    l1Loss = mean(abs(predicted - target), 'all');

    % Calculate Structural Similarity Index (SSIM) for each image and channel
    numImages = size(predicted, 4);
    numChannels = size(predicted, 3);
    ssimValues = zeros(numImages, numChannels, 'like', predicted);

    if ssim_loss
        for i = 1:numImages
            for ch = 1:numChannels
                ssimValues(i, ch) = ssim(predicted(:,:,ch,i), target(:,:,ch,i));
            end
        end
        ssimLoss = 1 - mean(ssimValues, 'all');
    else
        ssimLoss = 0;
    end

    % Combine the losses (adjust weights as needed)
    loss = 0.7 * l1Loss + 0.3 * ssimLoss;
end

% Function to display reconstructed vs ground truth patches
function displayReconstructedPatches(dlX, dlYPred, dlY)
    figure(2);  % Ensure plotting on the second figure
    clf;  % Clear the figure
    numPatchesToShow = min(5, size(dlX, 4));  % Show up to 5 patches
    for i = 1:numPatchesToShow
        % Extract and normalize patches for display
        inputPatch = gather(extractdata(dlX(:,:,:,i)));
        predictedPatch = gather(extractdata(dlYPred(:,:,:,i)));
        groundTruthPatch = gather(extractdata(dlY(:,:,:,i)));

        subplot(4, numPatchesToShow, i);
        imshow(inputPatch(:,:,1), []);
        title('Input Patch 1');

        subplot(4, numPatchesToShow, numPatchesToShow + i);
        imshow(inputPatch(:,:,2), []);
        title('Input Patch 2');

        subplot(4, numPatchesToShow, 2* numPatchesToShow + i);
        imshow(predictedPatch);
        title('Predicted Patch');

        subplot(4, numPatchesToShow, 3 * numPatchesToShow + i);
        imshow(groundTruthPatch);
        title('Ground Truth Patch');
    end
    drawnow;
end
function [augmentedX, augmentedY] = synchronizedAugmentation(X, Y, augmenter)
    % Apply random transformations from the augmenter to both input and output images
    augmentedX = zeros(size(X), 'like', X);
    augmentedY = zeros(size(Y), 'like', Y);

    for i = 1:size(X, 4)
        % Extract the current input and output images
        inputImage = X(:, :, :, i);
        targetImage = Y(:, :, :, i);
        
        % Apply the augmentations to the input image
        % Use the same augmentation parameters for both input and output
        [augmentedInput, params] = augment(augmenter, inputImage);
        
        % Apply the same parameters to the target image
        augmentedTarget = applyParamsToImage(targetImage, params);
        
        % Store the augmented images
        augmentedX(:, :, :, i) = augmentedInput;
        augmentedY(:, :, :, i) = augmentedTarget;
    end
end

function [augmentedImage, params] = augment(augmenter, image)
    % Augment the image using the specified augmenter
    augmentedImage = image;  % Start with the original image
    img_sz = size(augmentedImage);
    % Scale modification probability
    scaleProb=0.25;

    if ~isempty(augmenter.RandRotation)
        angle = rand() * (augmenter.RandRotation(2) - augmenter.RandRotation(1)) + augmenter.RandRotation(1);
        augmentedImage = imrotate(augmentedImage, angle, 'bilinear', 'crop');
    end
    xRefl=0;
    yRefl=0;
    if augmenter.RandXReflection && rand() > 0.5
        augmentedImage = flip(augmentedImage, 2); % Randomly flip horizontally
        xRefl=1;
    end

    if augmenter.RandYReflection && rand() > 0.5
        augmentedImage = flip(augmentedImage, 1); % Randomly flip vertically
        yRefl=1;
    end

    % Handle translations
    if ~isempty(augmenter.RandXTranslation) || ~isempty(augmenter.RandYTranslation)
        xTranslation = randi(augmenter.RandXTranslation);
        yTranslation = randi(augmenter.RandYTranslation);
        augmentedImage = imtranslate(augmentedImage, [xTranslation, yTranslation]);
    end

    % Store the parameters
    params.angle = angle;
    % params.scale = scale;
    params.xTranslation=xTranslation;
    params.yTranslation=yTranslation;
    params.xRefl=xRefl;
    params.yRefl=yRefl;

    %Mask same region as ground truth
    imageMask=ones(size(image,1),size(image,2));
    augmentedMask = imrotate(imageMask, params.angle,  'bilinear', 'crop');  % Rotate using the same angle
    if params.xRefl==1
        augmentedMask = flip(augmentedMask, 2); % Flip horizontally
    end
    if params.yRefl==1
        augmentedMask = flip(augmentedMask, 1); % Flip vertically
    end
    augmentedMask = imtranslate(augmentedMask, [params.xTranslation, params.yTranslation]);
    augmentedMask=logical(abs(floor(augmentedMask)-1));
    augmentedImage(repmat(augmentedMask,[1,1,2]))=0;

    % Handle scaling
    if ~isempty(augmenter.RandScale)
        if rand() > scaleProb
        scale = rand() * (augmenter.RandScale(2) - augmenter.RandScale(1)) + augmenter.RandScale(1);
        augmentedImage = imresize(augmentedImage, scale);
        augmentedImage = imresize(augmentedImage, img_sz(1:2));
        end
    end
end
function augmentedImage = applyParamsToImage(image, params)
    % This function applies the stored parameters to the target image
    imageMask=ones(size(image,1),size(image,2));
    augmentedImage = imrotate(image, params.angle,  'bilinear', 'crop');  % Rotate using the same angle
    augmentedMask = imrotate(imageMask, params.angle,  'bilinear', 'crop');  % Rotate using the same angle
    if params.xRefl==1
        augmentedImage = flip(augmentedImage, 2); % Flip horizontally
        augmentedMask = flip(augmentedMask, 2); % Flip horizontally
    end
    if params.yRefl==1
        augmentedImage = flip(augmentedImage, 1); % Flip vertically
        augmentedMask = flip(augmentedMask, 1); % Flip vertically
    end
    augmentedImage = imtranslate(augmentedImage, [params.xTranslation, params.yTranslation]);
    augmentedMask = imtranslate(augmentedMask, [params.xTranslation, params.yTranslation]);
    augmentedMask=logical(abs(floor(augmentedMask)-1));
    augmentedImage(repmat(augmentedMask,[1,1,3]))=1;
end