clc
close all


% Input
[image,path]=uigetfile('*.jpg','upload MRI image');
string=strcat(path,image);
img=imread(string);

figure;
imshow(img);
title('uploaded image','FontSize',10);

imageArray = double(s); % Convert image to double precision array for calculations

%% Calculate Mean, Variance, and Standard Deviation
meanValue = mean(imageArray(:));
varianceValue = var(imageArray(:));
stdDeviationValue = std(imageArray(:));

fprintf('Mean: %.2f\n', meanValue);
fprintf('Variance: %.2f\n', varianceValue);
fprintf('Standard Deviation: %.2f\n', stdDeviationValue);


% Filter
num_iter = 15;
    delta_t = 1/7;
    kappa = 20;
    option = 2;
    disp('please wait image under processing . . .');
    input = anisodiff_function(img,num_iter,delta_t,kappa,option);
    input = uint8(input);
    
input=imresize(input,[256,256]);
if size(input,3)>1
    input=rgb2gray(input);
end
figure;
imshow(input);
title('Filtered image','FontSize',10);

% thresholding
out_img=imresize(input,[256,256]);
t0=mean(img(:));
th=t0+((max(input(:))+min(input(:)))./2);
for i=1:1:size(input,1)
    for j=1:1:size(input,2)
        if input(i,j)>th
            out_img(i,j)=1;
        else
            out_img(i,j)=0;
        end
    end
end

% Morphological Operation

label=bwlabel(out_img);
stats=regionprops(logical(out_img),'Solidity','Area','BoundingBox');
density=[stats.Solidity];
area=[stats.Area];
high_dense_area=density>0.7;
max_area=max(area(high_dense_area));
tumor_label=find(area==max_area);
tumor=ismember(label,tumor_label);

if max_area>200
   figure;
   imshow(tumor)
   title('tumor alone','FontSize',10);
else
    h = msgbox('No Tumor!!','status');
    %disp('no tumor');
    return;
end
            
% Bounding box

box = stats(tumor_label);
wantedBox = box.BoundingBox;
figure
imshow(input);
title('Bounding Box','FontSize',10);
hold on;
rectangle('Position',wantedBox,'EdgeColor','y');
hold off;


% Getting Tumor Outline - image filling, eroding, subtracting
% erosion the walls by a few pixels

dilationAmount = 5;
rad = floor(dilationAmount);
[r,c] = size(tumor);
filledImage = imfill(tumor, 'holes');

for i=1:r
   for j=1:c
       x1=i-rad;
       x2=i+rad;
       y1=j-rad;
       y2=j+rad;
       if x1<1
           x1=1;
       end
       if x2>r
           x2=r;
       end
       if y1<1
           y1=1;
       end
       if y2>c
           y2=c;
       end
       erodedImage(i,j) = min(min(filledImage(x1:x2,y1:y2)));
   end
end
figure
imshow(erodedImage);
title('eroded image','FontSize',10);

% subtracting eroded image from original BW image

tumorOutline=tumor;
tumorOutline(erodedImage)=0;

figure;  
imshow(tumorOutline);
title('Tumor Outline','FontSize',10);

% Inserting the outline in filtered image in red color

rgb = input(:,:,[1 1 1]);
red = rgb(:,:,1);
red(tumorOutline)=255;
green = rgb(:,:,2);
green(tumorOutline)=0;
blue = rgb(:,:,3);
blue(tumorOutline)=0;

tumorOutlineInserted(:,:,1) = red; 
tumorOutlineInserted(:,:,2) = green; 
tumorOutlineInserted(:,:,3) = blue; 


figure
imshow(tumorOutlineInserted);
title('Detected Tumer','FontSize',10);


%% Background and Foreground Segmentation
backgroundMask = ~tumor;  % Invert the tumor mask to get the background mask
foregroundImage = imageArray;
foregroundImage(backgroundMask) = NaN;  % Set background pixels to NaN

%% Statistical Measures for Foreground and Background
meanForeground = nanmean(foregroundImage(:));
medianForeground = nanmedian(foregroundImage(:));
stdDevForeground = nanstd(foregroundImage(:));

backgroundPixels = imageArray(backgroundMask);
meanBackground = mean(backgroundPixels);
medianBackground = median(backgroundPixels);
stdDevBackground = std(backgroundPixels);

fprintf('Foreground Mean: %.2f\n', meanForeground);
fprintf('Foreground Median: %.2f\n', medianForeground);
fprintf('Foreground Standard Deviation: %.2f\n', stdDevForeground);

fprintf('Background Mean: %.2f\n', meanBackground);
fprintf('Background Median: %.2f\n', medianBackground);
fprintf('Background Standard Deviation: %.2f\n', stdDevBackground);


%% Create Histograms
figure;

% Histogram for Mean, Median, and StdDev for Foreground and Background
edges = linspace(min([meanForeground(:); meanBackground(:); medianForeground(:); medianBackground(:); stdDevForeground(:); stdDevBackground(:)]), ...
                 max([meanForeground(:); meanBackground(:); medianForeground(:); medianBackground(:); stdDevForeground(:); stdDevBackground(:)]), 50);

histogram(meanForeground, edges, 'FaceAlpha', 0.5, 'DisplayName', 'Mean Foreground');
hold on;
histogram(meanBackground, edges, 'FaceAlpha', 0.5, 'DisplayName', 'Mean Background');
histogram(medianForeground, edges, 'FaceAlpha', 0.5, 'DisplayName', 'Median Foreground');
histogram(medianBackground, edges, 'FaceAlpha', 0.5, 'DisplayName', 'Median Background');
histogram(stdDevForeground, edges, 'FaceAlpha', 0.5, 'DisplayName', 'StdDev Foreground');
histogram(stdDevBackground, edges, 'FaceAlpha', 0.5, 'DisplayName', 'StdDev Background');
hold off;

title('Comparison of Mean, Median, and StdDev for Foreground and Background', 'FontSize', 8);
xlabel('Value', 'FontSize', 12);
ylabel('Frequency', 'FontSize', 12);
legend('Location', 'best');


% Display Together

figure
subplot(231);imshow(img);title('uploaded image','FontSize',10);
subplot(232);imshow(input);title('Filtered image','FontSize',10);

subplot(233);imshow(input);title('Bounding Box','FontSize',10);
hold on;rectangle('Position',wantedBox,'EdgeColor','g');hold off;

subplot(234);imshow(tumor);title('tumor alone','FontSize',10);
subplot(235);imshow(tumorOutline);title('Tumor Outline','FontSize',10);
subplot(236);imshow(tumorOutlineInserted);title('Detected Tumor','FontSize',10);



