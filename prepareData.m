clear;
close all;
clc;

trainImgCount=15;
valImgCount=5;
imgRoot = '/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/allImgs_ref_distorted_val/';
imgList = getAllFiles(imgRoot,'bmp');
imgList = sort_nat(imgList);
imgNames = cellfun(@(x) strsplit(x,imgRoot),imgList,'uni',0);
imgNames = cellfun(@(x) x{2},imgNames,'uni',0);
distortClassLabels = reshape(repmat((0:120).',1,valImgCount),[],1);

% Prepare pairs from same class
pairsSameClass=[];
for i=min(distortClassLabels):max(distortClassLabels)
    ind=find(distortClassLabels==i);
    pairsSameClass = [pairsSameClass;allcomb(imgNames(ind),imgNames(ind))];
end

% Remove duplicate pairs
dupsInd = strcmp(pairsSameClass(:,1),pairsSameClass(:,2));
pairsSameClass(dupsInd,:)=[];
    
% prepare pairs from different classes
pairsDiffClasses = [];
% first create a list of classes which can be used to form pairs
pairedClasses = [];
for class1=0:119
    if class1 == 0
        class2 = 1:5:120;
        otherRandClass = setdiff(0:120,[0 1:5:120]);
        dummy=randperm(length(otherRandClass));
        r = sort(dummy(1:length(1:5:120)));
        otherRandClass = otherRandClass(r);
        class2 = sort([class2 otherRandClass]);
%         pairedClasses = union(pairedClasses,[class1*ones(length(class2),1) class2.'],'rows','stable');
        pairedClasses = [pairedClasses;[class1*ones(length(class2),1) class2.']];
    elseif class1 < 117
        if rem(class1,5) == 0
            continue;
        end
        class2 = class1 + 1;
        if rem(class2,5) ~= 0 
            otherRandClass = class2 + 1 : class2 + (5 - rem(class2,5));
        else
            otherRandClass = class2 - 4 : class2 - 2;
        end
        r = sort(randi(length(otherRandClass),1));
        otherRandClass = otherRandClass(r);
        class2 = sort([class2 otherRandClass]);
%         pairedClasses = union(pairedClasses,[class1*ones(length(class2),1) class2.'],'rows','stable');  
        pairedClasses = [pairedClasses;[class1*ones(length(class2),1) class2.']];
    elseif class1 == 117
        class2 = class1 + 1;
        otherRandClass = randi([119 120]);
        r = sort(randi(length(otherRandClass),1));
        otherRandClass = otherRandClass(r);
        class2 = sort([class2 otherRandClass]);
%         pairedClasses = union(pairedClasses,[class1*ones(length(class2),1) class2.'],'rows','stable');  
        pairedClasses = [pairedClasses;[class1*ones(length(class2),1) class2.']];
    elseif class1 == 118
        class2 = class1 + 1;
        otherRandClass = class2 + 1;
        r = sort(randi(length(otherRandClass),1));
        otherRandClass = otherRandClass(r);
        class2 = sort([class2 otherRandClass]);
%         pairedClasses = union(pairedClasses,[class1*ones(length(class2),1) class2.'],'rows','stable');  
        pairedClasses = [pairedClasses;[class1*ones(length(class2),1) class2.']];
    else
        class2 = class1 + 1;
        otherRandClass = randi([116 118]);
        r = sort(randi(length(otherRandClass),1));
        otherRandClass = otherRandClass(r);
        class2 = sort([class2 otherRandClass]);
%         pairedClasses = union(pairedClasses,[class1*ones(length(class2),1) class2.'],'rows','stable');  
        pairedClasses = [pairedClasses;[class1*ones(length(class2),1) class2.']];
    end
end

for i=1:size(pairedClasses,1)
    i
    ind1 = find(distortClassLabels == pairedClasses(i,1));
    ind2 = find(distortClassLabels == pairedClasses(i,2));
    pairsDiffClasses = [pairsDiffClasses; [imgNames(ind1) imgNames(ind2)]];
end
allPairs = [pairsSameClass(randi(size(pairsSameClass,1),[size(pairsDiffClasses,1) 1]),:);pairsDiffClasses];

 

class1;