% Rename all the images from .BMP to .bmp in the TID dataset

distortRoot='/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/distorted_images/';
moveToDir='/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/distorted_images_1/';
distorted_images = dir(distortRoot);
distorted_images(1:2)=[];
currDir=pwd;
cd(distortRoot);

for i=1:length(distorted_images)
    [~,name,ext]=fileparts(distorted_images(i).name);
    movefile([name ext],[moveToDir upper(name) lower(ext)]);
end

cd(currDir);