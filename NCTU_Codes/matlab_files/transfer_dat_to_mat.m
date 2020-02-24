clc;clear all;
% close all;

filelist_dat_6713 = dir('.\data\dat\DecRoomOption');

for no=1:length(filelist_dat_6713)-2
    fid = fopen(['.\data\dat\DecRoomOption\',filelist_dat_6713(no+2).name],'r');
    
    datacell = textscan(fid, '%f%f%f', 'HeaderLines', 1, 'Collect', 1);
    fclose(fid);
    A.data = datacell{1};
    s = A.data(:,1);
    
    save(['.\data\noise_mat\DecRoomOption\',filelist_dat_6713(no+2).name,'.mat'],'s');
    
end

%%

% load(['.\data\noise\record_6713\',filelist_dat_6713(no+2).name]);
% soundsc(s,16000);