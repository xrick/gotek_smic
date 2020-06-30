clc;clear all;
% close all;

%% filter design
fs = 16000;
windowSize = fs*0.025;  % 25ms
windowStep = fs*0.010;   % 10ms(shift)
nDims = 40;
context_l = 30;
context_r = 10;
% fftSize = 512;
% cepstralCoefficients = 40;
%% generate data
keyword1_path = '.\test_signal_dark';
keyword1_dir = dir(keyword1_path);
keyword2_path = '.\test_signal_suit';
keyword2_dir = dir(keyword2_path);
filler_path = '.\test_signal_filler';
filler_dir = dir(filler_path);

x_class1 = []; % keyword1
y_class1 = [];
for da=1:length(keyword1_dir)-2
    x = [];
    [signal fs] = audioread([keyword1_path,'\',keyword1_dir(da+2).name]);
    signal = resample(signal(:,1),16000,fs);
    % vad
    rest_signal = simpleSAD(signal,fs);
    
%     lens = length(rest_signal);
%     if lens>8000
%         rest_signal = rest_signal(lens-8000+1:lens,:);
%     elseif lens<8000
%         rest_signal = [0.5+rand(8000-lens,1).*10^-6;rest_signal];
%     end
    
    % mfcc
    coeff = mfcc(rest_signal, fs, windowSize, windowStep);
    temp = coeff(1,:)-min(coeff(1,:));
    coeff(1,:) = temp./max(temp);
    nframe = length(coeff(1,:));
    coeff = [zeros(nDims,context_l), coeff, zeros(nDims,context_r)];
    
    x = [];
    y = [];
    for context=1:nframe
        xx = [];
        window = coeff(:,context:context+context_l+context_r);
        for w=1:context_l+context_r+1
           xx = [xx;window(:,w)]; % concate
        end
        x = [x,xx];
        y = [y,1];
    end
    x_class1 = [x_class1;x'];
    y_class1 = [y_class1;y'];
end

x_class2 = []; % keyword2
y_class2 = [];
for da=1:length(keyword2_dir)-2
    x = [];
    [signal fs] = audioread([keyword2_path,'\',keyword2_dir(da+2).name]);
    signal = resample(signal(:,1),16000,fs);
    % vad
    rest_signal = simpleSAD(signal,fs);
    
%     lens = length(rest_signal);
%     if lens>8000
%         rest_signal = rest_signal(1:8000,:);
%     elseif lens<8000
%         rest_signal = [rest_signal;0.5+rand(8000-lens,1).*10^-6];
%     end
    
    % mfcc
    coeff = mfcc(rest_signal, fs, windowSize, windowStep);
    temp = coeff(1,:)-min(coeff(1,:));
    coeff(1,:) = temp./max(temp);
    nframe = length(coeff(1,:));
    coeff = [zeros(nDims,context_l), coeff, zeros(nDims,context_r)];
    
    x = [];
    y = [];
    for context=1:nframe
        xx = [];
        window = coeff(:,context:context+context_l+context_r);
        for w=1:context_l+context_r+1
           xx = [xx;window(:,w)]; % concate
        end
        x = [x,xx];
        y = [y,2];
    end
    x_class2 = [x_class2;x'];
    y_class2 = [y_class2;y'];
end

x_class3 = [];
y_class3 = [];
for da=1:length(filler_dir)-2
    x = [];
    [signal fs] = audioread([filler_path,'\',filler_dir(da+2).name]);
    signal = resample(signal(:,1),16000,fs);
    % vad
    rest_signal = simpleSAD(signal,fs);
    
%     lens = length(rest_signal);
%     if lens>8000
%         rest_signal = rest_signal(1:8000,:);
%     elseif lens<8000
%         rest_signal = [rest_signal;0.5+rand(8000-lens,1).*10^-6];
%     end
    
    % mfcc
    coeff = mfcc(rest_signal, fs, windowSize, windowStep);
    temp = coeff(1,:)-min(coeff(1,:));
    coeff(1,:) = temp./max(temp);
    nframe = length(coeff(1,:));
    coeff = [zeros(nDims,context_l), coeff, zeros(nDims,context_r)];
    
    x = [];
    y = [];
    for context=1:nframe
        xx = [];
        window = coeff(:,context:context+context_l+context_r);
        for w=1:context_l+context_r+1
           xx = [xx;window(:,w)]; % concate
        end
        x = [x,xx];
        y = [y,3];
    end
    x_class3 = [x_class3;x'];
    y_class3 = [y_class3;y'];
end

%% save data
x_test = [x_class1;x_class2;x_class3];
y_test = [y_class1;y_class2;y_class3];
save('.\train_data\test_ori_length.mat','x_test','y_test');
