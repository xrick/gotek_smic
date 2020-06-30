clc;clear all;

load('.\weight\w_3layer128');

count_pass_all = [];
count_fail_all = [];
w_smooth = 10;

threshold = 0.25;
% threshold = 0.15;
% threshold = 0.05;

target = 1;
datapath = '.\test_signal_dark_suit';
% datapath = '.\test_whole sentences(positive)';
% target = 0;
% datapath = '.\test_signal_filler';
% datapath = '.\test_whole sentences(negative)';

dir_datapath = dir(datapath);
for da=1:length(dir_datapath)-2
    [signal,fs] = audioread([datapath,'\',dir_datapath(da+2).name]);
    
% [signal,fs] = audioread('.\whole sentence\mzmb0_1-sa1.wav');
% [signal,fs] = audioread('.\whole sentence\mzmb0_2-sa2.wav');
% [signal,fs] = audioread('D:\Google ¶³ºÝµwºÐ\plan_´I¨¹\final project\data\rec_dark_suit\01\01.wav');
% [signal,fs] = audioread('D:\Google ¶³ºÝµwºÐ\plan_´I¨¹\final project\data\rec_enr_phrase\01_tell_apart-01.wav');
signal = resample(signal(:,1),16000,fs);
% [signal1,fs] = audioread('.\test_signal_dark\mzmb0_dark.wav');
% signal1 = resample(signal1(:,1),16000,fs);
% [signal2,fs] = audioread('.\test_signal_suit\mzmb0_suit.wav');
% signal2 = resample(signal2(:,1),16000,fs);
% signal = [signal1;signal2];
%% filter design
fs = 16000;
windowSize = fs*0.025;  % 25ms
windowStep = fs*0.010;   % 10ms(shift)
nDims = 40;
context_l = 30;
context_r = 10;
% fftSize = 512;
% cepstralCoefficients = 40;

%% pre-processing
x_data = [];
% vad
% rest_signal = simpleSAD(signal,fs);
% lens = length(rest_signal);
% if lens>8000
%     rest_signal = rest_signal(lens-8000+1:lens,:);
% elseif lens<8000
%     rest_signal = [0.5+rand(8000-lens,1).*10^-6;rest_signal];
% end

rest_signal = signal;
% rest_signal = [rand(8000,1).*10^-6;rest_signal;rand(8000,1).*10^-6];

% rest_signal = [zeros(8000,1);rest_signal;zeros(8000,1)];
% rest_signal = rest_signal+rand(length(rest_signal),1).*10^-6;

% mfcc
coeff = mfcc(rest_signal, fs, windowSize, windowStep);
temp = coeff(1,:)-min(coeff(1,:));
coeff(1,:) = temp./max(temp);
nframe = length(coeff(1,:));
coeff = [zeros(nDims,context_l), coeff, zeros(nDims,context_r)];

x = [];
for context=1:nframe
    xx = [];
    window = coeff(:,context:context+context_l+context_r);
    for w=1:context_l+context_r+1
        xx = [xx;window(:,w)]; % concate
    end
    x = [x,xx];
end
x_data = [x_data;x']; % ¤@­Ó­µÀÉªºdata


%% offline test
% w_smooth = 30;
% w_smooth = 10;
w_max = 100;
n_class = 3;

all_answer = [];
all_smooth_answer = [];
all_confi = [];
y_test = [];
for j=1:length(x_data(:,1))
    pred = poslin(poslin(poslin(x_data(j,:)*w1+b1)*w2+b2)*w3+b3)*w4+b4;
    answer = softmax(pred');
    all_answer = [all_answer,answer];
    % smooth
    if j<=w_smooth
        smooth_answer = (1/(j-1+1)).*sum(all_answer(:,1:j),2);
        all_smooth_answer = [all_smooth_answer,smooth_answer];
    else
        k = j-w_smooth+1;
        smooth_answer = (1/(j-k+1)).*sum(all_answer(:,k:j),2);
        all_smooth_answer = [all_smooth_answer,smooth_answer];
    end
    
    % confidence
    if j<=w_max
        confi = nthroot(max(all_smooth_answer(1,1:j))*max(all_smooth_answer(2,1:j)), n_class-1);
        all_confi = [all_confi,confi];
    else
        k = j-w_max+1;
        confi = nthroot(max(all_smooth_answer(1,k:j))*max(all_smooth_answer(2,k:j)), n_class-1);
        all_confi = [all_confi,confi];
    end    
  
    
    if(answer(1)>answer(2)&&answer(1)>answer(3))
        y_test = [y_test 1];
    elseif(answer(2)>answer(1)&&answer(2)>answer(3))
        y_test = [y_test 2];
    else
        y_test = [y_test 3];
    end
    
end

%% threshold

count_pass = 0;
count_fail = 0;
for l=1:length(all_confi)
    if all_confi(l)>threshold
%         fprintf('pass!\n');
        count_pass = count_pass+1;
        break;
    end
end
if count_pass==0
    count_fail = count_fail+1;
end

count_pass_all = [count_pass_all;count_pass];
count_fail_all = [count_fail_all;count_fail];

end

%% accuracy
c_pass = sum(count_pass_all);
c_fail = sum(count_fail_all);
if target==1
    acc = sum(count_pass_all)/length(count_pass_all);
else
    acc = sum(count_fail_all)/length(count_fail_all);
end
%% plot
figure;plot(all_answer(1,:));hold on;plot(all_answer(2,:));hold on;plot(all_answer(3,:));
ylim([-0.5 1.5]);
legend('class1','class2','class3');
figure;plot(all_smooth_answer(1,:));hold on;plot(all_smooth_answer(2,:));hold on;plot(all_smooth_answer(3,:));
ylim([-0.5 1.5]);
legend('class1','class2','class3');

figure;plot(all_confi);