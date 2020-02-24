clc;close all;
%clear all

slope = 12;
% load weight
load(['.\weight\8+2band(25ms)\w_1106a_sharp_',num2str(slope)]);

% load sound (.wav /.mp3)
% [s f] = audioread('.\data\noise\Office Noise\快速指敲桌子 1.mp3');
% s = resample(s(:,1),16000,f);

% load sound (.mat)
load('.\data\noise_mat\Speakers Train\Demo_開場_part1.dat.mat');

target = 0; % 0:for noise ; 1:for speech 

t = s;

% threshold = -250;
threshold = -145;
% threshold = -10;

%% icfilter design
inband = 4;
bandnum = 10;

% midfre = [16 20 26 36 48 60 80 101];  %1024 point 8band
% midfre = [6 7 9 12 16 20 27 35 46 61 80 105 141 184 243 321];  %1024 point 16band
% midfre = [16 20 26 36 48 60 80 101 223];  %1024 point 8+1 3~4k
% midfre = [16 20 26 36 48 60 80 101 353];  %1024 point 8+1 5~6k
midfre = [16 20 26 36 48 60 80 101 256 353];  %1024 point 8+2

for i=1:bandnum
    if i==9
        for j=1:513
            icfilter(i,j)=10^((-360*abs(log10(15.625*(j-1)+1)-log10(15.625*(midfre(i)-1))))/20);
        end
        
    elseif i==10
        for j=1:513
            icfilter(i,j)=10^((-360*abs(log10(15.625*(j-1)+1)-log10(15.625*(midfre(i)-1))))/20);
        end
        
    else
        for j=1:513
            icfilter(i,j)=10^((-20*slope*abs(log10(15.625*(j-1)+1)-log10(15.625*(midfre(i)-1))))/20);
        end
    end
end


%% count
count_answer_1 = 0;
count_answer_0 = 0;
count_test_1 = 0;
count_test_0 = 0;

y_answer = [];
y_test = [];

local_power = [];
local_power_normalize = [];
x_all = [];
x_normalize_all = [];

for i=1601:400:400*(floor(length(s)/400))+1 %25ms
    s_fft_1 = fft(s(i-1600:i-1201),1024);
    s_fft_2 = fft(s(i-1200:i-801),1024);
    s_fft_3 = fft(s(i-800:i-401),1024);
    s_fft_4 = fft(s(i-400:i-1),1024);
     
    x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
        log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
        log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
        log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];
    
    x_all = [x_all;x(1:inband*bandnum)];
    local_power = [local_power;sum(x)];
    
    % normalize
    x_normalize = (x(1:inband*bandnum)-min(x(1:inband*bandnum)))/(max(x(1:inband*bandnum))-min(x(1:inband*bandnum))+0.0001);
    x_normalize_all = [x_normalize_all;x_normalize];
    local_power_normalize = [local_power_normalize;sum(x_normalize)];
    
    % count test
    if sum(x)>threshold
        answer = softmax((poslin(poslin(x_normalize*w1+b1)*w2+b2)*w3+b3)');  % poslin = ReLU
        if(answer(1)>answer(2))
            y_test = [y_test 0];
            count_test_0 = count_test_0+1;
        else
            y_test = [y_test 1];
            count_test_1 = count_test_1+1;
        end
    else
        y_test = [y_test 0];
        count_test_0 = count_test_0+1;
    end
    
    % count answer
    if target==1
        if sum(x)>threshold
%         if(log(norm(t(i-1600:i-1))/norm(t)+0.0001)>-4)
            y_answer = [y_answer 1];
            count_answer_1 = count_answer_1+1;
        else
            y_answer = [y_answer 0];
            count_answer_0 = count_answer_0+1;
        end
    else
        y_answer = [y_answer 0];
        count_answer_0 = count_answer_0+1;
    end
           
end

%% accuracy
acc = 0;
LED = 0;
wrong = [];

if target==1
    for c=1:length(y_answer)
        acc = acc+abs(y_answer(c)-y_test(c));
    end
    acc = (length(y_answer)-acc)/length(y_answer);
    
else
    acc = count_test_0/(count_test_0+count_test_1);
    
%     % LED
%     for c=1:length(y_test)
%         if y_test(c)==1
%             %        LED = LED+1;
%             %        wrong = [wrong;c];
%             if y_test(c+1)==1
%                 %            LED = LED+1;
%                 %            wrong = [wrong;c];
%                 if y_test(c+2)==1
%                     %                LED = LED+1;
%                     %                wrong = [wrong;c];
%                     if y_test(c+3)==1
%                         LED = LED+1;
%                         wrong = [wrong;c];
%                     end
%                 end
%             end
%         end
%     end
%     
end
%% audiowrite
filter_answer = [];
filter_test = [];

for i=1:length(y_answer)
    if y_answer(i)==1
        filter_answer = [filter_answer;1*ones(400,1)];
    else
        filter_answer = [filter_answer;0*ones(400,1)];
    end
    
    if y_test(i)==1
        filter_test = [filter_test;1*ones(400,1)];
    else
        filter_test = [filter_test;0*ones(400,1)];
    end
    
end

for i=1:400*(floor(length(s)/400))
    s_temp(i,1) = s(i,1);
end
s_temp = s_temp(16000*0.075+1:length(s_temp));
s_answer = filter_answer.*s_temp;
s_test = filter_test.*s_temp;


% audiowrite('.\audiowrite\HD_StdTest_0618.wav',s_answer,16000);
% audiowrite('.\audiowrite\HD_StdTest_0618.wav',s_test,16000);

%% plot
figure;
plot(s);
title('original');
figure;
plot(s_answer);
title('answer');
figure;
plot(s_test);
title('test');

figure;
sp = spectrogram(s,800,400,1024);
% image((100*abs(sp)));axis xy; % for .wav /.mp3
image((0.01*abs(sp)));axis xy; %for .mat
title('Demo-開場-part1.dat');

figure;
plot(y_answer,'o');
title('answer');

figure;
plot(y_test,'o');
title('test');

axis_x = linspace(1,length(local_power_normalize),1000);
figure;
% plot(local_power,'bo-');hold on
plot(local_power,'b');hold on
plot(axis_x,mean(local_power)*ones(1000),'r');hold on
plot(axis_x,threshold*ones(1000),'g');
legend('power','maen',num2str(threshold));

% figure;
% plot(x_all);

%%
% soundsc(s_answer,16000)
% soundsc(s_test,16000)
% clear sound