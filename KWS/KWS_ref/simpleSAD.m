function rest_s = simpleSAD(s,fs)
% [s fs] = audioread(['C:\Users\User\Desktop\speech data\TIMIT\train\fsag0_4-si1953.wav']);
% s = resample(s(:,1),16000,fs);
% fs = 16000;

% normalize
s = s-min(s);
s = s./max(s);

% FrameSize = fs*0.032; % frame size 32ms
% ShiftSize = fs*0.016; % shift 16ms
FrameSize = fs*0.025; % frame size 32ms
ShiftSize = fs*0.010; % shift 16ms
Overlap = FrameSize-ShiftSize;
threshold = -1.63;
% threshold = -2;

s_temp = [];
temp = [];
temp_all = [];
new = [];
rest_s = [];

t = s;

%% frame size 32ms / shift 16ms (overlap 16ms)
n = floor((length(s)-FrameSize)/ShiftSize);
for i=FrameSize+1:ShiftSize:ShiftSize*n+FrameSize+1  %0.1s    
    temp = log(norm(t(i-FrameSize:i-1))/norm(t)+0.0001);
    temp_all = [temp_all;temp];
    if temp>threshold
        new = [new;1*ones(ShiftSize,1)];
    else
        new = [new;0*ones(ShiftSize,1)];
    end    
end

for i=1:ShiftSize*n+FrameSize
    s_temp(i,1) = s(i,1);
end
% s_temp = s_temp(Overlap+1:length(s_temp));
s_temp = s_temp(1:end-Overlap);
new_s = new.*s_temp;

for j=1:length(new)
    if new(j)==1
        rest_s = [rest_s;new_s(j)];
    end
end

%% plot
% figure;
% plot(s);
%
% figure;
% plot(rest_s);
%
% a = threshold*ones(400,1);
% figure;
% plot(temp_all);hold on
% plot(a);
%
% % soundsc(new_s,16000);
% soundsc(rest_s,16000);
% % clear sound
%
% figure;
% sp = spectrogram(s,512,256,1024);
% image((100*abs(sp)));axis xy;

end

