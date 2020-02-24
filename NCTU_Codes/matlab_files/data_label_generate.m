clc;clear all;close all;
% noise
filelist_noise = dir('.\data\noise');
filelist_noise_mat = dir('.\data\noise_mat');

% clean speech
% filelist_speech = dir('.\data\clean speech');
filelist_data = dir('.\data\clean speech\MIR-1K');
filelist_data_new1 = dir('.\data\clean speech\交大VAD人聲training');
filelist_data_new2 = dir('.\data\clean speech\WeiFangVocal_20180220');
filelist_data_new3 = dir('.\data\clean speech\中英字音_20180730');

x_data = [];
y_data = [];
%% icfilter design
inband = 4;
bandnum = 10;

% midfre = [16 20 26 36 48 60 80 101];  %1024 point 8band
% midfre = [6 7 9 12 16 20 27 35 46 61 80 105 141 184 243 321];  %1024 point 16band
% midfre = [16 20 26 36 48 60 80 101 223];  %1024 point 8+1 3~4k
% midfre = [16 20 26 36 48 60 80 101 353];  %1024 point 8+1 5~6k
midfre = [16 20 26 36 48 60 80 101 256 353];  %1024 point 8+2

slope = 12; %1 2 4 6 8 10 12 14 16 18 20

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

%% noise for .wav / .mp3
for N=1:length(filelist_noise)-2
    thisdir = filelist_noise(N+2).name;
    new_filelist_noise = dir(['.\data\noise\',thisdir]);
    for no=1:length(new_filelist_noise)-2        
        y = 0;
        x_all = zeros(1,inband*bandnum);        
        [s fs] = audioread(['.\data\noise\',thisdir,'\',new_filelist_noise(no+2).name]);
        s = resample(s(:,1),16000,fs);  %44100 to 16000
        fs = 16000;
        for i=1601:400:400*(floor(length(s)/400))+1  %0.25s
            %label
            y = [y,0];
            %8bins
            s_fft_1 = fft(s(i-1600:i-1201),1024);
            s_fft_2 = fft(s(i-1200:i-801),1024);
            s_fft_3 = fft(s(i-800:i-401),1024);
            s_fft_4 = fft(s(i-400:i-1),1024);
            
            x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];

            x_all = [x_all;x];
        end
        x_data = [x_data;x_all(2:length(x_all(:,1)),:)];
        y_data = [y_data,y(2:length(y))];
    end
end

%% noise for .mat
for N=1:length(filelist_noise_mat)-2
    thisdir = filelist_noise_mat(N+2).name;
    new_filelist_noise = dir(['.\data\noise_mat\',thisdir]);
    for no=1:length(new_filelist_noise)-2        
        y = 0;
        x_all = zeros(1,inband*bandnum);        
        load(['.\data\noise_mat\',thisdir,'\',new_filelist_noise(no+2).name]);
        for i=1601:400:400*(floor(length(s)/400))+1  %0.25s
            %label
            y = [y,0];
            %8bins
            s_fft_1 = fft(s(i-1600:i-1201),1024);
            s_fft_2 = fft(s(i-1200:i-801),1024);
            s_fft_3 = fft(s(i-800:i-401),1024);
            s_fft_4 = fft(s(i-400:i-1),1024);
            
            x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];

            x_all = [x_all;x];
        end
        x_data = [x_data;x_all(2:length(x_all(:,1)),:)];
        y_data = [y_data,y(2:length(y))];
    end
end

%% clean speech MIR-1k
y = 0;
x = [];
x_all = zeros(1,inband*bandnum);

for da=1:length(filelist_data)-2
    [s fs] = audioread(['.\data\clean speech\MIR-1K\',filelist_data(da+2).name]);
    s = resample(s,16000,fs);
    fs = 16000;
    
%     if length(s)>16000*40
%         s = s(1:16000*40);  % 110*8=880
%     end
    t = s;
    
    for i=1601:400:400*(floor(length(s)/400))+1  %0.25s
        if(log(norm(t(i-1600:i-1))/norm(t)+0.0001)>-3)  %-3%
            %label
            y = [y,1];
            
            s_fft_1 = fft(s(i-1600:i-1201),1024);
            s_fft_2 = fft(s(i-1200:i-801),1024);
            s_fft_3 = fft(s(i-800:i-401),1024);
            s_fft_4 = fft(s(i-400:i-1),1024);
            
            x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];

            x_all = [x_all;x];
        end
    end
end
x_data=[x_data;x_all(2:length(x_all(:,1)),:)];
y_data=[y_data,y(2:length(y))];

%% clean MIR-1k _ 2
y = 0;
x = [];
x_all = zeros(1,inband*bandnum);

for da=1:length(filelist_data)-2
    [s fs] = audioread(['.\data\clean speech\MIR-1K\',filelist_data(da+2).name]);
    s = resample(s,16000,fs);
    fs = 16000;
    
    if length(s)>16000*10
        s = s(1:16000*10);  % 110*8=880
    end
    t = s;
    
    for i=1601:400:400*(floor(length(s)/400))+1  %0.25s
        if(log(norm(t(i-1600:i-1))/norm(t)+0.0001)>-3)  %-3%
            %label
            y = [y,1];
            s_fft_1 = fft(s(i-1600:i-1201),1024);
            s_fft_2 = fft(s(i-1200:i-801),1024);
            s_fft_3 = fft(s(i-800:i-401),1024);
            s_fft_4 = fft(s(i-400:i-1),1024);            
            
            x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];

            x_all = [x_all;x];
        end
    end
end
x_data=[x_data;x_all(2:length(x_all(:,1)),:)];
y_data=[y_data,y(2:length(y))];

%% clean_new1
y = 0;
x = [];
x_all = zeros(1,inband*bandnum);

for da=1:length(filelist_data_new1)-2
    [s fs] = audioread(['.\data\clean speech\交大VAD人聲training\',filelist_data_new1(da+2).name]);
    s = resample(s(:,1),16000,fs);
    fs = 16000;
    
%     if length(s)>16000*120
%         s = s(1:16000*120);  % take 80 sec
%     end
    t = s;
    
    for i=1601:400:400*(floor(length(s)/400))+1  %0.25s
        if(log(norm(t(i-1600:i-1))/norm(t)+0.0001)>-3)  %-3%
            %label
            y = [y,1];
            
            s_fft_1 = fft(s(i-1600:i-1201),1024);
            s_fft_2 = fft(s(i-1200:i-801),1024);
            s_fft_3 = fft(s(i-800:i-401),1024);
            s_fft_4 = fft(s(i-400:i-1),1024);
                        
            x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];

            x_all = [x_all;x];
        end
    end
end
x_data=[x_data;x_all(2:length(x_all(:,1)),:)];
y_data=[y_data,y(2:length(y))];

%% clean_new2
y = 0;
x = [];
x_all = zeros(1,inband*bandnum);

for da=1:length(filelist_data_new2)-2
    [s fs] = audioread(['.\data\clean speech\WeiFangVocal_20180220\',filelist_data_new2(da+2).name]);
    s = resample(s(:,1),16000,fs);
    fs = 16000;
    
    %     if length(s)>16000*80
    %         s = s(1:16000*80);  % take 80 sec
    %     end
    t = s;
    
    for i=1601:400:400*(floor(length(s)/400))+1  %0.25s
        if(log(norm(t(i-1600:i-1))/norm(t)+0.0001)>-3)  %-3%
            %label
            y = [y,1];
            
            s_fft_1 = fft(s(i-1600:i-1201),1024);
            s_fft_2 = fft(s(i-1200:i-801),1024);
            s_fft_3 = fft(s(i-800:i-401),1024);
            s_fft_4 = fft(s(i-400:i-1),1024);
            
            x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];

            x_all = [x_all;x];
        end
    end
end
x_data=[x_data;x_all(2:length(x_all(:,1)),:)];
y_data=[y_data,y(2:length(y))];

%% clean_new3
y = 0;
x = [];
x_all = zeros(1,inband*bandnum);

for da=1:length(filelist_data_new3)-2
    
    [s fs] = audioread(['.\data\clean speech\中英字音_20180730\',filelist_data_new3(da+2).name]);
    s = resample(s(:,1),16000,fs);
    fs = 16000;
    
    %     if length(s)>16000*80
    %         s = s(1:16000*80);  % take 80 sec
    %     end
    t = s;
    
    for i=1601:400:400*(floor(length(s)/400))+1  %0.25s
        if(log(norm(t(i-1600:i-1))/norm(t)+0.0001)>-3)  %-3%
            %label
            y = [y,1];
            s_fft_1 = fft(s(i-1600:i-1201),1024);
            s_fft_2 = fft(s(i-1200:i-801),1024);
            s_fft_3 = fft(s(i-800:i-401),1024);
            s_fft_4 = fft(s(i-400:i-1),1024);            
          
            x = [log(icfilter*(abs(s_fft_1(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_2(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_3(1:513)).^2)+0.0001)'...
                log(icfilter*(abs(s_fft_4(1:513)).^2)+0.0001)'];

            x_all = [x_all;x];            
        end
    end
end
x_data=[x_data;x_all(2:length(x_all(:,1)),:)];
y_data=[y_data,y(2:length(y))];


%% normalize
x_temp=x_data;
x_data=x_temp;

for i=1:length(x_data)
    
    % normalize
    x_data(i,1:inband*bandnum) = (x_data(i,1:inband*bandnum)-min(x_data(i,1:inband*bandnum)))/...
        (max(x_data(i,1:inband*bandnum))-min(x_data(i,1:inband*bandnum))+0.0001);
    
end

%% save
save(['.\train_data\8+2band(25ms)\train_1106a_sharp_',num2str(slope),'.mat'],'x_data');
save(['.\train_label\8+2band(25ms)\label_1106a_sharp_',num2str(slope),'.mat'],'y_data');

%% count data
for count=1:length(y_data)
    if(y_data(count))==1
        count = count-1;
        break;
    end
end