% fid = fopen('icfilter.txt', 'w+');
% 
% fprintf(fid, '{');
% for i = 1: size(icfilter,1)
%     fprintf(fid, '{');
%     for j = 1:size(icfilter,2)-1
%         fprintf(fid, '%d,', icfilter(i,j) );
%     end
%     fprintf(fid, '%d', icfilter(i,j+1) );
%     fprintf(fid, '},\n');
% end
% fprintf(fid, '};');
% fclose(fid); 
% 
% fid = fopen('input_test.txt', 'w+');
% 
% fprintf(fid, '{');
% for i = 1: 10
%     fprintf(fid, '{');
%     for j = 1:1600-1
%         fprintf(fid, '%d,', s(j+(i-1)*1600) );
%     end
%     fprintf(fid, '%d', s(j+1+(i-1)*1600) );
%     fprintf(fid, '},\n');
% end
% fprintf(fid, '};');
% fclose(fid); 

%% 
slope = 12;
% load weight
load(['.\weight\8+2band(25ms)\w_1106a_sharp_',num2str(slope)]);


input = 40;
hiddenlayer1 = 128;
hiddenlayer2 = 32;

fid = fopen('weight_txt\nnCoeff_10band_sharp_12_1106a.txt', 'w+'); %%%%%%%%%%%%%%%%%%%%%%%%%%% name
fprintf(fid, 'float weight1[inband*bandnum][hiddenlayer1]={');
for i = 1: input 
    fprintf(fid, '{');
    for j = 1:hiddenlayer1-1
        fprintf(fid, '%d,', w1(i ,j) );
    end
    fprintf(fid, '%d', w1(i ,j+1) );
    fprintf(fid, '},\n');
end
fprintf(fid, '};\n\n');
fprintf(fid, 'float weight2[hiddenlayer1][hiddenlayer2]={');
for i = 1: hiddenlayer1
    fprintf(fid, '{');
    for j = 1:hiddenlayer2-1
        fprintf(fid, '%d,', w2(i ,j) );
    end
    fprintf(fid, '%d', w2(i ,j+1) );
    fprintf(fid, '},\n');
end
fprintf(fid, '};\n\n');
fprintf(fid, 'float weight3[hiddenlayer2][2]={');
for i = 1: hiddenlayer2
    fprintf(fid, '{');
    for j = 1:2-1
        fprintf(fid, '%d,', w3(i ,j) );
    end
    fprintf(fid, '%d', w3(i ,j+1) );
    fprintf(fid, '},\n');
end
fprintf(fid, '};\n\n');

fprintf(fid, 'float bias1[hiddenlayer1]={');
fprintf(fid, '%d,',b1);
fprintf(fid, '};\n');
fprintf(fid, 'float bias2[hiddenlayer2]={');
fprintf(fid, '%d,',b2);
fprintf(fid, '};\n');
fprintf(fid, 'float bias3[2]={');
fprintf(fid, '%d,',b3);
fprintf(fid, '};\n');
fclose(fid); 