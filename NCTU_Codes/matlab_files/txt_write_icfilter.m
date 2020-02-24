fid = fopen('icfilter_10band_sharp_12.txt', 'w+');%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf(fid, 'float filterCoeff[bandnum][fft_order/2+1]={');
for i = 1: size(icfilter,1)
    fprintf(fid, '{');
    for j = 1:size(icfilter,2)-1
        fprintf(fid, '%d,', icfilter(i,j) );
    end
    fprintf(fid, '%d', icfilter(i,j+1) );
    fprintf(fid, '},\n');
end
fprintf(fid, '};');
fclose(fid); 

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