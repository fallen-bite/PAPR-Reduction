%labeled_data_generation.m
%modulation 16-QAM; sub-blocks 6; 
%Jinan, Shangdong, China (Asia)
%2021/8/1
clear, figure(1), clf, clc
phase_fac=[cos(0)+1j*sin(0),cos(pi/3)+1j*sin(pi/3),cos(2*pi/3)+1j*sin(2*pi/3),cos(pi)+1j*sin(pi),cos(4*pi/3)+1j*sin(4*pi/3),cos(5*pi/3)+1j*sin(5*pi/3)];
[a1,b1,c1,d1,e1,f1]=ndgrid(1:6);
A=[a1(:),b1(:),c1(:),d1(:),e1(:),f1(:)];
for i=1:6
    A(A==i)=phase_fac(i);
end
%%
N=256; Nos=3; NNos=N*Nos;
b=4; M=2^b; % Number of bits per QAM symbol and Alphabet size
Nsbs = [6]; % number of sub-blocks v, by default.
%gss='*^<sd>v.'; % Numbers of subblocks and graphic symbols
dBs = [4:0.1:11]; dBcs = dBs+(dBs(2)-dBs(1))/2;
Nblk = 50000; % Number of OFDM blocks for iteration
CCDF_OFDMa = CCDF_OFDMA(N,Nos,b,dBs,Nblk);
semilogy(dBs,CCDF_OFDMa,'k')
hold on
Nsb=Nsbs(1); str(1,:)=sprintf('No of subblocks=%2d',Nsb);
CCDF=CCDF_PTS1(N,Nos,Nsb,b,dBs,Nblk,A);
%%
%
% pr2_2_8 
clear all; clc; close all;
 
fs=128;                         % 采样频率
% 第一部分
N=128;                          % 信号长度
t=(0:N-1)/fs;                   % 时间序列
y=cos(2*pi*30*t);               % 信号序列
Y=fft(y,N);                     % FFT
freq=(0:N/2)*fs/N;              
n2=1:N/2+1;                     % 计算正频率的索引号
Y_abs=abs(Y(n2))*2/N;           % 给出正频率部分的频谱幅值
% 作图
subplot 211; 
stem(freq,Y_abs,'k')
xlabel('频率(Hz)'); ylabel('幅值');
title('(a) Fs=128Hz, N=128')
axis([10 50 0 1.2]); 
 
% 第二部分
N1=100;                           % 信号长度
t1=(0:N1-1)/fs;                   % 时间序列
y1=cos(2*pi*30*t1);               % 信号序列
Y1=fft(y1,N1);                    % FFT
freq1=(0:N1/2)*fs/N1;             
n2=1:N1/2+1;                      % 计算正频率的索引号
Y_abs1=abs(Y1(n2))*2/N1;          % 给出正频率部分的频谱幅值
% 作图
subplot 212; 
stem(freq1,Y_abs1,'k')
xlabel('频率(Hz)'); ylabel('幅值');
title('(b) Fs=128Hz, N=100')
axis([10 50 0 1.2]); hold on
line([30 30],[0 1],'color',[.6 .6 .6],'linestyle','--');
plot(30,1,'ko','linewidth',5)
set(gcf,'color','w');






