%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%
% Copyright 2019 Mohammad Al-Sa'd
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
% Authors: Mohammad F. Al-Sa'd (mohammad.al-sad@tuni.fi)
%          Amr Mohamed         (amrm@qu.edu.qa)
%          Abdulla Al-Ali
%          Tamer Khattab
%
% The following reference should be cited whenever this script is used:
%     M. Al-Sa'd et al. "RF-based drone detection and identification using
%     deep learning approaches: an initiative towards a large open source
%     drone database", 2019.
%
% Last Modification: 12-02-2019
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

close all; clear; clc
current_directory_working = pwd;                    % [MR] Current working directory
filepath = fileparts(current_directory_working);    % [MR] Path for easier management
filepath = [filepath '\Data\'];

%% Parameters
opt = 2;                                      % Change to 1, 2, or 3 to alternate between the 1st, 2nd, and 3rd DNN inputs respectively.
BUI = {'00000','10000','10001','10010','10011',...
    '10100','10101','10110','10111','11000'}; % BUI of all RF data
M   = 2048;                                   % Total number of frequency bins
fs  = 80;                                     % Sampling frequency in MHz
f   = 2400+(0:fs/(M-1):fs);                   % Frequency array for plotting
S   = 10;                                     % Number of points in the moving average filter for smoothing
c   = [0 0 1 ; 1 0 0 ; 0 1 0 ; 0 0 0.1724 ;...
    1 0.1034 0.7241 ; 1 0.8276 0 ; 0 0.3448 0 ;...
    0.5172 0.5172 1 ; 0.6207 0.3103 0.2759 ;...
    0 1 0.7586];                              % 10 distinct colours for plotting

running_time = dictionary; % [MR] Timers

timer_total = tic; % [MR] Start timer for total program

%% Averaging spectra
timer_phase = tic; % [MR] Start timer for this phase

s = zeros(length(BUI),M);
for i = 1:length(BUI)
    x = load([filepath BUI{1,i} '.mat']);
    [M, N] = size(x.Data);
    s(i,:) = mean(x.Data,2);
end

elapsed_time = toc(timer_phase); % [MR] Stop timer for this phase
running_time('elapsed_time_averaging') = elapsed_time;      % [MR]
fprintf('Ended | Averaging \n');                            % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);    % [MR]

%% Aggregating and smoothing RF spectra
timer_phase = tic; % [MR] Start timer for this phase

if(opt == 1)
    sig             = zeros(2,M);
    sig_smooth      = zeros(2,M);
    sig(1,:)        = s(1,:);
    sig(2,:)        = mean(s(2:end,:));
    sig_smooth(1,:) = smooth(sig(1,:),S);
    sig_smooth(2,:) = smooth(sig(2,:),S);
elseif(opt == 2)
    sig             = zeros(4,M);
    sig_smooth      = zeros(4,M);
    sig(1,:)        = s(1,:);
    sig(2,:)        = mean(s(2:5,:));
    sig(3,:)        = mean(s(6:9,:));
    sig(4,:)        = s(10,:);
    sig_smooth(1,:) = smooth(sig(1,:),S);
    sig_smooth(2,:) = smooth(sig(2,:),S);
    sig_smooth(3,:) = smooth(sig(3,:),S);
    sig_smooth(4,:) = smooth(sig(4,:),S);
elseif(opt == 3)
    sig              = s;
    sig_smooth       = zeros(10,M);
    sig_smooth(1,:)  = smooth(sig(1,:),S);
    sig_smooth(2,:)  = smooth(sig(2,:),S);
    sig_smooth(3,:)  = smooth(sig(3,:),S);
    sig_smooth(4,:)  = smooth(sig(4,:),S);
    sig_smooth(5,:)  = smooth(sig(5,:),S);
    sig_smooth(6,:)  = smooth(sig(6,:),S);
    sig_smooth(7,:)  = smooth(sig(7,:),S);
    sig_smooth(8,:)  = smooth(sig(8,:),S);
    sig_smooth(9,:)  = smooth(sig(9,:),S);
    sig_smooth(10,:) = smooth(sig(10,:),S);
end

elapsed_time = toc(timer_phase); % [MR] Stop timer for this phase
running_time('elapsed_time_smoothing') = elapsed_time;      % [MR]
fprintf('Ended | Smoothing \n');                            % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);    % [MR]

%% Plotting
timer_phase = tic; % [MR] Start timer for this phase

figure('Color',[1,1,1],'position',[100, 60, 840, 600]);
a = [];
for i = 1:size(sig,1)
    a(i) = plot(f,20*log10(sig_smooth(i,:)./(max(sig_smooth(i,:)))),'Color',c(i,:),'linewidth',2); hold on;
    tt{i} = ['Class ' num2str(i)];
end
xlabel('Frequency (MHz)','fontsize',18); grid on; grid minor;
ylabel('Power (dB)','fontsize',18);
ylim([-110 5]);
if(opt == 3)
    legend(a,tt,'orientation','horizontal','NumColumns',5,'location','south')
else
    legend(a,tt,'orientation','horizontal','location','south')
end
set(gca,'fontweight','bold','fontsize',20,'FontName','Times');
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));

figure('Color',[1,1,1],'position',[100, 60, 700, 570]);
h = boxplot(20*log10(sig_smooth./(max(sig_smooth,[],2)))','Whisker',200);
set(h,{'linew'},{2});
set(gca,'fontweight','bold','fontsize',18,'FontName','Times');
xlabel('RF signal class','fontsize',18); grid on; grid minor;
ylabel('Power (dB)','fontsize',20);
ylim([-100 5]);
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));

elapsed_time = toc(timer_phase); % [MR] Stop timer for this phase
running_time('elapsed_time_plotting') = elapsed_time;       % [MR]
fprintf('Ended | Plotting \n');                             % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);    % [MR]

%% Saving
Q = input('Do you want to save the results (Y/N)\n','s');
if(Q == 'y' || Q == 'Y')
    timer_phase = tic;                  % [MR] Start timer for this phase

    print(1,['Spectrum_' num2str(opt)],'-dpdf','-r512');
    print(2,['Box_' num2str(opt)],'-dpdf','-r512');

    elapsed_time = toc(timer_phase);    % [MR] Stop timer for this phase
    running_time('elapsed_time_saving') = elapsed_time;         % [MR]
    fprintf('Ended | Saved results \n');                        % [MR]
    fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);    % [MR]
else
    return
end

%% [MR] Elapsed time
elapsed_time = toc(timer_total); % [MR] Stop timer for total program
running_time('elapsed_time_total') = elapsed_time;          % [MR]
fprintf('Ended | Total \n');                                % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);    % [MR]

%% [MR] Print running time
longest_name_length = max(cellfun(@length, ...
                            keys(running_time)));
longest_time_length = max(arrayfun(@(time) numel(num2str(time, '%.4f')), ...
                            values(running_time)));
fprintf('\nRunning Time:\n');
phases = keys(running_time);
for phase = 1:length(phases)
    phase_name = phases{phase};
    phase_elapsed_time = running_time(phase_name);
    fprintf('| %-*s = %*.4f seconds\n', longest_name_length, phase_name, longest_time_length, phase_elapsed_time);
end