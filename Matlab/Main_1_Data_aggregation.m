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
filepath = 'D:\ISCTE\Thesis\Program';               % [MR] Path for easier management
load_filename = [filepath '\Data\'];                % Path of raw RF data % [MR]
save_filename = fileparts(pwd);
save_filename = [save_filename '\Data\'];           % Path of aggregated data
log_filename = [fileparts(pwd) '\DataArchive\'];    % [MR] Path of archived result data

disp(['filepath      | ' filepath]);        % [MR] Print path
disp(['load_filename | ' load_filename]);   % [MR] Print path
disp(['save_filename | ' save_filename]);   % [MR] Print path
disp(['log_filename  | ' log_filename]);    % [MR] Print path
disp(['pwd           | ' pwd]);             % [MR] Print path

%% Parameters
BUI{1,1} = {'00000'};                         % BUI of RF background activities
BUI{1,2} = {'10000','10001','10010','10011'}; % BUI of the Bebop drone RF activities
BUI{1,3} = {'10100','10101','10110','10111'}; % BUI of the AR drone RF activities
BUI{1,4} = {'11000'};                         % BUI of the Phantom drone RF activities
M = 2048; % Total number of frequency bins
L = 1e5;  % Total number samples in a segment
Q = 10;   % Number of returning points for spectral continuity

%% Main
tic; % [MR] Start timer
for opt = 1:length(BUI)
    fprintf('%d | in for opt \n', opt); % [MR] Print for debugging
    % Loading and averaging
    for b = 1:length(BUI{1,opt})
        fprintf('%s | in for BUI ', BUI{1,opt}{b}); % [MR] Print for debugging
        %%disp(BUI{1,opt}{b})
        if(strcmp(BUI{1,opt}{b},'00000'))
            N = 40; % Number of segments for RF background activities
        elseif(strcmp(BUI{1,opt}{b},'10111'))
            N = 17;
        else
            N = 20; % Number of segments for drones RF activities
        end
        fprintf('|| N = %d \n', N); % [MR] Print N for debugging
        data = [];
        cnt = 1;
        for n = 0:N
            fprintf('||| n = %d ', n); % [MR] Print n for debugging
            % Loading raw csv files
            x = csvread([load_filename BUI{1,opt}{b} 'L_' num2str(n) '.csv']);
            y = csvread([load_filename BUI{1,opt}{b} 'H_' num2str(n) '.csv']);
            % re-segmenting and signal transformation
            for i = 1:length(x)/L
                st = 1 + (i-1)*L;
                fi = i*L;
                xf = abs(fftshift(fft(x(st:fi)-mean(x(st:fi)),M))); xf = xf(end/2+1:end);
                yf = abs(fftshift(fft(y(st:fi)-mean(y(st:fi)),M))); yf = yf(end/2+1:end);
                data(:,cnt) = [xf ; (yf*mean(xf((end-Q+1):end))./mean(yf(1:Q)))];
                cnt = cnt + 1; 
            end
            disp(100*n/N)
        end
        Data = data.^2;
        % Saving
        save([save_filename BUI{1,opt}{b} '.mat'],'Data');
        save([log_filename BUI{1,opt}{b} '.mat'],'Data'); % [MR] Save for archival
    end
end
disp('Ended.'); % [MR] Print for debugging
elapsed_time_total = toc; % [MR] Stop timer
disp(['Elapsed time (Total): ', num2str(elapsed_time_total), ' seconds']); % [MR] Print elapsed time