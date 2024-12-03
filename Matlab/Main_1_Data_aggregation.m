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
load_filename = [filepath '\Data\'];                % Path of raw RF data % [MR]
save_filename = load_filename;                      % [MR] Path of aggregated data

word_width = 14; % [MR]
fprintf('%-*s | %s \n', word_width, 'pwd', pwd);                        % [MR] Print path
fprintf('%-*s | %s \n', word_width, 'filepath', filepath);              % [MR] Print path
fprintf('%-*s | %s \n', word_width, 'load_filename', load_filename);    % [MR] Print path
fprintf('%-*s | %s \n', word_width, 'save_filename', save_filename);    % [MR] Print path
fprintf('\n'); % [MR] Separation

%% Parameters
BUI{1,1} = {'00000'};                         % BUI of RF background activities
BUI{1,2} = {'10000','10001','10010','10011'}; % BUI of the Bebop drone RF activities
BUI{1,3} = {'10100','10101','10110','10111'}; % BUI of the AR drone RF activities
BUI{1,4} = {'11000'};                         % BUI of the Phantom drone RF activities
M = 2048; % Total number of frequency bins
L = 1e5;  % Total number samples in a segment
Q = 10;   % Number of returning points for spectral continuity

running_time = dictionary; % [MR] Timers
timer_total = tic; % [MR] Start timer for total program

%% Main
for opt = 1:length(BUI)
    opt_length = length(BUI);                   % [MR] Help for printing
    bui_width = length(BUI{1,1}{1});            % [MR] Help for printing
    max_width = max([opt_length bui_width]);    % [MR] Help for printing
    fprintf('%*s | in for %-*s \n', max_width, num2str(opt), 3, 'opt'); % [MR] Print for debugging
    % Loading and averaging
    for b = 1:length(BUI{1,opt})
        timer_phase = tic; % [MR] Start timer for this phase
        fprintf('%*s | in for %-*s ', max_width, BUI{1,opt}{b}, 3, 'b'); % [MR] Print for debugging
        %%disp(BUI{1,opt}{b})
        if(strcmp(BUI{1,opt}{b},'00000'))
            N = 40; % Number of segments for RF background activities
        elseif(strcmp(BUI{1,opt}{b},'10111'))
            N = 17;
        else
            N = 20; % Number of segments for drones RF activities
        end
        fprintf('(N = %d)\n', N); % [MR] Print for debugging
        data = [];
        cnt = 1;
        for n = 0:N
            fprintf('| n = %-2d ', n); % [MR] Print for debugging
            % Loading raw csv files
            % Note: function must be 'csvread'
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
            fprintf('| pct. = %6.2f %% \n', 100*n/N); % [MR]
        end
        Data = data.^2;
        % Saving
        save([save_filename BUI{1,opt}{b} '.mat'],'Data');
        %% [MR] Elapsed time
        elapsed_time = toc(timer_phase); % [MR] Stop timer for this phase
        running_time(['elapsed_time_' BUI{1,opt}{b}]) = elapsed_time;   % [MR]
        fprintf('Ended | Saved %s \n', BUI{1,opt}{b});                  % [MR]
        fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);        % [MR]
    end
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
fprintf('\nRunning Time:');
phases = keys(running_time);
for phase = 1:length(phases)
    phase_name = phases{phase};
    phase_elapsed_time = running_time(phase_name);
    fprintf('| %-*s = %*.4f seconds\n', longest_name_length, phase_name, longest_time_length, phase_elapsed_time);
end