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
filepath = 'D:\ISCTE\Thesis\DroneRF';               % [MR] Path for easier management
load_filename = [filepath '\Data\'];                % Path of raw RF data % [MR]
save_filename = load_filename;

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

running_time = dictionary; % [MR] Timers

tic; % [MR] Start timer for total program

%% Loading and concatenating RF data
tic; % [MR] Start timer for loading phase
T = length(BUI);
DATA = [];
LN   = [];
for t = 1:T
    fprintf('%d | in for t \n', t); % [MR] Print for debugging
    for b = 1:length(BUI{1,t})
        fprintf('%s | in for BUI \n', BUI{1,t}{b}); % [MR] Print for debugging
        load([load_filename BUI{1,t}{b} '.mat']);
        Data = Data./max(max(Data));
        DATA = [DATA, Data];
        LN   = [LN size(Data,2)];
        clear Data;
    end
    disp(100*t/T)
end

elapsed_time_loading = toc; % [MR] Stop timer for loading phase
running_time('elapsed_time_loading') = elapsed_time_loading;    % [MR]
fprintf('Ended | Loading \n');                                  % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time_loading);  % [MR]

%% Labeling
tic; % [MR] Start timer for labeling phase
zeros(3,sum(LN));
Label(1,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:end)))];
Label(2,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:5))) 2*ones(1,sum(LN(6:9))) 3*ones(1,LN(10))];
temp = [];
for i = 1:length(LN)
    temp = [temp (i-1)*ones(1,LN(i))];
end
Label(3,:) = temp;

elapsed_time_labeling = toc; % [MR] Stop timer for labeling phase
running_time('elapsed_time_labeling') = elapsed_time_labeling;  % [MR]
fprintf('Ended | Labeling \n');                                 % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time_labeling); % [MR]

%% Saving
tic; % [MR] Start timer for saving phase
csvwrite([save_filename 'RF_Data.csv'],[DATA; Label]);

elapsed_time_saving = toc; % [MR] Stop timer for saving phase
running_time('elapsed_time_saving') = elapsed_time_saving;      % [MR]
fprintf('Ended | Saving \n');                                   % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time_saving);   % [MR]

%% [MR] Elapsed time
elapsed_time_total = toc; % [MR] Stop timer for total program
running_time('elapsed_time_total') = elapsed_time_total;        % [MR]
fprintf('Ended | Total \n');                                    % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time_total);    % [MR]

%% [MR] Print running time
longest_key_length = max(cellfun(@length, keys(running_time)));
fprintf('Running Time:\n');
phases = keys(running_time); % [MR] Get keys in stored order
for phase = 1:length(phases)
    phase_name = phases{phase};
    phase_elapsed_time = running_time(phase_name);
    fprintf('| %-*s = %.4f seconds\n', longest_key_length, phase_name, phase_elapsed_time);
end