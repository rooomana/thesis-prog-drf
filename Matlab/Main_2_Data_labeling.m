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
save_filename = load_filename;
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
disp(['Ended.', ' | Loading']); % [MR] Print for debugging
elapsed_time_loading = toc; % [MR] Stop timer for loading phase

%% Labeling
tic; % [MR] Start timer for labeling
zeros(3,sum(LN));
Label(1,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:end)))];
Label(2,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:5))) 2*ones(1,sum(LN(6:9))) 3*ones(1,LN(10))];
temp = [];
for i = 1:length(LN)
    temp = [temp (i-1)*ones(1,LN(i))];
end
Label(3,:) = temp;
disp(['Ended.', ' | Labeling']); % [MR] Print for debugging
elapsed_time_labeling = toc; % [MR] Stop timer for labeling phase

%% Saving
tic; % [MR] Start timer for saving
csvwrite([save_filename 'RF_Data.csv'],[DATA; Label]);
csvwrite([log_filename 'RF_Data.csv'],[DATA; Label]);
disp(['Ended.', ' | Saving']); % [MR] Print for debugging
elapsed_time_saving = toc; % [MR] Stop timer for saving phase

%% Elapsed time
elapsed_time_total = toc; % [MR] Stop timer for total program
disp(['Elapsed time (Loading)  | ', num2str(elapsed_time_loading), ' seconds']);    % [MR] Print elapsed time for loading phase
disp(['Elapsed time (Labeling) | ', num2str(elapsed_time_labeling), ' seconds']);   % [MR] Print elapsed time for labeling phase
disp(['Elapsed time (Saving)   | ', num2str(elapsed_time_saving), ' seconds']);     % [MR] Print elapsed time for saving phase
disp(['Elapsed time (Total)    | ', num2str(elapsed_time_total), ' seconds']);      % [MR] Print elapsed time for total program