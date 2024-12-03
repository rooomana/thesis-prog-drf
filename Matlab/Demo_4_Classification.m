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
%opt = 1;  % Change to 1, 2, or 3 to alternate between the 1st, 2nd, and 3rd DNN results respectively.
opt = 3; % [MR] DNN Results number

results_path = [pwd '\Results_' num2str(opt) '\']; % [MR]

running_time = dictionary; % [MR] Timers

timer_total = tic; % [MR] Start timer for total program

%% Main
timer_phase = tic;                  % [MR] Start timer for this phase

y = [];
for i = 1:10
    x = csvread([results_path 'Results_' num2str(opt) num2str(i) '.csv']); % [MR]
    y = [y ; x];
end

elapsed_time = toc(timer_phase);    % [MR] Stop timer for this phase
running_time('elapsed_time_reading') = elapsed_time;        % [MR]
fprintf('Ended | Reading \n');                              % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);    % [MR]

%% Plotting confusion matrix
timer_phase = tic;                  % [MR] Start timer for this phase

if(opt == 1)
    plotconfusion_mod(y(:,1:2)',y(:,3:4)');
elseif(opt == 2)
    plotconfusion_mod(y(:,1:4)',y(:,5:8)');
elseif(opt == 3)
    plotconfusion_mod(y(:,1:10)',y(:,11:20)');
    set(gcf,'position',[100, -100, 800, 800])
end
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));

elapsed_time = toc(timer_phase);    % [MR] Stop timer for this phase
running_time('elapsed_time_plotting') = elapsed_time;       % [MR]
fprintf('Ended | Plotting \n');                             % [MR]
fprintf('Elapsed time: %.4f seconds\n\n', elapsed_time);    % [MR]

%% Saving
Q = input('Do you want to save the results (Y/N)\n','s');
if(Q == 'y' || Q == 'Y')
    timer_phase = tic;                  % [MR] Start timer for this phase

    print(1,['confusion_matrix_' num2str(opt)],'-dpdf','-r512');

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
fprintf('\nRunning Time:');
phases = keys(running_time);
for phase = 1:length(phases)
    phase_name = phases{phase};
    phase_elapsed_time = running_time(phase_name);
    fprintf('| %-*s = %*.4f seconds\n', longest_name_length, phase_name, longest_time_length, phase_elapsed_time);
end