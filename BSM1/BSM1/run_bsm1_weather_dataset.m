% BSM1 batch simulation for dry/rain/storm weather datasets
% Output: MAT + CSV datasets for deep learning (LSTM/TCN/Transformer).

clearvars;
close all;
clc;

% ------------------------ User-configurable parameters ------------------------
warmup_days = 5;         % warm-up horizon (days), not exported
capture_days = 14;       % effective data horizon (days), exported
sample_step_day = 1/96;  % 15 min = 1/96 day
output_dir = fullfile(pwd, 'dataset_exports');
% -----------------------------------------------------------------------------

if ~exist('benchmark.mdl', 'file')
    error('Please run this script inside /BSM1/BSM1 where benchmark.mdl exists.');
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

scenario_names = {'dry', 'rain', 'storm'};
all_tables = cell(numel(scenario_names), 1);

for k = 1:numel(scenario_names)
    scenario = scenario_names{k};
    fprintf('\n=== Running scenario: %s ===\n', upper(scenario));

    clear in ASinput rec feed settler reac1 reac2 reac3 reac4 reac5 tout yout
    bdclose('all');
    benchmarkinit;  % reset model states and load standard influent files

    switch scenario
        case 'dry'
            selected_influent = DRYINFLUENT;
        case 'rain'
            selected_influent = RAININFLUENT;
        case 'storm'
            selected_influent = STORMINFLUENT;
        otherwise
            error('Unsupported scenario: %s', scenario);
    end

    % Force selected scenario independent of manual switch positions.
    DRYINFLUENT = selected_influent;
    RAININFLUENT = selected_influent;
    STORMINFLUENT = selected_influent;
    CONSTINFLUENT = selected_influent;

    % Warm-up run: no export.
    sim('benchmark', [0 warmup_days]);
    stateset;
    load states; %#ok<LOAD>

    % Formal 14-day run for export.
    clear in ASinput rec feed settler reac1 reac2 reac3 reac4 reac5 tout yout
    sim('benchmark', [0 capture_days]);

    if ~exist('tout', 'var') || ~exist('in', 'var') || ~exist('settler', 'var')
        error('Simulation outputs not found in workspace for scenario: %s', scenario);
    end

    [tout_unique, unique_idx] = unique(tout);
    in_unique = in(unique_idx, :);
    settler_unique = settler(unique_idx, :);

    t_uniform = (0:sample_step_day:(capture_days - sample_step_day))';
    in_uniform = interp1(tout_unique, in_unique, t_uniform, 'previous', 'extrap');
    settler_uniform = interp1(tout_unique, settler_unique, t_uniform, 'linear', 'extrap');

    % Requested X features (influent side)
    Qin = in_uniform(:, 15);
    Sso = in_uniform(:, 2);
    Sbs = in_uniform(:, 5);
    Sbh = in_uniform(:, 6);
    Sx = in_uniform(:, 4);
    Xi = in_uniform(:, 3);
    Sno = in_uniform(:, 9);
    Snh = in_uniform(:, 10);
    Snd = in_uniform(:, 11);
    Xnd = in_uniform(:, 12);
    Alk = in_uniform(:, 13);

    % Requested Y labels (effluent side)
    Se_Sso = settler_uniform(:, 18);
    Se_Snh = settler_uniform(:, 26);
    Se_Sno = settler_uniform(:, 25);
    COD_eff = sum(settler_uniform(:, 17:23), 2);

    % Same TN definition used in perf_plant.m
    i_XB = 0.08;
    i_XP = 0.06;
    TN_eff = settler_uniform(:, 25) + settler_uniform(:, 26) + settler_uniform(:, 27) + ...
             settler_uniform(:, 28) + i_XB * (settler_uniform(:, 21) + settler_uniform(:, 22)) + ...
             i_XP * (settler_uniform(:, 19) + settler_uniform(:, 23));

    % ASM1 does not include explicit TP state; use effluent X_P as TP proxy.
    TP_eff = settler_uniform(:, 23);

    scenario_col = repmat({scenario}, numel(t_uniform), 1);
    step_index = (1:numel(t_uniform))';

    dataset_table = table( ...
        scenario_col, step_index, t_uniform, ...
        Qin, Sso, Sbs, Sbh, Sx, Xi, Sno, Snh, Snd, Xnd, Alk, ...
        Se_Sso, Se_Snh, Se_Sno, COD_eff, TN_eff, TP_eff, ...
        'VariableNames', { ...
        'scenario', 'step_index', 'time_day', ...
        'Qin', 'Sso', 'Sbs', 'Sbh', 'Sx', 'Xi', 'Sno', 'Snh', 'Snd', 'Xnd', 'Alk', ...
        'Se_Sso', 'Se_Snh', 'Se_Sno', 'COD_eff', 'TN_eff', 'TP_eff'});

    scenario_data = struct();
    scenario_data.scenario = scenario;
    scenario_data.warmup_days = warmup_days;
    scenario_data.capture_days = capture_days;
    scenario_data.sample_step_day = sample_step_day;
    scenario_data.time_day = t_uniform;
    scenario_data.dataset_table = dataset_table;

    mat_path = fullfile(output_dir, sprintf('bsm1_%s_dataset.mat', scenario));
    csv_path = fullfile(output_dir, sprintf('bsm1_%s_dataset.csv', scenario));
    save(mat_path, 'scenario_data');
    writetable(dataset_table, csv_path);

    all_tables{k} = dataset_table;
    fprintf('Saved: %s\n', mat_path);
    fprintf('Saved: %s\n', csv_path);
end

all_dataset_table = vertcat(all_tables{:});
all_mat_path = fullfile(output_dir, 'bsm1_all_weather_dataset.mat');
all_csv_path = fullfile(output_dir, 'bsm1_all_weather_dataset.csv');
save(all_mat_path, 'all_dataset_table', 'warmup_days', 'capture_days', 'sample_step_day');
writetable(all_dataset_table, all_csv_path);

fprintf('\nAll scenarios complete.\n');
fprintf('Saved: %s\n', all_mat_path);
fprintf('Saved: %s\n', all_csv_path);
