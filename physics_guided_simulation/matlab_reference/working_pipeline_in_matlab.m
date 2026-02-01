% GRF-based Phantom k-Wave Simulation (Simplified)
% This script generates GRF-based tissue models for ultrasound simulation
% Similar approach to different_radii_location_bezier_new_phantoms.m

clear;
close all;
clc;

%% BASIC PARAMETERS (MODIFY THESE)
DATA_CAST = 'gpuArray-single';  % 'single' or 'gpuArray-single'

START_SAMPLE_ID = 0;               % First sample to process
END_SAMPLE_ID = 647;               % Last sample to process 
RUN_SIMULATION = true;             % Set to false to skip simulation and just load results
BUSI_DATA_PATH = 'D:\CMME\1_data_generation\data\BUSI\processed_data_for_simulation';
SAVE_RESULTS_PATH = 'D:\CMME\1_data_generation\data\BUSI\processed_grf_results_v4';

% Create results folder if it doesn't exist
if ~exist(SAVE_RESULTS_PATH, 'dir')
    mkdir(SAVE_RESULTS_PATH);
end

%% POST-PROCESSING PARAMETERS (MODIFY FOR DIFFERENT VISUALIZATIONS)
% Processing parameters for B-mode visualization
FUND_FILTER_BW = 100;              % Fundamental frequency filter bandwidth
HARM_FILTER_BW = 30;               % Harmonic frequency filter bandwidth
COMPRESSION_RATIO = 1;             % Log compression ratio (0-1)
SCALE_FACTOR = 2;                  % Upsampling factor for image rendering

%% Define scaling factor
sc = 1;                            % Grid size scaling factor (reduce for faster computation)

%% Define predefined noise parameter sets
% Define different noise level configurations for various tissue types
noise_configs = struct();

% Low noise configuration
noise_configs.low = struct();
noise_configs.low.background = 0.0001;
noise_configs.low.fatty = 0.0003;
noise_configs.low.glandular = 0.0002;
noise_configs.low.tumor = 0.00005;

% % Medium noise configuration
% noise_configs.medium = struct();
% noise_configs.medium.background = 0.005;
% noise_configs.medium.fatty = 0.015;
% noise_configs.medium.glandular = 0.01;
% noise_configs.medium.tumor = 0.002;

% % High noise configuration
% noise_configs.high = struct();
% noise_configs.high.background = 0.02;
% noise_configs.high.fatty = 0.06;
% noise_configs.high.glandular = 0.04;
% noise_configs.high.tumor = 0.01;

% Get all noise levels to loop through
noise_levels = fieldnames(noise_configs);

%% Define base acoustic properties for each tissue type
tissue_properties = struct();
tissue_properties.background = struct('sos', 1600, 'density', 1000);
tissue_properties.fatty = struct('sos', 1600, 'density', 1000);
tissue_properties.glandular = struct('sos', 1570, 'density', 1040);
tissue_properties.tumor = struct('sos', 1480, 'density', 1060);

%% Define constants
% Base acoustic properties
c0 = 1540;                         % [m/s] - Default speed of sound
rho0 = 1000;                       % [kg/m^3] - Default density

% GRF Parameters
grf_config = struct();
grf_config.grf_sigma = 8.0;        % Sigma for Gaussian kernel
grf_config.grf_kernel_size = 42;   % Kernel size for GRF
grf_config.coherence_level = 'very_high'; % Options: low, medium, high, very_high

%% Domain / k-Wave Grid parameters
pml_x_size = 20/sc;
pml_y_size = 10/sc;
pml_z_size = 1/sc;

Nx = 256/sc - 2 * pml_x_size;      % [grid points]
Ny = 256/sc - 2 * pml_y_size;      % [grid points]
Nz = 30/sc - 2 * pml_z_size;       % [grid points]

x = 40e-3;                         % [m]
dx = x / Nx;  
dy = dx;
dz = dx;

% Number of scan lines
number_scan_lines = 96/sc;
element_width = 2;                 % Width of each transducer element
Nx_tot = Nx;
Ny_tot = Ny + number_scan_lines * element_width;
Nz_tot = Nz;

%% Create k-Wave grid
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
t_end = (Nx * dx) * 2.2 / c0; 
kgrid.makeTime(c0, [], t_end);

%% Medium parameters
medium.alpha_coeff = 0.75; 
medium.alpha_power = 1.5;
medium.BonA = 6;

%% Input Signal
source_strength = 1e6;            % [Pa]
tone_burst_freq = 1.5e6/sc;       % [Hz]
tone_burst_cycles = 4;
input_signal = toneBurst(1 / kgrid.dt, tone_burst_freq, tone_burst_cycles);
input_signal = (source_strength / (c0 * rho0)) * input_signal;

%% Progress tracking file
progress_file = fullfile(SAVE_RESULTS_PATH, 'progress_tracking.mat');
if exist(progress_file, 'file')
    load(progress_file, 'processed_samples');
    disp(['Loaded progress tracking file. Found ' num2str(length(processed_samples)) ' processed samples.']);
else
    processed_samples = struct();
    disp('Created new progress tracking file.');
end

input_args = {...
    'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size, pml_z_size], ...
    'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false};

%% Main loop over sample IDs
for SAMPLE_ID = START_SAMPLE_ID:END_SAMPLE_ID
    % Check if this sample has already been fully processed
    sample_key = ['sample_' num2str(SAMPLE_ID)];
    if isfield(processed_samples, sample_key) && isfield(processed_samples.(sample_key), 'completed') && processed_samples.(sample_key).completed
        fprintf('Skipping sample %d - already completed all noise levels\n', SAMPLE_ID);
        continue;
    end
    
    % Setup the sample folder
    base_folder = fullfile(SAVE_RESULTS_PATH, sprintf('sample_%d', SAMPLE_ID));
    if ~exist(base_folder, 'dir')
        mkdir(base_folder);
    end
    
    % Initialize progress for this sample if needed
    if ~isfield(processed_samples, sample_key)
        processed_samples.(sample_key) = struct();
        processed_samples.(sample_key).completed = false;
        processed_samples.(sample_key).noise_levels = struct();
        for i = 1:length(noise_levels)
            processed_samples.(sample_key).noise_levels.(noise_levels{i}) = false;
        end
    end
    
    % Check if mask file exists
    input_phantom_file_path = fullfile(BUSI_DATA_PATH, sprintf('sample_%d', SAMPLE_ID), "mask.png");
    if ~exist(input_phantom_file_path, 'file')
        fprintf('Skipping sample %d - mask file not found\n', SAMPLE_ID);
        processed_samples.(sample_key).completed = true;
        save(progress_file, 'processed_samples');
        continue;
    end

    % Load and process the BUSI phantom (tumor/cyst mask)
    fprintf('Processing sample %d...\n', SAMPLE_ID);
    userPhantom2D = imread(input_phantom_file_path);
    if size(userPhantom2D, 3) > 1
        userPhantom2D = rgb2gray(userPhantom2D);
    end
    userPhantom2D = double(userPhantom2D);

    % Original dimensions and aspect ratio
    [rowsP, colsP] = size(userPhantom2D);
    original_aspect_ratio = colsP / rowsP;
    target_aspect_ratio = Ny_tot / Nx_tot;
    fprintf('Original phantom dimensions: %d x %d (aspect ratio: %.3f)\n', rowsP, colsP, original_aspect_ratio);
    fprintf('Target grid dimensions: %d x %d (aspect ratio: %.3f)\n', Nx_tot, Ny_tot, target_aspect_ratio);
    
    % Resize phantom while preserving aspect ratio
    userPhantom2D = resize_preserve_aspect_ratio(userPhantom2D, Nx_tot, Ny_tot, 'nearest');
    
    % Save a visualization of the resized phantom
    figure('visible', 'off');
    imagesc(userPhantom2D);
    axis image; colormap(gray);
    title(sprintf('Resized Phantom - Sample %d', SAMPLE_ID));
    xlabel('Width'); ylabel('Height');
    saveas(gcf, fullfile(base_folder, 'resized_phantom.png'));
    close;

    % Create 3D tumor mask by replicating along z
    tumor_mask_3d = repmat(userPhantom2D == 100, [1, 1, Nz_tot]);

    % Create Gaussian Random Field for tissue modeling
    fprintf('Generating GRF with sigma %.1f, kernel size %d, coherence level %s\n', ...
        grf_config.grf_sigma, grf_config.grf_kernel_size, grf_config.coherence_level);

    % Generate GRF
    grf = create_gaussian_random_field(Nx_tot, Ny_tot, Nz_tot, ...
        grf_config.grf_sigma, grf_config.grf_kernel_size, grf_config.coherence_level);

    % Create tissue type masks based on GRF thresholding
    threshold = 0.5;  % Threshold for fatty vs glandular tissue
    fatty_mask = (grf < threshold) & ~tumor_mask_3d;
    glandular_mask = (grf >= threshold) & ~tumor_mask_3d;
    background_mask = zeros(Nx_tot, Ny_tot, Nz_tot);

    % Create semantic map for visualization
    semantic_map = zeros(Nx_tot, Ny_tot, Nz_tot);
    semantic_map(background_mask > 0) = 1;  % Background
    semantic_map(fatty_mask) = 2;           % Fatty tissue
    semantic_map(glandular_mask) = 3;       % Glandular tissue
    semantic_map(tumor_mask_3d) = 4;        % Tumor/cyst

    % Create acoustic property maps
    sound_speed_map = zeros(Nx_tot, Ny_tot, Nz_tot);
    density_map = zeros(Nx_tot, Ny_tot, Nz_tot);

    % Apply base properties using masks
    sound_speed_map(background_mask > 0) = tissue_properties.background.sos;
    sound_speed_map(fatty_mask) = tissue_properties.fatty.sos;
    sound_speed_map(glandular_mask) = tissue_properties.glandular.sos;
    sound_speed_map(tumor_mask_3d) = tissue_properties.tumor.sos;

    density_map(background_mask > 0) = tissue_properties.background.density;
    density_map(fatty_mask) = tissue_properties.fatty.density;
    density_map(glandular_mask) = tissue_properties.glandular.density;
    density_map(tumor_mask_3d) = tissue_properties.tumor.density;
    
    % Loop through the different noise levels
    for noise_idx = 1:length(noise_levels)
        NOISE_LEVEL = noise_levels{noise_idx};
        
        % Skip if this noise level has already been processed for this sample
        if processed_samples.(sample_key).noise_levels.(NOISE_LEVEL)
            fprintf('Skipping noise level %s for sample %d - already processed\n', NOISE_LEVEL, SAMPLE_ID);
            continue;
        end
        
        fprintf('Processing sample %d with noise level %s\n', SAMPLE_ID, NOISE_LEVEL);
        
        % Output paths
        results_path = base_folder;
        simulation_filename = fullfile(results_path, sprintf('simulation_noise_%s.mat', NOISE_LEVEL));
        images_filename = fullfile(results_path, sprintf('images_data_%s.mat', NOISE_LEVEL));
        
        % Get the noise standard deviations for this configuration
        curr_noise_std = noise_configs.(NOISE_LEVEL);

        % Create copies of base maps
        current_sos_map = sound_speed_map;
        current_density_map = density_map;

        % Apply noise to each tissue region separately with predefined values
        % Background noise
        if any(background_mask(:))
            background_std = curr_noise_std.background;
            noise_multiplier = normrnd(1.0, background_std, size(sound_speed_map));
            current_sos_map(background_mask > 0) = current_sos_map(background_mask > 0) .* noise_multiplier(background_mask > 0);
            current_density_map(background_mask > 0) = current_density_map(background_mask > 0) .* noise_multiplier(background_mask > 0);
        end

        % Fatty tissue noise
        if any(fatty_mask(:))
            fatty_std = curr_noise_std.fatty;
            noise_multiplier = normrnd(1.0, fatty_std, size(sound_speed_map));
            current_sos_map(fatty_mask) = current_sos_map(fatty_mask) .* noise_multiplier(fatty_mask);
            current_density_map(fatty_mask) = current_density_map(fatty_mask) .* noise_multiplier(fatty_mask);
        end

        % Glandular tissue noise
        if any(glandular_mask(:))
            glandular_std = curr_noise_std.glandular;
            noise_multiplier = normrnd(1.0, glandular_std, size(sound_speed_map));
            current_sos_map(glandular_mask) = current_sos_map(glandular_mask) .* noise_multiplier(glandular_mask);
            current_density_map(glandular_mask) = current_density_map(glandular_mask) .* noise_multiplier(glandular_mask);
        end

        % Tumor noise
        if any(tumor_mask_3d(:))
            tumor_std = curr_noise_std.tumor;
            noise_multiplier = normrnd(1.0, tumor_std, size(sound_speed_map));
            current_sos_map(tumor_mask_3d) = current_sos_map(tumor_mask_3d) .* noise_multiplier(tumor_mask_3d);
            current_density_map(tumor_mask_3d) = current_density_map(tumor_mask_3d) .* noise_multiplier(tumor_mask_3d);
        end

        % Save base phantom data
        phantom_save_path = fullfile(results_path, sprintf('grf_phantom_base_%s.mat', NOISE_LEVEL));
        save(phantom_save_path, 'semantic_map', 'grf', 'tumor_mask_3d', 'fatty_mask', 'glandular_mask', ...
            'background_mask', 'sound_speed_map', 'density_map', 'tissue_properties');
        disp(['Base phantom data saved to ', phantom_save_path]);

        % Plot the phantom properties
        % Plot phantom cross-section
        figure;
        horz_axis_phantom = (0:number_scan_lines * element_width - 1) * dy * 1e3;
        imagesc(horz_axis_phantom, (0:Nx_tot-1) * dx * 1e3, current_sos_map(:, 1 + Ny/2:end - Ny/2, round(Nz_tot/2)));
        axis image; colormap(jet); colorbar;
        set(gca, 'YLim', [5, 40]);
        title(['Speed of Sound Map - Sample ' num2str(SAMPLE_ID) ' - Noise ' NOISE_LEVEL]);
        xlabel('Horizontal [mm]'); ylabel('Depth [mm]');
        saveas(gcf, fullfile(results_path, sprintf('phantom_sos_map_%s.png', NOISE_LEVEL)));

        % Plot the semantic map
        figure;
        imagesc(horz_axis_phantom, (0:Nx_tot-1) * dx * 1e3, semantic_map(:, 1 + Ny/2:end - Ny/2, round(Nz_tot/2)));
        axis image; colormap(jet); 
        c = colorbar;
        c.Ticks = [1.25, 2.25, 3.25, 4.25]; 
        c.TickLabels = {'Background', 'Fatty', 'Glandular', 'Tumor'};
        set(gca, 'YLim', [5, 40]);
        title(['Tissue Types - Sample ' num2str(SAMPLE_ID)]);
        xlabel('Horizontal [mm]'); ylabel('Depth [mm]');
        saveas(gcf, fullfile(results_path, sprintf('phantom_semantic_map_%s.png', NOISE_LEVEL)));

        %% Run simulation if needed, otherwise load saved data
        if RUN_SIMULATION
            % Define the transducer
            transducer = define_transducer(kgrid, c0, input_signal, sc, Ny, Nz);
            
            % Store element width
            element_width = transducer.element_width;
            
            % Initialize scan lines
            scan_lines = zeros(number_scan_lines, kgrid.Nt);
            medium_position = 1;
            
            % Run simulation for each scan line
            for scan_line_index = 1:number_scan_lines
                disp(['Computing scan line ' num2str(scan_line_index) ' of ' num2str(number_scan_lines)]);
                
                % Slice the portion of the medium
                medium.sound_speed = current_sos_map(:, medium_position:(medium_position + Ny - 1), :);
                medium.density = current_density_map(:, medium_position:(medium_position + Ny - 1), :);
                
                % % Run k-Wave simulation
                % sensor_data = kspaceFirstOrder3D(kgrid, medium, transducer, transducer, ...
                %     'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size, pml_z_size], ...
                %     'DataCast', 'gpuArray-single', 'DataRecast', true, 'PlotSim', false);
                sensor_data = kspaceFirstOrder3D(kgrid, medium, transducer, transducer, input_args{:});
                
                % Store scan line
                scan_lines(scan_line_index,:) = transducer.scan_line(sensor_data);
                medium_position = medium_position + transducer.element_width;
            end
            
            % Save raw simulation results (before processing)
            save(simulation_filename, 'scan_lines', 'current_sos_map', 'current_density_map', ...
                'kgrid', 'medium', 'transducer', 'input_signal', 'element_width', 'grf', 'semantic_map', 'NOISE_LEVEL');
            disp(['Raw simulation results saved to ', simulation_filename]);
            
            % Clean up
            clear transducer;
            reset(gpuDevice);
        else
            % Check if simulation results exist
            if ~exist(simulation_filename, 'file')
                fprintf('Simulation results not found for sample %d noise level %s, skipping\n', SAMPLE_ID, NOISE_LEVEL);
                continue;
            end
            
            % Load the simulation results
            disp(['Loading saved simulation results from ', simulation_filename]);
            loaded_data = load(simulation_filename);
            
            % Extract variables
            scan_lines = loaded_data.scan_lines;
            element_width = loaded_data.element_width;
            kgrid = loaded_data.kgrid;
            medium = loaded_data.medium;
            input_signal = loaded_data.input_signal;
        end

        %% POST-PROCESSING
        disp('Performing post-processing...');

        % Process results using the existing function
        [scan_line_example, t0, r, scan_lines_fund, scan_lines_harm] = process_results(...
            kgrid, input_signal, medium, scan_lines, tone_burst_freq, c0, number_scan_lines);

        % Save the processed images for further analysis
        save(images_filename, 'scan_lines_fund', 'scan_lines_harm', 'r', 'element_width', 'dy', 'SCALE_FACTOR', ...
            'FUND_FILTER_BW', 'HARM_FILTER_BW', 'COMPRESSION_RATIO');
        disp(['Processed images saved to ', images_filename]);

        % Create and save B-mode and harmonic images
        figure('visible', 'on');
        scf_plot = 2;
        horz_axis = (0:size(scan_lines_fund,1)-1) * element_width * dy / scf_plot * 1e3;
        imagesc(horz_axis, r*1e3, scan_lines_fund.');
        axis image; colormap(gray);
        colorbar;
        title(['Sample ' num2str(SAMPLE_ID) ' - B-mode Image - Noise ' NOISE_LEVEL]);
        xlabel('Horizontal [mm]'); ylabel('Depth [mm]');
        saveas(gcf, fullfile(results_path, sprintf('B_mode_image_noise_%s_BW%d.png', NOISE_LEVEL, FUND_FILTER_BW)));
        frame = getframe(gca);
        rendered_bmode_image = frame.cdata;
        save(fullfile(results_path, sprintf('B_mode_image_noise_%s_BW%d.mat', NOISE_LEVEL, FUND_FILTER_BW)), 'rendered_bmode_image');

        % Create and save harmonic images
        figure('visible', 'on');
        imagesc(horz_axis, r*1e3, scan_lines_harm.');
        axis image; colormap(gray);
        colorbar;
        title(['Sample ' num2str(SAMPLE_ID) ' - Harmonic Image - Noise ' NOISE_LEVEL]);
        xlabel('Horizontal [mm]'); ylabel('Depth [mm]');
        set(gca, 'YLim', [5, 40]);
        saveas(gcf, fullfile(results_path, sprintf('harmonic_image_noise_%s_BW%d.png', NOISE_LEVEL, HARM_FILTER_BW)));
        frame = getframe(gca);
        rendered_harmonic_image = frame.cdata;
        save(fullfile(results_path, sprintf('harmonic_image_noise_%s_BW%d.mat', NOISE_LEVEL, HARM_FILTER_BW)), 'rendered_harmonic_image');

        disp(['Processing completed for Sample ' num2str(SAMPLE_ID) ' with noise level ' NOISE_LEVEL]);
        
        % Mark this noise level as completed
        processed_samples.(sample_key).noise_levels.(NOISE_LEVEL) = true;
        save(progress_file, 'processed_samples');

        close all;
    end
    
    % Check if all noise levels are completed for this sample
    all_completed = true;
    for i = 1:length(noise_levels)
        if ~processed_samples.(sample_key).noise_levels.(noise_levels{i})
            all_completed = false;
            break;
        end
    end
    
    if all_completed
        processed_samples.(sample_key).completed = true;
        save(progress_file, 'processed_samples');
        fprintf('All noise levels completed for sample %d\n', SAMPLE_ID);
    end
end

disp('All samples processed!');