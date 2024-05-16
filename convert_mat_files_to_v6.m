
cd('/gpfs/milgram/pi/chang/pg496/data_dir/otnal');

% Get a list of all subfolders in the current directory
subfolders = dir(fullfile(pwd, '*'));

% Loop through each subfolder
for i = 1:length(subfolders)
    foldername = subfolders(i).name;
    disp(i);
    % Check if the subfolder is a directory and not '.' or '..'
    if subfolders(i).isdir...
            && ~strcmp(foldername, '.')...
            && ~strcmp(foldername, '..')...
            && ~strcmp(foldername, 'NoNeurophysiologyDataDoseResponseOnly')
        % Change to the subfolder
        if cd(fullfile(pwd, foldername)) == 0
            disp(['Failed to enter directory: ' foldername]);
            continue;
        end
        
        % List all .mat files
        files = dir('*.mat');

        if isempty(files)
            cd('..'); % Go back to the parent directory if no .mat files found
            continue;
        else
            % Loop through each .mat file
            for j = 1:length(files)
                filename = files(j).name;
                % Load the file
                data = load(filename);
                
                % Access the struct within data
                fields = fieldnames(data);
                
                % Create new filename with _regularFormat label
                [~, baseFilename, ~] = fileparts(filename);
                newFilename = [baseFilename '_regForm.mat'];
                
                % Create new struct with fieldname and save
                outStruct = struct();
                for k = 1:length(fields)
                    fieldname = fields{k};
                    outStruct.(fieldname) = data.(fieldname);
                end
                
                % Save the struct back without compression
                save(newFilename, '-struct', 'outStruct');
            end
        end
        % Go back to the parent directory
        cd('..');
    end
end
