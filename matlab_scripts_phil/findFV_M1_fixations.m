function findFV_M1_fixations(cfg, IDStr)

dataDir = fullfile(cfg.preProDataDir, sprintf('%s_%s_%s', cfg.project, cfg.task, IDStr));

try
    
    gazeFilePath = fullfile(dataDir,  sprintf('%s_%s_%s_M1_gaze.mat', cfg.project, cfg.task, IDStr));
    
    load(gazeFilePath, 'M1Xpx', 'M1Ypx');
    
catch loadErr
    warning('OTNAL Error (%s) --> Could not load M1 X/Y for session ID ''%s''!', mfilename, IDStr);
end

try
    
    runsFilePath = fullfile(dataDir,  sprintf('%s_%s_%s_runs.mat', cfg.project, cfg.task, IDStr));
    
    load(runsFilePath, 'runs');
    
catch loadErr
    warning('OTNAL Error (%s) --> Could not load FV runs for session ID ''%s''!', mfilename, IDStr);
end


fixStruct = struct();

for r = 1:size(runs,2)
   
    runStartMs = round(runs(r).startS*1e3);
    runStopMs = round(runs(r).stopS*1e3);
    
    runX = M1Xpx(runStartMs:runStopMs);
    runY = M1Ypx(runStartMs:runStopMs);
    
    [startIdx,stopIdx] = XY2Fix(runX,runY);
    
    fixStruct(r).relativeStartMs = startIdx;
    fixStruct(r).absoluteStartMs = startIdx+runStartMs;
    
    fixStruct(r).relativeStopMs = stopIdx;
    fixStruct(r).absoluteStopMs = stopIdx+runStartMs;
    
    fixStruct(r).runX = runX;
    fixStruct(r).runY = runY;
    
    fixStruct(r).runStartMs = runStartMs;
    fixStruct(r).runStopMs = runStopMs;
  
end

fixTable = struct2table(fixStruct);

fixationFilePath = fullfile(dataDir,  sprintf('%s_%s_%s_M1_fixations.mat', cfg.project, cfg.task, IDStr));

save(fixationFilePath, 'fixTable');

end






