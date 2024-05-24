function classify_M1_fixations(cfg,IDStr)
% %CLASSIFY_M1_FIXATIONS Summary of this function goes here
% %   Detailed explanation goes here
% outputArg1 = inputArg1;
% outputArg2 = inputArg2;
% end


scaleFactor = 1.3;

dataDir = fullfile(cfg.preProDataDir, sprintf('%s_%s_%s', cfg.project, cfg.task, IDStr));

try
    fixationFilePath = fullfile(dataDir,  sprintf('%s_%s_%s_M1_fixations.mat', cfg.project, cfg.task, IDStr));
    load(fixationFilePath, 'fixTable')
    
catch loadErr
    
    warning('OTNAL Error (%s) --> Could not load M1 fixations for session ID ''%s''!', mfilename, IDStr);
end


try
    farPlaneCalFilePath = fullfile(dataDir,  sprintf('%s_%s_%s_M1_farPlaneCal.mat', cfg.project, cfg.task, IDStr));
    load(farPlaneCalFilePath, 'farPlaneCal')
    
catch loadErr
    
    warning('OTNAL Error (%s) --> Could not load M1 far plane calibration for session ID ''%s''!', mfilename, IDStr);
end




[ROIs] = farPlaneCal2ROIs(farPlaneCal, scaleFactor);

fixStruct = table2struct(fixTable);

allFix = struct;
fc = 0;

for r = 1:size(fixStruct,1)
    
    runX = fixStruct(r).runX;
    runY = fixStruct(r).runY;
    
    fixStartMs = fixStruct(r).relativeStartMs;
    fixStopMs = fixStruct(r).relativeStopMs;
    
    fixAbsStartMs = fixStruct(r).absoluteStartMs;
    fixAbsStopMs = fixStruct(r).absoluteStopMs;
    
    
    numRunFix = size(fixStartMs,2);
    
    for f = 1:numRunFix
        
        fixClass = 'NONE';
        
        thisFixStartMs = fixStartMs(f);
        thisFixStopMs = fixStopMs(f);
        
        fixIdxs = [thisFixStartMs:thisFixStopMs];
        
        fixX = runX(fixIdxs);
        fixY = runY(fixIdxs);
        
      
        
        meanX = mean(fixX);
        meanY = mean(fixY);
        
        in_face = inpolygon(fixX,fixY,ROIs.faceROIe.Vertices(:,1),ROIs.faceROIe.Vertices(:,2));
        in_eyes = inpolygon(fixX,fixY,ROIs.eyesROIe.Vertices(:,1),ROIs.eyesROIe.Vertices(:,2));
       
        try
        in_lObj =  inpolygon(fixX,fixY,ROIs.leftObjROIe.Vertices(:,1),ROIs.leftObjROIe.Vertices(:,2));
        in_rObj =  inpolygon(fixX,fixY,ROIs.rightObjROIe.Vertices(:,1),ROIs.rightObjROIe.Vertices(:,2));
        catch err
           in_lObj = []; 
            in_rObj = [];
        end
        
        if any(in_eyes)
            
          
            fixClass = 'EYES';
            
        elseif any(in_face)
            fixClass = 'FACE';
            
        elseif any(in_lObj)
            fixClass = 'LOBJ';
        elseif any(in_rObj)
            fixClass = 'ROBJ';
        else
             fixClass = 'OUT';
        end
        
        
        
        fc = fc+1;
        
         uniqueFixID = sprintf('%s_%s_%s_M1_Run%02d_Fix%06d',  cfg.project, cfg.task, IDStr, r, fc);
        
        allFix(fc).Run = r;
        allFix(fc).AbsoluteIdx = fc;
        allFix(fc).RunIdx = f;
        allFix(fc).fixClass = fixClass;
        allFix(fc).ID = uniqueFixID;
        allFix(fc).lengthMs = fixAbsStopMs(f)-fixAbsStartMs(f);
        allFix(fc).meanX = meanX;
        allFix(fc).meanY = meanY;
        allFix(fc).fixX = fixX;
        allFix(fc).fixY = fixY;
        allFix(fc).relativeStartMs = thisFixStartMs;
        allFix(fc).relativeStopMs = thisFixStopMs;
        allFix(fc).absoluteStartMs = fixAbsStartMs(f);
        allFix(fc).absoluteStopMs = fixAbsStopMs(f);
        
       
        
        
        
        
        
    end


    
    
    
end


fixationsClassified = struct2table(allFix);

classifiedFixationFilePath = fullfile(dataDir,  sprintf('%s_%s_%s_M1_fixationsClassified.mat', cfg.project, cfg.task, IDStr));

save(classifiedFixationFilePath, 'fixationsClassified');

