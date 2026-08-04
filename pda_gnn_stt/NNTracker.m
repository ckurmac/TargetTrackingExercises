classdef NNTracker < BaseTracker
    methods
        function obj = NNTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig)
            obj = obj@BaseTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig);
        end

        function [isHit, isDetAssigned] = associateAndUpdate(obj, track, detections, isDetAssigned)
            minDist = inf;
            minDistDetIdx = 0;
            minDistDet = getDetectionStruct;
            for k = 1:numel(detections)
                if(isDetAssigned(k))
                    continue; % skip the assigned dets.
                end
                detObj = detections(k);
                dist = track.distance(detObj);
                if(dist<=obj.gateThreshold)
                   if(dist < minDist)
                    minDist = dist;
                    minDistDetIdx = k;
                    minDistDet = detObj;
                   end
                end
            end
            if(minDist<=obj.gateThreshold)
                isDetAssigned(minDistDetIdx) = true;
                track.updateNN(minDistDet);
                isHit = true;
            else
                isHit = false;
            end
        end
    end
end
