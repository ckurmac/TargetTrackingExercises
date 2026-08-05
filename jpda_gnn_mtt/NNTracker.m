classdef NNTracker < BaseTracker
    methods
        function obj = NNTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig)
            obj = obj@BaseTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig);
        end

        function [isDetAssigned, deletedTrackIDs] = associateAndUpdate(obj, tracks, detections, isDetAssigned, trackerCallTime) %#ok<INUSD>
            % trackerCallTime is accepted for a uniform interface; the NN
            % update propagates to the matched detection's own timestamp.
            deletedTrackIDs = [];
            for j = 1:numel(tracks)
                track = tracks{j};
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
                obj.checkTrackStatus(track.TrackID, isHit);
                if(track.InitiationState(1) == 0) % deleted track.
                    deletedTrackIDs(end+1) = track.TrackID; %#ok<AGROW>
                end
            end
        end
    end
end
