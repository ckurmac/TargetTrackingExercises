classdef PDATracker < BaseTracker
    properties
        P_d
        P_g
        beta_fa
    end

    methods
        function obj = PDATracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig,P_d,P_g,beta_fa)
            obj = obj@BaseTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig);
            obj.P_d = P_d;
            obj.P_g = P_g;
            obj.beta_fa = beta_fa;
        end

        function [isDetAssigned, deletedTrackIDs] = associateAndUpdate(obj, tracks, detections, isDetAssigned, trackerCallTime)
            deletedTrackIDs = [];
            for j = 1:numel(tracks)
                track = tracks{j};
                sampleDet = getDetectionStruct;
                associatedDets = repmat(sampleDet,0,1);
                detIdx = 0;
                for k = 1:numel(detections)
                    if(isDetAssigned(k))
                        continue; % skip the assigned dets.
                    end
                    detObj = detections(k);
                    dist = track.distance(detObj);
                    if(dist<=obj.gateThreshold)
                        detIdx = detIdx+1;
                        associatedDets(detIdx) = detObj;
                        isDetAssigned(k) = true;
                    end
                end
                if(detIdx>0)
                    % Predict to the tracker call time, not the detection's own
                    % timestamp.
                    track.updatePDA(associatedDets, obj.P_d, obj.P_g, obj.beta_fa, trackerCallTime);
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
