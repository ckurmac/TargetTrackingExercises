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

        function [isHit, isDetAssigned] = associateAndUpdate(obj, track, detections, isDetAssigned)
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
                % All detections in a scan share the same MeasurementTime.
                if(isempty(detections))
                    currTime = associatedDets(1).MeasurementTime; % safety fallback, unreachable when detIdx>0.
                else
                    currTime = detections(1).MeasurementTime;
                end
                track.updatePDA(associatedDets, obj.P_d, obj.P_g, obj.beta_fa, currTime);
                isHit = true;
            else
                isHit = false;
            end
        end
    end
end
