classdef GNNTracker < BaseTracker
    properties
        P_d
        P_g
        beta_fa
    end

    methods
        function obj = GNNTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig,P_d,P_g,beta_fa)
            obj = obj@BaseTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig);
            obj.P_d = P_d;
            obj.P_g = P_g;
            obj.beta_fa = beta_fa;
        end

        function [isDetAssigned, deletedTrackIDs] = associateAndUpdate(obj, tracks, detections, isDetAssigned, trackerCallTime)
            deletedTrackIDs = [];
            assignmentMatrix = -10^10*ones(numel(detections),numel(detections)+numel(tracks));
            isTrackUpdated = false(1,numel(tracks));
            for i = 1:numel(detections)
                for j = 1:numel(tracks)
                    track = tracks{j};
                    detObj = detections(i);
                    dist = track.distance(detObj);
                    if(dist<=obj.gateThreshold)
                        [predicted_meas,S_k] = track.getGateInfo(detObj.MeasurementCovariance,trackerCallTime);
                        likelihood = mvnpdf(detObj.Measurement,predicted_meas,S_k);
                        logLikelihood = log(obj.P_d*likelihood/(1-obj.P_d*obj.P_g));
                        assignmentMatrix(i,j) = logLikelihood;
                    end
                end
                assignmentMatrix(i,i+numel(tracks)) = log(obj.beta_fa);
            end
            [det2track,track2det] = Auction(assignmentMatrix);

            for i = 1:length(det2track)
                if(det2track(i)>numel(tracks))
                    continue;%det unassigned
                else
                    track = tracks{det2track(i)};
                    det = detections(i);              % i is the detection (customer) index.
                    track.updateNN(det);
                    isTrackUpdated(det2track(i)) = true;
                    isDetAssigned(i) = true;
                end
            end
            for i = 1:numel(tracks)
                track = tracks{i};
                obj.checkTrackStatus(track.TrackID, isTrackUpdated(i));
                if(track.InitiationState(1) == 0) % deleted track.
                    deletedTrackIDs(end+1) = track.TrackID; %#ok<AGROW>
                end
            end
        end
    end
end
