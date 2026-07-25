classdef BaseTracker < handle
    properties
        TracksList
        TrackNum
        LastTrackID
        maxTrackNum
        trackLUT
        confM
        confN
        delM
        delN
        confirmedTrackNum
        liveTrackIDs
    end

    methods
        function obj = BaseTracker(maxTrackNum,confM,confN,delM,delN)
            obj.TracksList = cell(maxTrackNum,1);
            sampleKF = CV_KF(eye(2,2),zeros(4,1),eye(4,4)); % state dim is 4 in this problem.
            sampleTrack = TrackObj(0,sampleKF);
            for i = 1:maxTrackNum
                obj.TracksList{i,1} = sampleTrack;
            end
            obj.TrackNum = 0;
            obj.LastTrackID = 0;
            obj.maxTrackNum = maxTrackNum;
            obj.confM = confM;
            obj.confN = confN;
            obj.delM = delM;
            obj.delN = delN;
            obj.confirmedTrackNum = 0;
            obj.liveTrackIDs = zeros(1,maxTrackNum,'uint32');
        end

        function  initiateTrack(obj,detection,posSig,velSig)
            for i = 1:obj.maxTrackNum
                if(obj.TracksList{i}.TrackID == 0)
                    state = [detection.Measurement(1);detection.Measurement(2);0;0];
                    obj.TracksList{i}.Filter.x = state;
                    stateCov = [posSig^2,0,0,0;...
                                0,posSig^2,0,0;...
                                0,0,velSig^2,0;...
                                0,0,0,velSig^2];
                    obj.TracksList{i}.Filter.xP = stateCov;
                    obj.LastTrackID = obj.LastTrackID+1;
                    obj.TracksList{i}.TrackID = obj.LastTrackID;
                    obj.TracksList{i}.UpdateTime = detection.MeasurementTime;
                    obj.TrackNum = obj.TrackNum+1;
                    obj.liveTrackIDs(obj.TrackNum) = obj.LastTrackID;
                end
            end
        end

        function removeTrack(obj,trackID)
            sampleKF = CV_KF(eye(2,2),zeros(4,1),eye(4,4)); % state dim is 4 in this problem.
            sampleTrack = TrackObj(0,sampleKF);
            trackIdx = obj.getTrackIndex(trackID);
            obj.TracksList{trackIdx} = sampleTrack;
            

            idx = 0;
            for i = 1:obj.TrackNum
                if(obj.liveTrackIDs(i) == trackID)
                    idx = i;
                    break;
                end
            end

            for i = idx:obj.TrackNum
                if(i<obj.TrackNum)
                    obj.liveTrackIDs(i) = obj.liveTrackIDs(i+1);
                else
                    obj.liveTrackIDs(i) = 0;
                end
            end

            obj.TrackNum = obj.TrackNum-1;
        end

        function checkTrackStatus(obj,trackID,isUpdated)
            trackIdx = obj.getTrackIndex(trackID);
            if(isUpdated)
                obj.TracksList{trackIdx}.markHit;
            else
                obj.TracksList{trackIdx}.markMiss;
            end
            trackAge = obj.TracksList{trackIdx}.InitiationState(4);
            missCount = obj.TracksList{trackIdx}.InitiationState(3);
            hitCount = obj.TracksList{trackIdx}.InitiationState(2);
            
            if(trackAge<=obj.delN)
                if(missCount>0)
                    obj.TracksList{trackIdx}.InitiationState(1) = 0; % deleted. Will do the removal after.
                end
            else
                if(missCount>(obj.confN-obj.confM))
                    obj.TracksList{trackIdx}.InitiationState(1) = 0; % deleted. Will do the removal after.
                elseif(hitCount>obj.delM+obj.confM)
                    obj.TracksList{trackIdx}.InitiationState(1) = 2; % confirmed
                end
            end

        end

        function trackIdx = getTrackIndex(obj,trackID)
            % hash table would be better, got lazy.
            trackIdx = 0;
            for i = 1:obj.maxTrackNum 
                if(obj.TracksList{i}.TrackID == trackID)
                    trackIdx = i;
                    return
                end
            end
        end
    end
end