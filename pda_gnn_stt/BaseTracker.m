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
    end

    methods
        function obj = BaseTracker(maxTrackNum,confM,confN,delM,delN)
            obj.TracksList = cell(maxTrackNum,1);
            sampleKF = CV_KF(eye(2,2),zeros(4,1),eye(4,4)); % state dim is 4 in this problem.
            sampleTrack = TrackObj(0,sampleKF);
            for i = 1:maxTrackNum
                obj.TracksList(i,1) = sampleTrack;
            end
            obj.TrackNum = 0;
            obj.LastTrackID = 0;
            obj.maxTrackNum = maxTrackNum;
            obj.confM = confM;
            obj.confN = confN;
            obj.delM = delM;
            obj.delN = delN;
            obj.confirmedTrackNum = 0;
        end

        function  initiateTrack(obj,detection,posSig,velSig)
            for i = 1:obj.maxTrackNum % hash table would be better, got lazy.
                if(obj.TracksList(i).TrackID == 0)
                    state = [detection(1);detection(2),0,0];
                    obj.TracksList(i).Filter.setState(state);
                    stateCov = [posSig^2,0,0,0;...
                                0,posSig^2,0,0;...
                                0,0,velSig^2,0;...
                                0,0,0,velSig^2];
                    obj.TracksList(i).Filter.setStateCovariance(stateCov);
                    obj.LastTrackID = obj.LastTrackID+1;
                    obj.TrackNum = obj.TrackNum+1;
                end
            end
        end

        function removeTrack(obj,trackID)
            sampleKF = CV_KF(eye(2,2),zeros(4,1),eye(4,4)); % state dim is 4 in this problem.
            sampleTrack = TrackObj(0,sampleKF);
            for i = 1:obj.maxTrackNum % hash table would be better, got lazy.
                if(obj.TracksList(i).TrackID == trackID)
                    obj.TracksList{i} = sampleTrack;
                    obj.TrackNum = obj.TrackNum-1;
                end
            end
        end

        function checkTrackStatus(trackID,isUpdated)
            
        end
    end
end