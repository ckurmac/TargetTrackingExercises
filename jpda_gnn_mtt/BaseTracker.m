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
        gateThreshold
        posSig
        velSig
    end

    methods
        function obj = BaseTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig)
            obj.TracksList = cell(maxTrackNum,1);
            for i = 1:maxTrackNum
                obj.TracksList{i,1} = TrackObj(0, CV_KF(eye(2,2), zeros(4,1), eye(4,4)));
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
            obj.gateThreshold = gateThreshold;
            obj.posSig = posSig;
            obj.velSig = velSig;
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
                    break;
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
            currStatus = obj.TracksList{trackIdx}.InitiationState(1);
            if(currStatus==2)
                if(sum(obj.TracksList{trackIdx}.trackHistoryBuffer) == 0)
                    obj.TracksList{trackIdx}.InitiationState(1) = 0; % delete if missed 3 consecutive.
                end
            else
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

        function [initiators, confirmed] = step(obj, detections, trackerCallTime)
            % Runs one scan of the tracker: jointly associate+update all
            % existing tracks against all detections, remove whatever the
            % status check flagged as deleted, then initiate new tracks from
            % the detections that remain unassigned. trackerCallTime is the
            % scan time used for prediction, so the filter is propagated to a
            % single common time rather than each detection's own timestamp.
            isDetAssigned = false(numel(detections),1);

            % Gather live track handles into a cell array so the subclass can
            % solve association jointly over all tracks (needed for GNN/JPDA).
            tracks = cell(1, obj.TrackNum);
            for j = 1:obj.TrackNum
                trackIdx = obj.getTrackIndex(obj.liveTrackIDs(j));
                tracks{j} = obj.TracksList{trackIdx};
            end

            % The subclass associates, updates, and runs checkTrackStatus for
            % each track. It returns which detections got claimed and the IDs
            % of tracks the status check flagged for deletion.
            [isDetAssigned, deletedTrackIDs] = obj.associateAndUpdate(tracks, detections, isDetAssigned, trackerCallTime);

            % Remove deleted tracks after the joint update so liveTrackIDs is
            % not mutated mid-association.
            for j = 1:numel(deletedTrackIDs)
                obj.removeTrack(deletedTrackIDs(j));
            end

            for k = 1:numel(detections)
                if(~isDetAssigned(k))
                    obj.initiateTrack(detections(k), obj.posSig, obj.velSig);
                end
            end

            [initiators, confirmed] = obj.getLiveTrackLists();
        end

        function [initiators, confirmed] = getLiveTrackLists(obj)
            initiators = {};
            confirmed = {};
            for j = 1:obj.TrackNum
                trackIdx = obj.getTrackIndex(obj.liveTrackIDs(j));
                if(trackIdx == 0)
                    continue;
                end
                currTrack = obj.TracksList{trackIdx};
                if(currTrack.InitiationState(1) == 1)
                    initiators{end+1} = currTrack; %#ok<AGROW>
                elseif(currTrack.InitiationState(1) == 2)
                    confirmed{end+1} = currTrack; %#ok<AGROW>
                end
            end
        end

        function [isDetAssigned, deletedTrackIDs] = associateAndUpdate(obj, tracks, detections, isDetAssigned, trackerCallTime)
            % Subclasses receive the full list of live track handles, all
            % detections, and the scan time trackerCallTime used for
            % prediction. For each track they must: associate, update the
            % filter, call obj.checkTrackStatus(track.TrackID, isHit), mark the
            % claimed detections in isDetAssigned, and append the IDs of any
            % track flagged deleted (InitiationState(1)==0) to deletedTrackIDs.
            error("subclass must implement associateAndUpdate");
        end
    end
end