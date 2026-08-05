classdef JPDATracker < BaseTracker
    properties
        P_d
        P_g
        beta_fa
    end

    methods
        function obj = JPDATracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig,P_d,P_g,beta_fa)
            obj = obj@BaseTracker(maxTrackNum,confM,confN,delM,delN,gateThreshold,posSig,velSig);
            obj.P_d = P_d;
            obj.P_g = P_g;
            obj.beta_fa = beta_fa;
        end

        function [isDetAssigned, deletedTrackIDs] = associateAndUpdate(obj, tracks, detections, isDetAssigned, trackerCallTime)
            deletedTrackIDs = [];
            trackGateList = zeros(numel(tracks),numel(detections));
            Ntracks = numel(tracks);
            m = numel(detections);
            if m > 0
                MeasNoise = detections(1).MeasurementCovariance;
            else
                MeasNoise = zeros(2,2); % unused when there are no detections.
            end
            for j = 1:Ntracks
                track = tracks{j};
                for k = 1:m
                    detObj = detections(k);
                    dist = track.distance(detObj);
                    if(dist<=obj.gateThreshold)
                        trackGateList(j,k) = 1;
                    end
                end
            end
            
            TracksPossibleAssociations=cell(1,Ntracks);
            TracksNofPossibleAssociations=zeros(1,Ntracks);
            for i=1:Ntracks
                MeasurementIndices=find(trackGateList(i,:)>0);
                TracksPossibleAssociations{i}=[0 MeasurementIndices];
                TracksNofPossibleAssociations(i)=length(MeasurementIndices)+1;
            end
            Nhypotheses=prod(TracksNofPossibleAssociations);
            JPDAhypothesisMatrix=zeros(Nhypotheses,Ntracks);
            HypothesisProbabilities=ones(Nhypotheses,1);

            IndexMatrix=zeros(Nhypotheses,Ntracks);
            Numbers=0:(Nhypotheses-1);
            ModeNumber=Nhypotheses;
            for i=1:Ntracks
                ModeNumber=ModeNumber/TracksNofPossibleAssociations(i);
                IndexMatrix(:,i)=floor(Numbers/ModeNumber)+1;
                Numbers=rem(Numbers,ModeNumber);
            end

            for i=1:Ntracks
                TrackPossibleAssociations=TracksPossibleAssociations{i};
                TrackNofPossibleAssociations=TracksNofPossibleAssociations(i);  
                ProbabilityFactors=zeros(TrackNofPossibleAssociations,1);
                track = tracks{i};
                [yhat,S] = track.getGateInfo(MeasNoise,trackerCallTime);
                sqrtS=cholcov(S);
                for j=1:TrackNofPossibleAssociations 
                    MeasurementIndex=TrackPossibleAssociations(j);
                    if MeasurementIndex>0 
                        y=detections(MeasurementIndex).Measurement;
                        ytilda=y-yhat;
                        ProbabilityFactors(j)=obj.P_d*exp(-0.5*sum((ytilda'/sqrtS).^2,2))/sqrt(det(2*pi*S));
                    else 
                        ProbabilityFactors(j)=1-obj.P_d*obj.P_g;
                    end        
                end
                JPDAhypothesisMatrix(:,i)=TrackPossibleAssociations(IndexMatrix(:,i));
                HypothesisProbabilities=HypothesisProbabilities.*ProbabilityFactors(IndexMatrix(:,i));
            end
            HypothesisProbabilities=HypothesisProbabilities.*(obj.beta_fa.^(-sum(JPDAhypothesisMatrix>0,2)));
            
            for i=1:m
                HypothesisProbabilities(sum(JPDAhypothesisMatrix==i,2)>1)=0;
            end
            HypothesisProbabilities=HypothesisProbabilities/sum(HypothesisProbabilities); 

            JPDAprobs=zeros(Ntracks,m+1);

            for i=1:Ntracks
                TrackPossibleAssociations=TracksPossibleAssociations{i};
                TrackNofPossibleAssociations=TracksNofPossibleAssociations(i); 
                for j=1:TrackNofPossibleAssociations
                    MeasIndex=TrackPossibleAssociations(j);
                    if MeasIndex>0 
                        JPDAprobs(i,MeasIndex)=sum(HypothesisProbabilities(IndexMatrix(:,i)==j));
                    else 
                        JPDAprobs(i,m+1)=sum(HypothesisProbabilities(IndexMatrix(:,i)==j));
                    end
                end
            end
            for i=1:Ntracks
                track = tracks{i};
                track.updateJPDA(detections,JPDAprobs(i,:),trackerCallTime);
                isHit = sum(trackGateList(i,:)); % If any det in gate.
                isDetAssigned(trackGateList(i,:)>0) = true;
                obj.checkTrackStatus(track.TrackID, isHit);
                if(track.InitiationState(1) == 0) % deleted track.
                    deletedTrackIDs(end+1) = track.TrackID; %#ok<AGROW>
                end
            end
            
        end
    end
end
