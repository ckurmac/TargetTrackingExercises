classdef TrackObj < handle

    properties
        TrackID
        Filter
        InitiationState
        UpdateTime
        trackHistoryBuffer % operate as a shift register for confirmed tracks.
    end

    methods
        function obj = TrackObj(ID,filter)
            obj.TrackID = ID;
            obj.Filter = filter;
            obj.InitiationState = [1,1,0,1]; % [state,hit_count,miss_count,age]
            obj.trackHistoryBuffer = ones(1,3);
        end

        function updateNN(obj, detection)
            dt = detection.MeasurementTime - obj.UpdateTime;
            obj.Filter.update(detection.Measurement,detection.MeasurementCovariance,dt);
            obj.UpdateTime = detection.MeasurementTime;
        end

        function updatePDA(obj,dets,PD,PG,BFA,currTime)

            weights = zeros(1,length(dets)+1);
            weights(1) = (1-PD*PG)*BFA; % u_0;
            states = cell(1,length(dets));
            stateCovariances = cell(1,length(dets));

            dt = currTime - obj.UpdateTime;
            [predicted_state,predicted_stateCov] = obj.Filter.predict(dt);

            for i = 1:length(dets)
                currDet = dets(i);

                S_k = obj.Filter.C*predicted_stateCov*obj.Filter.C' + currDet.MeasurementCovariance;
                predicted_meas = obj.Filter.C*predicted_state;
                weights(i+1) = PD*mvnpdf(currDet.Measurement,predicted_meas,S_k);
                K_k = predicted_stateCov * obj.Filter.C' / S_k;
                states{i} = predicted_state + K_k*(currDet.Measurement-predicted_meas);
                stateCovariances{i} = predicted_stateCov - K_k*S_k*K_k';
            end
            weights = weights./sum(weights);
            corrected_state = zeros(4,1);
            corrected_state = corrected_state + weights(1)*predicted_state; 
            corrected_stateCov = zeros(4,4);
            for i = 1:length(dets)
                corrected_state = corrected_state + weights(i+1)*states{i};
            end

            corrected_stateCov = corrected_stateCov + weights(1)*(predicted_stateCov + ...
                                (predicted_state-corrected_state)*(predicted_state-corrected_state)');
            for i = 1:length(dets)
                corrected_stateCov = corrected_stateCov + weights(i+1)*(stateCovariances{i} + ...
                                (states{i}-corrected_state)*(states{i}-corrected_state)');
            end

            obj.Filter.x = corrected_state;
            obj.Filter.xP = corrected_stateCov;
            obj.UpdateTime = currTime;
        end
        
        function dist = distance(obj,detection)
            dt = detection.MeasurementTime - obj.UpdateTime;
            [predicted_state,predicted_state_cov] = obj.Filter.predict(dt);
            predicted_meas = obj.Filter.C*predicted_state;
            dz = detection.Measurement - predicted_meas;
            S_k = obj.Filter.C * predicted_state_cov * obj.Filter.C' + detection.MeasurementCovariance;
            dist = dz'* S_k^-1*dz;
        end

        function markMiss(obj)
            if(obj.InitiationState(1) == 2)
                for i = 1:length(obj.trackHistoryBuffer)-1
                    obj.trackHistoryBuffer(i+1) = obj.trackHistoryBuffer(i+1);
                end
                obj.trackHistoryBuffer(1) = 0;
            else
                obj.InitiationState(3) = obj.InitiationState(3)+1;
                obj.InitiationState(4) = obj.InitiationState(4)+1;
            end
        end
        
        function markHit(obj)
            if(obj.InitiationState(1) == 2)
                for i = 1:length(obj.trackHistoryBuffer)-1
                    obj.trackHistoryBuffer(i+1) = obj.trackHistoryBuffer(i+1);
                end
                obj.trackHistoryBuffer(1) = 1;
            else
                obj.InitiationState(2) = obj.InitiationState(2)+1;
                obj.InitiationState(4) = obj.InitiationState(4)+1;
            end
        end
    end
end
