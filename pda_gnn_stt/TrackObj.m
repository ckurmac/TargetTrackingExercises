classdef TrackObj < handle

    properties
        TrackID
        Filter
        InitiationState
        UpdateTime
    end

    methods
        function obj = TrackObj(ID,filter)
            obj.TrackID = ID;
            obj.Filter = filter;
            obj.InitiationState = [1,1,0,1]; % [state,hit_count,miss_count,age]
        end

        function updateNN(obj, detection)
            dt = detection.MeasurementTime - obj.UpdateTime;
            obj.Filter.update(detection.Measurement,detection.MeasurementCovariance,dt);
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
            obj.InitiationState(3) = obj.InitiationState(3)+1;
            obj.InitiationState(4) = obj.InitiationState(4)+1;
        end
        
        function markHit(obj)
            obj.InitiationState(2) = obj.InitiationState(2)+1;
            obj.InitiationState(4) = obj.InitiationState(4)+1;
        end
    end
end
