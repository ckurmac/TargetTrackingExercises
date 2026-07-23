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

        function update(obj, z)
            
        end

        function markMiss(obj)
            
        end
    end
end
