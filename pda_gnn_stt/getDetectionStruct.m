function [sampleDet] = getDetectionStruct()
    sampleDet = struct("Measurement",zeros(2,1),"MeasurementCovariance",zeros(2,2),"MeasurementTime",0);
end