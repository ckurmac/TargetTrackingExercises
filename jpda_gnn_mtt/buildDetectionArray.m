function dets = buildDetectionArray(detSet, measCov, t)
    sampleDet = getDetectionStruct();
    dets = repmat(sampleDet,0,1);
    for k = 1:size(detSet,1)
        detObj = getDetectionStruct();
        detObj.Measurement = detSet(k,:)';
        detObj.MeasurementCovariance = measCov;
        detObj.MeasurementTime = t;
        dets(k,1) = detObj;
    end
end
