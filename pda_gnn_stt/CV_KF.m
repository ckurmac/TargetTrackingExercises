classdef CV_KF < handle
    properties
        Q
        C
        x
        xP
    end

    methods
        function obj = CV_KF(process_noise,state,state_cov)
            obj.Q = process_noise;
            obj.x = state;
            obj.xP = state_cov;
            obj.C = [1,0,0,0;...
                     0,1,0,0];
        end
        
        function [x_predict,xP_predict] = predict(obj,dt)
            A_k = getCVStateTransitionMtx(dt);
            B_k = getCVNoiseGainMtx(dt);

            x_predict = A_k*obj.x;
            xP_predict = A_k*obj.xP*A_k' + B_k*obj.Q*B_k';
        end

        function [x_update,xP_update] = update(obj,y_k,R_k,t_y)
            [obj.x,obj.xP] = obj.predict(t_y);
            S_k = obj.C*obj.xP*obj.C'+R_k;
            K_k = obj.xP*obj.C'*S_k^-1;
            x_update = obj.x+K_k*(y_k-obj.C*obj.x);
            xP_update = obj.xP - K_k*S_k*K_k';
            obj.x = x_update;
            obj.xP = xP_update;
        end
        
    end
end