classdef IMMFilter < handle

    properties
        models
        modelNum 
        transitionMtx
        modelProbabilities;
        x
        xP
    end

    methods
        function obj = IMMFilter(init_x,init_xP,models,transitionMtx)
            %models as a cell array of kalman filters.
            obj.models = models;
            obj.modelNum = length(models);
            obj.transitionMtx = transitionMtx;
            obj.modelProbabilities = ones(1,length(models))/length(models); % initiate with equal prob.
            obj.x = init_x;
            obj.xP = init_xP;
        end

        function [x_mixed,xP_mixed] = calcMixedStateEst(obj)
            x_mixed = cell(1,obj.modelNum);
            xP_mixed = cell(1,obj.modelNum);
            u_all = zeros(obj.modelNum,obj.modelNum);
            for i = 1:obj.modelNum
                x_mixed{i} = 0.*obj.x;
                xP_mixed{i} = 0.*obj.xP;
            end

            for i = 1:obj.modelNum
                for j = 1:obj.modelNum
                    u_all(j,i) = (obj.transitionMtx(j,i)*obj.modelProbabilities(j));
                end
                u_all(:,i) = u_all(:,i)./sum(u_all(:,i));
            end
            
            for i = 1:obj.modelNum
                for j = 1:obj.modelNum
                    x_j = obj.models{j}.x;
                    x_mixed{i} = x_mixed{i} + u_all(j,i)*x_j;
                end
            end

            for i = 1:obj.modelNum
                for j = 1:obj.modelNum
                    x_j = obj.models{j}.x;
                    xP_j = obj.models{j}.xP;
                    xP_mixed{i} = xP_mixed{i} + u_all(j,i)*(xP_j+(x_j-x_mixed{i})*(x_j-x_mixed{i})');
                end
            end
        end

        function [x_merged,xP_merged,model_prob,y_k_predict,S_k] = update(obj,y_k,R_k,dt)

            [x_mixed,xP_mixed] = obj.calcMixedStateEst;
            for i = 1:obj.modelNum
                obj.models{i}.setState(x_mixed{i});
                obj.models{i}.setStateCovariance(xP_mixed{i});
            end
            model_prob = zeros(1,obj.modelNum);


            for i = 1:obj.modelNum
                [x_i_predict,xP_i_predict] = obj.models{i}.predict(dt);
                Sk_i = obj.models{i}.C*xP_i_predict*obj.models{i}.C' + R_k;
                y_k_pred_i = obj.models{i}.C*x_i_predict;
                likelihood = calcNormalLikelihood(y_k,y_k_pred_i,Sk_i);
                u = 0;
                for j = 1:obj.modelNum
                    u = u + obj.transitionMtx(j,i)*obj.modelProbabilities(j);
                end
                model_prob(i) = likelihood*u;
            end
            
            % for gate calculations.
            y_k_predict = 0;
            for i = 1:obj.modelNum
                x_pred = obj.models{i}.x;
                y_k_pred_i = obj.models{i}.C*x_pred;
                u = 0;
                for j = 1:obj.modelNum
                    u = u + obj.transitionMtx(j,i)*obj.modelProbabilities(j);
                end
                y_k_predict = y_k_predict + u*y_k_pred_i;
            end
            
            S_k = 0.*R_k;
            for i = 1:obj.modelNum
                xP_pred = obj.models{i}.xP;
                x_pred = obj.models{i}.x;
                y_k_pred_i = obj.models{i}.C*x_pred;
                Sk_i = obj.models{i}.C*xP_pred*obj.models{i}.C' + R_k;
                u = 0;
                for j = 1:obj.modelNum
                    u = u + obj.transitionMtx(j,i)*obj.modelProbabilities(j);
                end
                S_k = S_k + u*(Sk_i+(y_k_pred_i-y_k_predict)*(y_k_pred_i-y_k_predict)');
            end

            
            model_prob = model_prob./sum(model_prob);
            obj.modelProbabilities = model_prob;
            x_merged = 0.*obj.x;
            xP_merged = 0.*obj.xP;
            for i = 1:obj.modelNum
                [x_k_i_updated,~] = obj.models{i}.update(y_k,R_k);
                x_merged = x_merged + model_prob(i)*x_k_i_updated;
            end
            obj.x = x_merged;
            for i = 1:obj.modelNum
                x_k_i = obj.models{i}.x;
                xP_k_i = obj.models{i}.xP;
                xP_merged = xP_merged + model_prob(i)*(xP_k_i+(x_k_i-x_merged)*(x_k_i-x_merged)');
            end
            obj.xP = xP_merged;
        end
    end
end