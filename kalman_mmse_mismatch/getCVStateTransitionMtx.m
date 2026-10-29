function [A] = getCVStateTransitionMtx(dt)
            A = [1,0,dt,0;... %x
                0,1,0,dt;... %y
                0,0,1,0; ... %vx 
                0,0,0,1];    %vy
end