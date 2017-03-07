clear, clc, close all

% Racing on a simulated track. The agent learns to navigate and complete a 
% non-trivial racetrack without crashing into the borders in the shortest
% time possible. In order to do this, the agent must manage its
% velocity vector in 2-D. It can accelerate in either component by +- 1
% pixel per iteration, and neither component can be negative. On every
% iteration the agent recieves a reward of -1. Episodes conclude when the
% agent crosses the finish line. If it intersects with the track's boundary 
% on any iteration it must return to the starting line. This simulation
% utilizes an off-policy Monte Carlo learning algorithm.

GR1 = imformat('track1.png', [30 32]);
GR2 = imformat('track2.png', [32 17]);

figure, colormap(gray), surf(GR1), title('GR1'), axis equal % 30 x 32
figure, colormap(gray), surf(GR2), title('GR2'), axis equal % 32 x 17

% initialize learning parameters for GR1
% (rows, columns, y', x'[, y'', x''])
[m, n] = size(GR1);

target_policy = zeros(m, n, m - 1, n - 2);
% behavior_policy = randomly accelerate forward / backward
Qsa = zeros(m, n, m - 1, n - 2, 3, 3);
% returns = zeros(m, n, m - 1, n - 2, 3, 3);
import_ratios = zeros(m, n, m - 1, n - 2);

converging = true;
while converging
    reward = 1;
    stata = [];
    
    % initialize agent
    rowv = 0; % y' velocity
    colv = 0; % x' velocity
    start = datasample(find(GR1(end, :) == 1.5), 1); % start line index
    [row, col] = ind2sub(size(GR1), position);
    
    race_in_progress = true;
    while race_in_progress
        state_action_pair = [row col; rowv colv; 0 0];
        reward = reward - 1;
        
        % agent follows behavior policy
        if rowv > 0
            action_taken = randi([-1 1]);
        else
            action_taken = randi([0 1]);
        end
        
        state_action_pair(3, 1) = action_taken;
        rowv = rowv + action_taken;
        
        if colv > 0
            action_taken = randi([-1 1]);
        else
            action_taken = randi([0 1]);
        end
        
        state_action_pair(3, 2) = action_taken;
        colv = colv + action_taken;
        
        stata = cat(3, stata, state_action_pair);
        
        % update position
        row = row + rowv;
        col = col + colv;
        
        % check for collision with barrier or finish line
        if GR1(row, col) == 1
            start = datasample(find(GR1(end, :) == 1.5), 1);
            [row, col] = ind2sub(size(GR1), position);
        elseif GR1(row, col) == 1.5 && row ~= 1
            race_in_progress = false;
        end
    end
    
    % assign returns
%     SA = sub2ind(size(returns), stata(1, 1, :), stata(1, 2, :), ...
%         stata(2, 1, :), stata(2, 2, :), stata(3, 1, :), stata(3, 2, :));
%     returns(SA) = returns(SA) + (-1 : -1 : reward)';
    
    
    % increment importance sampling ratios
    SS = sub2ind(size(import_ratios), stata(1, 1, :), stata(1, 2, :), ...
        stata(2, 1, :), stata(2, 2, :));
    import_ratios(SS) = import_ratios(SS) + prod(policy(SS) ./ ...
        (((stata(2, 1, :) > 0) * 1 / 3 + (stata(2, 1, :) == 0) * 1 / 2) .* ...
        ((stata(2, 2, :) > 0) * 1 / 3 + (stata(2, 2, :) == 0) * 1 / 2)));
    
    
end