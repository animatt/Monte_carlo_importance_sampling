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
% GR2 = imformat('track2.png', [32 17]);

% figure, colormap(gray), surf(GR1), title('GR1'), axis equal % 30 x 32
% figure, colormap(gray), surf(GR2), title('GR2'), axis equal % 32 x 17

% initialize learning parameters for GR1
% ([y'', x'', ][u'', ]rows, columns, y', x')
[m, n] = size(GR1);

r = roots([1, 1, 2 * (1 - m)]);
r1 = ceil(r(r > 0));
r = roots([1, 1, 2 * (2 - n)]);
r2 = ceil(r(r > 0));

Qsa = zeros(3, 3, m, n, r1, r2);
import_ratios = zeros(3, 3, m, n, r1, r2);
% behavior_policy = randomly accelerate forward/backward
target_policy = ones(2, m, n, r1, r2);

converging = true;
counter = 0;
while converging
    reward = 1;
    ep_hist = [];
    
    % initialize agent
    row = m;
    col = datasample(find(GR1(end, :) == 1.5), 1); % start line index
    rowv = 0; % y' velocity
    colv = 0; % x' velocity
    
    race_in_progress = true;
    while race_in_progress
        
        reward = reward - 1;
        
        % update importance ratio
        behavior_policy = (row > 0) * 1 / 3 + (row == 0) * 1 / 2;
        next_ratio = 1 / behavior_policy;
        
        % agent follow behavior policy
        if rowv > 0
            action_taken = randi([-1 1]);
        else
            action_taken = randi([0 1]);
        end

        episode = [action_taken, 0, row, col, rowv, colv, next_ratio]';
        rowv = rowv + action_taken;
        
        if colv > 0
            action_taken = randi([-1 1]);
        else
            action_taken = randi([0 1]);
        end

        episode(2) = action_taken;
        colv = colv + action_taken;
        
        % update history
        ep_hist = [ep_hist, episode]; % grow array efficiently in last Dim
        
        % update position
        row = row - rowv; % y velocity is negative
        col = col + colv;
        
        % check collision with barrier/finish line
        % must allow agent to exit bounds across the finish line however
        if ~inbounds(size(GR1), row, col) || GR1(row, col) == 1
            row = m;
            col = datasample(find(GR1(end, :) == 1.5), 1);
            rowv = 0;
            colv = 0;
        elseif GR1(row, col) == 1.5 && row ~= m
            race_in_progress = false;
        end
    end
    % formatting
    ep_hist(1 : 2, :) = ep_hist(1 : 2, :) + 2;
    ep_hist(5 : 6, :) = ep_hist(5 : 6, :) + 1;
    T = array2table(ep_hist', 'VariableNames', ...
        {'Ra', 'Ca', 'R', 'C', 'Rv', 'Cv', 'importance_ratio'});
    
    SA = sub2ind(size(Qsa), T.Ra, T.Ca, T.R, T.C, T.Rv, T.Cv);
    
    W = cumprod(T.importance_ratio, 'reverse');
    G = (0 : -1 : reward)';
    import_ratios(SA) = import_ratios(SA) + W;
    
    % use importance ratio to adjust expected value
    Qsa(SA) = Qsa(SA) + W ./ import_ratios(SA) .* (G - Qsa(SA));
    
    % improve policy
    sz = size(Qsa);
    SS = sub2ind(sz(3 : 6), T.R, T.C, T.Rv, T.Cv);
    
    QSub = permute(Qsa(:, :, SS), [3 1 2]);
    QSub(QSub == 0) = NaN;
    
    [~, II] = max(QSub(:, :), [], 2); % may need to set some vals NaN
    [Rbest, Cbest] = ind2sub(size(QSub), II);
    target_policy(:, SS) = [Rbest'; Cbest'];
    
    counter = counter + 1;
end