% Hebbian Learning Simulation with SimpleClass Dataset and Multiple Shape Windows

% Load the SimpleClass dataset
load simpleclass_dataset

% Initialize parameters
num_neurons = size(inputs, 1);
num_iterations = size(inputs, 2);
learning_rate = 0.0025;

% Define shapes
shapes = {'circle', 'triangle', 'rectangle'};

% Create figures for visualization
figure_handles = zeros(1, length(shapes));
for i = 1:length(shapes)
    figure_handles(i) = figure('Name', shapes{i});
end

% Main simulation loop for each shape
for shape_idx = 1:length(shapes)
    % Initialize weight matrix
    weights = randn(num_neurons) * 0.01;  % Initialize with small random values
    
    % Initialize error histories for each error function
    mse_history = zeros(1, num_iterations);
    mae_history = zeros(1, num_iterations);
    rmse_history = zeros(1, num_iterations);
    weight_history = zeros(num_neurons, num_neurons, num_iterations);
    
    % Set current shape
    current_shape = shapes{shape_idx};
    
    % Simulation loop for current shape
    for iter = 1:num_iterations
        % Get input from the SimpleClass dataset
        input = inputs(:, iter);
        
        % Calculate neuron activations
        activations = tanh(weights * input);  % Use tanh activation function
        
        % Get target output from the SimpleClass dataset
        target_output = targets(:, iter);
        
        % Calculate current output
        current_output = activations;
        
        % Calculate errors using different error functions
        mse_error = mean((current_output - target_output).^2);  % Mean Squared Error
        mae_error = mean(abs(current_output - target_output));   % Mean Absolute Error
        rmse_error = sqrt(mse_error);                           % Root Mean Squared Error
        
        % Store errors in history
        mse_history(iter) = mse_error;
        mae_history(iter) = mae_error;
        rmse_history(iter) = rmse_error;
        
        % Update weights using modified Hebbian learning rule
        delta_weights = learning_rate * (activations * input' - 0.01 * weights);  % Added weight decay term
        weights = weights + delta_weights;
        
        % Store weight history
        weight_history(:, :, iter) = weights;
        
        % Display error values in the command window
        fprintf('Shape: %s, Iteration %d:\n', current_shape, iter);
        fprintf('  MSE: %.4f, MAE: %.4f, RMSE: %.4f\n', mse_error, mae_error, rmse_error);
        
        % Visualize the network (every 10 iterations to reduce computational load)
        if mod(iter, 10) == 0
            figure(figure_handles(shape_idx));
            visualize_network(weights, activations, iter, mse_error, current_shape, learning_rate);
        end
    end
    
    % Display final weight matrix for current shape
    disp(['Final weight matrix for ', current_shape, ':']);
    disp(weights);
    
    % Plot error histories for current shape
    figure;
    subplot(3,1,1);
    plot(1:num_iterations, mse_history);
    xlabel('Iteration');
    ylabel('MSE');
    title(['Mean Squared Error History - ', current_shape]);
    
    subplot(3,1,2);
    plot(1:num_iterations, mae_history);
    xlabel('Iteration');
    ylabel('MAE');
    title(['Mean Absolute Error History - ', current_shape]);
    
    subplot(3,1,3);
    plot(1:num_iterations, rmse_history);
    xlabel('Iteration');
    ylabel('RMSE');
    title(['Root Mean Squared Error History - ', current_shape]);
    
    % Analyze weight evolution for current shape
    figure;
    hold on;
    for i = 1:num_neurons
        for j = i+1:num_neurons
            plot(1:num_iterations, squeeze(weight_history(i,j,:)));
        end
    end
    xlabel('Iteration');
    ylabel('Weight Strength');
    title(['Weight Evolution - ', current_shape]);
    legend('W1-2', 'W1-3', 'W2-3');

    % Calculate and display final statistics for current shape
   final_mse = mse_history(end);
    final_mae = mae_history(end);
    final_rmse = rmse_history(end);
    avg_weight = mean(weights(:));
    max_weight = max(weights(:));
    min_weight = min(weights(:));
    
    fprintf('\n%s\n', repmat('=', 1, 50));
    fprintf('Final statistics for %s:\n', current_shape);
    fprintf('%s\n', repmat('-', 1, 50));
    fprintf('  MSE: %.4f\n', final_mse);
    fprintf('  MAE: %.4f\n', final_mae);
    fprintf('  RMSE: %.4f\n', final_rmse);
    fprintf('  Average Weight: %.4f\n', avg_weight);
    fprintf('  Maximum Weight: %.4f\n', max_weight);
    fprintf('  Minimum Weight: %.4f\n', min_weight);
    fprintf('%s\n', repmat('-', 1, 50));
    fprintf('LEARNING RATE: %.4f', learning_rate);  % Highlighted in yellow
    fprintf('%s\n\n', repmat('=', 1, 50));
end

% Function definitions

% Visualization function for network
function visualize_network(weights, activations, iter, error, shape, learning_rate)
    clf;
    hold on;
    
    num_neurons = length(activations);
    
    % Draw neurons based on the shape
    if strcmp(shape, 'circle')
        for i = 1:num_neurons
            theta = linspace(0, 2*pi, 100);
            x = cos(theta) * 0.1 + cos(2*pi*i/num_neurons);
            y = sin(theta) * 0.1 + sin(2*pi*i/num_neurons);
            color = [0.5 + 0.5 * activations(i), 0.5 - 0.5 * abs(activations(i)), 0.5 - 0.5 * activations(i)];
            color = max(0, min(1, color));  % Ensure color values are between 0 and 1
            fill(x, y, color);
        end
    elseif strcmp(shape, 'triangle')
        for i = 1:num_neurons
            x = [cos(2*pi*i/num_neurons), cos(2*pi*i/num_neurons) + 0.1, cos(2*pi*i/num_neurons) - 0.1];
            y = [sin(2*pi*i/num_neurons), sin(2*pi*i/num_neurons) + 0.15, sin(2*pi*i/num_neurons) + 0.15];
            color = [0.5 + 0.5 * activations(i), 0.5 - 0.5 * abs(activations(i)), 0.5 - 0.5 * activations(i)];
            color = max(0, min(1, color));  % Ensure color values are between 0 and 1
            fill(x, y, color);
        end
    else % rectangle
        for i = 1:num_neurons
            x = [cos(2*pi*i/num_neurons) - 0.1, cos(2*pi*i/num_neurons) + 0.1, cos(2*pi*i/num_neurons) + 0.1, cos(2*pi*i/num_neurons) - 0.1];
            y = [sin(2*pi*i/num_neurons) - 0.1, sin(2*pi*i/num_neurons) - 0.1, sin(2*pi*i/num_neurons) + 0.1, sin(2*pi*i/num_neurons) + 0.1];
            color = [0.5 + 0.5 * activations(i), 0.5 - 0.5 * abs(activations(i)), 0.5 - 0.5 * activations(i)];
            color = max(0, min(1, color));  % Ensure color values are between 0 and 1
            fill(x, y, color);
        end
    end
    
    % Draw connections
    for i = 1:num_neurons
        for j = i+1:num_neurons
            x1 = cos(2*pi*i/num_neurons);
            y1 = sin(2*pi*i/num_neurons);
            x2 = cos(2*pi*j/num_neurons);
            y2 = sin(2*pi*j/num_neurons);
            
            % Determine connection strength and color
            strength = abs(weights(i,j));
            color = [0.5 + 0.5 * weights(i,j), 0.5 - 0.5 * abs(weights(i,j)), 0.5 - 0.5 * weights(i,j)];
            color = max(0, min(1, color));  % Ensure color values are between 0 and 1
            
            % Draw connection
            line([x1, x2], [y1, y2], 'Color', color, 'LineWidth', 1 + 3 * strength);
        end
    end
    
    % Set plot properties
    axis equal;
    axis off;
    title(sprintf('Iteration %d, Error: %.4f, Shape: %s', iter, error, shape));
    
    % Add learning rate information
    text(-1.5, -1.5, sprintf('Learning Rate: %.4f', learning_rate), 'FontSize', 10, 'FontWeight', 'bold');
    
    % Pause to create animation effect
    pause(0.01);
end