%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%
V = 10;          % Vocabulary size

m = 50;          % Number of distinct contexts
n = 5000;        % Total number of samples
r=n/m;           % (fixed) number that each context repeats in data

Sigma_len = 4;   % Size of the support set for each context


d = 60;   % Dimensionality of the embeddings



% %%%%%%%%%%%%%%%%%%%% low-dim %%%%%%%%%%%%%%%%%%%%
% V = 5;          % Vocabulary size
%
% m = 3;          % Number of distinct contexts
% n = 600;        % Total number of samples
% r=n/m;           % (fixed) number that each context repeats in data
%
% Sigma_len = 2;   % Size of the support set for each context
%
%
% d = 2;   % Dimensionality of the embeddings
%
%
%
learning_rate = 0.5; % Step size for gradient descent
iterations = 1e3; % Number of iterations

%%%%%%%%%%%%%%%% PMF of next-token %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize vectors to store generated data
supportSets = zeros(m, Sigma_len);         % support set of next-token distribution
probabilityVectors = zeros(m, Sigma_len);  % pmf of next-token distribution

% Generate data for each distinct context
for j = 1:m
    % Generate support set Sigma_j
    supportSets(j, :) = randperm(V, Sigma_len);

    % Generate probability distribution vector p in a random way

    % split (0,1) in Sigma_len pieces
    a = rand(Sigma_len-1,1);
    sortedAs = [0;sort(a)]; % each piece determines mass for respective token

    p=zeros(Sigma_len,1);
    for k=1:Sigma_len-1
        p(k) = round(r * (sortedAs(k+1)-sortedAs(k))) / r;
        if p(k)==0
            % avoid zero probabilities so that support set is of same length for each sequence
            p(k)=1/r;
            if k>1
                [~,I]=max(p(1:k-1));
                p(I)=p(I)-1/r;
            end
        end
    end
    p(Sigma_len)= 1 - sum(p(1:end-1));
    if p(Sigma_len)==0
            p(Sigma_len)=1/r;
            [~,I]=max(p(1:Sigma_len-1));
            p(I)=p(I)-1/r;
    end

    % Store the generated probability vector
    probabilityVectors(j, :) = p;
end

% sum(sum(abs(probabilityVectors)<1e-4))

% Display the generated next-token distributions
disp('Support Sets:');
disp(supportSets);

disp('Probability Vectors:');
disp(probabilityVectors);

% display 8 randomly chosen next-token distributions
figure
rand_set_disp=randi(m,8,1);
for ind=1:length(rand_set_disp)
    subplot(2,4,ind)
    ind_set_disp=rand_set_disp(ind);
    bar(supportSets(ind_set_disp,:),probabilityVectors(ind_set_disp,:))
    title(['p_',int2str(ind)])
end

%%%%%%%%%%
% Generate embedding matrix H_m
H_m = randn(d, m);
% Repeat columns of H_m n/m times to construct H_n
H_n = repelem(H_m, 1, r);

%%
% Initialize label vector y
y = zeros(n, 1);

% Generate label set for each j
for j = 1:m
    % Extract support set and probability vector for context j
    supportSet_j = supportSets(j, :);
    probabilityVector_j = probabilityVectors(j, :);

    % Generate label set y_j
    y_j = zeros(r, 1);

    count = 1;
    for k = 1:Sigma_len
        % Generate repeated indices based on the probability
        indices = repmat(supportSet_j(k), 1, round(probabilityVector_j(k) * r));

        % Ensure the number of elements matches the target count
        len = length(indices);
        indices = indices(1:min(len, r));

        % Assign the indices to the label set
        y_j(count : count + length(indices) - 1) = indices;
        count = count + length(indices);
    end

    % Assign label set to y
    y((j-1)*r + 1 : j*r) = y_j(1:r);
end



disp('Label Vector y:');
disp(y');

% Convert y to one-hot encoding
y_one_hot = ind2vec(y', V);

%%%%%%%%%%%%%%%%%%%%%  THEORY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Construct the subspace of non-separability
Iden = eye(V);
Smat = [];  % measurement vectors (e_z-e_z')^Th_j
avec = [];  % vector of log-probabilities
for j = 1:m
    indices = (2:Sigma_len);
    tempmat = -Iden(:, supportSets(j,indices)); % One-hot encoding of the complement set
    tempmat(supportSets(j,1),:) = ones(1,Sigma_len-1);
    avec = [avec, log(probabilityVectors(j, 1)./probabilityVectors(j, indices))];
    for i=1:length(indices)
        Smat = [Smat , vec(tempmat(:,i)*H_m(:,j)')];
    end
end
[US,~,~] = svd(Smat,0);

%solve linear sys of eqns to find unique Wp on subspace that satisfies
%log-probability equations
Wstar = reshape( [(eye(V*d)-US*US') ; Smat']\[zeros(V*d,1);avec'] , V , d);


% %% Wfin 2  ---<<< this is another way to find Wstar. simply minimize fro
% norm subject to log-prob equations.
cvx_begin
variable Wfin(V,d)
minimize(norm(Wfin,'fro'))
    for j = 1:m
            indices = (2:Sigma_len); % Logical index to exclude the i-th element
            (repmat(Wfin(supportSets(j, 1),:),Sigma_len-1,1)-Wfin(supportSets(j, indices),:))*H_m(:,j)...
                        ==log(probabilityVectors(j, 1)./probabilityVectors(j, indices))';

    end
cvx_end
Wfin2=Wfin;  %%% <<<===  Wfin2 is the same as Wstar

if norm(Wfin2-Wstar)>1e-5
    error('norm(Wfin2-Wstar)>1e-5')
end


% %% Wmm
cvx_begin
variable Wmm(V,d)
minimize(norm(Wmm,'fro'))
    for j = 1:m
            indices = (2:Sigma_len); % Logical index to exclude the i-th element
                    (repmat(Wmm(supportSets(j, 1),:),Sigma_len-1,1)-Wmm(supportSets(j, indices),:))*H_m(:,j)...
                        ==0;
            indices = setdiff(1:V, supportSets(j,:)); % Logical index to exclude support
                    (repmat(Wmm(supportSets(j, 1),:),V-Sigma_len,1)-Wmm(indices,:))*H_m(:,j)...
                        >=1;
    end
cvx_end


%%%%%%%%%%%%%%%%% ENTROPY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Compute conditional entropy
conditional_entropy = 0;

for j = 1:m
    % Identify indices corresponding to the j-th context
    indices_j = (1:r) + (j-1)*(r);

    % Extract labels and features for the j-th context
    y_j = y(indices_j);
    X_j = H_n(:, indices_j);

    % Compute conditional probability of labels given features for the j-th context
    unique_labels = unique(y_j);
    label_probs_given_X = zeros(length(unique_labels), 1);

    for i = 1:length(unique_labels)
        label_probs_given_X(i) = sum(y_j == unique_labels(i)) / length(y_j);
    end

    % Compute conditional entropy for the j-th context
    conditional_entropy_j = -sum(label_probs_given_X .* log(label_probs_given_X + eps));

    % Weight the conditional entropy by the probability of the j-th context
    conditional_entropy = conditional_entropy + (1 / m) * conditional_entropy_j;
end

% Display the computed conditional entropy
fprintf('Conditional Entropy: %f\n', conditional_entropy);


%%%%%%%%%%%%%%%%%%  TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize arrays to store results
loss_values = zeros(iterations, 1);
norm_values = zeros(iterations, 1);
corr2_values = zeros(iterations, 1);      % <----- correlation of Wt with Wmm-Wstar
corr_values_raw = zeros(iterations, 1);   % <----- corelation of Wt with Wmm
proj_error = zeros(iterations,1);         % <------ error on finite component

% Initialize weight matrix W
W = randn(V, d)/sqrt(d);

% Convert sparse label vector y to full matrix for display
y_full = full(y_one_hot);

% Empirical cross-entropy loss function
loss_function = @(W, X, y) -(1/n)*sum(sum(log(softmax(W * X)) .* y));

% Softmax function
softmax = @(x) exp(x - max(x, [], 1)) ./ sum(exp(x - max(x, [], 1)));

% Gradient descent
for iter = 1:iterations
    % Compute predicted output
    y_pred = softmax(W * H_n);

    % Compute the gradient of the loss with respect to W
    gradient = (y_pred - y_full) * H_n' / (n);

    % Update W using gradient descent
    W = W - learning_rate * gradient/norm(gradient,'fro');

    % Compute and store the current loss
    loss_values(iter) = loss_function(W, H_n, y_full);
    fprintf('Iteration %d, Loss: %f\n', iter, loss_values(iter));

    % Compute and store the norm of W
    norm_values(iter) = norm(W, 'fro');


    corr2_values(iter) = trace((W-Wfin2)'/norm((W-Wfin2),'fro')*Wmm/norm(Wmm,'fro'));
    corr_values_raw(iter) = trace((W)'/norm((W),'fro')*Wmm/norm(Wmm,'fro'));

    W_S = reshape(US*US'*vec(W),V,d);  % project on subspace of log-prob measurements
    proj_error(iter) = norm(W_S-Wfin2)^2/norm(Wfin2)^2;

end
% Display the computed conditional entropy
fprintf('Conditional Entropy: %f\n', conditional_entropy);

%%
% Plot loss versus iterations
figure;
subplot(4, 1, 1);
semilogx(1:iterations, loss_values, 'b', 'LineWidth', 1.5);
title('CE loss');
% xlabel('Iterations');
% ylabel('Empirical Cross-Entropy Loss');

% Plot conditional entropy as a dashed line for comparison
hold on;
semilogx(1:iterations, conditional_entropy*ones(iterations,1), '--r', 'LineWidth', 1.5);
legend('CE$(\mathbf{W}_k)$', 'Entropy $\mathcal{H}$');
grid on
hold off;
% ylim([0,2])

% Plot the norm of W versus iterations
subplot(4, 1, 2);
semilogx(1:iterations, norm_values, 'b', 'LineWidth', 1.5);
title('$\|\mathbf{W}_k\|$');
grid on
% xlabel('Iterations');
% ylabel('Norm of W');


% Plot the norm of W versus iterations
subplot(4, 1, 3);
% loglog(1:iterations,1- corr_values, 'g', 'LineWidth', 1.5);
% hold on;
loglog(1:iterations,1- corr2_values, 'b', 'LineWidth', 1.5);
hold on;
% loglog(1:iterations,1- corr3_values, 'k--', 'LineWidth', 1.5);
hold on;
grid on
loglog(1:iterations,1- corr_values_raw, 'r', 'LineWidth', 1.5);
title('$1-\rm{corr}\left(\mathbf{W}_k-\mathbf{W}^{\rm{p}},\mathbf{W}^{\rm{mm}}\right)$');
% xlabel('Iterations $(k)$');
% legend('$\mathbf{W}^{\rm{p}}$ feas.', '$\mathbf{W}^{\rm{p}}$ min-norm', '$\mathbf{W}^{\rm{p}}=0$.');
legend('$\mathbf{W}^{\rm{p}}$ min-norm', '$\mathbf{W}^{\rm{p}}=0$.');
% ylabel('Norm of W');

% Plot the norm of W versus iterations
subplot(4, 1, 4);
loglog(1:iterations, proj_error, 'b', 'LineWidth', 1.5);
title('$\|\mathbf{W}_k-\mathbf{W}_*\|^2/\|\mathbf{W}_*\|^2$');
grid on
xlabel('Iterations');
% ylabel('Norm of W');



%% ONLY if d=2 do plots

if d==2,
% Plot points
% Plot points
figure;

% Define marker types for each point
markerTypes = {'o', '^', 's', 'd', 'v'}; % You can use other marker types as needed

% Plot points with different marker types
figure;
for i = 1:m
    scatter(H_m(1, i), H_m(2, i), markerTypes{i}, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', 'LineWidth', 2, 'SizeData', 100, 'HandleVisibility', 'off');
    hold on;
end

% Define distinct colors for hyperplanes
hyperplaneColors = lines(V); % You can use other color maps as needed


% Plot hyperplanes
for i = 1:V
    normal = Wmm(i, :);
    normal =  [-normal(2), normal(1)];
    pointOnLine = [0, 0];
    scale = 10; % Length of the line

    % Calculate endpoint of the line
    endpoint = pointOnLine + scale * normal;
    starpoint = pointOnLine - scale * normal;

    % Plot line
    plot([starpoint(1), endpoint(1)], [starpoint(2), endpoint(2)], 'LineWidth', 2, 'Color', hyperplaneColors(i, :) ,'DisplayName', ['$w_',int2str(i),'$']);
end

% % Annotate each point with SupportSets
% textOffset = 0; % Adjust this value based on the desired offset
% for i = 1:m
%     orderedElements = sort(supportSets(i, :));
%     text(H_m(1, i), H_m(2, i) + textOffset, sprintf('(%d,%d)', orderedElements(1), orderedElements(2)), 'FontSize', 16, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
% end

% Annotate each point with SupportSets using color-coded indices
textOffset = -0.25; % Adjust this value based on the desired offset
for i = 1:m
    orderedElements = sort(supportSets(i, :));

    % Use different colors for each index with separate text objects
    text(H_m(1, i), H_m(2, i) + textOffset, sprintf('%d)', orderedElements(1)), 'FontSize', 16,'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Color', hyperplaneColors(orderedElements(1), :));
    text(H_m(1, i), H_m(2, i) + textOffset, sprintf('(%d,', orderedElements(2)), 'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Color', hyperplaneColors(orderedElements(2), :));
end


hold off;
grid on;
legend;
% title('Points and Hyperplanes');

% Set axis limits close to the points
% xlim([-1.5,1.5]);
% ylim([-1.5,1.5]);
% axis square
xlim([min(H_m(1, :)) - 0.5, max(H_m(1, :)) + 0.5]);
ylim([min(H_m(2, :)) - 0.5, max(H_m(2, :)) + 0.5]);

end