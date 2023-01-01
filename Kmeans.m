function [M,obj] = Kmeans(X, k, T)

[d,m]=size(X);

% Norms of the samples
norm_x = full(sum(X.^2,1));
% precalculate this quantity that will be used in each iteration
repmat_norm_x=repmat(norm_x,k,1);

% k-means++ initialization
M=zeros(d,k);
%first center at random
idx=randperm(m);
M(:,1)=X(:,idx(1));
% the other ones drawn with a certain probability distribution
for i=2:k
    % calculate matrix of squared distances between centers we have till now and points
    norm_c = full(sum(M(:,1:i-1).^2,1));
    squared_dists = repmat_norm_x(1:i-1,:) + repmat(norm_c',1,m) - 2*full(M(:,1:i-1)'*X);
    [mn]=min(squared_dists,[],1);
    % we might get some negative numbers due to numerical errors, let's
    % remove them
    mn=max(mn,0);
    prob_distr=mn/sum(mn);
    M(:,i)=X(:,randsample(1:m,1,true,prob_distr)); 
end

% Fake initial assignment
cluster=ones(1,m);

for iter=1:T
    
    % store previous centers
    old_cluster=cluster;
    % calculate matrix of squared distances between centers and points
    norm_c = full(sum(M.^2,1));
    squared_dists = repmat_norm_x + repmat(norm_c',1,m) - 2*full(M'*X);
    % calculate assignments
    [mn,cluster]=min(squared_dists);
    % calculated objective function
    obj(iter)=sum(mn);
    if old_cluster==cluster
        break
    end

    % update centers
    for i=1:k
        M(:,i)=mean(X(:,cluster==i),2);
    end
end