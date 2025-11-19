


%make synthetic data assuming d = a*log(1+t/tR)
%where tR is relaxation time and t is time

a=1;
tR=1;

t=linspace(0,60,20)';
d = a*log(1+t/tR);

%add Gaussian noise
d = d + .15*randn(size(d));

%error vector
sigma = 0.15*ones(size(d));

%initial guess
a=1;
tR=1;
X = [a tR];


%number of MC samples
NumSamples = 1e4;
%step sizes for Metropolis random walk
stepsize=[0.05 0.05]; 


%include Gaussian prior
include_prior = true;
prior_m = [1.4, 2.5];  %prior mean for a, tR
prior_sigma = [.1, .1];  %prior standard deviation for a, tR

%begin Monte Carlo Metropolis walk
M=zeros(length(X),NumSamples); %store up samples
logprobs = zeros(1,NumSamples);
dhats = zeros(length(d),NumSamples);  %store up all dhat

accept_sample = 0;  %keep track of acceptance rate
for k=1:NumSamples


    %take a random step in model space to select trial value   
    r=(-1).^round(rand(size(X))).*rand(size(X));  %rand (-1,1)
    r=r.*stepsize; 
    X=X+r;  %trial value (X prime)

    a = X(1); tR = X(2);
    dhat2 = a*log(1+t/tR);
    resid = (d - dhat2)./sigma;  %weighted residual vector
    
    %compute log probability of trial sample
    if include_prior

          logprior = -0.5*((X'-prior_m')./prior_sigma')'*((X'-prior_m')./prior_sigma');
         logprob2 = -0.5*resid'*resid + logprior;  %assumes Gaussian likelihood 


    else

        logprob2 = -0.5*resid'*resid ;

    end

    
    %use Metropolis rule to decide whether or not to accept model
    if k==1
       logprob=logprob2;
       dhat = dhat2;
    end
    accept=metropolis(logprob,logprob2);
    
    if accept==1  %if accept==1, keep the model
       logprob=logprob2;
       dhat = dhat2;
       Xprev=X;
       logprobprev=logprob;
       
       accept_sample = accept_sample+1;
       
    else  %if accept==0, discard this model and retain previous            
        X=Xprev;
        logprob=logprobprev;
    end

    %store results matrices
    M(:,k) = X';

    %store values
    logprobs(k) = logprob;
    dhats(:,k) = dhat;
 
    
end


acceptance_rarte = accept_sample/NumSamples

%evaulate quality of sampling
figure
plot(logprobs)
title('log probility of each sample','fontsize',15)
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;

figure
subplot(121)
plot(M(1,:))
title('samples of parameter, a','fontsize',15)
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;

subplot(122)
plot(M(2,:))
title('samples of parameter, t_R','fontsize',15)
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;



%Now compare with gridsearch method


%define range of parameters to search
as = linspace(0.1,2,100);
tRs = linspace(.1,5,100);


for loop1=1:length(as)
    
    for loop2=1:length(tRs)
        
        dhat = as(loop1)*log(1+t/tRs(loop2));
        residual(loop1,loop2) = norm(d./sigma-dhat./sigma); 
    
    end
end

    
%plot (weighted) residual
figure
imagesc(tRs,as,residual)
xlabel('tR')
ylabel('a')
hold on
title('norm(d-dhat)')
colormap(jet)
c=colorbar;
c.FontSize = 15;
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;


%plot bootstrap samples on grid search
plot(M(2,:),M(1,:),'w.')

%plot marginal distributions
figure
subplot(121)
histogram(M(1,:),50)
title('marginal posterior distribution of a','fontsize',15)
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;

subplot(122)
histogram(M(2,:),50)
title('marginal posterior distribution of tR','fontsize',15)
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;


%plot fit to data
figure
plot(t,dhats(:,1:100:end),'r')
hold on
errorbar(t,d,sigma,'bo')
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
title('fit to data','fontsize',15)
xlabel('time, t')
ylabel('data, d')
