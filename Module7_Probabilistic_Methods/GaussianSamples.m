
%This script samples a 1D Gaussian distribution using MCMC-Metropolis


%define Gaussian distribution to be sampled
dist_mean = 10;  %mean of distribution
sig = 2.5;  %standard deviation of distribution

X=1; %starting value
stepsize=[2]; %step sizes for Metropolis random walk
NumSamples = 10^5;  %number of samples


%evaluate prob for first sample
logprob=-.5/(sig^2)*(X-dist_mean)'*(X-dist_mean);


%intialize radnom number generator
%rand('state', sum(100*clock)); %reset random number generator
rng('default')
rng('shuffle') 


%define 'previous values'
Xprev=X;
logprobprev=logprob;


%begin Monte Carlo Metropolis walk
M=zeros(length(X),NumSamples);
logprobs = zeros(1,NumSamples);

for loop=1:NumSamples

    %take a random step in model space    
    r=(-1).^round(rand(size(X))).*rand(size(X));  %rand (-1,1)
    r=r.*stepsize; 
    X=X+r;  %trial value (X prime)

    %compute log probability
    logprob2=-.5/(sig^2)*(X-dist_mean)'*(X-dist_mean);

    %use Metropolis rule to decide whether or not to accept model
    accept=metropolis(logprob,logprob2);
    
    if accept==1  %if accept==1, keep the model
       logprob=logprob2;
       Xprev=X;
       logprobprev=logprob;
    else  %if accept==0, discard this model and retain previous            
        X=Xprev;
        logprob=logprobprev;
    end

    %store results matrices
    M(loop) = X;

    %store logprob for determining burn-in period
    logprobs(loop) = logprob;
    
end %loop


%discard burn-in samples
t0 = 1000;  %t0 is the number of burn-in samples
M(:,1:t0) = [];

%make histogram
Nbins=50;  %number of bins in histogram
figure
hist(M,Nbins)


estimate_mean = mean(M)
estimate_std = std(M)



