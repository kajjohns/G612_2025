clear all


%define Gaussian distribution to be sampled
Cov = [[4 1];[1 2]];
InvCov = inv(Cov);

%generate values for plotting
x = linspace(-10,10);
y = linspace(-10,10);
[X,Y]=meshgrid(x,y);
for loop1=1:length(x)
    for loop2=1:length(y)
        P(loop1,loop2) = exp(-.5*( [X(loop1,loop2) Y(loop1,loop2)]*InvCov*[X(loop1,loop2); Y(loop1,loop2)]));
    end
end
contour(x,y,P,25)
hold on

X=[8,8]; %starting value
stepsize=3*[.5 .5]; %step sizes for Metropolis random walk
NumSamples = 10^5;  %number of samples


%evaluate prob for first sample
logprob = -.5*( [X(1) X(2)]*InvCov*[X(1); X(2)]);


%intialize radnom number generator
rand('state', sum(100*clock)); %reset random number generator

%define 'previous values'
Xprev=X;
logprobprev=logprob;


%begin Monte Carlo Metropolis walk
M=zeros(length(X),NumSamples);


all_logprob = zeros(1,NumSamples);

for loop=1:NumSamples

    %take a random step in model space    
    r=(-1).^round(rand(size(X))).*rand(size(X));
    r=r.*stepsize; 
    X=X+r;

    %compute log probability
   logprob2 = -.5*( [X(1) X(2)]*InvCov*[X(1); X(2)]);

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

    ll_logprob(loop) = logprob;
    
    %plot sample
    plot(X(1),X(2),'k.');
    pause(0.1)
    drawnow

end %loop


