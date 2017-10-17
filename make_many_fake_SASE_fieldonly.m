function make_many_fake_SASE_fieldonly


%-------------------------------------
% Variable Definition
%-------------------------------------
Nruns=5e4;
lambda=1;       % wavelength
c=1;            % speed of light
sig_t=1;        % coherence length
N_p=20;        % number of wave packets
T=20;          % time interval
N_t=200;       % number of time points
omega0=2*pi*c/lambda;   % frequency of light


mypath='/Users/dratner/Desktop/Data/GhostImaging/SASE_sim/12fs/';
%run_name=[mypath 'SASE_12fs_2p3nm_phase_' num2str(Nruns) 'runs.mat'];
run_name=[mypath 'SASE_12fs_2p3nm_field_' num2str(Nruns) 'runs.mat'];


t=linspace(-T/10,T*1.1,N_t);
tic
for j=1:Nruns
    [field_j]=fake_sase_fieldonly(omega0,T,N_p,sig_t,N_t,t);
    
%     if j==1
%         power=zeros(Nruns,length(power_j));
%         pspec=zeros(Nruns,length(pspec_j));        
%         phase=zeros(Nruns,length(phase_j)); 
%         C1=zeros(Nruns,length(C1_j)); C2=C1;
%     end
%     
%     power(j,:)=power_j;
%     pspec(j,:)=pspec_j;
%     phase(j,:)=phase_j;    
%     C1(j,:)=C1_j;    
%     C2(j,:)=C2_j;    

    if j==1
        field=zeros(Nruns,length(field_j));
    end
    field(j,:)=field_j;

    if mod(j,100)==0
        disp([num2str(j) ' runs complete, ' num2str(toc) ' sec'])
    end    
end

save(run_name,'field','omega0','t')    






function [power,pspec,phase,C1,C2]=fake_sase(omega0,T,N_p,sig_t,N_t)


%-------------------------
% Construct waves
%-------------------------

% distribute wave packets randomly across T
tau=rand(1,N_p)*T;

% look at 1000 time steps from about 0 to T
t=linspace(-T/5,T*1.2,N_t);

% Add up wave packets
E = zeros(size(t));
for j=1:N_p
    E = E+exp(-(t-tau(j)).^2/(2*sig_t^2)-1i*omega0*(t-tau(j)));
end

%E=E.*exp(1i*omega0*t);


% plot (t,EE*)
power = E.*conj(E);

pfft = fft(E);
pspec = pfft.*conj(pfft);
phase = angle(E);
figure(2); hold off; plot(t,power); xlabel('time'); ylabel('power')

[C1,C2]=WK_check(E);
E;


function [E]=fake_sase_fieldonly(omega0,T,N_p,sig_t,N_t,t)


%-------------------------
% Construct waves
%-------------------------

% distribute wave packets randomly across T
tau=rand(1,N_p)*T;


% Add up wave packets
E = zeros(size(t));
for j=1:N_p
    E = E+exp(-(t-tau(j)).^2/(2*sig_t^2)-1i*omega0*(t-tau(j)));
end

% remove oscillating component if desired
E=E.*exp(1i*omega0*t);

WK_check(E);

%figure(2); plot(abs(E).^2,'.-')


%figure(3); hold off; plot(t,phase); xlabel('time'); ylabel('phase')
%figure(4); hold off; plot(pspec); xlabel('time'); ylabel('phase')


function [C1,C2] = WK_check(x)
% x: field

% pad x so circular autocorr works
nx=size(x,2);
x2=cat(2,x,zeros(size(x)));

p=x2.*conj(x2);   % power

xw=fft(x2);

C1=GI_autocorr(p);
C1f=GI_autocorr(x2).^2;
Sw=xw.*conj(xw)/length(x2);
C2=fft(Sw).^2;

C1=C1(1:nx); C2=C2(1:nx); C1f=C1f(1:nx);

show_plot=1;
if show_plot
    figure(11); plot(p(1:nx));
    figure(10); hold off;
    plot(abs(C1)/max(abs(C1)));  hold on;
    plot(abs(C2)/max(abs(C2)),'r');  
    plot(abs(C1f/max(abs(C1f))),'k');
end

C1=abs(C1);
C2=abs(C2);


function [C] = GI_autocorr(x)
%autocorrelator for vector x

%x=x-mean(x)

%number of row elements
nelts=length(x);

% % full autocorr; (comment out to just do overlapping portion
% x=cat(2,x,zeros(size(x)));

C=zeros(1,nelts);

for k=1:nelts
    C(k)=sum(x.*circshift(conj(x),[0,k]),2);
end



function [D1,D2]=crosscorr(C1,C2)

[n,m]=size(C1);

D1=zeros(n,1);
D2=ones(n,m);
for j=1:n
    D=circshift(C1,[j-1,0]);
    CDprod=D.*C2;
    D1(j)=sum(sum(CDprod));
    if j==1
        D2norm=sum(CDprod,1);
    else
        D2(j,:)=sum(CDprod,1)./D2norm;
    end
end

figure(20); plot(D1);
figure(21); imagesc(D2);

Z=mean(D2(2:end,:),1);
figure(22); plot(Z);





