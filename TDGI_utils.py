import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scis
import scipy.interpolate as scii
import scipy.io as sciio

def save_data(npts,M_meas,power,power_meas,R2):

    GD={}
    GD['M_measured']=M_meas[:npts,:]
    GD['power']=power[:npts,:]
    GD['power_measured']=power_meas[:npts,:]
    GD['R'] = R2
    GD['t'] = t_sase
    GD['t_measured'] = t_sase2
    sciio.savemat('GordonData_%d'%npts,GD)


def calc_impulse(P):
    # Input: n x t matrix of Power with 'n' examples and 't' times per example
    # Output: n x t matrix of Impulses with 'n' examples and 't' delays per example
    # (seems should be faster with fftconvolve, but possible on just 1 axis?

    t = P.shape[1]      # number of time steps

    I = np.zeros(P.shape)
    delays = np.arange(1,t)
    norm = np.sum(P*np.conj(P),axis=1)

    I[:,0] = 1.
    for d in delays:
        imp = np.sum(P[:,:-d]*np.conj(P[:,d:]),axis=1)
        I[:,d] = imp/norm

    return I


def calc_PDF(P):
    # Input: n x t matrix of Power with 'n' examples and 't' times per example
    # Output: n x t matrix of PDFs with 'n' examples and 't' integrated signals per example

    t = P.shape[1]      # number of time steps

    I = np.zeros(P.shape)
    it = np.arange(1,t)

    I[:,0] = 1.
    for i in it:
        I[:,i] = I[:,i-1]+P[:,i]

    return I

def nanoxtal_damage(q=np.logspace(-1,0,20),dt=np.linspace(0,50,25)):
    # make up nanoxtal damage response
    # use resolution range q [nm] and delay dt [fs]

    A=30
    tc=100

    R=np.ones([q.shape[0],dt.shape[0]])

    for j,qj in enumerate(q):
        
        R[j,:]= 1/(1+np.exp((dt-tc*np.sqrt(qj))/(A*np.sqrt(qj))))
        Rj0=R[j,0]
        R[j,:]=R[j,:]/Rj0
    
    
    
    plt.imshow(R, aspect='auto',extent=[dt[0],dt[-1],q[-1],q[0]],cmap=cmap);
#    plt.xlim([dt[0],dt[-1]]); plt.ylim([q[0],q[-1]]);
    plt.xlabel('delay [fs]',fontsize=15); plt.ylabel('resolution [nm]',fontsize=15)
    plt.title('Normalized scattering vs. delay/resolution', fontsize=15)
    plt.colorbar()
    plt.savefig('figs/damage/nanoxtal_damage.png');
    plt.show()
    
    return R,q,dt



def yag_sat(x,c=2, A=1):
    # make up YAG saturation response

    s=1-2/(1+np.exp(-(x-c)/A))
    return s

def add_noise(I,noise=5e-0):
    # include multiplicative white noise
    # average taken as half max of each example
    # then scaled by 'noise'
    
    #sig_noise = np.max(I,axis=1)/2*noise
    #Inoise = np.random.random(I.shape)*sig_noise[:,np.newaxis]
    #Inoise = np.random.random(I.shape)*sig_noise
    Inoise = np.random.randn(I.shape[0],I.shape[1])*noise
    I2 = I+Inoise

    return I2

def matrix_samp(M,n):

    M_samp = np.zeros(np.shape(M))
    for row in np.arange(M.shape[0]):
        M_samp[row,:] = poisson_sample(M[row,:],n)

    return M_samp


def poisson_sample(P,n):
    # sample n particles from a distribution P
    # return sampled distribution, P_samp

    Pnorm=P/np.sum(P)
    ind = np.arange(len(Pnorm))
    n=np.int(n)
    samp = np.random.choice(ind,size=n,p=Pnorm)

    bins = np.arange(len(Pnorm)+1)
    hist=np.histogram(samp,bins=bins)

    P_samp = hist[0]
    #plt.plot(bins,P_samp); plt.show()

    return P_samp


def gauss_smear(P,t,sigma=.5):
    # add Gaussian smearing (not finished)

    examples = np.arange(P.shape[0])
    
    g = np.exp(-(t-np.mean(t))**2/(2*sigma**2))
    g = g/np.sum(g)
    P_out = np.zeros(P.shape)
    for ex in examples:
        P_out[ex,:] = scis.convolve(P[ex,:],g,mode='same')

    return P_out


def max_filt(R,size):

    [n,m]=np.shape(R)
    ns=size[0]; ms=size[1]

    R2=R.copy()
    for j in np.arange(ns,n-ns):
        for k in np.arange(ms,m-ms):

            Rtemp=R[j-ns:j+ns+1,k-ms:k+ms+1]

            #R2[j,k]=np.max(Rtemp)
            R2[j,k]=np.mean(Rtemp[Rtemp>=np.mean(Rtemp)])

    return R2



def resample_time(R,t_in,t_out):
    # take response matrix (R), input times (t_in), and output time (t_out)
    # output is resampled R using 't_out' times

    y = np.arange(R.shape[0])

    x_in,y_in = np.meshgrid(t_in,y)
    x_out,y_out = np.meshgrid(t_out,y)

    R2 = scii.griddata(np.array([x_in.ravel(),y_in.ravel()]).T, R.ravel(),
                                    (x_out,y_out), method='linear')   # def

    return R2



def make_heatmap(dat,extent=None,cmap='inferno',xlim=None,ylim=None,title=None,
            xlabel='Delay (fsec)',ylabel='Electron energy (eV)',savename=None,
                 vmin=None,vmax=None,fontsize=15,interpolation=None):

    plt.imshow(dat, aspect='auto',extent=extent,cmap=cmap,interpolation=interpolation);
    plt.clim(vmin,vmax)
    plt.xlim(xlim); plt.ylim(ylim);
    plt.title(title,fontsize=fontsize);
    plt.xlabel(xlabel,fontsize=fontsize);
    plt.ylabel(ylabel,fontsize=fontsize)
    if savename is not None:
        plt.savefig(savename);
    plt.show()


def make_plot(x1,y1,x2=None,y2=None,l1=None,l2=None,xlim=None,ylim=None,title=None,
              xlabel=None,ylabel=None,savename=None,fontsize=15):
    
    if len(y1.shape)==2:
        plt.plot(x1,y1.T)
    else:
        y1=np.max(y2)*y1/np.max(y1)*1;    # normalize two plots
        plt.plot(x1,y1,linewidth=2); plt.plot(x2,y2,'--');
        plt.legend([l1,l2])

    if ylim is None:
        ylim=[0,np.max(y1)*1.05]

    if xlim is None:
        xlim=[np.min(x1),np.max(x1)]

    plt.xlim(xlim); plt.ylim(ylim);
    plt.title(title,fontsize=fontsize);
    plt.xlabel(xlabel,fontsize=fontsize); plt.ylabel(ylabel,fontsize=fontsize)
    if savename is not None:
        plt.savefig(savename);
    plt.show()


