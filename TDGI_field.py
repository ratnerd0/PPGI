import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sciio
import scipy.optimize as sciopt
import scipy.signal as scis
import TDGI_utils as tu

from sklearn.decomposition import NMF
import sklearn.linear_model as sklm

##################################################
# Parameters
##################################################


hard_xray_path = '/Users/dratner/Desktop/Data/GhostImaging/SASE_sim/2AA'
sase_path = '/Users/dratner/Desktop/Data/GhostImaging/SASE_sim/12fs'
sase_path = '/Users/dratner/Desktop/Data/GhostImaging/SASE_sim/25fs'
amo_path = '/Users/dratner/Dropbox/ResearchTopics/GhostImaging/Data/amo_sim/'

sase_data = 'SASE_10fs_2p3nm_8500runs.mat'
sase_data = 'SASE_12fs_2p3nm_5000runs.mat'
sase_data = 'SASE_12fs_2p3nm_14000runs.mat'
sase_data = 'SASE_25fs_2p3nm2.mat'
sase_data = 'SASE_25fs_2p3nm_field_50000runs.mat'
#sase_data = 'SASE_8fs_2p3A_5runs.mat'; sase_path = hard_xray_path
amo_data = 'CO_sims_6fs_2eV.mat'

sase_file = os.path.join(sase_path,sase_data)
amo_file = os.path.join(amo_path,amo_data)


alpha = 1e-6    # regularization parameter
l1_ratio = 1e-4
XTCAV_FWHM = 1.*2.  # resolution in fsec (FWHM)
XTCAV_noise = 0e-2   # rms noise on XTCAV
spec_noise = 10e-1   # noise fraction on spectrometer
n_elec = 100          # number of electrons measured
N_samp = -1


# plotting parameters
cmap='inferno'
t_plot=[2,20]
vmin=0; vmax=20;


##################################################
# Analysis
##################################################


# load SASE simulations
sase_contents = sciio.loadmat(sase_file)
field = sase_contents['field']
power=np.real(field*np.conj(field))
t_sase = sase_contents['t'][0]


# load AMO simulations
amo_contents = sciio.loadmat(amo_file)
t_amo = amo_contents['time'][0]
eV = amo_contents['energy'][0]
R = amo_contents['Sims']

# select number of examples
if N_samp == -1:
    N_samp = power.shape[0]
else:
    N_samp = np.min([N_samp,power.shape[0]]).astype('int')
    power=power[:N_samp,:]




t_amo = t_amo - 12
#plt.imshow(R,extent=[t_amo[0],t_amo[-1],eV[0],eV[-1]],interpolation='nearest',cmap=cmap); plt.show()


# downsample time resolution
t_samp=np.arange(0,len(t_sase),12)    # downsample time resolution
t_sase2=t_sase[t_samp]
power2=power[:,t_samp]
field2=field[:,t_samp]

# sample AMO sims to have same time dimension as SASE sims
R2=tu.resample_time(R,t_amo,t_sase2)
R2=R2.T[:,170:-25]         # cut out un-excited fraction
eV=np.flipud(eV[170:-25])

# calculate impulse
I = tu.calc_impulse(power2)
I_true0 = tu.calc_impulse(power[0:1,:])

# Estimate measurement
M = np.dot(I,R2)
#M2 = tu.add_noise(M,noise=spec_noise)
#M_sig = tu.matrix_samp(M2,n=n_elec*(1+spec_noise))
#M_meas=M_sig
M_noise = np.mean(np.mean(M,axis=1))*spec_noise
M_sig = tu.matrix_samp(M+M_noise,n=n_elec*(1+spec_noise))
M_meas=M_sig
#M_noise = tu.matrix_samp(np.ones(M_sig.shape),n=spec_noise*n_elec)
#M_meas=(M_sig+M_noise)


# Estimate measured SASE
power_smear = tu.gauss_smear(power,t_sase,sigma=XTCAV_FWHM/2.35)  # add gaussian smearing
power_meas = power_smear[:,t_samp]                                # downsample in time
XTCAV_noise=XTCAV_noise*np.max(power_meas)/2
power_meas = tu.add_noise(power_meas,noise=XTCAV_noise)           # add white noise
I_meas = tu.calc_impulse(power_meas)                              # calculate measured impulse
I_field = (np.abs(tu.calc_impulse(field2)))**2

# Save data to .mat
# tu.save_data

#G=np.zeros(I_meas.shape)
#for j in range(4):
#    for i in range(I_meas.shape[1]):
#        G[j,i]=np.sum(I_meas[j,i:])/np.sum(I_meas[j,:])


R_extent=[t_sase2[0],t_sase2[-1],eV[0],eV[-1]]

## extract estimate R from NNLS
#n=I.shape[0]
#e=R2.shape[1]
#R_est=np.zeros(R2.shape)
#for j in np.arange(e):
#    b=M_meas[:,j]
#    R_est[:,j] = sciopt.nnls(I_meas,b)[0]
#    print('%d of %d finished' %(j+1,e))
##
#plt.imshow(R_est[:-1,:].T, aspect='auto',interpolation='nearest',extent=R_extent,cmap=cmap);
#plt.xlim(t_plot); plt.show()


# extract estimate R from ElasticNet
R_est=np.zeros(R2.shape)
clf = sklm.ElasticNet(alpha=alpha,l1_ratio=l1_ratio,positive=True,max_iter=1e6)
#clf = sklm.ElasticNet(alpha=alpha,l1_ratio=l1_ratio,positive=True,max_iter=1e4,
#    fit_intercept=False,normalize=True,selection='random')
clf.fit(I_field,M_meas)
R_est=clf.coef_

#plt.imshow(R_est[:-1,:], aspect='auto',interpolation='nearest',extent=R_extent,cmap=cmap);
#plt.xlim(t_plot); plt.show()



##################################################
# Plotting
##################################################

#----------
# R_mas
tu.make_heatmap(R2.T,extent=R_extent,xlim=t_plot,title='simulated ground truth',savename='figs/AMO/25fs_sim.png')

#----------
# R_est
savename='figs/AMO/Rfield_est_%delec_%0.1ffsec_%dnoise.png' % (n_elec,XTCAV_FWHM,XTCAV_noise*100)
title='Reconstructed response'
tu.make_heatmap(R_est,extent=R_extent,vmin=vmin,vmax=vmax,xlim=t_plot,title=title,savename=savename)

R_est2=tu.max_filt(R_est,size=[1,2])*10
z=np.arange(t_sase.shape[0])
R_est3 = tu.gauss_smear(R_est2.T,z,sigma=5).T
tu.make_heatmap(R_est3,extent=R_extent,vmin=vmin,vmax=vmax,xlim=t_plot,title=title,savename=savename)


#----------
# SASE power
savename='figs/AMO/SASE_12fs_%0.1ffsec_%dnoise.png' % (XTCAV_FWHM,XTCAV_noise*100)
title='SASE: %0.1f fsec res' % (XTCAV_FWHM)
xlabel='Time (fsec)'; ylabel='Power (arb. units)'
l1='True power'; l2='XTCAV meas.'
tu.make_plot(x1=t_sase,y1=power[0,:],x2=t_sase2,y2=power_meas[0,:],l1=l1,l2=l2,
             xlabel=xlabel,ylabel=ylabel,title=title,savename=savename)

#----------
# SASE power, 50 imshow
savename='figs/AMO/SASE_imshow_12fs_%0.1ffsec_%dnoise_%dshots.png' % (XTCAV_FWHM,XTCAV_noise*100,N_shots)
title='Measured power: %0.1f fsec res' % (XTCAV_FWHM)
xlabel='Time (fsec)';
tu.make_heatmap(dat=power_meas[:N_shots,:],extent=[t_sase2[0],t_sase2[-1],N_shots,0],xlabel=xlabel,ylabel='Shot number',
                title=title,savename=savename,interpolation='nearest')



#----------
# SASE Impulse
savename='figs/AMO/Impulse_12fs_%0.1ffsec_%dnoise.png' % (XTCAV_FWHM,XTCAV_noise*100)
title='Impulse: %0.1f fsec res' % (XTCAV_FWHM)
xlabel='Delay (fsec)'; ylabel='Amplitude (arb. units)'
l1='True impulse'; l2='Meas. impulse'
tu.make_plot(x1=t_sase,y1=I_true0[0],x2=t_sase2,y2=I_meas[0,:],l1=l1,l2=l2,
             xlabel=xlabel,ylabel=ylabel,title=title,savename=savename)


#----------
# SASE Impulse, 20
N_shots=20
savename='figs/AMO/Impulse_%0.1ffsec_%dnoise_%dshots.png' % (XTCAV_FWHM,XTCAV_noise*100,N_shots)
title='Impulse: %d shots, %0.1f fsec res' % (N_shots,XTCAV_FWHM)
xlabel='Delay (fsec)'; ylabel='Amplitude (arb. units)'
tu.make_plot(x1=t_sase2,y1=I_meas[:N_shots,:],
             xlabel=xlabel,ylabel=ylabel,title=title,savename=savename)

#----------
# SASE Impulse, 50 imshow
N_shots=50
savename='figs/AMO/Impulse_imshow_%0.1ffsec_%dnoise_%dshots.png' % (XTCAV_FWHM,XTCAV_noise*100,N_shots)
title='Impulse: %d shots, %0.1f fsec res' % (N_shots,XTCAV_FWHM)
xlabel='Delay (fsec)';
tu.make_heatmap(dat=I_meas[:N_shots,:],extent=[t_sase2[0],t_sase2[-1],N_shots,0],xlabel=xlabel,ylabel='Shot number',
                title=title,savename=savename,interpolation='nearest')




#----------
# M (GT)
savename='figs/AMO/Meas_%delec_%0.1ffsec_%dnoise.png' % (n_elec,XTCAV_FWHM,XTCAV_noise*100)
title='Electron spectrum, %d electrons' % (n_elec)
xlabel='Electron energy (eV)'; ylabel='Counts (arb. units)'
l1='True signal'; l2='Single shot, %d e-' % n_elec;
tu.make_plot(x1=np.flipud(eV),y1=M[0,:],x2=np.flipud(eV),y2=M_meas[0,:],l1=l1,l2=l2,
             xlabel=xlabel,ylabel=ylabel,title=title,savename=savename)

#----------
# M (Avg)
savename='figs/AMO/AvgMeas_%delec_%0.1ffsec_%dnoise.png' % (n_elec,XTCAV_FWHM,XTCAV_noise*100)
title='Electron spectrum, %d electrons' % (n_elec)
xlabel='Electron energy (eV)'; ylabel='Counts (arb. units)'
l1='True signal'; l2='Single shot, %d e-' % n_elec;
tu.make_plot(x1=np.flipud(eV),y1=np.sum(M_meas,axis=0),x2=np.flipud(eV),y2=M_meas[0,:],l1=l1,l2=l2,
             xlabel=xlabel,ylabel=ylabel,title=title,savename=savename)


#----------
# M, 20
N_shots=20
savename='figs/AMO/Meas20_%delec_%0.1ffsec_%dnoise.png' % (n_elec,XTCAV_FWHM,XTCAV_noise*100)
title='Electron spectrum'
xlabel='Electron energy (eV)'; ylabel='Counts (arb. units)'
l1='True signal'; l2='Single shot, %d e-' % n_elec;
tu.make_plot(x1=np.flipud(eV),y1=M[:N_shots,:],l1=l1,l2=l2,
             xlabel=xlabel,ylabel=ylabel,title=title,savename=savename)


#----------
# M, 50 imshow
N_shots=50
savename='figs/AMO/Meas20_imshow_%delec_%0.1ffsec_%dnoise.png' % (n_elec,XTCAV_FWHM,XTCAV_noise*100)
title='Electron spectrum'
xlabel='Electron energy (eV)';
l1='True signal'; l2='Single shot, %d e-' % n_elec;
tu.make_heatmap(dat=M[:N_shots,:],extent=[eV[-1],eV[0],N_shots,0],xlabel=xlabel,ylabel='Shot number',
                title=title,savename=savename,interpolation='nearest')







