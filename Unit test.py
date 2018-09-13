

# r(data,data)

resultdir = '/home/tahereh/Documents/Research/Results/Neural-Dynamics/'
neuralfeaturesdir = '/home/tahereh/Documents/Research/features/neural_features/'
datadir = '/home/tahereh/Documents/Research/Data/DiCarlo/'

import pickle
import time
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as ss
import importlib
import MappingV36
importlib.reload(MappingV36)
from MappingV36 import MappingV36 as Mapping
Mapping = Mapping()

# UNIT TEST: RHS & LHS Perfect model + map + Gaussian noise
# NOTE: Uses pseudo-inverse regression function, replace all instances of pinv with regression
# method of choice (i.e. ridge, PLS, etc)

from ReadMeta import ReadMeta
from ReadData import ReadData
from MappingV36 import MappingV36 as Mapping
Mapping = Mapping()

# Read Meta
Meta = ReadMeta(neuralfeaturesdir)
DF_img = Meta.get_DF_img()
DF_neu = Meta.get_DF_neu()
times = Meta.get_times()

# Read Neural data
Data = ReadData(datadir, DF_neu)
IT, V4 = Data.get_data()

##
# ------------------------------------------------------------
# Parameters
# -----------------

report_sitefit = False
reg_method = 'ridge'#'OMP'# 'PLS'#'ridge'
spearman_brown = False

# ------------------------------------------------------------
trainfraci = 0.8 # image trainfrac
splitfract= 0.5   # trial splitfrac
nfoldi = 5
nfoldt = 5
ni = 20

Data = 'synthetic'#'HvM'

if Data == 'HvM':
    nf = IT[0].shape[0]
    D = Mapping.get_Neu_trial_V36(IT[1:], [70, 170], times)
    D = D[:, :ni, :]
    nf = D.shape[0]
    nt = D.shape[2]
    sd = np.arange(2)
    Collinearity = 'HvM'
    # Create model features MA = D from data
    A = np.random.rand(nf, nf)
    M = np.matmul(D.mean(2), A)

elif Data == 'synthetic':

    nf = 20  # # of features
    nt = 10  # # of trials
    sd = np.logspace(-1,1, num=10) #np.arange(0.5, 10, 1)
    D = np.random.rand(ni, nf)  # model M: nf feat x ni images
    Collinearity = False

    if Collinearity:

        collinearity_r = np.random.uniform(low=0.5, high=1, size=nf - 1)
        # collinearity_r = np.random.uniform(low=0., high=0.1, size=nf-1)

        for ir, r in enumerate(collinearity_r):
            D[ir + 1] = D[0] * r + D[ir + 1] * np.sqrt(1 - r ** 2)
            # print(r, ss.pearsonr(M[0], M[ir+1])[0], end="")
    # Create model features M = D*W from data
    A = np.random.rand(nf, nf)
    M = np.matmul(D, A)


  # model M: nf feat x ni images


# regularization parameters

if reg_method == 'PLS':
    n_components = nf
    reg_params = n_components
    report_popfit = True
    
elif reg_method == 'ridge':
    reg_params = []
    report_popfit = True
    
elif reg_method == 'OMP':
    reg_params = []
    report_popfit = False
    
zscored_observations = False

r12 = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))
r11 = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))
r22 = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))

r12_reg = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))
r11_reg = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))
r22_reg = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))


r12_reg_sitfit = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))
r11_reg_sitfit = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))
r22_reg_sitfit = np.zeros((sd.shape[0], nf, nfoldi, nfoldt))

start = time.time()

for s in range(sd.shape[0]):
    print(s, time.time()-start, end="")
    for fi in range(nfoldi):
        
        # train/test image split
        ind = np.random.permutation(ni)
        indtraini = ind[:int(ni*trainfraci)]
        indtesti = ind[int(ni*trainfraci):]
        
        for ft in range(nfoldt):
            # % add Gaussian noise and create two sets of trials
            if Data == 'HvM':
                D1 = D[:, :, :int(nt * splitfract)].mean(2)
                D2 = D[:, :, int(nt * splitfract):].mean(2)
            else:
                D1 = D + np.mean(np.random.normal(0, sd[s], size=[ni, nf, int(nt*splitfract)]), 2)
                D2 = D + np.mean(np.random.normal(0, sd[s], size=[ni, nf, int(nt*splitfract)]), 2)

            # NUMERATOR: Fit on train, test on test
            Ahat = np.dot(np.linalg.pinv(M[indtraini, :]), D1[indtraini, :])
            D1_test, D1_pred = D1[indtesti, :], np.dot(M[indtesti, :], Ahat)
            r12[s, :, fi, ft] = [ss.pearsonr(D1_pred[:, indf], D1_test[:, indf])[0] for indf in range(nf)]

            # DENOMINATOR consistency between trial sets 1 & 2 on test images
            D2_test = D2[indtesti, :]
            r22[s, :, fi, ft] = [ss.pearsonr(D1_test[:, indf], D2_test[:, indf])[0] for indf in range(nf)]

            # DENOMINATOR LHS map consistency between trial sets 1 & 2 on test images
            Ahat1 = np.dot(np.linalg.pinv(M[indtraini, :]), D1[indtraini, :])
            Ahat2 = np.dot(np.linalg.pinv(M[indtraini, :]), D2[indtraini, :])
            lhs1, lhs2 = np.dot(M[indtesti, :], Ahat1), np.dot(M[indtesti, :], Ahat2)
            r11[s, :, fi, ft] = [ss.pearsonr(lhs1[:, indf], lhs2[:, indf])[0] for indf in range(nf)]
            

            # Regression
            train_inds, test_inds = indtraini, indtesti
            model_features_X, half1, half2 = M, D1, D2

            # population fit
            if report_popfit:
                start_popfit = time.time()
                return_fitted_reg = False
                if spearman_brown:
                    r_Nom, r_Nom_sites = Mapping.Nominator(train_inds, test_inds, model_features_X, np.mean([half2, half2], axis=0), reg_method,
                                                       reg_params, zscored_observations, return_fitted_reg)
                else:
                    r_Nom, r_Nom_sites = Mapping.Nominator(train_inds, test_inds, model_features_X, half1, reg_method, 
                                                       reg_params, zscored_observations, return_fitted_reg)
                    
                r_RHS, r_RHS_sites = Mapping.Denom_RHS(train_inds, test_inds, half1, half2)
                r_LHS, r_LHS_sites = Mapping.Denom_LHS(train_inds, test_inds, model_features_X, half1, half2, reg_method, reg_params,
                      zscored_observations, return_fitted_reg)
                r12_reg[s, :, fi, ft] = r_Nom_sites
                
                if spearman_brown:
                    r_RHS_sites_sb = [Mapping.spearman_brown_correction(r) for r in r_RHS_sites]
                    r_LHS_sites_sb = [Mapping.spearman_brown_correction(r) for r in r_LHS_sites]
                    
                    r22_reg[s, :, fi, ft] = r_RHS_sites_sb
                    r11_reg[s, :, fi, ft] = r_LHS_sites_sb
                else:
                    r22_reg[s, :, fi, ft] = r_RHS_sites
                    r11_reg[s, :, fi, ft] = r_LHS_sites
                time_popfit = time.time() - start_popfit
                print('popfit took %.2f seconds' % (time_popfit))
                
                
                    
            
            # site fit
            if report_sitefit:
                start_sitefit = time.time()
                for n in range(nf):
                    return_fitted_reg = False
                    r_Nom, _ = Mapping.Nominator(train_inds, test_inds, model_features_X, half1[:, n], reg_method, reg_params, 
                                          zscored_observations, return_fitted_reg)
                    
                    r_LHS, _ = Mapping.Denom_LHS(train_inds, test_inds, model_features_X, half1[:, n], half2[:, n], reg_method,
                                                 reg_params, zscored_observations, return_fitted_reg)
                    r12_reg_sitfit[s, n, fi, ft] = r_Nom
                    
                    r11_reg_sitfit[s, n, fi, ft] = r_LHS
                    
                    if report_popfit:
                        r22_reg_sitfit[s, :, fi, ft] = r_RHS_sites
                    else:
                        r_RHS, _ = Mapping.Denom_RHS(train_inds, test_inds, half1[:, n], half2[:, n])
                        r22_reg_sitfit[s, n, fi, ft] = r_RHS
                time_sitefit = time.time()-start_sitefit
                print('sitefit took %.2f seconds' % (time_sitefit))

r12, r11, r22 = np.mean(r12, 3), np.mean(r11, 3), np.mean(r22, 3)
r12_reg, r11_reg, r22_reg = np.mean(r12_reg, 3), np.mean(r11_reg, 3), np.mean(r22_reg, 3)
r12_reg_sitfit, r11_reg_sitfit, r22_reg_sitfit = np.mean(r12_reg_sitfit, 3), np.mean(r11_reg_sitfit, 3),np.mean(r22_reg_sitfit, 3)




data_list = [r12, r11, r22, r12_reg, r11_reg, r22_reg, r12_reg_sitfit, r11_reg_sitfit, r22_reg_sitfit]
pickle.dump(data_list, open(resultdir+'unit_test_%s_%s_ni%d_nf%d_nt%d_collinearity%s_SB%s.pickle'%(reg_method, reg_params, ni, nf, nt, Collinearity, spearman_brown), 'wb'))



# In[8]:


# ----------------
# Read data
# ----------------
# ni,nf,nt = 20,20,10
# reg_method,reg_params,Collinearity = 'ridge','[]',True

# ni,nf,nt = 20,20,10
# reg_method,reg_params,Collinearity = 'PLS',nf,True


# ni,nf,nt = 200,20,10
# reg_method,reg_params,Collinearity = 'ridge','[]',True

# ni,nf,nt = 200,20,10
# reg_method,reg_params,Collinearity = 'PLS',nf,True
# 
# 
# 
# file = open(resultdir+'unit_test_%s_%s_ni%d_nf%d_nt%d_collinearity%s.pickle'%\
#                             (reg_method,reg_params,ni,nf,nt,Collinearity),'rb')
# data_list = pickle.load(file)
# file.close()
# r12,r11,r22,r12_reg,r11_reg,r22_reg,r12_reg_sitfit,r11_reg_sitfit,r22_reg_sitfit = data_list
# ----------------------------------------------------------

if reg_method == 'ridge':
    color = 'r'
elif reg_method == 'PLS':
    color = 'gray'
elif reg_method == 'OMP':
    color = 'blue'
    
    
fig = plt.figure(figsize=[12,5])
ax0 = fig.add_axes([0.1,0.11,0.25,0.70])
ax1 = fig.add_axes([0.4,0.11,0.25,0.70])
ax2 = fig.add_axes([0.7,0.11,0.25,0.70])

for s in range(sd.shape[0]):
    
    ax0.scatter(np.sqrt(r11[s].mean(1)*r22[s].mean(1)), r12[s].mean(1), color = 'k', label ='Nom. vs. Denom.')
    ax0.scatter(r22[s].mean(1), r11[s].mean(1), color=color, label = 'LHS vs. RHS')
    ax0.plot([0, 1], [0, 1], ls='--', color='gray')
    
    if reg_method != 'OMP':
        ax1.scatter(np.sqrt(r11_reg[s].mean(1)*r22_reg[s].mean(1)), r12_reg[s].mean(1), color='k',
                    label='Nom. vs. Denom.')
        ax1.scatter(r22_reg[s].mean(1), r11_reg[s].mean(1), color=color, label='LHS vs. RHS')
        ax1.plot([0, 1], [0, 1], ls='--', color='gray')
    
    ax2.scatter(np.sqrt(r11_reg_sitfit[s].mean(1)*r22_reg_sitfit[s].mean(1)), r12_reg_sitfit[s].mean(1), 
                color='k', marker='x', label='Nom. vs. Denom.')
    ax2.scatter(r22_reg_sitfit[s].mean(1), r11_reg_sitfit[s].mean(1), color=color, marker='x',
                label='LHS vs. RHS')
    ax2.plot([0, 1], [0, 1], ls='--', color='gray')
       
    
for ax in [ax0, ax2]:    
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc='upper left')


ax0.set_ylabel('r(model,data)_est OR r(model,model)')
ax1.set_yticks([])
ax0.set_title('VE=%.2f, N=%.2f, L= %.2f, R=%0.2f'%(np.nanmedian(r12/np.sqrt(r11*r22)), np.median(r12),
                                                   np.median(r11), np.median(r22))) 
ax0.text(0.0, 0.75, 'inversion', fontsize=14)

if reg_method != 'OMP':
    ax1.set_xlabel('r(model,model),r(data,data))^0.5 OR r(data,data)')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], labels[:2], loc='upper left')
    ax1.text(0.0, 0.75, 'regression:%s'%reg_method, fontsize=14)
    ax1.set_title('VE=%.2f, N=%.2f, L= %.2f, R=%0.2f'%(np.nanmedian(r12_reg/np.sqrt(r11_reg*r22_reg)),np.median(r12_reg),                                                   np.median(r11_reg),np.median(r22_reg)))
else:
    ax1.set_xticks([])
    ax1.axis('off')
   
    ax0.set_xlabel('r(model,model),r(data,data))^0.5 OR r(data,data)')
    ax2.set_xlabel('r(model,model),r(data,data))^0.5 OR r(data,data)')

print('Pinv: VarExp=%.3f, Num =%.3f, LHS= %.3f, RHS=%0.3f'%(np.nanmedian(r12/np.sqrt(r11*r22))            ,np.median(r12), np.median(r11), np.median(r22))) 
print('Reg: VarExp=%.3f, Num =%.3f, LHS= %.3f, RHS=%0.3f'%(np.nanmedian(r12_reg/np.sqrt(r11_reg*r22_reg)), np.median(r12_reg), np.median(r11_reg), np.median(r22_reg))) 
ax0.text(0.25,1.2,'#images = %d, #features = %d, #trials = %d, Collinearity = %s'%(ni, nf, nt, Collinearity), fontsize=14)

ax2.set_title('VE=%.2f, N=%.2f, L= %.2f, R=%0.2f'%(np.nanmedian(r12_reg_sitfit/np.sqrt(r11_reg_sitfit*r22_reg_sitfit))                ,np.median(r12_reg_sitfit), np.median(r11_reg_sitfit),np.median(r22_reg_sitfit))) 
ax2.text(0.0,0.75,'regression-sitefit', fontsize=14)
ax2.text(0.0,0.65,'%s'%reg_method, fontsize = 14)

print('Reg-sitefit: VarExp=%.3f, Num =%.3f, LHS= %.3f, RHS=%0.3f'%(np.nanmedian(r12_reg_sitfit/np.sqrt(r11_reg_sitfit*r22_reg_sitfit))            ,np.median(r12_reg_sitfit), np.median(r11_reg_sitfit), np.median(r22_reg_sitfit)))

plt.show()
fig.savefig(resultdir+'unit_test_%s_ni%d_nf%d_nt%d_Collin%s_SB%s.png'%(reg_method, ni, nf, nt, Collinearity, spearman_brown), dpi=300)
fig.savefig(resultdir+'unit_test_%s_ni%d_nf%d_nt%d_Collin%s_SB%s.pdf'%(reg_method, ni, nf, nt, Collinearity, spearman_brown), dpi=300)


# In[18]:


r12_reg_sitfit[s].mean(1)

