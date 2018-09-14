

# r(data,data)

resultdir = '/home/tahereh/Documents/Research/Results/Mapping_unit_test/'
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


from get_mappings_unit_test import get_mappings_unit_test
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


# ------------------------------------------------------------
# Parameters
# -----------------

trainfraci = 0.8  # image trainfrac
splitfract = 0.5  # trial splitfrac
nfoldi = 1
nfoldt = 1
ni = 2000

Data_type = 'HvM'  # 'synthetic'#'HvM'

if Data_type == 'HvM':
    nf = IT[0].shape[0]
    D = Mapping.get_Neu_trial_V36(IT[1:], [70, 170], times)
    D = D[:, :ni, :]
    D = np.swapaxes(D, 0, 1)
    nf = D.shape[1]
    nt = D.shape[2]
    sd = np.arange(2)
    Collinearity = 'HvM'
    # Create model features MA = D from data
    A = np.random.rand(nf, nf)
    M = np.matmul(D.mean(2), A)

elif Data_type == 'synthetic':

    nf = 20  # # of features
    nt = 10  # # of trials
    sd = np.logspace(-1, 1, num=10)  # np.arange(0.5, 10, 1)
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


#reg_method = 'ridge'#'OMP'# 'PLS'#'ridge
report_sitefit = True

spearman_brown = False
for reg_method in ['ridge', 'OMP', 'PLS']:



# ------------------------------------------------------------


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

    Data_params = [ni,nf,nt,nfoldi,nfoldt,trainfraci,splitfract, sd, Collinearity]

    data_list = get_mappings_unit_test(M, D, Data_type, Data_params, reg_method, reg_params, spearman_brown, report_sitefit, report_popfit)
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
    r12,r11,r22,r12_reg,r11_reg,r22_reg,r12_reg_sitfit,r11_reg_sitfit,r22_reg_sitfit = data_list
    # ----------------------------------------------------------

    if reg_method == 'ridge':
        color = 'r'
    elif reg_method == 'PLS':
        color = 'gray'
    elif reg_method == 'OMP':
        color = 'blue'


    fig = plt.figure(figsize=[12, 5])
    ax0 = fig.add_axes([0.1, 0.11, 0.25, 0.70])
    ax1 = fig.add_axes([0.4, 0.11, 0.25, 0.70])
    ax2 = fig.add_axes([0.7, 0.11, 0.25, 0.70])

    for s in range(sd.shape[0]):

        ax0.scatter(np.sqrt(r11[s].mean(1)*r22[s].mean(1)), r12[s].mean(1), color = 'k', label ='Nom. vs. Denom.')
        ax0.scatter(r22[s].mean(1), r11[s].mean(1), color=color, label='LHS vs. RHS')
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

