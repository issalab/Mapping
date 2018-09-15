

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

# ------------------------------------------------------------
# Parameters
# -----------------

trainfraci = 0.8  # image trainfrac
splitfract = 0.5  # trial splitfrac
nfoldi = 5
nfoldt = 5
ni = 1000

Data_type = 'HvM'  # 'synthetic'#'HvM'

if Data_type == 'HvM':

    # Read Meta
    Meta = ReadMeta(neuralfeaturesdir)
    DF_img = Meta.get_DF_img()
    DF_neu = Meta.get_DF_neu()
    times = Meta.get_times()

    # Read Neural data
    Data = ReadData(datadir, DF_neu)
    IT, V4 = Data.get_data()


    D = Mapping.get_Neu_trial_V36(IT[1:], [70, 170], times)
    image_indices = np.random.randint(low=0, high=D.shape[0], size=ni)
    D = D[:, image_indices, :]
    D = np.swapaxes(D, 0, 1)
    nf = D.shape[1]
    nt = D.shape[2]
    sds = []
    Collinearity = 'HvM'
    noise_dist = 'HvM'
    # Create model features MA = D from data
    A = np.random.rand(nf, nf)
    M = np.matmul(D.mean(2), A)

elif Data_type == 'synthetic':

    nf = 200  # # of features
    nt = 50  # # of trials
    sds = np.logspace(-1, 1, num=int(nf))  # np.arange(0.5, 10, 1)
    noise_dist = 'normal'  # 'poisson'  # 'normal'
    D = np.random.rand(ni, nf)  # model M: nf feat x ni images
    Collinearity = True

    if Collinearity:

        collinearity_r = np.random.uniform(low=0.8, high=1, size=nf - 1)
        # collinearity_r = np.random.uniform(low=0., high=0.1, size=nf-1)

        for ir, r in enumerate(collinearity_r):
            D[ir + 1] = D[0] * r + D[ir + 1] * np.sqrt(1 - r ** 2)
            # print(r, ss.pearsonr(M[0], M[ir+1])[0], end="")
    # Create model features M = D*W from data
    A = np.random.rand(nf, nf)
    M = np.matmul(D, A)

# ------------------------------------------------------------
# reg_method = 'ridge'#'OMP'# 'PLS'#'ridge
report_sitefit = True

spearman_brown = False
corr_method_for_inv = 'pearson'  # 'spearman'
for reg_method in ['ridge']:

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
        report_sitefit = True

    Data_params = [ni, nf, nt, nfoldi, nfoldt, trainfraci,splitfract, noise_dist,  sds, Collinearity, corr_method_for_inv]

    data_list = get_mappings_unit_test(M, D, Data_type, Data_params, reg_method, reg_params, spearman_brown, report_sitefit, report_popfit)

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
    r12, r11, r22, r12_reg, r11_reg, r22_reg, r12_reg_sitfit, r11_reg_sitfit, r22_reg_sitfit = data_list
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

    ax0.scatter(np.sqrt(r11.mean(1)*r22.mean(1)), r12.mean(1), color = 'k', label ='Nom. vs. Denom.')
    ax0.scatter(r22.mean(1), r11.mean(1), color=color, label='LHS vs. RHS')
    ax0.plot([0, 1], [0, 1], ls='--', color='gray')

    if reg_method != 'OMP':
        ax1.scatter(np.sqrt(r11_reg.mean(1)*r22_reg.mean(1)), r12_reg.mean(1), color='k',
                    label='Nom. vs. Denom.')
        ax1.scatter(r22_reg.mean(1), r11_reg.mean(1), color=color, label='LHS vs. RHS')
        ax1.plot([0, 1], [0, 1], ls='--', color='gray')

    ax2.scatter(np.sqrt(r11_reg_sitfit.mean(1)*r22_reg_sitfit.mean(1)), r12_reg_sitfit.mean(1),
                color='k', marker='x', label='Nom. vs. Denom.')
    ax2.scatter(r22_reg_sitfit.mean(1), r11_reg_sitfit.mean(1), color=color, marker='x',
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
    ax0.text(0.0, 0.71, corr_method_for_inv+'corr', fontsize=14)

    if reg_method != 'OMP':
        ax1.set_xlabel('r(model,model),r(data,data))^0.5 OR r(data,data)')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[:2], labels[:2], loc='upper left')
        ax1.text(0.0, 0.75, 'regression:%s'%reg_method, fontsize=14)
        ax1.set_title('VE=%.2f, N=%.2f, L= %.2f, R=%0.2f'%(np.nanmedian(r12_reg/np.sqrt(r11_reg*r22_reg)), np.median(r12_reg),                                                   np.median(r11_reg),np.median(r22_reg)))
    else:
        ax1.set_xticks([])
        ax1.axis('off')

        ax0.set_xlabel('r(model,model),r(data,data))^0.5 OR r(data,data)')
        ax2.set_xlabel('r(model,model),r(data,data))^0.5 OR r(data,data)')

    print('Pinv: VarExp=%.3f, Num =%.3f, LHS= %.3f, RHS=%0.3f'%(np.nanmedian(r12/np.sqrt(r11*r22)),np.median(r12), np.median(r11), np.median(r22)))
    print('Reg: VarExp=%.3f, Num =%.3f, LHS= %.3f, RHS=%0.3f'%(np.nanmedian(r12_reg/np.sqrt(r11_reg*r22_reg)), np.median(r12_reg), np.median(r11_reg), np.median(r22_reg)))
    ax0.text(0.25, 1.2, '#images = %d, #features = %d, #trials = %d, Collinearity = %s, %s noise'%(ni, nf, nt, Collinearity,noise_dist), fontsize=14)

    ax2.set_title('VE=%.2f, N=%.2f, L= %.2f, R=%0.2f'%(np.nanmedian(r12_reg_sitfit/np.sqrt(r11_reg_sitfit*r22_reg_sitfit)),np.median(r12_reg_sitfit), np.median(r11_reg_sitfit),np.median(r22_reg_sitfit)))
    ax2.text(0.0,0.75,'regression-sitefit', fontsize=14)
    ax2.text(0.0,0.65,'%s'%reg_method, fontsize=14)

    print('Reg-sitefit: VarExp=%.3f, Num =%.3f, LHS= %.3f, RHS=%0.3f'%(np.nanmedian(r12_reg_sitfit/np.sqrt(r11_reg_sitfit*r22_reg_sitfit)),np.median(r12_reg_sitfit), np.median(r11_reg_sitfit), np.median(r22_reg_sitfit)))

    plt.show()
    fig.savefig(resultdir+'unit_test_%s_ni%d_nf%d_nt%d_Collin%s_%s_SB%s_%s.png'%(reg_method, ni, nf, nt, Collinearity, noise_dist, spearman_brown, corr_method_for_inv), dpi=300)
    fig.savefig(resultdir+'unit_test_%s_ni%d_nf%d_nt%d_Collin%s_%s_SB%s_%s.pdf'%(reg_method, ni, nf, nt, Collinearity, noise_dist, spearman_brown, corr_method_for_inv), dpi=300)



