import pickle
import numpy as np
from DataMapModel import DataMapModel
from MappingUnitTest import MappingUnitTest

#

resultdir = '/home/tahereh/Documents/Research/Results/Mapping_unit_test/'
neuralfeaturesdir = '/home/tahereh/Documents/Research/features/neural_features/'
datadir = '/home/tahereh/Documents/Research/Data/DiCarlo/'

# ------------------------------------------------------------
# Data Parameters
# ------------------------------------------------------------
imag_feat_ratio = 10 # 10, 100
nf = 20
ni = imag_feat_ratio*nf  # # of features
nt = 46

trainfraci = 0.8  # image trainfrac
splitfract = 0.5  # trial splitfrac
nfoldi = 5
nfoldt = 5
noisy_map = False
various_unit_stds = True
Collinearity = True
noise_dist = 'normal'

if various_unit_stds:
    sds = np.logspace(-1, 1, num=int(nf))  # np.arange(0.5, 10, 1)
else:
    sds = np.ones((nf))*sd

Data_type = 'synthetic'  # 'synthetic'#'HvM'


# ------------------------------------------------------------
# Regression Parameters
# ------------------------------------------------------------

spearman_brown = False
corr_method_for_inv = 'pearson'  # 'spearman'


# regularization parameters

# if reg_method == 'PLS':
#     n_components = nf
#     reg_params = n_components
#     report_popfit = True
#
# elif reg_method == 'ridge':
#     reg_params = []
#     report_popfit = True
#
# elif reg_method == 'OMP':
#     reg_params = []
#     report_popfit = False
#     report_sitefit = True

reg_methods = ['OMP', 'PLS', 'ridge', 'ridge']
reg_params_list = [[], 10, [10, -6, 6], [10, -6, 6]]
report_popfit = [False, True, True, True]  # [False, True, True]
report_sitefit = [False, False, False, False]  # [True, True, True]


# ------------------------------------------------------------
# Map Parameters
# ------------------------------------------------------------
# PCA_ncomponents = 0 means no PCA will be applied on the model
PCA_ncomponents_list = [0,0, 0, 0, 10]  # The first one is for pinv and the rest for the regressions

# ------------------------------------------------------------
#
# ------------------------------------------------------------
DataMapModel = DataMapModel(ni, nf, nt)

if Data_type == 'HvM':
    D = DataMapModel.get_HvM()
    Dmu = D.mean(2)

elif Data_type == 'synthetic':
    D, Dtruth = DataMapModel.get_syntheic(sds, splitfract, Collinearity, noise_dist)

    if noisy_map:
        Dmu = D.mean(2)
    else:
        Dmu = Dtruth

A = np.random.rand(nf, nf)

MappingUnitTest = MappingUnitTest(D, Dmu,  A, PCA_ncomponents_list)

Data_params = [ni, nf, nt, nfoldi, nfoldt, trainfraci, splitfract, noise_dist,  sds, Collinearity, various_unit_stds, corr_method_for_inv, noisy_map]

data_list = MappingUnitTest.get_mappings_unit_test(Data_params, reg_methods, reg_params_list, spearman_brown, report_sitefit, report_popfit)

pickle.dump(data_list, open(resultdir + 'unit_test_%s_%s_%s, ni%d_nf%d_nt%d_collinearity%s_%s_SB%s_VariousSd%s_noisymap%s.pickle' % (
    reg_methods, reg_params_list, PCA_ncomponents_list , ni, nf, nt, Collinearity, noise_dist, spearman_brown, various_unit_stds, noisy_map), 'wb'))