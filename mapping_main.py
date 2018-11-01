import pickle
import time
import numpy as np
import MappingV36
from MappingV36 import MappingV36 as Mapping
from ReadMeta import ReadMeta
from ReadData import ReadData
from ReadModel import ReadModel

import h5py


host = 'SYNPAI'  # 'habanero'
if host == 'habanero':
    resultdir = '/rigel/issa/users/Tahereh/Results/'
    neuraldir = '/rigel/issa/users/Tahereh/models/neural_features/'
    modeldir = '/rigel/issa/users/Tahereh/models/'
    datadir = '/rigel/issa/users/Tahereh/Data/DiCarlo/'
elif host == 'Mac':
    resultdir = '/Users/tahereh/Documents/Results/invertibility/error coding in visual cortex/'
    neuraldir = '/Users/tahereh/Documents/Results/invertibility/error coding in visual cortex/models/neural_features/'
    modeldir = '/Users/tahereh/Documents/Results/invertibility/error coding in visual cortex/models/'
    datadir = '/Users/Tahereh/Documents/Data/DiCarlo'
elif host == 'SYNPAI':
    resultdir = '/home/tahereh/Documents/Research/Results/Neural-Dynamics/'
    neuralfeaturesdir = '/home/tahereh/Documents/Research/features/neural_features/'
    modeldir = '/home/tahereh/Documents/Research/features/'
    datadir = '/home/tahereh/Documents/Research/Data/DiCarlo/'
    # %pylab inline
    # pylab.rcParams['xtick.color'] = 'white'
    # pylab.rcParams['ytick.color'] = 'white'
    # # pylab.rcParams['figsize'] = [6,4]



Mapping = Mapping()


# Read Meta
Meta = ReadMeta(neuralfeaturesdir)
DF_img = Meta.get_DF_img()
DF_neu = Meta.get_DF_neu()
times = Meta.get_times()

# Read Neural data
# Data = ReadData(datadir, DF_neu)
# IT, V4 = Data.get_data()



# get_consistencies
region = 'IT'
time_interval_fixed = [70, 130]
flexible_times = np.arange(70, 280, 10)
itr = 0
pseed = itr

# trial_split = Mapping.trial_split
# inds1, inds2 = trial_split(pseed)
# mean_trial_split = Mapping.mean_trial_split
# get_Neu_trial_V36 = Mapping.get_Neu_trial_V36
#
# half1, half2 = mean_trial_split(inds1, inds2, get_Neu_trial_V36(IT[1:], time_interval_fixed, times))
# train_test_split_objV36 = Mapping.train_test_split_objV36
# test_size = 0.2
# train_inds, test_inds = train_test_split_objV36(DF_img, test_size)
#
# Neu_trial = IT[1:]
# Neu_features = Mapping.get_Neu_trial_V36(Neu_trial, time_interval_fixed, times)
# Neu_features = Neu_features.swapaxes(0,1)
#
# h5 = h5py.File(resultdir+ 'IT_standard.h5','w')
# h5.create_dataset('IT', data=Neu_features)
# h5.close()

hf = h5py.File(resultdir+ 'IT_standard.h5', 'r')
Neu_features = np.array(hf.get('IT'))
hf.close()

n_components_PCA = 500
folder_name = 'HvM_forward_features/'
# Model 1
# Read Model
model_layer1 = 'conv5'
Model = ReadModel(model_layer1, modeldir, True, n_components_PCA)
model1 = Model.get_model()[640:,:]


# filename = modeldir+'%s/500PCA/%dcmps_%s.h5'%(folder_name, n_components_PCA, model_layer1)
# h5f = h5py.File(filename, 'w')
# h5f.create_dataset('model_layer', data=model1)
# h5f.close()

start_time = time.time()
# [0,4,8,12,16]#np.random.permutation(20)[0:5]
D = Neu_features
M = model1

ni = Neu_features.shape[0]  # # of features
nf = Neu_features.shape[1]
nt = Neu_features.shape[2]

trainfraci = 0.8  # image trainfrac
splitfract = 0.5  # trial splitfrac
nfoldi = 1
nfoldt = 1

data_unit_indices = range(nf)
purpose_of_this_run = 'testpinvsite'  #  'masterplotwithpinv'
n_components_range = [5, 12, 25, 35, 50,80,100,128, 168]

for nc in n_components_range:
    print(nc)
    # ------------------------------------------------------------
    # Regression Parameters
    # ------------------------------------------------------------

    spearman_brown = False

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

    reg_methods = ['PLS', 'ridge', 'ridge']
    reg_params_list = [nc, [20, -10, 10], [20, -10, 10]]  # for ridge [n_alpha, alpha0, alpha1]
    report_popfit = [True, True, True]  # [False, True, True]
    report_sitefit = [False, False, True]  # [False, False, False, False]  # [True, True, True]

    # ------------------------------------------------------------
    # Map Parameters
    # ------------------------------------------------------------
    # PCA_ncomponents = -1 means no PCA will be applied on the model,
    # PCA_ncomponents = 0 means refer to the explained_var_ratio to calculate the number of components for PCA
    PCA_ncomponents_list = [-1, nc, -1]  # The first one is for pinv and the rest for the regressions
    explained_var_ratio_list = [0, 0, 0]
    # ------------------------------------------------------------
    #
    # ------------------------------------------------------------

    from MappingModelToData import MappingModelToData as MappingModelToDataClass

    MappingUnitTest = MappingModelToDataClass(M, D, PCA_ncomponents_list, explained_var_ratio_list)

    Data_params = [ni, nf, nt, nfoldi, nfoldt, trainfraci, splitfract, data_unit_indices]

    data_list = MappingUnitTest.get_mappings(Data_params, reg_methods, reg_params_list, spearman_brown, report_sitefit,
                                             report_popfit)

    pickle.dump(data_list, open(resultdir + 'MappingMtoD_%s_%s_%s, ni%d_nf%d_nt%d_SB%s_%dcmp_%s.pickle' % (
        reg_methods, reg_params_list, PCA_ncomponents_list, ni, nf, nt, spearman_brown, nc, purpose_of_this_run), 'wb'))

print(time.time()-start_time)