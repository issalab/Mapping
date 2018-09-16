# import h2o4gpu as sklearn
import scipy.stats as ss
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import OrthogonalMatchingPursuitCV
import random
import numpy as np
import scipy.io
import pickle
import pandas as pd
import time


class MappingV36:

    def __init__(self):
        print('')

    def split_objV36(self,DF_img, test_size_of_80objs=16):
        objects = np.unique(DF_img['obj']).tolist()
        test_size = test_size_of_80objs
        test_inds = []
        train_inds = []
        for obj in objects:
            inds_obj = np.where(DF_img['obj'] == obj)[0]
            inds_objV36 = [i for i in inds_obj if DF_img.iloc[i]['var'] in ['V3', 'V6']]
            for i in inds_objV36:
                if DF_img.iloc[i]['var'] == 'V0':
                    print('Buzzzz')
            #     print(len(inds_obj),len(inds_objV36))
            random.shuffle(inds_objV36)
            test_inds_obj = inds_objV36[0:test_size]
            train_inds_obj = inds_objV36[test_size:]
            test_inds.extend(test_inds_obj)
            train_inds.extend(train_inds_obj)
        return train_inds, test_inds

    def train_test_split_objV36(self,DF_img, test_size=0.2):
        test_size_of_80objs = int(80 * test_size)
        train_inds, test_inds = self.split_objV36(DF_img, test_size_of_80objs)

        train_indsV36 = []
        for i in train_inds:
            train_indsV36.extend([i - 640])

        test_indsV36 = []
        for i in test_inds:
            test_indsV36.extend([i - 640])

        train_inds, test_inds = train_indsV36, test_indsV36

        return train_inds, test_inds

    def trial_split(self, pseed):
        self.pseed = pseed
        nsplits = 2
        perm_trindx = np.random.RandomState(seed=pseed).permutation(46)
        idx = int(46 / nsplits)
        inds1, inds2 = perm_trindx[:idx], perm_trindx[idx:]
        if list(set(inds1).intersection(inds2)):
            print('Error! Shared trial in both halves!')
        return inds1, inds2

    def mean_trial_split(self,inds1, inds2, IT_trial_V36):
        if list(set(inds1).intersection(inds2)):
            print('Error! Shared trial in both halves!')
        IT_trial_V36_dum = np.swapaxes(IT_trial_V36, 0, 2)
        np.random.shuffle(IT_trial_V36_dum)
        IT_trial_V36 = np.swapaxes(IT_trial_V36_dum, 0, 2)
        half1 = IT_trial_V36[:, :, inds1].mean(2).T
        half2 = IT_trial_V36[:, :, inds2].mean(2).T
        return half1, half2

    def mapping_half_V36(self, features1, features2_half, train_inds, test_inds, reg_method,
                         reg_params=[], zscored_observations=True, return_pred=False, return_fitted_reg=False):

        X = features1
        y = features2_half

        if X.shape[0] != y.shape[0]:
            print('Error! The number of images should match.')
            print(X.shape, y.shape)

        if len(X.shape) == 2 and len(y.shape) == 2:
            X_train, X_test,y_train, y_test = \
            X[train_inds,:],X[test_inds,:],y[train_inds,:],y[test_inds,:]


        elif len(X.shape) == 2 and len(y.shape) == 1:
            X_train, X_test,y_train, y_test = X[train_inds],X[test_inds],y[train_inds],y[test_inds]

        if zscored_observations:
            y_train = self.zscored_over_images(y_train)
            y_test = self.zscored_over_images(y_test)

        if reg_method == 'ridge':
            # n_alphas,alpha1,alpha2 = reg_params[0],reg_params[1],reg_params[2]
            n_alphas = 10
            alphas = np.logspace(-6, 6, n_alphas)
            reg = RidgeCV(alphas=alphas, cv=10)
            reg.fit(X_train, y_train)
            reg_param = reg.alpha_

        elif reg_method == 'PLS':
            n_components_PLS = reg_params
            reg = PLSRegression(n_components=n_components_PLS)
            reg.fit(X_train, y_train)
            reg_param = [reg.n_components, reg.n_iter_]

        elif reg_method == 'OMP':
            reg = OrthogonalMatchingPursuitCV()
            # print(y_train)
            # # y_train = np.argmax(y_train)
            # print(y_train)
            # print(reg, X_train.shape, y_train.shape)
            reg.fit(X_train, y_train)
            reg_param = [reg.n_iter_]


        y_pred = reg.predict(X_test)

        if zscored_observations:
            y_pred = self.zscored_over_images(y_pred)

        r_train = self.r_corr(y_train, reg.predict(X_train))
        r_test = self.r_corr(y_test, y_pred)

        if len(y_test.shape) == 2:
            r_test_sites = self.r_corr_persite(y_test, y_pred)
        elif len(y_test.shape) == 1:
            r_test_sites = [r_test]


        if (return_fitted_reg == False) and (return_pred == False):
            return r_test,reg_param,r_test_sites
        elif (return_fitted_reg == False) and (return_pred == True):
            return r_test,reg_param,r_test_sites, y_pred
        elif (return_fitted_reg == True) and (return_pred == False):
            return r_test,reg_param,r_test_sites, reg
        elif (return_fitted_reg == True) and (return_pred == True):
            return r_test,reg_param,r_test_sites, y_pred, reg


    def demean(self,X):
        if len(X.shape) ==2:
            return X-np.tile(X.mean(0),(X.shape[0],1))
        elif len(X.shape) ==1:
            return X-X.mean(0)

    def r_corr(self,X,Y):
        return ss.pearsonr(np.ravel(X),np.ravel(Y))[0]

    def r_corr_persite(self, X, Y):
        r_sites = np.zeros((X.shape[1]))
        for c in range(X.shape[1]):
             r_sites[c] = ss.pearsonr(X[:, c], Y[:, c])[0]
        return r_sites

    def zscored_over_images(self,features):
        if len(features.shape) == 2:
            features = ss.zscore(features, axis=0)
        return features

    def r_corrected_consis(self, numerator, RHS, LHS):
        return numerator/(np.sqrt(LHS*RHS))

    def spearman_brown_correction(self, r):
        return 2*r/(1+r)

    def Numerator(self,train_inds, test_inds, model_features_X, half1, reg_method, reg_params,
                  zscored_observations, return_fitted_reg):

        return_pred = False

        if return_fitted_reg == False:
            r_test, reg_param, r_test_sites = self.mapping_half_V36(model_features_X, half1, train_inds, test_inds,
                                                               reg_method, reg_params, zscored_observations,
                                                               return_pred, return_fitted_reg)
            return r_test, r_test_sites

        elif return_fitted_reg:
            r_test, reg_param, r_test_sites, reg = self.mapping_half_V36(model_features_X, half1, train_inds, test_inds,
                                                                         reg_method, reg_params, zscored_observations,
                                                                         return_pred, return_fitted_reg)
            return r_test, r_test_sites, reg

    def Denom_LHS(self,train_inds, test_inds, model_features_X, half1, half2, reg_method, reg_params,
                  zscored_observations, return_fitted_reg):
        return_pred = True

        if return_fitted_reg == False:

            _, _, _, y1_pred = self.mapping_half_V36(model_features_X, half1,
                                                      train_inds, test_inds, reg_method, reg_params,
                                                      zscored_observations, return_pred, return_fitted_reg)
            _, _, _, y2_pred = self.mapping_half_V36(model_features_X, half2,
                                                      train_inds, test_inds, reg_method, reg_params,
                                                      zscored_observations, return_pred, return_fitted_reg)

            if (len(half1.shape) == 2) and (len(half2.shape) == 2):
                r_test = self.r_corr(y1_pred, y2_pred)
                r_test_sites = self.r_corr_persite(y1_pred, y2_pred)
            else:
                r_test = self.r_corr(y1_pred, y2_pred)
                r_test_sites = r_test
            return r_test, r_test_sites

        elif return_fitted_reg:
            _, _, _, y1_pred, reg1 = self.mapping_half_V36(model_features_X, half1,
                                                            train_inds, test_inds, reg_method, reg_params,
                                                            zscored_observations, return_pred, return_fitted_reg)
            _, _, _, y2_pred, reg2 = self.mapping_half_V36(model_features_X, half2,
                                                            train_inds, test_inds, reg_method, reg_params,
                                                            zscored_observations, return_pred, return_fitted_reg)

            if (len(half1.shape) == 2) and (len(half2.shape) == 2):
                r_test = self.r_corr(y1_pred, y2_pred)
                r_test_sites = self.r_corr_persite(y1_pred, y2_pred)
            else:
                r_test = self.r_corr(y1_pred, y2_pred)
                r_test_sites = r_test

            return r_test, r_test_sites, reg1, reg2

    def Denom_RHS(self, train_inds, test_inds, half1, half2):

        if (len(half1.shape) == 2) and (len(half2.shape) == 2):
            r_test = self.r_corr(half1[test_inds, :], half2[test_inds, :])
            r_test_sites = self.r_corr_persite(half1[test_inds, :], half2[test_inds, :])
        else:
            r_test = self.r_corr(half1[test_inds], half2[test_inds])
            r_test_sites = r_test

        return r_test, r_test_sites

    def Numerator_fixedmap(self, train_inds, test_inds, features1, features2_half, reg_fitted, zscored_observations):

        X = features1
        y = features2_half

        if (X.shape[0] != y.shape[0]):
            print('Error! The number of images should match.')

        if (len(X.shape) == 2) and (len(y.shape) == 2):
            X_train, X_test, y_train, y_test = \
                X[train_inds, :], X[test_inds, :], y[train_inds, :], y[test_inds, :]


        elif (len(X.shape) == 2) and (len(y.shape) == 1):
            X_train, X_test, y_train, y_test = \
                X[train_inds], X[test_inds], y[train_inds], y[test_inds]

        if zscored_observations:
            y_train = self.zscored_over_images(y_train)
            y_test = self.zscored_over_images(y_test)

        y_pred = reg_fitted.predict(X_test)

        if zscored_observations:
            y_pred = self.zscored_over_images(y_pred)

        r_train = self.r_corr(y_train, reg_fitted.predict(X_train))
        r_test = self.r_corr(y_test, y_pred)

        if len(y_test.shape) == 2:
            r_test_sites = self.r_corr_persite(y_test, y_pred)
        elif len(y_test.shape) == 1:
            r_test_sites = [r_test]

        return r_test, r_test_sites

    def Denom_LHS_fixedmap(self,train_inds, test_inds, features1, features2_half1, features2_half2, reg1_fitted, reg2_fitted, \
                           zscored_observations):

        X = features1
        y1 = features2_half1
        y2 = features2_half2

        if (X.shape[0] != y1.shape[0]):
            print('Error! The number of images should match.')

        if (len(X.shape) == 2) and (len(y1.shape) == 2):
            X_train, X_test, y1_train, y1_test, y2_train, y2_test = X[train_inds, :], X[test_inds, :], \
                                                                    y1[train_inds, :], y1[test_inds, :],\
                                                                    y2[train_inds, :],y2[test_inds,:]

        elif (len(X.shape) == 2) and(len(y2.shape) == 1):
            X_train, X_test, y1_train, y1_test, y2_train, y2_test = \
                X[train_inds], X[test_inds], y1[train_inds], y1[test_inds], y2[train_inds], y2[test_inds]

        if zscored_observations:
            y1_test = self.zscored_over_images(y1_test)
            y2_test = self.zscored_over_images(y2_test)

        y1_pred = reg1_fitted.predict(X_test)
        y2_pred = reg2_fitted.predict(X_test)

        if zscored_observations:
            y1_pred = self.zscored_over_images(y1_pred)
            y2_pred = self.zscored_over_images(y2_pred)

        if (len(y1.shape) == 2) and (len(y2.shape) == 2):
            r_test = self.r_corr(y1_pred, y2_pred)
            r_test_sites = self.r_corr_persite(y1_pred, y2_pred)
        else:
            r_test = self.r_corr(y1_pred, y2_pred)
            r_test_sites = r_test

        return r_test, r_test_sites

    def get_Neu_trial_V36(self, Neu_trial, time_interval, times):
        # gives the 46 trials of V3&6 in given time interval

        Neu_trial_V3, Neu_trial_V6 = Neu_trial[0], Neu_trial[1]

        t0, t1 = time_interval
        it0 = times.index(t0)
        it1 = times.index(t1)
        # For the sake of consistency we only use 46 first trials of both V3(51) and V6 (47)
        Neu_features = np.concatenate(
            (Neu_trial_V3[:, it0:it1, :, 0:46].mean(1), Neu_trial_V6[:, it0:it1, :, 0:46].mean(1)),
            axis=1)
        return Neu_features

    def get_consistency(self, reg_method, reg_params, model_layer, inds1, inds2,  Neu_trial, train_inds, test_inds, time_interval_fixed, flexible_times, times):

        return_fitted_reg = True
        zscored_observations = True
        n_imsplits = 1
        n_trsplits = 1

        half1, half2 = self.mean_trial_split(inds1, inds2,
                                             self.get_Neu_trial_V36(Neu_trial, time_interval_fixed, times))

        start = time.time()

        dum, dum, reg_fitted = self.Numerator(train_inds, test_inds, model_layer, half1, reg_method,
                                              reg_params, zscored_observations, return_fitted_reg)
        dum, dum, reg1_fitted, reg2_fitted = self.Denom_LHS(train_inds, test_inds, model_layer, half1, half2,
                                                            reg_method,
                                                            reg_params, zscored_observations, return_fitted_reg)
        print((time.time() - start) / 60, 'minutes')
        n_neurons = half1.shape[1]

        r_pop_fixed = np.zeros((4, len(flexible_times)))
        r_sites_fixed = np.zeros((4, n_neurons, len(flexible_times)))

        r_pop_flexible = np.zeros((4, len(flexible_times)))
        r_sites_flexible = np.zeros((4, n_neurons, len(flexible_times)))

        print((time.time() - start) / 60, 'minutes')

        for indt, t in enumerate(flexible_times):
            print(t, (time.time() - start) / 60, 'minutes')
            time_interval = [t, t + 10]
            half1, half2 = self.mean_trial_split(inds1, inds2, self.get_Neu_trial_V36(Neu_trial, time_interval, times))

            # fixed map
            r_Nom, r_Nom_sites = self.Numerator_fixedmap(train_inds, test_inds, model_layer, half1, reg_fitted,
                                                    zscored_observations)

            r_LHS, r_LHS_sites = self.Denom_LHS_fixedmap(train_inds, test_inds, model_layer, half1, half2,
                                                    reg1_fitted, reg2_fitted, zscored_observations)
            r_RHS, r_RHS_sites = self.Denom_RHS(train_inds, test_inds, half1, half2)

            r_corrected_sites = [self.r_corrected_consis(r_Nom_sites[s], r_RHS_sites[s], r_LHS_sites[s])
                                 for s in range(n_neurons)]
            r_corrected = self.r_corrected_consis(r_Nom, r_RHS, r_LHS)

            r_pop_fixed[:, indt] = [r_Nom, r_RHS, r_LHS, r_corrected]
            r_sites_fixed[0, :, indt] = r_Nom_sites
            r_sites_fixed[1, :, indt] = r_RHS_sites
            r_sites_fixed[2, :, indt] = r_LHS_sites
            r_sites_fixed[3, :, indt] = r_corrected_sites

            # flexible map

            return_fitted_reg = False
            r_Nom_fl, r_Nom_sites_fl = self.Numerator(train_inds, test_inds, model_layer, half1, reg_method,
                                                 reg_params, zscored_observations, return_fitted_reg)

            r_LHS_fl, r_LHS_sites_fl = self.Denom_LHS(train_inds, test_inds, model_layer, half1, half2, reg_method,
                                                 reg_params, zscored_observations, return_fitted_reg)

            r_corrected_fl = self.r_corrected_consis(r_Nom_fl, r_RHS, r_LHS_fl)
            r_corrected_sites_fl = [self.r_corrected_consis(r_Nom_sites_fl[s], r_RHS_sites[s], r_LHS_sites_fl[s])
                                    for s in range(n_neurons)]
            r_pop_flexible[:, indt] = [r_Nom_fl, r_RHS, r_LHS_fl, r_corrected_fl]
            r_sites_flexible[0, :, indt] = r_Nom_sites_fl
            r_sites_flexible[1, :, indt] = r_RHS_sites
            r_sites_flexible[2, :, indt] = r_LHS_sites_fl
            r_sites_flexible[3, :, indt] = r_corrected_sites_fl

        neurons_negative_denom = set(np.where(np.isnan(r_sites_fixed[3]))[0]).union(
            set(np.where(np.isnan(r_sites_flexible[3]))[0]))

        list_acceptable_neurons = list(range(n_neurons))
        [list_acceptable_neurons.remove(n) for n in neurons_negative_denom]

        return r_sites_fixed, r_sites_flexible, list_acceptable_neurons

# if __name__ == "__main__":
#     MappingV36()