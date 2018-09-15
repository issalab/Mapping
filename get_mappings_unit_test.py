import pickle
import numpy as np
import scipy.stats as ss
import time
from MappingV36 import MappingV36 as Mapping
Mapping = Mapping()

resultdir = '/home/tahereh/Documents/Research/Results/Mapping_unit_test/'


def get_mappings_unit_test(M, D, Data_type, Data_params, reg_method, reg_params, spearman_brown, report_sitefit, report_popfit):
    ni,nf,nt,nfoldi,nfoldt,trainfraci,splitfract, noise_dist, sds, Collinearity, corr_method_for_inv= Data_params

    r12 = np.zeros((nf, nfoldi, nfoldt))
    r11 = np.zeros((nf, nfoldi, nfoldt))
    r22 = np.zeros((nf, nfoldi, nfoldt))

    r12_reg = np.zeros((nf, nfoldi, nfoldt))
    r11_reg = np.zeros((nf, nfoldi, nfoldt))
    r22_reg = np.zeros((nf, nfoldi, nfoldt))

    r12_reg_sitfit = np.zeros((nf, nfoldi, nfoldt))
    r11_reg_sitfit = np.zeros((nf, nfoldi, nfoldt))
    r22_reg_sitfit = np.zeros((nf, nfoldi, nfoldt))

    start = time.time()
    time_sitefit = []
    time_popfit = []


    print(time.time() - start, end="")
    for fi in range(nfoldi):

        # train/test image split
        ind = np.random.permutation(ni)
        indtraini = ind[:int(ni * trainfraci)]
        indtesti = ind[int(ni * trainfraci):]

        for ft in range(nfoldt):
            # % add Gaussian noise and create two sets of trials
            if Data_type == 'HvM':
                indices = np.random.permutation(nt)
                D1 = D[:, :, indices[:int(nt * splitfract)]].mean(2)
                D2 = D[:, :, indices[int(nt * splitfract):]].mean(2)
            else:
                if noise_dist == 'normal':

                    noise1 = np.array([np.random.normal(0, sd, size=[ni, int(nt*splitfract)]) for sd in sds])
                    noise1 = np.mean(noise1, 2).T

                    noise2 = np.array([np.random.normal(0, sd, size=[ni, int(nt*splitfract)]) for sd in sds])
                    noise2 = np.mean(noise2, 2).T

                elif noise_dist == 'poisson':
                    # TODO:change poisson sds
                    noise1 = np.mean(np.random.poisson(sds[s], size=[ni, nf, int(nt * splitfract)]), 2)
                    noise2 = np.mean(np.random.poisson(sds[s], size=[ni, nf, int(nt * splitfract)]), 2)

                D1 = D + noise1
                D2 = D + noise2

            # NUMERATOR: Fit on train, test on test
            Ahat = np.dot(np.linalg.pinv(M[indtraini, :]), D1[indtraini, :])
            D1_test, D1_pred = D1[indtesti, :], np.dot(M[indtesti, :], Ahat)
            if corr_method_for_inv == 'pearson':
                r12[:, fi, ft] = [ss.pearsonr(D1_pred[:, indf], D1_test[:, indf])[0] for indf in range(nf)]
            else:
                r12[:, fi, ft] = [ss.spearmanr(D1_pred[:, indf], D1_test[:, indf])[0] for indf in range(nf)]

            # DENOMINATOR consistency between trial sets 1 & 2 on test images
            D2_test = D2[indtesti, :]
            if corr_method_for_inv == 'pearson':
                r22[:, fi, ft] = [ss.pearsonr(D1_test[:, indf], D2_test[:, indf])[0] for indf in range(nf)]
            else:
                r22[:, fi, ft] = [ss.spearmanr(D1_test[:, indf], D2_test[:, indf])[0] for indf in range(nf)]

            # DENOMINATOR LHS map consistency between trial sets 1 & 2 on test images
            Ahat1 = np.dot(np.linalg.pinv(M[indtraini, :]), D1[indtraini, :])
            Ahat2 = np.dot(np.linalg.pinv(M[indtraini, :]), D2[indtraini, :])
            lhs1, lhs2 = np.dot(M[indtesti, :], Ahat1), np.dot(M[indtesti, :], Ahat2)
            if corr_method_for_inv == 'pearson':
                r11[:, fi, ft] = [ss.pearsonr(lhs1[:, indf], lhs2[:, indf])[0] for indf in range(nf)]
            else:
                r11[:, fi, ft] = [ss.spearmanr(lhs1[:, indf], lhs2[:, indf])[0] for indf in range(nf)]

            # Regression
            train_inds, test_inds = indtraini, indtesti
            model_features_X, half1, half2 = M, D1, D2
            zscored_observations = False
            return_fitted_reg = False

            # population fit
            if report_popfit:
                start_popfit = time.time()

                if spearman_brown:
                    _, r_Nom_sites = Mapping.Nominator(train_inds, test_inds, model_features_X, np.mean([half2, half2], axis=0), reg_method,
                                                           reg_params, zscored_observations, return_fitted_reg)
                else:
                    _ , r_Nom_sites = Mapping.Nominator(train_inds, test_inds, model_features_X, half1, reg_method,
                                                           reg_params, zscored_observations, return_fitted_reg)

                _, r_RHS_sites = Mapping.Denom_RHS(train_inds, test_inds, half1, half2)
                _, r_LHS_sites = Mapping.Denom_LHS(train_inds, test_inds, model_features_X, half1, half2, reg_method, reg_params,
                                                       zscored_observations, return_fitted_reg)
                r12_reg[:, fi, ft] = r_Nom_sites

                if spearman_brown:
                    r_RHS_sites_sb = [Mapping.spearman_brown_correction(r) for r in r_RHS_sites]
                    r_LHS_sites_sb = [Mapping.spearman_brown_correction(r) for r in r_LHS_sites]

                    r22_reg[:, fi, ft] = r_RHS_sites_sb
                    r11_reg[:, fi, ft] = r_LHS_sites_sb
                else:
                    r22_reg[:, fi, ft] = r_RHS_sites
                    r11_reg[:, fi, ft] = r_LHS_sites
                time_popfit.append([time.time() - start_popfit])



            # site fit
            if report_sitefit:
                start_sitefit = time.time()
                for n in range(nf):
                    return_fitted_reg = False
                    r_Nom, _ = Mapping.Nominator(train_inds, test_inds, model_features_X, half1[:, n], reg_method, reg_params,
                                                 zscored_observations, return_fitted_reg)

                    r_LHS, _ = Mapping.Denom_LHS(train_inds, test_inds, model_features_X, half1[:, n], half2[:, n], reg_method,
                                                 reg_params, zscored_observations, return_fitted_reg)
                    r12_reg_sitfit[n, fi, ft] = r_Nom

                    r11_reg_sitfit[n, fi, ft] = r_LHS

                    if report_popfit:
                        r22_reg_sitfit[:, fi, ft] = r_RHS_sites
                    else:
                        r_RHS, _ = Mapping.Denom_RHS(train_inds, test_inds, half1[:, n], half2[:, n])
                        r22_reg_sitfit[n, fi, ft] = r_RHS
                time_sitefit.append([time.time() -start_sitefit])

    print('popfit for %s took %.2f seconds' %(reg_method, np.mean(time_popfit)))
    print('sitefit for %s took %.2f seconds' %(reg_method, np.mean(time_sitefit)))

    r12, r11, r22 = np.mean(r12, 2), np.mean(r11, 2), np.mean(r22, 2)
    r12_reg, r11_reg, r22_reg = np.mean(r12_reg, 2), np.mean(r11_reg, 2), np.mean(r22_reg, 2)
    r12_reg_sitfit, r11_reg_sitfit, r22_reg_sitfit = np.mean(r12_reg_sitfit, 2), np.mean(r11_reg_sitfit, 2), np.mean(
        r22_reg_sitfit, 2)

    data_list = [r12, r11, r22, r12_reg, r11_reg, r22_reg, r12_reg_sitfit, r11_reg_sitfit, r22_reg_sitfit]
    pickle.dump(data_list, open(resultdir + 'unit_test_%s_%s_ni%d_nf%d_nt%d_collinearity%s_%s_SB%s.pickle' % (
    reg_method, reg_params, ni, nf, nt, Collinearity,noise_dist,spearman_brown), 'wb'))

    return data_list