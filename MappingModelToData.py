import numpy as np
import scipy.stats as ss
import time
from sklearn.decomposition import PCA
from MappingV36 import MappingV36 as Mapping
Mapping = Mapping()

resultdir = '/home/tahereh/Documents/Research/Results/Mapping_unit_test/'


class MappingModelToData:

    def __init__(self, M, D, PCA_ncomponents_list, explained_var_ratio_list):
        self.D = D
        self.M = M
        self.PCA_ncomponents_list = PCA_ncomponents_list
        self.explained_var_ratio_list = explained_var_ratio_list

    def get_transformed_model(self, PCA_ncomponents, explained_var_ratio):
        # PCA_ncomponents=-1 means no PCA will be applied
        # PCA_ncomponents=0 means we required a given explained_var_ratio
        # PCA_ncomponents>0 means perform PCA with PCA_ncomponents components

        if PCA_ncomponents == 0:
            ncomponents = int(self.M.shape[1]*explained_var_ratio)
            evp = 0
            while evp < explained_var_ratio:

                pca = PCA(n_components=ncomponents)
                pca.fit(self.M)
                evp = pca.explained_variance_ratio_
                ncomponents += 1
            M_PCA = pca.transform(self.M)

        elif PCA_ncomponents > 0:

            pca = PCA(n_components=PCA_ncomponents)
            pca.fit(self.M)
            evp = pca.explained_variance_ratio_
            M_PCA = pca.transform(self.M)

        elif PCA_ncomponents == -1:
            M_PCA = self.M

        return M_PCA

    def get_mappings(self, Data_params, reg_methods, reg_params_list, spearman_brown, report_sitefit, report_popfit):
        ni,nf,nt,nfoldi,nfoldt,trainfraci,splitfract, data_unit_indices = Data_params

        r12 = np.zeros((len(data_unit_indices), nfoldi, nfoldt))
        r11 = np.zeros((len(data_unit_indices), nfoldi, nfoldt))
        r22 = np.zeros((len(data_unit_indices), nfoldi, nfoldt))


        r12_reg = np.zeros((len(data_unit_indices), len(reg_methods), nfoldi, nfoldt))
        r11_reg = np.zeros((len(data_unit_indices), len(reg_methods), nfoldi, nfoldt))
        r22_reg = np.zeros((len(data_unit_indices), len(reg_methods), nfoldi, nfoldt))

        r12_reg_sitfit = np.zeros((len(data_unit_indices), len(reg_methods), nfoldi, nfoldt))
        r11_reg_sitfit = np.zeros((len(data_unit_indices), len(reg_methods), nfoldi, nfoldt))
        r22_reg_sitfit = np.zeros((len(data_unit_indices), len(reg_methods), nfoldi, nfoldt))

        for fi in range(nfoldi):

            # train/test image split
            ind = np.random.permutation(ni)
            indtraini = ind[:int(ni * trainfraci)]
            indtesti = ind[int(ni * trainfraci):]

            for ft in range(nfoldt):

                indices = np.random.permutation(nt)
                D1 = self.D[:, np.array(data_unit_indices)[:, np.newaxis], indices[:int(nt * splitfract)]].mean(2)
                D2 = self.D[:, np.array(data_unit_indices)[:, np.newaxis], indices[int(nt * splitfract):]].mean(2)

                # Pseudo-inverse
                # NUMERATOR: Fit on train, test on test
                Ahat = np.dot(np.linalg.pinv(self.M[indtraini, :]), D1[indtraini, :])
                D1_test, D1_pred = D1[indtesti, :], np.dot(self.M[indtesti, :], Ahat)

                r12[:, fi, ft] = [ss.pearsonr(D1_pred[:, indf], D1_test[:, indf])[0] for indf in range(len(data_unit_indices))]

                # DENOMINATOR consistency between trial sets 1 & 2 on test images
                D2_test = D2[indtesti, :]

                r22[:, fi, ft] = [ss.pearsonr(D1_test[:, indf], D2_test[:, indf])[0] for indf in range(len(data_unit_indices))]

                # DENOMINATOR LHS map consistency between trial sets 1 & 2 on test images
                Ahat1 = np.dot(np.linalg.pinv(self.M[indtraini, :]), D1[indtraini, :])
                Ahat2 = np.dot(np.linalg.pinv(self.M[indtraini, :]), D2[indtraini, :])
                lhs1, lhs2 = np.dot(self.M[indtesti, :], Ahat1), np.dot(self.M[indtesti, :], Ahat2)

                r11[:, fi, ft] = [ss.pearsonr(lhs1[:, indf], lhs2[:, indf])[0] for indf in range(len(data_unit_indices))]

                # Regression
                start = time.time()
                time_sitefit = []
                time_popfit = []
                print(time.time() - start, end="")

                for r, reg_method in enumerate(reg_methods):
                    reg_params = reg_params_list[r]

                    train_inds, test_inds = indtraini, indtesti
                    PCA_ncomponents = self.PCA_ncomponents_list[r]
                    explained_var_ratio = self.explained_var_ratio_list[r]
                    M_PCA = self.get_transformed_model(PCA_ncomponents, explained_var_ratio)
                    model_features_X, half1, half2 = M_PCA, D1, D2
                    zscored_observations = False
                    return_fitted_reg = False

                    # population fit
                    if report_popfit[r]:
                        start_popfit = time.time()

                        if spearman_brown:
                            _, r_Nom_sites = Mapping.Numerator(train_inds, test_inds, model_features_X, np.mean([half2, half2], axis=0), reg_method,
                                                                   reg_params, zscored_observations, return_fitted_reg)
                        else:
                            _, r_Nom_sites = Mapping.Numerator(train_inds, test_inds, model_features_X, half1, reg_method,
                                                                   reg_params, zscored_observations, return_fitted_reg)

                        _, r_RHS_sites = Mapping.Denom_RHS(test_inds, half1, half2)
                        _, r_LHS_sites = Mapping.Denom_LHS(train_inds, test_inds, model_features_X, half1, half2, reg_method, reg_params,
                                                               zscored_observations, return_fitted_reg)
                        r12_reg[:, r, fi, ft] = r_Nom_sites

                        if spearman_brown:
                            r_RHS_sites_sb = [Mapping.spearman_brown_correction(r) for r in r_RHS_sites]
                            r_LHS_sites_sb = [Mapping.spearman_brown_correction(r) for r in r_LHS_sites]

                            r22_reg[:, r,  fi, ft] = r_RHS_sites_sb
                            r11_reg[:, r, fi, ft] = r_LHS_sites_sb
                        else:
                            r22_reg[:, r, fi, ft] = r_RHS_sites
                            r11_reg[:, r, fi, ft] = r_LHS_sites
                        time_popfit.append([time.time() - start_popfit])

                    # site fit
                    if report_sitefit[r]:
                        start_sitefit = time.time()
                        for n in range(len(data_unit_indices)):
                            return_fitted_reg = False

                            r_Nom, _ = Mapping.Numerator(train_inds, test_inds, model_features_X, half1[:, n], reg_method, reg_params,
                                                         zscored_observations, return_fitted_reg)

                            r_LHS, _ = Mapping.Denom_LHS(train_inds, test_inds, model_features_X, half1[:, n], half2[:, n], reg_method,
                                                         reg_params, zscored_observations, return_fitted_reg)
                            r12_reg_sitfit[n, r, fi, ft] = r_Nom

                            r11_reg_sitfit[n, r, fi, ft] = r_LHS

                            if report_popfit[r]:
                                r22_reg_sitfit[:, r, fi, ft] = r_RHS_sites
                            else:
                                r_RHS, _ = Mapping.Denom_RHS(test_inds, half1[:, n], half2[:, n])
                                r22_reg_sitfit[n, r, fi, ft] = r_RHS
                        time_sitefit.append([time.time() -start_sitefit])

                    print('popfit for %s took %.2f seconds' %(reg_method, np.mean(time_popfit)))
                    print('sitefit for %s took %.2f seconds' %(reg_method, np.mean(time_sitefit)))

        r12, r11, r22 = np.mean(r12, 2), np.mean(r11, 2), np.mean(r22, 2)
        r12_reg, r11_reg, r22_reg = np.mean(r12_reg.mean(3), 2), np.mean(r11_reg.mean(3), 2), np.mean(r22_reg.mean(3), 2)
        r12_reg_sitfit, r11_reg_sitfit, r22_reg_sitfit = np.mean(r12_reg_sitfit.mean(3), 2), np.mean(r11_reg_sitfit.mean(3), 2), np.mean(r22_reg_sitfit.mean(3), 2)

        data_list = [r12,r11,r22,r12_reg,r11_reg,r22_reg,r12_reg_sitfit,r11_reg_sitfit,r22_reg_sitfit]

        return data_list

