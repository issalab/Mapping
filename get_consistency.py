
def get_consistency(model_layer, Neu_trial, train_inds, test_inds, time_interval_fixed, flexible_times, times):

    return_fitted_reg = True
    zscored_observations = True
    n_imsplits = 1
    n_trsplits = 1

    half1, half2 = self.mean_trial_split(inds1, inds2, self.get_Neu_trial_V36(Neu_trial, time_interval_fixed, times))

    start = time.time()

    dum, dum, reg_fitted = self.Numerator(train_inds, test_inds, model_layer, half1, reg_method,
                                     reg_params, zscored_observations, return_fitted_reg)
    dum, dum, reg1_fitted, reg2_fitted = self.Denom_LHS(train_inds, test_inds, model_layer, half1, half2, reg_method,
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
        r_Nom, r_Nom_sites = Numerator_fixedmap(train_inds, test_inds, model_layer, half1, reg_fitted,
                                                zscored_observations)

        r_LHS, r_LHS_sites = Denom_LHS_fixedmap(train_inds, test_inds, model_layer, half1, half2,
                                                reg1_fitted, reg2_fitted, zscored_observations)
        r_RHS, r_RHS_sites = Denom_RHS(train_inds, test_inds, half1, half2)

        r_corrected_sites = [r_corrected_consis(r_Nom_sites[s], r_RHS_sites[s], r_LHS_sites[s])
                             for s in range(n_neurons)]
        r_corrected = r_corrected_consis(r_Nom, r_RHS, r_LHS)

        r_pop_fixed[:, indt] = [r_Nom, r_RHS, r_LHS, r_corrected]
        r_sites_fixed[0, :, indt] = r_Nom_sites
        r_sites_fixed[1, :, indt] = r_RHS_sites
        r_sites_fixed[2, :, indt] = r_LHS_sites
        r_sites_fixed[3, :, indt] = r_corrected_sites

        # flexible map

        return_fitted_reg = False
        r_Nom_fl, r_Nom_sites_fl = Numerator(train_inds, test_inds, model_layer, half1, reg_method,
                                             reg_params, zscored_observations, return_fitted_reg)

        r_LHS_fl, r_LHS_sites_fl = Denom_LHS(train_inds, test_inds, model_layer, half1, half2, reg_method,
                                             reg_params, zscored_observations, return_fitted_reg)

        r_corrected_fl = r_corrected_consis(r_Nom_fl, r_RHS, r_LHS_fl)
        r_corrected_sites_fl = [r_corrected_consis(r_Nom_sites_fl[s], r_RHS_sites[s], r_LHS_sites_fl[s])
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