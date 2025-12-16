def preprocess_project(D):
    # remove invalid rows (missing / non-numeric), remove duplicates
    # restrict to common feature set shared across all projects
    # project-wise min-max normalize features
    return X, y_or_none


def gaussian_copula_transform(X):
    # for each feature column:
    #   compute ranks
    #   map ranks to (0, 1) via u = rank / (n + 1)
    #   map u to Gaussian space via z = inv_normal_cdf(u)
    return Z


def sinkhorn_ot(mu_a, mu_b):
    # compute entropic-regularized OT cost between two empirical measures
    return ot_cost


def symmetrized_sinkhorn_divergence(mu_s, mu_t):
    # S = OT(mu_s, mu_t) - 0.5*OT(mu_s, mu_s) - 0.5*OT(mu_t, mu_t)
    ot_st = sinkhorn_ot(mu_s, mu_t)
    ot_ss = sinkhorn_ot(mu_s, mu_s)
    ot_tt = sinkhorn_ot(mu_t, mu_t)
    return ot_st - 0.5 * ot_ss - 0.5 * ot_tt


def select_source_project(candidate_sources, target_project):
    Xt, _ = preprocess_project(target_project)
    Zt = gaussian_copula_transform(Xt)
    mu_t = make_empirical_measure(Zt)  # uniform weights

    best_idx = None
    best_score = +infinity

    for k, Ds in enumerate(candidate_sources):
        Xs, _ = preprocess_project(Ds)
        Zs = gaussian_copula_transform(Xs)
        mu_s = make_empirical_measure(Zs)  # uniform weights

        score = symmetrized_sinkhorn_divergence(mu_s, mu_t)
        if score < best_score:
            best_score = score
            best_idx = k

    return best_idx


def make_empirical_measure(Z):
    # represent rows of Z as support points with uniform weights
    return mu
