import numpy as np


def object_data_gen(ground_truth, motion_model, noisy):
    K = ground_truth.K
    X = [[] for _ in range(K)]
    N = [0] * K

    for i in range(ground_truth.nbirths):
        state = ground_truth.xstart[:,i]

        for k in range(ground_truth.tbirth[i], min(ground_truth.tdeath[i], K)):
            mean = motion_model.f(state)
            cov = motion_model.Q
            state = np.random.multivariate_normal(mean, cov) if noisy else mean
            X[k].append(state)
            N[k] += 1

    X = [np.array(Z).T for Z in X]
    return X, N


def meas_data_gen(X, N, sensor_model, meas_model):
    K = len(X)
    meas_data = [[] for _ in range(K)]

    for k in range(K):
        n_c = np.random.poisson(sensor_model.lambda_c)
        a = sensor_model.range_c.dot(np.array([[-1],[1]])) * np.eye(meas_model.dim)
        b = np.random.randn(meas_model.dim, n_c)
        C = np.tile(sensor_model.range_c[:,0], (1, n_c)).reshape((-1, n_c)) + a.dot(b)

        if N[k] > 0:
            index = np.random.randn(N[k]) <= sensor_model.P_D
            states = X[k][:,index]

            for i in range(states.shape[1]):
                mean = meas_model.h(states[:,i])
                cov = meas_model.R
                meas = np.random.multivariate_normal(mean, cov)
                meas_data[k].append(meas)

        if len(meas_data[k]):
            meas_data[k] = np.array(meas_data[k]).T
            meas_data[k] = np.hstack((meas_data[k], C))
        else:
            meas_data[k] = C

    return meas_data


def norm_log_weights(log_w):
    if log_w.size == 1:
        log_sum_w = log_w[0]
    else:
        idx = np.argsort(log_w)[::-1]
        log_w_aux = log_w[idx]
        maxi = log_w_aux[0]
        log_sum_w = maxi + np.log(1 + np.sum(np.exp(log_w_aux[1:] - maxi)))

    return log_w - log_sum_w, log_sum_w


def prune_hyp(w, hyp, w_min):
    idx = w >= w_min
    return w[idx], hyp[idx]


def cap_hyp(w, hyp, M):
    if w.size <= M:
        return w, hyp

    idx = np.argsort(w)[::-1]
    return w[idx][:M], hyp[idx][:M]


def RMSE(estimates, ground_truth):
    N = len(estimates)
    m = estimates[0].size
    rmse = 0

    for i in range(N):
        diff = estimates[i] - ground_truth[i][:,0]
        rmse += np.dot(diff, diff)

    return np.sqrt(rmse / (N * m))
