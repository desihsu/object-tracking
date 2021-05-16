import numpy as np
from utils import norm_log_weights, prune_hyp, cap_hyp
from models import State


class SingleObjectTracker:
    def __init__(self, density, sensor_model, motion_model, meas_model, w_min, M):
        self.density = density
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.meas_model = meas_model
        self.w_min = np.log(w_min)
        self.M = M

    def nearest_neighbor(self, state, Z):
        N = len(Z)
        P_D = self.sensor_model.P_D
        int_c = self.sensor_model.intensity_c
        estimates = []

        for i in range(N):
            z_ingate, meas_in_gate = self.density.ellips_gating(state, Z[i])

            if z_ingate.size:
                likelihood = self.density.pred_likelihood(state, z_ingate)
                index = np.argmax(likelihood)
                w_NN = likelihood[index] + np.log(P_D / int_c)

                if w_NN >= np.log(1 - P_D):
                    state = self.density.update(state, z_ingate[:,index])

            estimates.append(state.x)
            state = self.density.predict(state)

        return estimates
    
    def prob_data_assoc(self, state, Z):
        N = len(Z)
        P_D = self.sensor_model.P_D
        int_c = self.sensor_model.intensity_c
        estimates = []

        for i in range(N):
            z_ingate, meas_in_gate = self.density.ellips_gating(state, Z[i])
            preds = self.density.pred_likelihood(state, z_ingate) if z_ingate.size else np.array([])
            m = preds.size
            w = np.zeros(m + 1)
            hyp = np.array([None] * (m + 1), dtype=np.object)

            hyp[m] = State(x=state.x, P=state.P)
            w[:m] = preds + np.log(P_D / int_c)
            w[m] = np.log(1 - P_D)
            w, W = norm_log_weights(w)

            for j in range(m):
                hyp[j] = self.density.update(state, z_ingate[:,j])

            w, hyp = prune_hyp(w, hyp, self.w_min)
            w, W = norm_log_weights(w)
            state = self.density.moment_matching(w, hyp)
            estimates.append(state.x)
            state = self.density.predict(state)

        return estimates

    def gaussian_sum(self, state, Z):
        N = len(Z)
        P_D = self.sensor_model.P_D
        int_c = self.sensor_model.intensity_c
        estimates = []
        w, W = norm_log_weights(np.array([np.log(1)]))
        hyp = np.array([State(state.x, state.P)], dtype=np.object)

        for k in range(N):
            H = w.size
            w_h = []
            hyp_h = []

            for i in range(H):
                z_ingate, meas_in_gate = self.density.ellips_gating(hyp[i], Z[k])
                preds = self.density.pred_likelihood(hyp[i], z_ingate) if z_ingate.size else np.array([])
                m = preds.size
                w_h.append(w[i] + np.log(1 - P_D))
                hyp_h.append(State(hyp[i].x, hyp[i].P))
                
                for j in range(m):
                    state_upd = self.density.update(hyp[i], z_ingate[:,j])
                    w_h.append(w[i] + preds[j] + np.log(P_D / int_c))
                    hyp_h.append(state_upd)

            w = np.array(w_h)
            hyp = np.array(hyp_h, dtype=np.object)

            w, W = norm_log_weights(w)
            w, hyp = prune_hyp(w, hyp, self.w_min)
            w, W = norm_log_weights(w)

            w, hyp = self.density.mixture_reduction(w, hyp)
            w, hyp = cap_hyp(w, hyp, self.M)

            w, W = norm_log_weights(w)
            idx = np.argmax(w)
            estimates.append(hyp[idx].x)

            for i in range(w.size):
                hyp[i] = self.density.predict(hyp[i])

        return estimates
        