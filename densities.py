import numpy as np
from scipy.stats import multivariate_normal
from models import State
from utils import norm_log_weights


class GaussianDensity:
	def __init__(self, motion_model, meas_model, gating, merge_thresh):
		self.motion_model = motion_model
		self.meas_model = meas_model
		self.gating = gating
		self.merge_thresh = merge_thresh

	def predict(self, state):
		F = self.motion_model.F(state.x)
		x = self.motion_model.f(state.x)
		P = np.dot(np.dot(F, state.P), F.T) + self.motion_model.Q
		return State(x, P)

	def update(self, state, z):
		H = self.meas_model.H(state.x)
		S = np.dot(np.dot(H, state.P), H.T) + self.meas_model.R
		S = (S + S.T) / 2.
		S_inv = np.linalg.inv(S)
		K = np.dot(np.dot(state.P, H.T), S_inv)
		I = np.eye(self.motion_model.dim) - np.dot(K, H)
		x = state.x + np.dot(K, z - self.meas_model.h(state.x))
		P = np.dot(I, state.P)
		return State(x, P)

	def ellips_gating(self, state, z):
		n, m = z.shape
		mean = self.meas_model.h(state.x)
		z_ingate = []

		H = self.meas_model.H(state.x)
		S =	np.dot(np.dot(H, state.P), H.T) + self.meas_model.R
		S = (S + S.T) / 2.
		S_inv = np.linalg.inv(S)

		for i in range(m):
			dz = z[:,i] - mean

			if np.dot(np.dot(dz, S_inv), dz) <= self.gating:
				z_ingate.append(z[:,i])

		return np.array(z_ingate).T

	def pred_likelihood(self, state, z_ingate):
		mean = self.meas_model.h(state.x)
		H = self.meas_model.H(state.x)
		S = np.dot(np.dot(H, state.P), H.T) + self.meas_model.R
		S = (S + S.T) / 2.
		preds = np.log(multivariate_normal.pdf(z_ingate.T, mean=mean, cov=S))
		return preds if z_ingate.shape[1] > 1 else np.array([preds])
		
	def moment_matching(self, w, hyp):
		if w.size == 1:
			state = hyp[0]
		else:
			w = np.exp(w)
			m = w.size
			n = self.motion_model.dim
			state = State(x=np.zeros(n), P=np.zeros((n, n)))

			for i in range(m):
				state.x += w[i] * hyp[i].x

			for i in range(m):
				diff = state.x - hyp[i].x
				state.P += w[i] * (hyp[i].P + np.outer(diff, diff.T))

		return state

	def mixture_reduction(self, w, hyp):
		if w.size == 1:
			return w, hyp

		indices = set(range(w.size))
		w_hat = []
		hyp_hat = []

		while indices:
			ij = set()
			j = np.argmax(w)
			mask = np.array([False] * w.size)

			for i in indices:
				temp = hyp[i].x - hyp[j].x
				ls, _, _, _ = np.linalg.lstsq(hyp[j].P, temp, rcond=None)
				val = np.dot(temp, ls)

				if val < self.merge_thresh:
					ij.add(i)
					mask[i] = True

			indices -= ij
			temp_w, temp_W = norm_log_weights(w[mask])
			temp_hyp = self.moment_matching(temp_w, hyp[mask])
			w_hat.append(temp_W)
			hyp_hat.append(temp_hyp)
			w[mask] = np.log(2.2204e-16)

		return np.array(w_hat), np.array(hyp_hat, dtype=np.object)