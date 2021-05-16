import numpy as np


class State:
	def __init__(self, x, P):
		self.x = x
		self.P = P


class SensorModel:
	def __init__(self, P_D, lambda_c, range_c):
		if len(range_c.shape) == 1:
			V = range_c[1] - range_c[0]
		else:
			V = (range_c[0,1] - range_c[0,0]) * (range_c[1,1] - range_c[1,0])

		self.pdf_c = 1. / V
		self.P_D = P_D
		self.lambda_c = lambda_c
		self.range_c = range_c
		self.intensity_c = lambda_c * self.pdf_c


class GroundTruth:
	def __init__(self, nbirths, xstart, tbirth, tdeath, K):
		self.nbirths = nbirths
		self.xstart = xstart
		self.tbirth = tbirth
		self.tdeath = tdeath
		self.K = K


######################
# Measurement Models #
######################

class MeasModel:
	def __init__(self, dim):
		self.dim = dim

	def h(self, x):
		pass

	def H(self, x):
		pass


class CVMeasModel(MeasModel):
	def __init__(self, sigma, dim=2):
		super().__init__(dim)

		self.observation_matrix = np.array([[1, 0, 0, 0],
										    [0, 1, 0, 0]])

		self.R = sigma**2 * np.eye(2)

	def h(self, x):
		return np.dot(self.observation_matrix, x)

	def H(self, x):
		return self.observation_matrix


class CTMeasModel(MeasModel):
	def __init__(self, sigma, dim=2):
		super().__init__(dim)

		self.observation_matrix = np.array([[1, 0, 0, 0, 0],
			                                [0, 1, 0, 0, 0]])

		self.R = sigma**2 * np.eye(2)

	def h(self, x):
		return np.dot(self.observation_matrix, x)

	def H(self, x):
		return self.observation_matrix


#################
# Motion Models #
#################

class MotionModel:
	def __init__(self, T, dim):
		self.T = T
		self.dim = dim

	def f(self, x):
		pass

	def F(self, x):
		pass


class CVMotionModel(MotionModel):
	def __init__(self, T, sigma, dim=4):
		super().__init__(T, dim)

		self.motion_matrix = np.array([[1, 0, T, 0],
			                           [0, 1, 0, T],
			                           [0, 0, 1, 0],
			                           [0, 0, 0, 1]])

		self.Q = sigma**2 * np.array([[T**4/4, 0, T**3/2, 0],
			                          [0, T**4/4, 0, T**3/2],
			                          [T**3/2, 0, T**2, 0],
			                          [0, T**3/2, 0, T**2]])

	def f(self, x):
		return np.dot(self.motion_matrix, x)

	def F(self, x):
		return self.motion_matrix


class CTMotionModel(MotionModel):
	def __init__(self, T, sigmaV, sigmaOmega, dim=5):
		super().__init__(T, dim)

		self.Q = np.array([[0, 0, 0, 0, 0],
			               [0, 0, 0, 0, 0],
			               [0, 0, sigmaV**2, 0, 0],
			               [0, 0, 0, 0, 0],
			               [0, 0, 0, 0, sigmaOmega**2]])

	def f(self, x):
		return x + np.array([self.T*x[2]*np.cos(x[3]), 
							 self.T*x[2]*np.sin(x[3]), 
							 0,
							 self.T*x[4],
							 0])

	def F(self, x):
		return np.array([[1, 0, self.T*np.cos(x[3]), -self.T*x[2]*np.sin(x[3]), 0],
			             [0, 1, self.T*np.sin(x[3]), self.T*x[2]*np.cos(x[3]), 0],
			             [0, 0, 1, 0, 0],
			             [0, 0, 0, 1, self.T],
			             [0, 0, 0, 0, 1]])