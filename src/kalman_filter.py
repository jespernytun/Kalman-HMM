import numpy as np

from dataclasses import dataclass

# Our states
@dataclass
class KalmanState:
    price: float # price right now
    trend: float # derivate
    P: np.ndarray # state covariance

# Dataclass representing the filter itself
# Q the performance- and R the cost matrix
@dataclass
class KalmanMatrices:
    Q: np.array
    R: float

    ## We create a function that let's us create the Q and R matrices
    @classmethod
    def create(cls, q_price: float, q_trend: float, r:float) -> "KalmanMatrices":
        q = np.array([[q_price, 0.0],
                      [0.0,     q_trend]])
        return cls(Q=q, R=r)

# Creating the filter logic, as inspired by the filterpy.kalman github repo

# Predicts the next state using the kalman equations
def predict(state: KalmanState, F: np.ndarray, noise: KalmanMatrices, dt: float) -> KalmanState:

    # If we assume that our x = [price, trend], and trend is a constant (for simplification) we get
    # x_dot = A@x,   A = [0, dt; 0, 0]
    # x_t = F*x_t-1, F = [1, dt; 0, 1]

    # if ever I find our that trend is not constant I add it as a optional argument
    if F is None:
        F = np.array([[1, dt], [0, 1]])

    # We calculate the predition
    x = np.array([state.price, state.trend])

    x_prior = F@x
    P_prior = F@state.P@F.T + noise.Q

    return KalmanState(price=x_prior[0], trend=x_prior[1], P=P_prior)

#def update(state: KalmanState, observation: float, H: np.ndarray, noise: KalmanMatrices) -> KalmanState:

#def run_kalman(prices: np.ndarray, noise: KalmanMatrices) -> KalmanResult: