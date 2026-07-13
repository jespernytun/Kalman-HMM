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
def predict(state: KalmanState, A: np.ndarray, noise: KalmanMatrices, dt: float) -> KalmanState:

    # If we assume that our x = [price, trend], and trend is a constant (for simplification) we get
    # x_dot = A@x,   A = [0, dt; 0, 0]
    # x_t = F*x_t-1, F = [1, dt; 0, 1]

    # if ever I find our that trend is not constant I add it as a optional argument
    # We keep engineering syntax, A=F and C=H, just so I don't confuse myself
    if A is None:
        A = np.array([[1, dt], [0, 1]])

    # We calculate the predition
    x = np.array([state.price, state.trend])

    x_prior = A@x
    P_prior = A@state.P@A.T + noise.Q

    return KalmanState(price=x_prior[0], trend=x_prior[1], P=P_prior)

# Takes on a new mesurement for our Kalman filter
def update(state: KalmanState, observation: float, noise: KalmanMatrices, dt: float) -> KalmanState:

    # We only observe price directly, trend is inferred
    C = np.array([[1.0, 0.0]])

    x_prior = np.array([state.price, state.trend])
    P_prior = state.P

    # Residual: difference between the measurement and our prediction
    y = observation - C@x_prior

    # Combined incertainty of obeservation and mesurement (R)
    S = C@P_prior@C.T + noise.R

    # Kalman gain
    # We compare the two incertainties to know which value to trust the most
    K = P_prior@C.T@np.linalg.inv(S)

    x_post = x_prior + (K@y)
    P_post = (np.eye(2) - K@C)@P_prior

    return KalmanState(price=x_post[0], trend=x_post[1], P=P_post)

# Runs predict/update over a full price series, one measurement at a time
# Returns a KalmanState whose fields hold the full trajectory (arrays) rather than a single instant
def run_kalman(prices: np.ndarray, noise: KalmanMatrices, dt: float = 1.0, A: np.ndarray = None, P0: np.ndarray = None) -> KalmanState:

    n = len(prices)

    # We start on the first observed price with no known trend
    # and a wide initial uncertainty since we have no prior belief yet
    if P0 is None:
        P0 = np.eye(2) * 1.0

    state = KalmanState(price=prices[0], trend=0.0, P=P0)

    price_filt = np.empty(n)
    trend_filt = np.empty(n)
    P_filt = np.empty((n, 2, 2))

    price_filt[0] = state.price
    trend_filt[0] = state.trend
    P_filt[0] = state.P

    for t in range(1, n):
        state = predict(state, A, noise, dt)
        state = update(state, prices[t], noise, dt)

        price_filt[t] = state.price
        trend_filt[t] = state.trend
        P_filt[t] = state.P

    return KalmanResult(price=price_filt, trend=trend_filt, P=P_filt)