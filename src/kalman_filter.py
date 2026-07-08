import numpy as np

from dataclasses import dataclass

# Our states
@dataclass
class KalmanState:
    price: float # price right now
    trend: float # derivate

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
