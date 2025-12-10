import numpy as np
from .base import Pipeline

class VolatilityPipeline(Pipeline):
    def __init__(self, mu=-3.0, phi=0.95, sigma=0.15, seed=43):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.last_log_val = self.mu

    def get_latest_data(self) -> dict:
        noise = self.rng.normal(0.0, self.sigma)
        new_log_val = self.mu + self.phi * (self.last_log_val - self.mu) + noise
        self.last_log_val = new_log_val
        return {"har_rv": float(np.exp(new_log_val))}
