import numpy as np
from .base import Pipeline

class NewsPipeline(Pipeline):
    def __init__(self, mu=-3.2, phi=0.96, sigma=0.18, seed=42):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.last_log_val = self.mu  # Start at mean

    def get_latest_data(self) -> dict:
        noise = self.rng.normal(0.0, self.sigma)
        new_log_val = self.mu + self.phi * (self.last_log_val - self.mu) + noise
        self.last_log_val = new_log_val
        return {"news_rv": float(np.exp(new_log_val))}
