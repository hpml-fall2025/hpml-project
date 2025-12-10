from .base import Pipeline

class VolatilityPipeline(Pipeline):
    def get_latest_data(self) -> dict:
        raise NotImplementedError
