from .base import Pipeline

class NewsPipeline(Pipeline):
    def get_latest_data(self) -> dict:
        raise NotImplementedError
