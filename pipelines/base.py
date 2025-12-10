from abc import ABC, abstractmethod

class Pipeline(ABC):
    @abstractmethod
    def get_latest_data(self) -> dict:
        """Returns the latest data point from the pipeline."""
        pass
