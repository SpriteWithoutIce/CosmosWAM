from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass

    def set_normalizer_from_stats(self, stats):
        pass
