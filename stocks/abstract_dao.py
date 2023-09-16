from abc import ABC, abstractmethod


class AbstractDao(ABC):

    @abstractmethod
    def create_key(self, identifier):
        pass

    @abstractmethod
    def get_key_field(self):
        pass

    @abstractmethod
    def save(self, entity, table_name):
        pass

    @abstractmethod
    def find(self, entity):
        pass

    @abstractmethod
    def delete(self, entity):
        pass

