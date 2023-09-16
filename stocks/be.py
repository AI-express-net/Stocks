import enum
from abc import ABC, abstractmethod
from datetime import date
from stocks import identifier


def get_timestamp():
    return date.today().strftime("%Y-%m-%d:%H:%M:%S.%Z")


class BaseEntity(dict):
    class Period(enum.Enum):
        Yearly = 1
        Quarterly = 2

    def __init__(self, name):
        self.name = name
        self["data"] = {}
        self["_id"] = identifier.get_identifier(name, self.get_table_name())
        self.change_counter = 0

    @abstractmethod
    def get_table_name(self):
        pass

    def get_id(self):
        return self["_id"]

    def get_data(self):
        return self["data"]

    def set_api_data(self, data):
        self["data"] = data
        self.change_counter += 1

    def set_db_data(self, data):
        for k in list(self.keys()):
            self[k] = data[k]
        self.change_counter += 1

    def get_creation_date(self):
        return self["_creation_date"]

    def get_modification_date(self):
        return self["_modification_date"]

    # def from_json(self, json, entity):
    #     entity._id = json["_id"]
    #     entity.set_creation_date = json["creation_date"]
    #     entity.set_modification_date = json["modification_date"]
    #     entity.set_date = json["data"]

