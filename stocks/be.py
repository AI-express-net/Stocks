import enum
from abc import ABC, abstractmethod
from datetime import date, datetime
from stocks import identifier


def get_timestamp():
    return date.today().strftime("%Y-%m-%d:%H:%M:%S.%Z")


class BaseEntity(dict):
    class Period(enum.Enum):
        Yearly = 1
        Quarterly = 2
        Daily = 3

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
        
        # Ensure dates are properly set from database data
        if "_creation_date" in data:
            self["_creation_date"] = data["_creation_date"]
        if "_modification_date" in data:
            self["_modification_date"] = data["_modification_date"]
        
        self.change_counter += 1

    def get_creation_date(self):
        return self.get("_creation_date")

    def get_modification_date(self):
        return self.get("_modification_date")

    def set_creation_date(self, creation_date):
        self["_creation_date"] = creation_date

    def set_modification_date(self, modification_date):
        self["_modification_date"] = modification_date

    def is_data_stale(self):
        """
        Check if the data is more than a day old.
        Returns True if data should be refreshed, False otherwise.
        """
        modification_date = self.get_modification_date()
        if modification_date is None:
            return True
        
        try:
            # Parse the modification date
            if isinstance(modification_date, str):
                mod_date = datetime.strptime(modification_date, "%Y-%m-%d").date()
            else:
                mod_date = modification_date
            
            # Check if it's more than a day old
            today = date.today()
            return (today - mod_date).days > 1
        except (ValueError, TypeError):
            # If we can't parse the date, consider it stale
            return True

    # def from_json(self, json, entity):
    #     entity._id = json["_id"]
    #     entity.set_creation_date = json["creation_date"]
    #     entity.set_modification_date = json["modification_date"]
    #     entity.set_date = json["data"]

