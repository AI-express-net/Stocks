import pymongo
from stocks.abstract_dao import AbstractDao
from stocks.db_error import DbError
from stocks import identifier

"""
Mongo DB connection class
"""

class MongoDbDao(AbstractDao):

    KEY_FIELD = "_id"

    def __init__(self, database_name):
        self.database_name = database_name
        self.db_client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.db_client[database_name]

    def create_key(self, id_str):
        key = {self.KEY_FIELD: id_str}
        table_name = identifier.get_table_name(id_str)
        return key, table_name

    def get_key_field(self):
        return self.KEY_FIELD

    def save(self, entity):
        try:
            key, table_name = self.create_key(entity.get_id())
            collection = self.db[table_name]
            
            # Convert entity to dictionary
            entity_dict = dict(entity)
            
            # Ensure dates are properly formatted as strings
            if entity.get_creation_date():
                entity_dict["_creation_date"] = entity.get_creation_date()
            if entity.get_modification_date():
                entity_dict["_modification_date"] = entity.get_modification_date()
            
            if collection.find_one(key) is None:
                collection.insert_one(entity_dict)
            else:
                collection.delete_one(key)
                collection.insert_one(entity_dict)
        except Exception as exception:
            raise DbError(table_name, "save", str(exception))

    def find(self, identifier):
        try:
            key, table_name = self.create_key(identifier)
            collection = self.db[table_name]
            return collection.find_one(key)
        except Exception as exception:
            raise DbError(table_name, "find", str(exception))

    def delete(self, identifier):
        try:
            key, table_name = self.create_key(identifier)
            collection = self.db[table_name]
            collection.delete(key)
        except Exception as exception:
            raise DbError(table_name, "delete", str(exception))

