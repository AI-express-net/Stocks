class DbError(Exception):
    """Exception raised for errors when calling the DB

    Attributes:
        collection_name -- collection for which he DB call caused the error
        message -- explanation of the error
    """

    def __init__(self, collection_name, operation, message):
        self.collection_name = collection_name
        self.message = "DB failed to {} for {}. Reason: {}".format(operation, collection_name, message)
        super().__init__(self.message)

    def __str__(self):
        return f'{self.collection_name} -> {self.message}'
