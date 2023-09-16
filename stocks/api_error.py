class ApiError(Exception):
    """Exception raised for errors when calling the REST API

    Attributes:
        api_name -- api call which caused the error
        message -- explanation of the error
    """

    def __init__(self, api_name, message):
        self.api_name = api_name
        self.message = "API call failed. Reason: {}".format(message)
        super().__init__(self.message)

    def __str__(self):
        return f'{self.api_name} -> {self.message}'
