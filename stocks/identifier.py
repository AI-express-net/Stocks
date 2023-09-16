# Common definitions how to manipulate an identifier.

SEPARATOR = '/'


def get_table_name(identifier):
    return identifier.split(SEPARATOR)[0]


def get_identifier(id_str, table_name):
    return table_name + SEPARATOR + id_str  # What's more efficient, this or "".join([table_name, SEPARATOR, id_str]) ?
