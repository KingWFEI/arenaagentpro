from loguru import logger
default_log = logger


def print_dict(data, data_name="", prefix="", log=default_log):
    """Prints the contents of a dictionary in a human-readable format.

    `
    Info of #data_dict:
    key1: value1
    key2: value2
    key3:
        subkey1: subvalue1
        subkey2: subvalue2
    ...
    `

    Args:
        data_dict (dict): The dictionary to be printed.
    """
    # Check if data_dict is a dictionary, or can be converted to one
    if not isinstance(data, dict):
        try:
            data_dict = data.__dict__
        except AttributeError:
            log.info(f"{data_name} is not a dictionary and has no __dict__ attribute.")
    else:
        data_dict = data
    if data_name:
        log.info(f"================Info of {data_name}================")
    for key, value in data_dict.items():
        # if value is a dictionary or can be converted to one, print it recursively
        if isinstance(value, dict) or hasattr(value, "to_dict"):
            log.info(f"{prefix}{key}:")
            dict_value = value if isinstance(value, dict) else value.to_dict()
            print_dict(dict_value, prefix=prefix + "\t", log=log)
        else:
            log.info(f"{prefix}{key}: {value}")
    if data_name:
        log.info("===============================================")


def print_list(data, data_name="", prefix="", log=default_log):
    """Prints the contents of a list in a human-readable format.

    Args:
        data (list): The list to be printed.
        data_name (str): The name of the list.
        prefix (str): The prefix to be added to each line.
        log (logging.Logger): The logger to be used.
    """
    if data_name:
        log.info(f"================Info of {data_name}================")
    for i, item in enumerate(data):
        # if item is a dictionary or can be converted to one, print it recursively
        if isinstance(item, dict) or hasattr(item, "to_dict"):
            log.info(f"{prefix}{i}:")
            dict_item = item if isinstance(item, dict) else item.to_dict()
            print_dict(dict_item, prefix=prefix + "\t", log=log)
        else:
            log.info(f"{prefix}{i}: {item}")
    if data_name:
        log.info("================Info of {data_name}================")


def get_variable_name(var):
    """Get the name of a variable as a string.

    Args:
        var: The variable whose name is to be retrieved.

    Returns:
        str: The name of the variable.
    """
    for name in globals():
        if eval(name) == var:
            return name
    return "Empty Name"
