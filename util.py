from datetime import datetime


def days_between(d1, d2):
    """
    get the days interval between two dates
    :param d1: date1
    :param d2: date2
    :return: days interval
    """
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return abs((d2 - d1).days)


def remove_duplicates(values):
    """
    remove duplicate value in a list
    :param values: input list
    :return: no duplicate entry list
    """
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def getPrice(price):
    """
    Get the numeric price in a string format, which contains currency symbol
    :param price:
    :return:
    """
    price = float( filter( lambda x: x in '0123456789.', price) )
    return price