from itertools import product
from typing import Any, Dict, List



def dict_product(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compute the Cartesian product of dictionary values, supporting nested dictionaries.

    For a dictionary where each value is a single item, a list of items, or a nested dictionary,
    this function returns a list of flattened dictionaries representing the Cartesian product
    of all possible combinations of values. Nested dictionaries are expanded recursively, with their
    results used in the Cartesian product.

    Parameters
    ----------
    d : dict of {str: Any or list of Any or nested dict}
        The input dictionary. Each key maps to:
        - a single item;
        - a list of items;
        - or another dictionary (which will be recursively expanded).

    Returns
    -------
    : list of dict of {str: Any}
        A list of flattened dictionaries, each representing one combination from the Cartesian
        product of the input values.

    Notes
    -----
    Nested dictionaries are recursively expanded. Their result is treated as a value
    in the Cartesian product. Each nested dictionary must itself resolve to a list
    of dictionaries after recursion.

    Examples
    --------    
    >>> dict_product({'a': 1, 'b': [2, 3]})
    [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
    
    >>> dict_product({'x': {'a': [1, 2]}, 'b': [3, 4]})
    [{'x': {'a': 1}, 'b': 3}, {'x': {'a': 1}, 'b': 4}, {'x': {'a': 2}, 'b': 3}, {'x': {'a': 2}, 'b': 4}]
    """
    items = list(d.items())

    for k, v in items:
        if isinstance(v, dict):
            d[k] = dict_product(d[k])
        elif not isinstance(v, list):
            d[k] = [v]

    res = [dict(zip(d.keys(), combo)) for combo in product(*d.values())]
    return res
