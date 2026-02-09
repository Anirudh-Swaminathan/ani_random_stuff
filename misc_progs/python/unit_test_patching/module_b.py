"""
This module contains a function that calls a function from another module.
This is the function that we will be testing in our unit test.
"""
from module_a import real_function


def run_real() -> str:
    """
    A function that calls module_a.real_function
    This is the function that we will be testing in our unit test.
    We will be patching module_a.real_function to return a fake value,
    and we will check if run_real returns the fake value.

    :return: Description
    :rtype: str
    """
    return real_function()
