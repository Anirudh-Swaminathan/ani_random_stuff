"""
Unit tests for module_b functions with mocking.

WARNING: This module uses importlib.reload() as a learning exercise
to demonstrate how Python's module caching interacts with unittest.mock.patch.
DO NOT use this pattern in production code. The idiomatic approach is to
patch where the function is looked up, not where it is defined:
    @patch('module_b.real_function', return_value="fake")
"""
import importlib
from unittest.mock import patch
import module_b

@patch('module_a.real_function', return_value="fake")
def test_run_real_mocked(_mock_function):
    """
    This is a unit test for the run_real function in module_b.
    We are patching the real_function in module_a
    to return "fake" instead of "real".
    We will then call run_real and check if it returns "fake".

    We must reload module_b so that its 'from module_a import real_function'
    re-executes while the patch is active, binding to the mock.
    """
    importlib.reload(module_b)
    result = module_b.run_real()
    assert result == "fake"


def test_run_real_original():
    """
    This is a unit test for the run_real function in module_b.
    We are not patching anything, so it should return "real".

    We must reload module_b here too, in case a previous test left it
    with a stale reference to a mock that has since been cleaned up.
    """
    importlib.reload(module_b)
    result = module_b.run_real()
    assert result == "real"
