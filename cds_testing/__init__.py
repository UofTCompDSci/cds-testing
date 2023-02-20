from __future__ import annotations
from typing import Any, Callable
from notebook_helper import importer
import pytest
import pandas as pd
import numpy as np

def load_code(request):
    cells = importer.get_cells(request.param)
    for cell in cells:
        cell.run(raise_on_error=False)


def make_variable_names_test(hw, expected_vars: dict[str, dict]) -> Callable:
    """ Returns a function that tests that the hw created by the students contains the expected
    variable name.
    """
    expected_var_names = expected_vars.keys()

    @pytest.mark.parametrize("expected_var_name", expected_var_names)
    def test_variable_name(expected_var_name: str):
        msg = f"ISSUE FOUND: The required variable name \"{expected_var_name}\" is missing from your submission.\n"

        assert hasattr(hw, expected_var_name), msg

    return test_variable_name


def assert_type(var_name, var_value, expected_value):
    """ Verifies that the students answer is of the correct type."""
    msg = f"ISSUE FOUND: Your variable {var_name} has type {type(var_value)} but should have type {type(expected_value)}.\n"
    assert isinstance(var_value, type(expected_value)), msg


def make_answer_equality_test(hw, soln, expected_vars: dict[str, dict[str, bool]]) -> Callable:
    """Returns a function that tests that the values of variables created by the student match the
    expected value from the sample solutions.
    """
    parameters = [(hw, soln, var, expected_vars[var]) for var in expected_vars]

    @pytest.mark.parametrize("student_hw,soln_nb,var_name,args", parameters)
    def test_answer_equality(student_hw, soln_nb, var_name, args):
        msg = f"ISSUE FOUND: The required variable name \"{var_name}\" is missing from your submission.\n"
        assert hasattr(student_hw, var_name), msg

        student_value = getattr(student_hw, var_name)
        soln_value = getattr(soln_nb, var_name)

        # If either variable is of numpy generic type, then we can skip the type check
        # and go straight to the equality check. As some of the values in the hw/soln may be correct
        # but the difference might be <int> vs <numpy.int64>.
        if not(isinstance(student_value, np.generic) or isinstance(soln_value, np.generic)):
            assert_type(var_name, student_value, soln_value)

        if isinstance(soln_value, pd.Series):
            pd.testing.assert_series_equal(student_value, soln_value, obj=var_name, **args)
        elif isinstance(soln_value, pd.DataFrame):
            pd.testing.assert_frame_equal(student_value, soln_value, obj=var_name, **args)
        elif isinstance(soln_value, list):
            assert_list_equality(student_value, soln_value, var_name)
        else:
            msg = (
                f"ISSUE FOUND: The value of your variable {var_name} does not match the expected value,\n\n"
                f"\n"
                f"Yours:     {student_value}"
                f"\n"
                f"Expected:     {soln_value}"
                f"\n\n"
                f"In case it helps, your variable {var_name} has type {type(student_value).__name__}.\n"
            )

            if isinstance(soln_value, float):
                soln_value = pytest.approx(soln_value, **args)
            assert student_value == soln_value, msg

    return test_answer_equality

def assert_list_equality(student_value, soln_value, var_name: str):
    """
    Utility function to test the equality of lists. If the lists are large, then only
    display the first value that is incorrect.
    """
    msg = (
        f"ISSUE FOUND: Your list variable {var_name} has a length of {len(student_value)} "
        f"but should have a length of {len(soln_value)}.\n"
    )
    assert len(student_value) == len(soln_value), msg

    # For smaller size lists, we can display the entire list when showing the difference
    if len(soln_value) < 25:
        msg = (
            f"ISSUE FOUND: The value of your list variable {var_name} does not match the expected value.\n\n"
            f"Yours:\n\n"
            f"     {student_value}"
            f"\n\n"
            f"Expected:\n\n"
            f"     {soln_value}\n"
        )
        assert student_value == soln_value, msg
    else:
        # If the lists are large, then only search for the first error instead of
        # displaying the entire list difference
        incorrect_index = -1
        for i in range(len(student_value)):
            if student_value[i] != soln_value[i]:
                incorrect_index = i
                break
        msg = (
            f"ISSUE FOUND: The value of your list variable {var_name} is not equal to what we expect.\n"
            f"The first incorrect value in your list occurred at index {incorrect_index}, "
            f"that is, at {var_name}[{incorrect_index}].\n\n"
            f"Yours:\n\n"
            f"     {student_value[incorrect_index]}"
            f"\n\n"
            f"Expected:\n\n"
            f"     {soln_value[incorrect_index]}"
            f"\n"
        )
        assert incorrect_index == -1, msg


# Example functions (not to be used in final version)

# @pytest.fixture
# def test_all_code_cells(load_code):
#     error_cells_str = f'{"-" * 50}\n'.join(cell._source for cell in error_cells)
#     msg = f'''
# ISSUE FOUND: Some of your cells didn't have valid code,
# either because of a mistake in this code or because of an
# earlier error that this code relies on.

# You should re-run all the Code cells in your notebook to
# help you debug the issue. Here is the problematic code,
# each cell separated by a line of hyphens:

# {error_cells_str}
# '''

#     assert len(error_cells) == 0, msg
