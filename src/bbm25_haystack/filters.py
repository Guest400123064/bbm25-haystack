# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from functools import wraps
from typing import Any, Callable, Final, Optional

import pandas as pd
from haystack.dataclasses import Document
from haystack.errors import FilterError


def apply_filters_to_document(
    filters: Optional[dict[str, Any]], document: Document
) -> bool:
    """
    Apply filters to a document.

    :param filters: The filters to apply to the document.
    :type filters: dict[str, Any]
    :param document: The document to apply the filters to.
    :type document: Document

    :return: True if the document passes the filters.
    :rtype: bool
    """
    if filters is None or not filters:
        return True
    return _run_comparison_condition(filters, document)


def _get_document_field(document: Document, field: str) -> Optional[Any]:
    """
    Get the value of a field in a document.

    If the field is not found within the document then, instead of
    raising an error, `None` is returned. Note that here we do not
    implicitly add 'meta' prefix for fields that are not a direct
    attribute of the document, not supporting legacy behavior anymore.

    :param document: The document to get the field value from.
    :type document: Document
    :param field: The field to get the value of.
    :type field: str

    :return: The value of the field in the document.
    :rtype: Optional[Any]
    """
    if "." not in field:
        return getattr(document, field)

    attr = document.meta
    for f in field.split(".")[1:]:
        attr = attr.get(f)
        if attr is None:
            return None
    return attr


def _run_logical_condition(condition: dict[str, Any], document: Document) -> bool:
    if "operator" not in condition:
        msg = "Logical condition must have an 'operator' key."
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = "Logical condition must have a 'conditions' key."
        raise FilterError(msg)

    conditions = condition["conditions"]
    reducer = LOGICAL_OPERATORS[condition["operator"]]

    return reducer(document, conditions)


def _run_comparison_condition(condition: dict[str, Any], document: Document) -> bool:
    if "field" not in condition:
        return _run_logical_condition(condition, document)

    if "operator" not in condition:
        msg = "Comparison condition must have an 'operator' key."
        raise FilterError(msg)
    if "value" not in condition:
        msg = "Comparison condition must have a 'value' key."
        raise FilterError(msg)

    field: str = condition["field"]
    value: Any = condition["value"]
    comparator = COMPARISON_OPERATORS[condition["operator"]]

    return comparator(_get_document_field(document, field), value)


def _and(document: Document, conditions: list[dict[str, Any]]) -> bool:
    """
    Return True if all conditions are met.

    :param document: The document to check the conditions against.
    :type document: Document
    :param conditions: The conditions to check against the document.
    :type conditions: list[dict[str, Any]]

    :return: True if not all conditions are met.
    :rtype: bool
    """
    return all(
        _run_comparison_condition(condition, document) for condition in conditions
    )


def _or(document: Document, conditions: list[dict[str, Any]]) -> bool:
    """
    Return True if any condition is met.

    :param document: The document to check the conditions against.
    :type document: Document
    :param conditions: The conditions to check against the document.
    :type conditions: list[dict[str, Any]]

    :return: True if not all conditions are met.
    :rtype: bool
    """
    return any(_run_comparison_condition(cond, document) for cond in conditions)


def _not(document: Document, conditions: list[dict[str, Any]]) -> bool:
    """
    Return True if not all conditions are met.

    The 'NOT' operator is under-specified when supplied with a
    set of conditions instead of a single condition. Because we
    can have the semantics of 'at least one False' versus
    'all False'. Here we choose to comply with the official
    implementation of Haystack (the 'at least one False' semantics).

    :param document: The document to check the conditions against.
    :type document: Document
    :param conditions: The conditions to check against the document.
    :type conditions: list[dict[str, Any]]

    :return: True if not all conditions are met.
    :rtype: bool
    """
    return not _and(document, conditions)


def _check_comparator_inputs(
    comparator: Callable[[Any, Any], bool]
) -> Callable[[Any, Any], bool]:
    """
    A decorator to check and preprocess input attribute values.

    ALL COMPARISON OPERATORS SHOULD BE WRAPPED WITH THIS DECORATOR.
    because a `False` may be returned by both input validation and
    the actual comparison. This decorator ensures that the comparison
    function is only called if the input values are valid.

    :param comparator: The comparator function to wrap.
    :type comparator: Callable[[Any, Any], bool]

    :return: The wrapped comparator function.
    :rtype: Callable[[Any, Any], bool]
    """

    @wraps(comparator)
    def _wrapper(dv: Any, fv: Any) -> bool:

        # I think allowing comparison between DataFrames would
        # be a really bad idea because it would create unexpected
        # behavior, but I am open to discussion on this.
        if isinstance(dv, pd.DataFrame) or isinstance(fv, pd.DataFrame):
            msg = (
                "Cannot compare DataFrames. Please convert them to "
                "simpler data structures before comparing."
            )
            raise FilterError(msg)

        # I think comparison between missing values is ambiguous,
        # but again, I am open to discussion on this. Here I choose
        # to return False if either value is None because from a
        # logical perspective, we really cannot say anything about
        # the comparison between a missing value and a non-missing.
        if dv is None or fv is None:
            return False

        try:
            return comparator(dv, fv)
        except TypeError as exc:
            msg = (
                f"Cannot compare document value of {type(dv)} type "
                f"with filter value of {type(fv)} type."
            )
            raise FilterError(msg) from exc

    return _wrapper


@_check_comparator_inputs
def _eq(dv: Any, fv: Any) -> bool:
    """
    Conservative implementation of equal comparison.

    There are two major differences between this implementation
    and the default Haystack filter implementation:
        - If both values are None, we return False, instead of True.
        - If any value is a DataFrame, we raise an error, instead
            of converting them to JSON.
    """
    return dv == fv


@_check_comparator_inputs
def _ne(dv: Any, fv: Any) -> bool:
    return not _eq(dv, fv)


@_check_comparator_inputs
def _gt(dv: Any, fv: Any) -> bool:
    """
    A more liberal implementation with less surprises.

    Simply compare the two values with default Python comparison.
    We do not perform any conversion here to have the behavior
    more predictable. If we want to compare the dates, we should
    just convert the document value and filter value explicitly
    to dates before comparing them.
    """
    return dv > fv


@_check_comparator_inputs
def _lt(dv: Any, fv: Any) -> bool:
    return dv < fv


@_check_comparator_inputs
def _gte(dv: Any, fv: Any) -> bool:
    return _gt(dv, fv) or _eq(dv, fv)


@_check_comparator_inputs
def _lte(dv: Any, fv: Any) -> bool:
    return _lt(dv, fv) or _eq(dv, fv)


@_check_comparator_inputs
def _in(dv: Any, fv: Any) -> bool:
    """
    Allowing iterable filter values not just lists.

    This implementation permits a larger set of filter values
    such as tuples, sets, and other iterable objects.
    """
    if not isinstance(fv, Iterable):
        msg = "Filter value must be an iterable for 'in' comparison."
        raise FilterError(msg)

    return any(_eq(dv, v) for v in fv)


@_check_comparator_inputs
def _nin(dv: Any, fv: Any) -> bool:
    return not _in(dv, fv)


LOGICAL_OPERATORS: Final = {"NOT": _not, "AND": _and, "OR": _or}

COMPARISON_OPERATORS: Final = {
    "==": _eq,
    "!=": _ne,
    ">": _gt,
    "<": _lt,
    ">=": _gte,
    "<=": _lte,
    "in": _in,
    "not in": _nin,
}
