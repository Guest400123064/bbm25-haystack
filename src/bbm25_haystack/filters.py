# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Any, Dict, Callable, Optional, Final

from functools import wraps

from haystack.dataclasses import Document
from haystack.errors import FilterError


def apply_filters_to_document(
    filters: Dict[str, Any],
    document: Document
) -> bool:
    """
    Apply filters to a document.

    :param filters: The filters to apply to the document.
    :type filters: Dict[str, Any]
    :param document: The document to apply the filters to.
    :type document: Document
    
    :return: True if the document passes the filters.
    :rtype: bool
    """
    if filters is None:
        return True

    assert isinstance(filters, dict)
    if not filters:
        return True

    return _run_comparison_condition(filters, document)


def _get_document_field(document: Document, field: str) -> Optional[Any]:
    """
    Get the value of a field in a document.

    If the field is not found within the document then, instead of
    raising an error, `None` is returned. Note that here we do not
    support '.meta' prefix for legacy compatibility any more.

    :param document: The document to get the field value from.
    :type document: Document
    :param field: The field to get the value of.
    :type field: str

    :return: The value of the field in the document.
    :rtype: Optional[Any]
    """
    if r"." not in field:
        return getattr(document, field)

    attr = document.meta
    for f in field.split(r".")[1:]:
        attr = attr.get(f)
        if attr is None:
            return None
    return attr


def _run_logical_condition(
    condition: Dict[str, Any],
    document: Document
) -> bool:
    """
    """
    if "operator" not in condition:
        msg = "Logical condition must have an 'operator' key."
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = "Logical condition must have a 'conditions' key."
        raise FilterError(msg)

    conditions = condition["conditions"]
    reducer = LOGICAL_OPERATORS[condition["operator"]]

    return reducer(document, conditions)


def _run_comparison_condition(
    condition: Dict[str, Any],
    document: Document
) -> bool:
    """
    """
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


def _and(
    document: Document,
    conditions: List[Dict[str, Any]]
) -> bool:
    """
    Return True if all conditions are met.

    :param document: The document to check the conditions against.
    :type document: Document
    :param conditions: The conditions to check against the document.
    :type conditions: List[Dict[str, Any]]

    :return: True if not all conditions are met.
    :rtype: bool
    """
    return all(_run_comparison_condition(condition, document)
               for condition in conditions)


def _or(
    document: Document,
    conditions: List[Dict[str, Any]]
) -> bool:
    """
    Return True if any condition is met.

    :param document: The document to check the conditions against.
    :type document: Document
    :param conditions: The conditions to check against the document.
    :type conditions: List[Dict[str, Any]]

    :return: True if not all conditions are met.
    :rtype: bool
    """
    return any(_run_comparison_condition(condition, document)
               for condition in conditions)


def _not(
    document: Document,
    conditions: List[Dict[str, Any]]
) -> bool:
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
    :type conditions: List[Dict[str, Any]]

    :return: True if not all conditions are met.
    :rtype: bool
    """
    return not _and(document, conditions)


def _comparator_input_type_check_wrapper(
    comparator: Callable[[Any, Any], bool]
) -> Callable[[Any, Any], bool]:
    """
    A wrapper function to check the input types of the comparator function.
    if the input types are not compatible with a comparison binary operator,
    then a FilterError is raised.

    :param comparator: The comparator function to wrap.
    :type comparator: Callable[[Any, Any], bool]

    :return: The wrapped comparator function.
    :rtype: Callable[[Any, Any], bool]
    """
    @wraps(comparator)
    def wrapper(dv: Any, fv: Any) -> bool:
        if dv is None or fv is None:
            return False

        try:
            return comparator(dv, fv)
        except TypeError:
            msg = (
                f"Cannot compare document value of {type(dv)} type "
                f"with filter value of {type(fv)} type."
            )
            raise FilterError(msg)

    return wrapper


def _eq(dv: Any, fv: Any) -> bool:
    """Note that we do not apply the input check wrapper to
    equality comparison too, which means `equal(None, None)`
    return True."""
    return dv == fv


@_comparator_input_type_check_wrapper
def _gt(dv: Any, fv: Any) -> bool:
    return dv > fv


@_comparator_input_type_check_wrapper
def _geq(dv: Any, fv: Any) -> bool:
    return dv >= fv


@_comparator_input_type_check_wrapper
def _in(dv: Any, fv: Any) -> bool:
    return dv in fv


LOGICAL_OPERATORS = {
    "NOT": _not,
    "AND": _and,
    "OR": _or,
}

COMPARISON_OPERATORS = {
    "==": _eq,
    "!=": lambda dv, fv: not _eq(dv, fv),
    ">": _gt,
    ">=": _geq,
    "<": lambda dv, fv: not _geq(dv, fv),
    "<=": lambda dv, fv: not _gt(dv, fv),
    "in": _in,
    "not_in": lambda dv, fv: not _in(dv, fv),
}
