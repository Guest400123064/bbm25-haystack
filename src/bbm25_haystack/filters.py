# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Any, Dict, Optional, Final

from haystack.dataclasses import Document
from haystack.errors import FilterError


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

    attr = document
    for f in field.split(r"."):
        attr = getattr(attr, f)
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
    operator = COMPARISON_OPERATORS[condition["operator"]]

    return operator(value, _get_document_field(document, field))


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


def _eq():
    pass


LOGICAL_OPERATORS = {
    "NOT": _not,
    "AND": _and,
    "OR": _or,
}

COMPARISON_OPERATORS = {
}
