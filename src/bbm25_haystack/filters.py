# SPDX-FileCopyrightText: 2024-present Yuxuan Wang <wangy49@seas.upenn.edu>
#
# SPDX-License-Identifier: Apache-2.0
"""
Returns the documents that match the filters provided.

Filters are defined as nested dictionaries that can be of two types:
- Comparison
- Logic

Comparison dictionaries must contain the keys:

- `field`
- `operator`
- `value`

Logic dictionaries must contain the keys:

- `operator`
- `conditions`

The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

The `operator` value in Comparison dictionaries must be one of:

- `==`
- `!=`
- `>`
- `>=`
- `<`
- `<=`
- `in`
- `not in`

The `operator` values in Logic dictionaries must be one of:

- `NOT`
- `OR`
- `AND`


A simple filter:
```python
filters = {"field": "meta.type", "operator": "==", "value": "article"}
```

A more complex filter:
```python
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.type", "operator": "==", "value": "article"},
        {"field": "meta.date", "operator": ">=", "value": 1420066800},
        {"field": "meta.date", "operator": "<", "value": 1609455600},
        {"field": "meta.rating", "operator": ">=", "value": 3},
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
            ],
        },
    ],
}

:param filters: the filters to apply to the document list.
:return: a list of Documents that match the given filters.
"""