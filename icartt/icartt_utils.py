from __future__ import print_function, absolute_import, division, unicode_literals

import re
import pandas as pd


def _get_fields_dataframe(icartt_df):
    return icartt_df.columns.levels[0]


def search_icartt_fields(icartt_data, field_re, ignore_case=True, require_single_match=False):
    if isinstance(icartt_data, pd.DataFrame):
        fieldnames = _get_fields_dataframe(icartt_data)
    else:
        raise TypeError('icartt_data not expected to be of type "{}"'.format(type(icartt_data).__name__))

    re_flags = 0
    if ignore_case:
        re_flags |= re.IGNORECASE

    matching_names = []
    for name in fieldnames:
        if re.search(field_re, name, re_flags):
            matching_names.append(name)

    if not require_single_match:
        return matching_names
    elif len(matching_names) == 1:
        return matching_names[0]
    else:
        raise RuntimeError('Multiple fields matching "{}" found'.format(field_re))