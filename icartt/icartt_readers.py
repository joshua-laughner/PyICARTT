from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import pandas as pd
import re
#from xlrd import open_workbook


class ICARTTError(Exception):
    """
    Base class for ICARTT errors
    """
    pass


class ICARTTParsingError(ICARTTError):
    """
    Error class to use when illegal or unexpected format is found in an ICARTT file
    """
    pass


def _parse_na(str_in):
    if re.match('N/A', str_in, re.IGNORECASE):
        return None
    else:
        return float(str_in)


_special_comment_fields = {'PI_CONTACT_INFO': dict(),
                           'PLATFORM': dict(),
                           'LOCATION': dict(),
                           'ASSOCIATED_DATA': dict(),
                           'INSTRUMENT_INFO': dict(),
                           'DATA_INFO': dict(),
                           'UNCERTAINTY': dict(),
                           'ULOD_FLAG': {'type_out': float},
                           'ULOD_VALUE': {'type_out': _parse_na},
                           'LLOD_FLAG': {'type_out': float},
                           'LLOD_VALUE': {'type_out': _parse_na},
                           'DM_CONTACT_INFO': dict(),
                           'PROJECT_INFO': dict(),
                           'STIPULATIONS_ON_USE': dict(),
                           'OTHER_COMMENTS': dict(),
                           'REVISION': dict(),
                           r'R\d+': dict()}


def read_icartt_file(icartt_file, data_ret_fmt='dataframe', index='indep', encoding='utf-8'):
    """
    Read a single ICARTT file.

    Read an ICARTT file with the format specifed in the document:
    https://www-air.larc.nasa.gov/missions/etc/IcarttDataFormat.htm

    :param icartt_file: the path to the ICARTT file
    :type icartt_file: str

    :param data_ret_fmt: how to return the ICARTT data. Possible values are "dataframe", "df", or "dict". "dataframe" or
     "df" will both return the data in a Pandas dataframe with the primary variable as the index, the variable names as
     column headers, and the units as the second header. Any values within default numpy tolerance to the fill values
     will be replaces with NA value. "dict" will return a dictionary with the unit names as keywords, each containing
     a dictionary with the keys "name", "unit", "fill", and "values". Fill values will still be replaced with NaNs.
    :type data_ret_fmt: str

    :param index: how to index the dataframe, if a dataframe is returned. "indep" will use the independent variable,
     "datetime" will generate a datetime index.

    :param encoding: text encoding of the ICARTT file. Try specifying an alternate encoding (e.g. ISO-8859-1) with this 
     keyword if you encounter unicode decoding errors. A list of encodings recognized by Python can be found 
     `here <https://docs.python.org/3/library/codecs.html#standard-encodings>`_. To identify the encoding of a file,
     try using the ``file`` command line program on Linux systems.

    :return: the ICARTT data, in the format defined by ``data_ret_fmt`` and the ICARTT metadata, as a dict.
    :rtype: dict or :class:`pandas.DataFrame`, dict
    """
    nheader, file_type, metadata, variable_info, indep_varname, delimiter = _read_icartt_header(icartt_file, var_info_fmt=data_ret_fmt, encoding=encoding)
    if delimiter == ',':
        read_kws = {'sep': r'\s*,\s*', 'engine': 'python'}
    elif delimiter == ' ':
        read_kws = {'sep': r'\s+'}
    df = pd.read_csv(icartt_file, header=nheader-1, encoding=encoding, **read_kws)

    # Convert fill values, ULOD flags, and LLOD flags to NaNs
    ulod_flag = metadata['normal_comments']['ULOD_FLAG']
    llod_flag = metadata['normal_comments']['LLOD_FLAG']
    for colname, info in variable_info.items():
        if info['fill'] is not None:
            where_fxn = lambda column: ~np.isclose(column, info['fill']) & ~np.isclose(column, llod_flag) & ~np.isclose(column, ulod_flag)
        else:
            where_fxn = lambda column: ~np.isclose(column, llod_flag) & ~np.isclose(column, ulod_flag)

        df.loc[:, colname].where(where_fxn, inplace=True)

    if index == 'indep':
        # Make the independent variable the index. I cannot seem to get this to work after the multiindex is assigned
        # so unfortunately the units of the index will just have to be known.
        df.set_index(indep_varname, inplace=True, verify_integrity=True)
    elif index == 'datetime':
        df.index = _make_icartt_datetimes(metadata['data_start_date'], df[indep_varname])
    else:
        raise ValueError('index == "{}" is not allowed. Only allowed values are "indep" and "datetime"'.format(index))

    return df, metadata, variable_info


def _read_icartt_header(icartt_file, var_info_fmt='dataframe', encoding='utf-8'):
    def parse_one_line(line, split_on=None, type_out=str):
        if split_on:
            return tuple(type_out(v.strip()) for v in line.split(split_on))
        else:
            return type_out(line)

    def get_one_line(fhandle, split_on=None, type_out=str):
        line = fhandle.readline().strip()
        return parse_one_line(line, split_on=split_on, type_out=type_out)

    def parse_var_line(fhandle):
        parts = get_one_line(fhandle, split_on=',')
        if len(parts) == 1:
            # ARCTAS 10s multiday merge files don't include a unit
            return parts[0], 'none'
        else:
            # FIREX-AQ 10s multiday merge file includes more than name and unit
            return parts[:2]

    def get_variable_info(fhandle, split_on=','):
        # Get the independent variable name and units
        independent_var_name, independent_var_unit = parse_var_line(fhandle)
        var_names = [independent_var_name]
        var_units = [independent_var_unit]
        # Assume that the independent variable is not scaled and does not have a fill value
        var_scale_factors = [1.0]
        var_fill_values = [None]

        nvars = get_one_line(fhandle, type_out=int)
        var_scale_factors += get_one_line(fhandle, split_on=split_on, type_out=float)
        var_fill_values += get_one_line(fhandle, split_on=split_on, type_out=float)

        for i in range(nvars):
            this_var_name, this_var_unit = parse_var_line(fhandle)
            var_names.append(this_var_name)
            var_units.append(this_var_unit)

        if var_info_fmt == 'dict':
            # Convert to a list of dictionaries for cleaner organization
            var_info = dict()
            for name, unit, scale, fill in zip(var_names, var_units, var_scale_factors, var_fill_values):
                var_info[name] = {'unit': unit, 'scale': scale, 'fill': fill}
        elif var_info_fmt == 'dataframe':
            var_info = pd.DataFrame({'unit': var_units, 'scale': var_scale_factors, 'fill': var_fill_values}, index=var_names).T
        else:
            raise ValueError('var_info_fmt "{}" is not allowed. Must be "dict" or "dataframe".')

        return var_info, independent_var_name

    def read_special_comments(fhandle):
        n_comment_lines = get_one_line(fhandle, type_out=int)
        comments = []
        for i in range(n_comment_lines):
            comments.append(get_one_line(fhandle))
        return comments

    def read_normal_comments(fhandle):
        n_comment_lines = get_one_line(fhandle, type_out=int)
        normal_comments = dict()
        for i in range(n_comment_lines):
            line = get_one_line(fhandle)
            for regex, parse_args in _special_comment_fields.items():
                match = re.match(r'({}):?\s*(.+)'.format(regex), line, re.IGNORECASE)
                if match is not None:
                    key = match.groups()[0]
                    if key in normal_comments:
                        raise ICARTTParsingError('The special comment "{}" is defined multiple times; this is not '
                                                 'allowed'.format(key))
                    normal_comments[key] = match.groups()[1]
                    if re.match(r'[UL]LOD_FLAG', key, re.IGNORECASE):
                        normal_comments[key] = float(normal_comments[key])

        return normal_comments

    with open(icartt_file, 'r', encoding=encoding) as ict:
        # The INTEX-NA 10s merge "all" file is very inconsistent in its use of delimiter; everything EXCEPT the lines
        # defining the variable units are space delimited, but those lines are comma delimited. We can infer the overall
        # delimiter from the first line. If any future file has the variable info lines be space delimited, we'll have to
        # infer those separately.
        line = ict.readline()
        delimiter = ',' if ',' in line else ' '
        ict.seek(0)

        nheader, file_type = get_one_line(ict, split_on=delimiter)
        nheader = int(nheader)
        if file_type != '1001':
            raise NotImplementedError('File types other than "1001" are not yet supported')

        metadata = dict()
        metadata['PI'] = get_one_line(ict)
        metadata['organization'] = get_one_line(ict)
        metadata['data_description'] = get_one_line(ict)
        metadata['mission_name'] = get_one_line(ict)
        metadata['volume_num'], metadata['total_num_volumes'] = get_one_line(ict, split_on=delimiter, type_out=int)

        # Dates require special handling
        data_start_yr, data_start_month, data_start_day, rev_year, rev_month, rev_day = get_one_line(ict, split_on=delimiter, type_out=int)
        metadata['data_start_date'] = pd.Timestamp(data_start_yr, data_start_month, data_start_day)
        metadata['revision_date'] = pd.Timestamp(rev_year, rev_month, rev_day)
        metadata['data_interval_s'] = get_one_line(ict, type_out=float)

        # Variable information will be used when parsing the data, but not explicitly included in the metadata
        var_info, indep_varname = get_variable_info(ict, split_on=delimiter)

        metadata['special_comments'] = read_special_comments(ict)
        metadata['normal_comments'] = read_normal_comments(ict)

    return nheader, file_type, metadata, var_info, indep_varname, delimiter


def _make_icartt_datetimes(data_start_date, utc_seconds):
    datetimes = [data_start_date + pd.Timedelta(seconds=s) for s in utc_seconds]
    return pd.DatetimeIndex(datetimes)


def read_tbl_file(tbl_file, data_dict_file=None):
    data = pd.read_csv(tbl_file, sep='\s+')
    if data_dict_file is None:
        return data

    extra_info = _read_tbl_data_dict(data_dict_file, data.keys())
    data.index = _make_tbl_datetime(data['Year'], data['DOY'], data['UTC'])

    return data, extra_info


def _read_tbl_data_dict(data_dict_file, variables):
    def read_column_as_array(s, col_idx):
        arr = [s.cell(r, col_idx).value for r in range(1, s.nrows)]
        return np.array(arr)

    indices = {'unit': 'unit'}
    values = dict()

    book = open_workbook(data_dict_file)
    sheet = book.sheet_by_index(0)

    # First row should be the headers. Look for the one that says "Unit"
    for out_key, re_str in indices.items():
        for i in range(1, sheet.ncols):
            header_str = sheet.cell(0, i).value
            if re.match(re_str, header_str, re.IGNORECASE):
                indices[out_key] = i
                break

    # Assume the variable names are in the first column
    var_names = read_column_as_array(sheet, 0)

    for key, ind in indices.items():
        values[key] = read_column_as_array(sheet, ind)

    df = pd.DataFrame(data=values, index=var_names)
    return df.loc[variables, :].T


def _make_tbl_datetime(years, doy, utc):
    datetimes = [pd.Timestamp(y.item(), 1, 1) + pd.Timedelta(days=d.item()-1, seconds=s.item()) for y, d, s in
                 zip(years.to_numpy(), doy.to_numpy(), utc.to_numpy())]
    return pd.DatetimeIndex(datetimes)
