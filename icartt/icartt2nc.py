from abc import ABC, abstractmethod
from argparse import ArgumentParser
import json
import re
import sys
import netCDF4 as ncdf
import pandas as pd
from pathlib import Path
from warnings import warn

from .icartt_readers import read_icartt_file

from typing import Optional, Callable, Tuple, Union

_JDAY_UNITS = {'day', 'days'}
_UTC_UNITS = {'s', 'second', 'seconds'}


class AbstractIcarttTimeConversion(ABC):
    """Base class for computing timestamps from ICARTT data

    Concrete classes must implement the :meth:`convert_time` method.
    """
    converter_type = ''

    @abstractmethod
    def convert_time(self, icartt_df: pd.DataFrame, icartt_metadata: dict, icartt_attributes: dict) -> pd.TimedeltaIndex:
        """Compute timestamps from ICARTT data

        Parameters
        ----------
        icartt_df
            The dataframe containing ICARTT timeseries data.

        icartt_metadata
            A dictionary containing the header metadata for the ICARTT file (i.e. the global metadata, not the per-variable attributes)

        icartt_attributes
            A dicationary containing the per-variable attributes (unit, scale, fill) from an ICARTT file.

        Notes
        -----
        The three inputs are the output of :func:`icartt_reader.read_icartt_file` with ``data_ret_fmt = "dict"``.
        """
        pass
    
    @staticmethod
    def _compute_time_from_jday_utc(base_date, jdays, utc_sec):
        if (jdays.diff().dropna() < -1e-6).any():
            raise NotImplementedError('Julian day decreases between two measurements')
            
        td_days = pd.TimedeltaIndex(jdays, unit='D')
        td_secs = pd.TimedeltaIndex(utc_sec, unit='s')
        return base_date + td_days + td_secs

    @classmethod
    def _get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def create_from_type(cls, converter_type, **kwargs):
        types = dict()
        for klass in cls._get_subclasses():
            if klass.converter_type in types:
                other = types[klass.converter_type]
                warn(f'converter_type {klass.converter_type} is used on both {klass.__class__.__name__} and {other.__class__.__name__}. {other.__class__.__name__} will always be used.')
            else:
                types[klass.converter_type] = klass

        if converter_type in types:
            return types[converter_type](**kwargs)
        else:
            raise TypeError(f'No time converter class defined for type "{converter_type}"')
    
    
class MultidayTimeConversion(AbstractIcarttTimeConversion):
    """Class to compute timestamps from an ICARTT file that provides a single time per data point and spans multiple days

    Parameters
    ----------
    start_date_attr
        The key in the dictionary of global attributes that gives the date of the first data point in the ICARTT file.

    julian_day_var 
        The variable name in the ICARTT file (and its dataframe) that gives the day-of-year variable.

    utc_var
        The variable name in the ICARTT file (and its dataframe) that gives the second after midnight in UTC.
    """
    converter_type = 'multiday'

    def __init__(self, start_date_attr: str = 'data_start_date', julian_day_var: str = 'JDAY', utc_var: str = 'UTC'):
        self.start_date_attr = start_date_attr
        self.julian_day_var = julian_day_var
        self.utc_var = utc_var
        
    def convert_time(self, icartt_df: pd.DataFrame, icartt_metadata: dict, icartt_attributes: dict) -> pd.TimedeltaIndex:
        jday_units = icartt_attributes[self.julian_day_var]['unit']
        if jday_units.lower() not in _JDAY_UNITS:
            warn(f'{self.julian_day_var} does not have units of days, has "{jday_units}". Time calculation may be incorrect')
        utc_units = icartt_attributes[self.utc_var]['unit']
        if utc_units.lower() not in _UTC_UNITS:
            warn(f'{self.utc_var} does not have units of seconds, has "{utc_units}". Time calculation may be incorrect')
            
        start_year = icartt_metadata[self.start_date_attr].year
        base_date = pd.Timestamp(start_year-1, 12, 31)
        return self._compute_time_from_jday_utc(base_date, icartt_df[self.julian_day_var], icartt_df[self.utc_var])
    
    
class RangeTimeConversion(AbstractIcarttTimeConversion):
    """Class to compute timestamps from an ICARTT file that provides a start and stop time for each data point and spans multiple days

    Parameters
    ----------
    start_date_attr
        The key in the dictionary of global attributes that gives the date of the first data point in the ICARTT file.

    julian_day_var 
        The variable name in the ICARTT file (and its dataframe) that gives the day-of-year variable.

    start_time_var
        The variable name in the ICARTT file (and its dataframe) that gives the start time for each data point in seconds after
        UTC midnight.

    stop_time_var
        The variable name in the ICARTT file (and its dataframe) that gives the end time for each data point in seconds after
        UTC midnight.
    """
    converter_type = 'range'

    def __init__(self, start_date_attr: str = 'data_start_date', julian_day_var: str = 'JDAY', start_time_var: str = 'Time_Start', stop_time_var: str = 'Time_Stop'):
        self.start_date_attr = start_date_attr
        self.julian_day_var = julian_day_var
        self.start_time_var = start_time_var
        self.stop_time_var = stop_time_var
        
    def convert_time(self, icartt_df: pd.DataFrame, icartt_metadata: dict, icartt_attributes: dict) -> pd.TimedeltaIndex:
        jday_units = icartt_attributes[self.julian_day_var]['unit']
        if jday_units not in _JDAY_UNITS:
            warn(f'{self.julian_day_var} does not have units of days, has "{jday_units}". Time calculation may be incorrect')
        tstart_units = icartt_attributes[self.start_time_var]['unit']
        if tstart_units not in _UTC_UNITS:
            warn(f'{self.start_time_var} does not have units of seconds, has "{tstart_units}". Time calculation may be incorrect')
        tstop_units = icartt_attributes[self.stop_time_var]['unit']
        if tstop_units not in _UTC_UNITS:
            warn(f'{self.stop_time_var} does not have units of seconds, has "{tstop_units}". Time calculation may be incorrect')
            
        start_year = icartt_metadata[self.start_date_attr].year
        base_date = pd.Timestamp(start_year-1, 12, 31)
        jdays = icartt_df[self.julian_day_var]
        utc_sec = self.get_time(icartt_df)
        return self._compute_time_from_jday_utc(base_date, jdays, utc_sec)
        
    def get_time(self, icartt_df: pd.DataFrame) -> pd.Series:
        """Compute the single time to represent each data point in the dataframe.

        This class computes the midpoint time. To return different times (e.g. start or stop time),
        make a subclass and override this method.
        """
        return 0.5*(icartt_df[self.start_time_var] + icartt_df[self.stop_time_var])

def write_nc_file(nc_file: str, 
                  icartt_df: pd.DataFrame, 
                  icartt_metadata: dict, 
                  icartt_attributes: dict, 
                  time_converter: Optional[AbstractIcarttTimeConversion] = None, 
                  var_rename_map: dict = dict(), 
                  var_rename_op: Optional[Callable[[str], str]] = None,
                  var_comments: dict = dict(),
                  extra_nc_atts: dict = dict()):
    """Create a netCDF version of ICARTT data

    Parameters
    ----------
    nc_file
        The path to the netCDF file to write. If it exists, it will be overwritten.

    icartt_df
        The dataframe containing ICARTT timeseries data.

    icartt_metadata
        A dictionary containing the header metadata for the ICARTT file (i.e. the global metadata, not the per-variable attributes)

    icartt_attributes
        A dicationary containing the per-variable attributes (unit, scale, fill) from an ICARTT file.

    time_converter
        An instance of a :class:`AbstractIcarttTimeConversion` subclass which will be used to generate timestamps for the netCDF file.
        If this is ``None``, no timestamps will be created.

    var_rename_map
        A dictionary that has ICARTT variable names as keys and desired variable names as values. Any ICARTT variables present as keys
        in this dictionary will be renamed to the corresponding value in the netCDF file. The original ICARTT name will be stored as an
        attribute.

    var_rename_op
        A function that takes in a variable name and outputs a transformed name. If this is ``None``, then no transformation is applied.
        Common examples are making variable names lower case, or removing special characters. Note that this is applied AFTER ``var_rename_map``,
        so if the netCDF variable name in the map needs to be transformed, it will be.

    var_comments
        A dictionary with ICARTT variable names as keys and comments to add to those variables as values. The comment will be added as the 
        "comment" attribute for that variable in the netCDF file. (Variables not included in this dictionary will not receive comments.)

    extra_nc_atts
        A dictionary of attribute names and values to add to as global attributes to the netCDF file.
    """
    primary_dim = 'time'
    try:
        with ncdf.Dataset(nc_file, 'w') as ds:
            ds.createDimension(primary_dim, icartt_df.shape[0])
            if time_converter is not None:
                times = time_converter.convert_time(icartt_df, icartt_metadata, icartt_attributes)
                times = ncdf.date2num(times.to_pydatetime(), 'seconds since 1970-01-01', calendar='gregorian')
                tvar = ds.createVariable('time', times.dtype, (primary_dim,))
                tvar[:] = times
                tvar.units = 'seconds since 1970-01-01'
                tvar.calendar = 'gregorian'
                
            for orig_colname, data in icartt_df.iteritems():
                varname = var_rename_map.get(orig_colname, orig_colname)
                if var_rename_op is not None:
                    varname = var_rename_op(varname)
                
                var = ds.createVariable(varname, data.dtype, (primary_dim,))
                var[:] = data.to_numpy()
                if orig_colname in icartt_attributes:
                    var.units = icartt_attributes[orig_colname]['unit']
                else:
                    warn(f'Could not find units for {orig_colname}, {varname} will have unknown units in the netCDF file. '
                        f'(This can happen if there is a typo between the header and table OR if a variable is duplicated.)')
                    var.units = 'UNKNOWN'
                if varname != orig_colname:
                    var.original_icartt_name = orig_colname
                cmt = var_comments.get(orig_colname, '')
                if cmt:
                    var.comment = cmt
                    
            for attname, attval in icartt_metadata.items():
                if attname == 'normal_comments':
                    for k, v in attval.items():
                        ds.setncattr(k, v)
                elif isinstance(attval, list):
                    if attval:
                        ds.setncattr(attname, [str(v) for v in attval])
                elif isinstance(attval, dict):
                    for k, v in attval.items():
                        ds.setncattr(f'{attname}_{k}', v)
                elif isinstance(attval, (bool, int, float, str)):
                    ds.setncattr(attname, attval)
                else:
                    ds.setncattr(attname, str(attval))

            ds.setncatts(extra_nc_atts)
    except Exception:
        print(f'Error occurred while writing {nc_file}, cleaning up incomplete netCDF file', file=sys.stderr)
        Path(nc_file).unlink()
        print(f'{nc_file} deleted', file=sys.stderr)
        raise


def write_nc_from_json(icartt_file: str, nc_file: str, options_json: Union[str, Path, dict], clobber: bool = False):
    """Convert an ICARTT file to netCDF using a JSON file/dictionary of options

    Parameters
    ----------
    icartt_file
        Path to the ICARTT file to convert.

    nc_file
        Path to write the netCDF file to. Will not be overwritten unless ``clobber = True``.

    options_json
        Either a dictionary containing options for reading the ICARTT file and writing the netCDF file
        or a path pointing to a JSON file that contains that dictionary. See :func:`parse_options_json` 
        for details on what this dictionary can contain.

    clobber
        Whether to allow overwriting the netCDF file if it exists or not.
    """
    if not isinstance(options_json, dict):
        with open(options_json) as f:
            options_json = json.load(f)

    if Path(nc_file).exists() and not clobber:
        raise IOError(f'nc_file {nc_file} already exists, rename/remove this file or use clobber option to overwrite')

    read_opts, nc_opts = parse_options_json(options_json)
    print(f'Reading ICARTT file {icartt_file}')
    icartt_df, icartt_meta, icartt_atts = read_icartt_file(icartt_file, data_ret_fmt='dict', **read_opts)
    print(f'Writing netCDF file {nc_file}')
    write_nc_file(nc_file=nc_file, icartt_df=icartt_df, icartt_attributes=icartt_atts, icartt_metadata=icartt_meta, **nc_opts)


def batch_write_nc_from_json(options_json, skip_bad_icartt=False, clobber=False, skip_existing=False):
    """Convert multiple ICARTT files to netCDF files

    Parameters
    ----------
    options_json
        A list that contains the list of ICARTT and netCDF files plus their conversion options
        or a path to a JSON file that contains that list. Each element in the list must be a dictionary
        suitable to pass as ``options_json`` to :func:`write_nc_from_json` with the additional keys
        "icartt_file" and "nc_file".

    skip_bad_icartt
        If any ICARTT file cannot be read, skip it and go to the next. When this is ``False`` (default),
        an error reading an ICARTT file results in a crash.

    clobber
        Whether to allow overwriting the netCDF file if it exists or not.

    skip_existing
        Setting this to ``True`` means that if the output netCDF file for a given ICARTT file already
        exists, we just go onto the next one. This is useful for restarting a partially completed batch
        run, and this supercedes ``clobber``.
    """
    def get_req(d, k, i, pop=True):
        if k not in d:
            raise IOError(f'Options for file {i} missing required key "{k}"')
        elif pop:
            return d.pop(k)
        else:
            return d[k]

    if not isinstance(options_json, dict):
        with open(options_json) as f:
            options_json = json.load(f)

    if not isinstance(options_json, (list, tuple)):
        raise TypeError('A batch JSON must have a list as the top level element')

    nfiles = len(options_json)
    for ifile, file_opts in enumerate(options_json, start=1):
        print(f'Processing file {ifile}/{nfiles}')
        icartt_file = get_req(file_opts, 'icartt_file', ifile)
        nc_file = get_req(file_opts, "nc_file", ifile)
        if Path(nc_file).exists() and skip_existing:
            print(f'nc_file {nc_file} already exists, skipping\n')
            continue

        if not Path(icartt_file).exists():
            if skip_bad_icartt:
                print(f'  -> {icartt_file} does not exist, skipping')
                continue
            else:
                raise IOError(f'{icartt_file} does not exist')

        try:
            write_nc_from_json(icartt_file, nc_file, file_opts, clobber=clobber)
        except Exception as err:
            if skip_bad_icartt:
                print(f'Problem converting {icartt_file}, skipping: {err}', file=sys.stderr)
            else:
                raise
        print('') # put a blank line between file outputs



def parse_options_json(options_json: dict) -> Tuple[dict, dict]:
    """Parse a JSON options dict into read and write keyword arguments

    The sole input is a dictionary with the following structure. Note that anything not
    indicated as required is optional.

    * "time": a dictionary containing options that define how to convert time from the ICARTT format to
      CF conventions. Has the following sub keys:
        * "type" (required if this section present) = what type of conversion to do. This relies on the 
          ``converter_type`` attribute of :class:`AbstractIcarttTimeConversion` subclasses, it can be any
          of those values. Currently that is "multiday" (for ICARTT files that give a single time per row)
          or "range" (for ICARTT files that give a start/stop time per row.)
        * "start_date_attr" = which ICARTT global metadata attribute has the starting date for that ICARTT file.
        * "julian_day_var" = which ICARTT variable has the day-of-year value
        * "utc_var" (multiday type only) = which ICARTT variable has the seconds after UTC midnight value
        * "start_time_var" (range type only) = which ICARTT variable has the seconds after UTC midnight value for
          the start time of that row
        * "stop_time_var" (range type only) = which ICARTT variable has the seconds after UTC midnight value for
          the end time of that row
    * "var_rename_map": a dictionary mapping ICARTT names to desired netCDF name
    * "var_comments": a dictionary mapping ICARTT variable names to comment to add to them
    * "var_rename_op": one of the strings 'upper', 'lower', 'remove-special-chars', 'remove-special-chars+upper',
      or 'remove-special-chars+lower' that indicates how to transform all ICARTT variables names (upper/lower case,
      remove non letter/number/_ characters)
    * "extra_nc_atts": a dictionary of global attributes to add to the netCDF file
    * "icartt_encoding": what encoding to use when reading the ICARTT file, e.g. 'utf-16' or 'ISO-8859-1'. Default
      is 'utf-8'

    Returns
    -------
    read_options
        A dictionary of keywords for :func:`read_icartt_file`

    write_options
        A dictionary of keywords for :func:`write_nc_file`.
    """
    def get_req(sect, key):
        try:
            return options_json[sect][key]
        except KeyError as err:
            raise IOError(f'Missing required key in {sect} section of JSON: {key}')

    read_options = {
        'encoding': options_json.get('icartt_encoding', 'utf-8')
    }

    nc_options = dict()

    if 'time' not in options_json:
        nc_options['time_converter'] = None
    else:
        conv_type = get_req('time', 'type')
        kws = {k: v for k, v in options_json['time'].items() if k != 'type'}
        nc_options['time_converter'] = AbstractIcarttTimeConversion.create_from_type(conv_type, **kws)

    if 'var_rename_map' not in options_json:
        nc_options['var_rename_map'] = dict()
    else:
        nc_options['var_rename_map'] = options_json['var_rename_map']

    if 'var_comments' not in options_json:
        nc_options['var_comments'] = dict()
    else:
        nc_options['var_comments'] = options_json['var_comments']

    if 'var_rename_op' not in options_json:
        nc_options['var_rename_op'] = None
    elif options_json['var_rename_op'] == 'lower':
        nc_options['var_rename_op'] = lambda name: name.lower()
    elif options_json['var_rename_op'] == 'upper':
        nc_options['var_rename_op'] = lambda name: name.upper()
    elif options_json['var_rename_op'] == 'remove-special-chars':
        nc_options['var_rename_op'] = lambda name: re.sub(r'[^\w]', '', name)
    elif options_json['var_rename_op'] == 'remove-special-chars+lower':
        nc_options['var_rename_op'] = lambda name: re.sub(r'[^\w]', '', name).lower()
    elif options_json['var_rename_op'] == 'remove-special-chars+upper':
        nc_options['var_rename_op'] = lambda name: re.sub(r'[^\w]', '', name).upper()
    else:
        raise IOError(f'Unknown var_rename_op: {options_json["var_rename_op"]}')

    if 'extra_nc_atts' not in options_json:
        nc_options['extra_nc_atts'] = dict()
    else:
        nc_options['extra_nc_atts'] = options_json['extra_nc_atts']

    return read_options, nc_options


_JSON_HELP = """A single file JSON that uses all of the options would look like:

{
    "time": {
        "type": "range",
        "julian_day_var": "Day_Of_Year_YANG"
    },
    "var_rename_map": {
        "HAE_GPS_Altitude_YANG": "alt_gps",
        "Pressure_Altitude_YANG": "alt_pres",
        "Radar_Altitude_YANG": "alt_radar"
    },
    "var_rename_op": "lower",
    "icartt_encoding": "utf-16"
}

* "time" must have the key "type", which can currently be "range" (for ICARTT 
files that give a start and stop time) or "multiday" (for ICARTT files that 
give a single time per row). The other keys depend on the type, and are the
class keywords for the corresponding time converter. If this section is missing,
the output netCDF files will not have a time variable.

* "var_rename_map" must have ICARTT variables as keys and desired netCDF variable
names as values. If not given, no remapping occurs.

* "var_rename_op" must be one of:
    - "lower" = lower case variable names in the netCDF file
    - "upper" = upper case variable names in the netCDF file
    - "remove-special-chars" = remove any non-letter, number, or _ characters in
      variable names.
    - "remove-special-chars+lower", "remove-special-chars+upper" combine both ops.
  If this is omitted, no operation is applied. Note that any operations happen after
  "var_rename_map" is applied, so the values in that map will be processed with this
  function.

* "icartt_encoding" indicates which encoding to use to read the ICARTT file. The 
  default is "utf-8"; if given, this must be an encoding Python recognizes.

A batch file JSON could be as simple as:

[
    {
        "icartt_file": "intex-na_all.ict",
        "nc_file": "intex-na_all.nc"
    },
    {
        "icartt_file": "arctas_all.ict",
        "icartt_file": "arctas_all.nc"
    }
]

where each element in the top list is a map that has at a minimum of the keys
"icartt_file" (the path to the ICARTT file to convert) and "nc_file" (the path
to write the netCDF file to). Relative paths are interpreted relative to where
you run this program, NOT relative to the JSON file.

Any of the keys from the single file JSONs can be included, e.g.:

[
    {
        "icartt_file": "intex-na_all.ict",
        "nc_file": "intex-na_all.nc",
        "time": {"type": "multiday"},
        "icartt_encoding": "ISO-8859-1"
    },
    {
        "icartt_file": "firex_all.ict",
        "icartt_file": "firex_all.nc",
        "time": {"type": "range", "julian_day_var": "Day_Of_Year_YANG"}
    }
]
"""


def main():
    p = ArgumentParser(description='Convert ICARTT files into netCDF files')
    p.add_argument('--pdb', action='store_true', help='Launch Python debugger')
    
    subp = p.add_subparsers()
    single = subp.add_parser('single', help='Convert one ICARTT file')
    single.add_argument('icartt_file', help='Path to the ICARTT file to convert')
    single.add_argument('nc_file', help='Path to write the netCDF file as')
    single.add_argument('options_json', nargs='?', default=dict(), help='Path to a JSON options file controlling how to do the conversion. Use the json-help subcommand for details on the JSON structure.')
    single.add_argument('-c', '--clobber', action='store_true', help='Overwrite the netCDF file if it exists')
    single.set_defaults(driver=write_nc_from_json)

    batch = subp.add_parser('batch', help='Convert multiple ICARTT files')
    batch.add_argument('options_json', help='Path to a batch JSON options file controlling how to do the conversion. Use the json-help subcommand for details on the JSON structure.')
    batch.add_argument('-b', '--skip-bad-icartt', action='store_true', help='If one of the ICARTT files is missing or there is an error when converting, skip it rather than crashing')
    bmut = batch.add_mutually_exclusive_group()
    bmut.add_argument('-c', '--clobber', action='store_true', help='Overwrite each netCDF file if it exists')
    bmut.add_argument('-s', '--skip-existing', action='store_true', help='If a netCDF file already exists, move on to the next ICARTT file')
    batch.set_defaults(driver=batch_write_nc_from_json)

    json_help = subp.add_parser('json-help', help='Provide information about the options_json structure')
    json_help.set_defaults(driver=lambda: print(_JSON_HELP))

    clargs = vars(p.parse_args())
    driver_fxn = clargs.pop('driver')
    if clargs.pop('pdb'):
        import pdb
        pdb.set_trace()

    driver_fxn(**clargs)


if __name__ == '__main__':
    main()
