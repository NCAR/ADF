"""
Time series file discovery helpers.

Kept apart from adf_utils so they can be unit tested without the scientific
stack: the ADF unit test workflow installs only PyYAML and pytest, so anything
that imports xarray/geocat at module level cannot be exercised in CI.  This
module imports nothing but os and pathlib.

Functions
---------
describe_dir_problem(path, need_write=False)
    Report why a configured directory cannot be used, permissions included.
find_ts_files(ts_loc, pattern, recursive=True)
    Locate time series files matching a glob pattern under a directory.
select_ts_files(fils, syr, eyr)
    Narrow a set of time series files to those needed for a year range.
ts_files_overlap(fils)
    Report whether a set of time series files cover overlapping periods.
ts_files_need_combining(fils)
    Report whether a set of time series files has to be combined into one.
ts_file_span(fils)
    Report the period a set of time series files covers, taken together.
as_hist_str_list(value)
    Normalize a configured history stream setting to a list.
pick_hist_str(value, wanted)
    Choose the one history stream a case should be searched for.

Notes
-----
Re-exported by adf_utils, so ``utils.find_ts_files(...)`` keeps working for
every existing caller.
"""

import os
from pathlib import Path


def describe_dir_problem(path, need_write=False):
    """
    Report why a configured directory cannot be used, or ``None`` if it can.

    Existence is not the only way a configured path fails.  A directory the
    user cannot read reports ``is_dir() == True`` and then globs to nothing,
    so a caller that only checks existence tells the user their data is
    missing when the truth is that it cannot be read -- which is the wrong
    thing to go looking for.  Reading another user's output makes that the
    likely case rather than a rare one.

    Parameters
    ----------
    path : str or Path
        directory named in the config file
    need_write : bool, optional
        Whether the ADF has to write into it as well as read it.  Default is
        ``False``.

    Returns
    -------
    str or None
        A phrase naming the problem, suitable for appending to a caller's own
        message, or ``None`` when the directory is usable.
    """
    ppath = Path(path)
    if not ppath.is_dir():
        # An unsearchable parent makes is_dir() report False for a directory
        # that is really there, so saying "does not exist" would send the user
        # after the wrong problem again:
        parent = ppath.parent
        if parent != ppath and parent.is_dir() and not os.access(parent, os.X_OK):
            return (
                f"'{ppath}' cannot be reached, because this user does not have"
                f" permission to search '{parent}'"
            )
        # End if
        return f"'{ppath}' does not exist, or is not a directory"
    # End if
    # Listing a directory needs both read and search permission, and a missing
    # search bit is the one that produces an empty glob rather than an error:
    if not os.access(ppath, os.R_OK | os.X_OK):
        return f"'{ppath}' exists but this user does not have permission to read it"
    # End if
    if need_write and not os.access(ppath, os.W_OK):
        return f"'{ppath}' exists but this user does not have permission to write in it"
    # End if
    return None


def as_hist_str_list(value):
    """
    Normalize a configured history stream setting to a list.

    The ADF holds one stream as a plain string, several as a list, and an
    empty string when none was configured, so all three have to be accepted.
    Iterating a plain string would walk its characters instead of its value.

    Parameters
    ----------
    value : str or list or None
        a history stream setting as the ADF stores it

    Returns
    -------
    list
        the streams it names; empty when none were configured
    """
    if not value:
        return []
    # End if
    if isinstance(value, str):
        return [value]
    # End if
    return list(value)


def pick_hist_str(value, wanted):
    """
    Choose the one history stream a case should be searched for.

    Parameters
    ----------
    value : str or list or None
        the case's configured stream(s)
    wanted : set
        the streams the caller can use, e.g. ``{"cam.h0", "cam.h0a"}``

    Returns
    -------
    str
        the first configured stream the caller can use, or ``""`` when the
        case has none.

    Notes
    -----
    Every case has to yield exactly one answer.  Callers build a list of these
    and index it by case number, alongside the case names, file locations and
    years, so a case that contributed nothing would shift every case after it
    onto the wrong entry.  An empty string leaves the file search unanchored to
    a stream, which finds the files whichever stream they are in.
    """
    matches = [stream for stream in as_hist_str_list(value) if stream in wanted]
    return matches[0] if matches else ""


def find_ts_files(ts_loc, pattern, recursive=True):
    """
    Locate time series files matching ``pattern`` underneath ``ts_loc``.

    Searches ``ts_loc`` itself first, which is where ADF's own time series step
    and a flat GenTS run both put their files.  Only if that finds nothing does
    it fall back to a recursive search, so that a GenTS archive laid out as
    <component>/proc/tseries/<frequency>/ can be used by pointing ``cam_ts_loc``
    at the top of the tree instead of the frequency sub-directory.

    Parameters
    ----------
    ts_loc : str or Path
        directory to search
    pattern : str
        glob pattern, e.g. "case.cam.h0a.T.*nc"
    recursive : bool, optional
        Whether to fall back to a recursive search when the flat one finds
        nothing.  Default is ``True``.  A missing variable is a normal
        condition, so a caller trying many patterns in a row should pass
        ``False`` and make a single recursive attempt at the end rather than
        walking the whole tree once per pattern.

    Returns
    -------
    list of Path
        Matching files, sorted; empty if nothing matches.
    """
    ts_loc = Path(ts_loc)
    found = sorted(ts_loc.glob(pattern))
    if found or not recursive:
        return found
    #End if
    return sorted(ts_loc.rglob(pattern))


def select_ts_files(fils, syr, eyr):
    """
    Narrow a set of time series files to those needed for a year range.

    A time series directory can hold more than one set of files for the same
    variable: a run post-processed over years 1-20 and then, once it had been
    extended, over years 1-40 leaves both sets behind.  Those sets cannot be
    opened together -- the combined time axis would have duplicate times --
    but either one alone is fine as long as it covers the years being plotted,
    so the configured ``start_year``/``end_year`` are enough to choose between
    them.

    Only a set that cannot be opened as it stands is narrowed.  Files that
    combine cleanly, including a variable split into consecutive chunks (what
    GenTS produces with 'slice_years'), are returned untouched: the plots that
    show a whole record ask for every file a case has, not only the years the
    climatologies use, and there is nothing to resolve for them anyway.

    Where a choice is needed, files are picked greedily: at each step take the
    file that starts at or before the first year not yet covered, preferring
    the smallest one that holds all of the years still needed and otherwise
    the one that reaches furthest.

    Parameters
    ----------
    fils : list
        strings or paths to time series files
    syr : int or str or None
        first year needed
    eyr : int or str or None
        last year needed

    Returns
    -------
    list
        The files needed for [syr, eyr], in chronological order.  The input
        list is returned unchanged whenever this function has nothing to add:
        fewer than two files, files that already combine cleanly, no years
        given, a year range given backwards, names whose dates could not be
        read, dates that would have to be compared at finer than year
        resolution, a gap in the requested range, or a set whose files cannot
        be reduced to a combinable choice.  In each case the caller is left
        with the behavior it had before.
    """
    fils = list(fils)
    if len(fils) < 2:
        return fils
    #End if

    #Files that can be opened together need no choosing:
    if not ts_files_overlap(fils):
        return fils
    #End if

    if syr is None or eyr is None or syr == "" or eyr == "":
        return fils
    #End if
    syr, eyr = int(syr), int(eyr)
    if syr > eyr:
        #A backwards range covers nothing, so there is no choice to make:
        return fils
    #End if

    pairs = _ts_file_span_pairs(fils)
    if pairs is None:
        #Unrecognized names, so make no promises about the set:
        return fils
    #End if

    #The requested years as dates, so that what is dropped can be checked
    #below at the resolution the file names actually use:
    fill = {4: ("", ""), 6: ("01", "12"), 8: ("0101", "1231")}
    width = len(pairs[0][0])
    if width not in fill:
        #An unfamiliar date width, so make no promises about the set:
        return fils
    #End if
    req = (f"{syr:04d}{fill[width][0]}", f"{eyr:04d}{fill[width][1]}")

    #Files that hold some of the requested years:
    candidates = [(int(start[:4]), int(end[:4]), fil) for start, end, fil in pairs
                  if int(start[:4]) <= eyr and int(end[:4]) >= syr]

    chosen = []
    needed = syr
    while needed <= eyr:
        reaching = [c for c in candidates if c[0] <= needed <= c[1]]
        if not reaching:
            #The requested years are not covered, which is a configuration
            #problem rather than a choice to be made here:
            return fils
        #End if
        #Prefer a file that holds all of what is left to cover, and the
        #smallest such, so that a 40-year file is not read to build a 20-year
        #climatology.  Failing that take the one reaching furthest, which is
        #what lets consecutive chunks be picked up in turn:
        covering = [c for c in reaching if c[1] >= eyr]
        if covering:
            _, end_yr, fil = min(covering, key=lambda c: c[1] - c[0])
        else:
            _, end_yr, fil = max(reaching, key=lambda c: c[1])
        #End if
        chosen.append(fil)
        needed = end_yr + 1
    #End while

    #A greedy walk can still end up holding two files that overlap (a 25-year
    #set beside 10-year chunks of the same run, say).  Handing back a set that
    #cannot be opened would be no better than not choosing at all:
    if ts_files_overlap(chosen):
        return fils
    #End if

    #Only years were compared above, so make sure nothing was dropped that
    #the chosen files do not hold.  Two halves of a year
    #(00010101-00010630 beside 00010701-00011231) look alike by year, and
    #dropping one of them would quietly lose half the data; a file that merely
    #runs on past the requested years (years 10-40 beside years 1-20, with
    #years 1-20 asked for) is a different matter and is dropped safely:
    kept = [(start, end) for start, end, fil in pairs if fil in chosen]
    for start, end, fil in pairs:
        if fil in chosen:
            continue
        #End if
        lower, upper = max(start, req[0]), min(end, req[1])
        if lower > upper:
            #Holds none of the requested period:
            continue
        #End if
        if not any(k_start <= lower and k_end >= upper for k_start, k_end in kept):
            return fils
        #End if
    #End for

    return chosen


def _ts_file_span_pairs(fils):
    """
    Parse the {start}-{end} date token out of each name, keeping the file.

    Parameters
    ----------
    fils : list
        strings or paths to time series files

    Returns
    -------
    list of tuple or None
        (start, end, file) in the order given, or ``None`` if any name could
        not be parsed or the dates do not all use the same width.
    """
    pairs = []
    for fil in fils:
        #Last dot-separated token of the stem -- second-to-last of the file
        #name -- e.g. "001001-001112" in "case.cam.h0a.T.001001-001112.nc":
        date_str = Path(fil).stem.split(".")[-1]
        start, sep, end = date_str.partition("-")
        if not sep or not start.isdigit() or not end.isdigit():
            return None
        #End if
        pairs.append((start, end, fil))
    #End for

    #Zero-padded dates of equal width sort chronologically as strings, but
    #mixed widths (e.g. YYYY next to YYYYMM) would not, so bail out on those:
    if len({len(d) for start, end, _ in pairs for d in (start, end)}) != 1:
        return None
    #End if

    return pairs


def _ts_file_spans(fils):
    """
    Parse the {start}-{end} date token out of each time series file name.

    Parameters
    ----------
    fils : list
        strings or paths to time series files

    Returns
    -------
    list of tuple or None
        (start, end) string pairs sorted chronologically, or ``None`` if any
        name could not be parsed or the dates do not all use the same width.
        Callers treat ``None`` as "make no promises about these files".
    """
    pairs = _ts_file_span_pairs(fils)
    if not pairs:
        #Unrecognized names, so make no promises about them:
        return None
    #End if
    spans = [(start, end) for start, end, _ in pairs]

    return sorted(spans)


def ts_files_overlap(fils):
    """
    Report whether time series files cover overlapping periods.

    ADF and GenTS both name time series files
    ``{case}.{stream}.{variable}.{start}-{end}.nc``, where the dates are
    zero-padded and all use the same width for a given variable.  A variable
    split into consecutive chunks (e.g. 001001-001912 then 002001-002912) can
    safely be opened together; two overlapping sets (e.g. years 1-20 alongside
    years 1-40, which happens when a run is extended and the time series are
    remade over a longer period) cannot, because the combined time axis would
    contain duplicates.

    Parameters
    ----------
    fils : list
        strings or paths to time series files

    Returns
    -------
    bool
        True if the files overlap, or if the dates could not be read from the
        names -- in both cases the caller should not blindly combine them.
        False if the files are non-overlapping and safe to open together.
    """
    if len(fils) < 2:
        return False
    #End if

    spans = _ts_file_spans(fils)
    if spans is None:
        return True
    #End if

    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        if next_start <= prev_end:
            return True
    #End for

    return False


def ts_files_need_combining(fils):
    """
    Report whether time series files have to be combined into a single file.

    Answers the question a caller that wants one file per variable has to ask,
    which is not the same as whether the files overlap: one file needs no
    combining, several consecutive files do, and several overlapping files
    cannot be combined at all and so must not be attempted.

    Parameters
    ----------
    fils : list
        strings or paths to time series files

    Returns
    -------
    bool
        True only when there is more than one file and the files can be opened
        together.  False for a single file, for files that overlap, and for
        names whose dates could not be read -- in each of those cases combining
        is either unnecessary or unsafe.
    """
    return len(fils) > 1 and not ts_files_overlap(fils)


def ts_file_span(fils):
    """
    Report the period covered by a set of time series files, taken together.

    Used to name a file derived from several chunked constituent files: the
    derived file has to advertise the whole span it actually contains, not the
    span of whichever chunk happened to sort first.  For a single file this is
    just that file's own dates, so names are unchanged for the common case.

    Parameters
    ----------
    fils : list
        strings or paths to time series files

    Returns
    -------
    tuple of str or None
        (start, end) as they appear in the file names, or ``None`` if the
        dates could not be read.
    """
    spans = _ts_file_spans(fils)
    if not spans:
        return None
    #End if

    #Sorted by start, so the earliest start leads; take the latest end
    #explicitly rather than assuming the last entry carries it:
    return spans[0][0], max(end for _, end in spans)
