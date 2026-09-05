"""
Unit string handling.

Kept apart from adf_utils so it can be unit tested without the scientific
stack: the ADF unit test workflow installs only PyYAML and pytest, so anything
that imports xarray/geocat at module level cannot be exercised in CI.  This
module imports nothing but re.

Functions
---------
normalize_units(units)
    Reduce a unit string to a canonical form for comparison.
units_equivalent(first, second)
    Report whether two unit strings mean the same thing.

Notes
-----
The ADF compares units to decide whether the conversion named in the variable
defaults has already been applied to a file.  The two strings being compared
come from different places -- one written by CAM, the other by whoever edited
`adf_variable_defaults.yaml` -- so they agree on the physics far more often
than they agree character for character.  `W/m2`, `W m-2`, `Wm^-2` and the
LaTeX `W m$^{-2}$` are one unit written four ways, and the shipped defaults
alone contain both `Wm$^{-2}$` and `W m$^{-2}$`.  Comparing the raw strings
answers "were these typed the same way", which is not the question being
asked, and getting it wrong scales the data twice.

Rendering a unit for somewhere with no LaTeX renderer, such as a table cell,
is `adf_utils.plain_text_units`; this module only compares.
"""

import re

# Unit names that mean the same thing.  Keys and values are compared after
# normalization, so only real synonyms belong here, not spelling variants:
_ALIASES = {
    "fraction": "1",
    "frac": "1",
    "unitless": "1",
    "dimensionless": "1",
    "none": "1",
    "-": "1",
    "[-]": "1",
    "percent": "%",
    "pct": "%",
    "degrees_kelvin": "k",
    "deg_k": "k",
    "kelvin": "k",
    "degrees_celsius": "degc",
    "celsius": "degc",
    "deg_c": "degc",
    "degrees_east": "degrees_east",
    "meters": "m",
    "meter": "m",
    "metre": "m",
    "seconds": "s",
    "second": "s",
    "sec": "s",
    "days": "d",
    "day": "d",
    "grams": "g",
    "gram": "g",
    "micron": "um",
    "microns": "um",
}

# Factors written without a space between them.  Case matters and is still
# intact here, which is the point: "Nm" is newton-metre while "nm" is
# nanometre, and lower-casing first makes them the same string.  Only forms
# that actually occur are listed -- a general splitter guessing where to cut a
# run of letters reads "Sv" as siemens-volt and "nm" as newton-metre, the
# second of which reports two different units as equal.  Note that "ms" is
# read as metre-second: no CAM field is reported in milliseconds, and
# "ms$^{-1}$" for wind speed is in the shipped variable defaults today.
_RUN_TOGETHER = {
    "Wm": "W m",
    "Nm": "N m",
    "Jm": "J m",
    "Km": "K m",
    "kgm": "kg m",
    "gm": "g m",
    "ms": "m s",
    "Pas": "Pa s",
}

# LaTeX and unicode fragments that carry no meaning for a comparison:
_LATEX = (
    (r"\mathrm", ""),
    (r"\text", ""),
    (r"\mu", "u"),
    (r"\,", " "),
    (r"\;", " "),
    (r"\ ", " "),
    ("µ", "u"),  # micro sign
    ("μ", "u"),  # greek small letter mu
    ("·", " "),  # middle dot, used as a multiplication sign
    ("**", "^"),
)

# Superscript digits, which appear in units copied out of documents:
_SUPERSCRIPTS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
    }
)


def _strip_markup(units):
    """Return `units` with LaTeX, superscripts and stray braces removed."""
    text = units.translate(_SUPERSCRIPTS)
    for old, new in _LATEX:
        text = text.replace(old, new)
    # A LaTeX exponent is written ^{-2}; the braces are noise once the
    # superscript digits are gone:
    text = re.sub(r"\^\s*\{([^}]*)\}", r"^\1", text)
    text = text.replace("$", "").replace("{", "").replace("}", "")
    # Split the run-together factors before anything lower-cases the string,
    # because which factors they are depends on their case:
    for joined, apart in _RUN_TOGETHER.items():
        text = re.sub(rf"(?<![A-Za-z]){joined}(?=[-^0-9\s]|$)", apart, text)
    # End for
    return text.strip()


def _tokenize(text, sign):
    """Return (name, exponent) pairs for one side of a solidus."""
    tokens = []
    for chunk in re.split(r"[\s*.]+", text):
        if not chunk:
            continue
        # A trailing exponent, written either "m2", "m^2", "m-2" or "m^-2":
        match = re.fullmatch(r"([a-z%_]+)\^?([+-]?\d+)?", chunk)
        if match is None:
            # Not something this understands; keep it whole so that two
            # identical odd strings still compare equal:
            tokens.append((chunk, sign))
            continue
        name, exponent = match.group(1), match.group(2)
        tokens.append((name, sign * int(exponent if exponent else 1)))
    return tokens


def normalize_units(units):
    """
    Reduce a unit string to a canonical form for comparison.

    Parameters
    ----------
    units : str or None
        a unit string as it appears in a file or in the variable defaults

    Returns
    -------
    str
        A canonical form: lower case, no LaTeX, every factor written
        ``name^exponent`` and sorted, so that the many spellings of one unit
        reduce to a single string.  An empty string for ``None`` or for a
        string that holds nothing.

    Examples
    --------
    ``W/m2``, ``W m-2``, ``Wm^-2`` and ``W m$^{-2}$`` all give ``m^-2 w^1``.
    """
    if units is None:
        return ""
    text = _strip_markup(str(units)).lower()
    if not text:
        return ""
    text = _ALIASES.get(text, text)
    # Split on the solidus: everything after the first one is a denominator.
    # "a/b/c" is read the way it is written, left to right.
    parts = text.split("/")
    tokens = _tokenize(parts[0], 1)
    for part in parts[1:]:
        tokens += _tokenize(part, -1)

    # Fold the aliases once more, now that the factors are separated, so that
    # "m/s" and "meters/second" agree:
    folded = {}
    for name, exponent in tokens:
        name = _ALIASES.get(name, name)
        folded[name] = folded.get(name, 0) + exponent
    # A factor that cancels out carries no dimension, but it does carry an
    # identity: "kg/kg" and "kg kg-1" are the same thing, and neither is
    # "mol/mol".  A mass mixing ratio and a volume mixing ratio are both
    # dimensionless and are not the same number, so what cancelled is kept.
    remaining = {
        name: exponent
        for name, exponent in folded.items()
        if exponent != 0 and name != "1"
    }
    if not remaining:
        cancelled = sorted(name for name in folded if name != "1")
        return f"1[{' '.join(cancelled)}]" if cancelled else "1"
    # End if
    return " ".join(
        f"{name}^{exponent}" for name, exponent in sorted(remaining.items())
    )


def units_equivalent(first, second):
    """
    Report whether two unit strings mean the same thing.

    Parameters
    ----------
    first, second : str or None
        the unit strings to compare

    Returns
    -------
    bool
        ``True`` when the two describe the same unit, however they are
        spelled.  Two strings that are both empty or ``None`` are not
        equivalent to anything, including each other: nothing is known about
        a variable that does not say what its units are.
    """
    left = normalize_units(first)
    right = normalize_units(second)
    if not left or not right:
        return False
    return left == right


##############
# END OF FILE
