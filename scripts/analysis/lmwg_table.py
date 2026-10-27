"""Land (CLM) name for the ADF variable table.

The table itself is made by ``amwg_table``, which makes the land working
group's (LMWG) version when the cases are land cases.  This name is kept so
that land config files can list ``lmwg_table``.
"""

from amwg_table import amwg_table


def lmwg_table(adf):
    """Make the LMWG table of global land means or totals.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object that contains all the configuration information.

    Notes
    -----
    Uses ``adf.model_component`` and ``amwg_table``; see ``amwg_table`` for
    what the table holds and which variable defaults it reads.
    """
    if adf.model_component != "lnd":
        print("\t WARNING: 'lmwg_table' is for land cases, so an AMWG table will be made instead.")
    # End if
    amwg_table(adf)
