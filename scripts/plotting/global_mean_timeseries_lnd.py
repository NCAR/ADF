"""Land (CLM) name for the global mean time series plots.

The plots are made by ``global_mean_timeseries``, which averages over land
when the cases are land cases.  This name is kept so that land config files
can list ``global_mean_timeseries_lnd``.
"""

from global_mean_timeseries import global_mean_timeseries


def global_mean_timeseries_lnd(adfobj):
    """Make global mean (or total) time series plots of land variables.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information.

    Notes
    -----
    Uses ``adfobj.model_component`` and ``global_mean_timeseries``; see
    ``global_mean_timeseries`` for the land averaging and the variable
    defaults it reads.
    """
    if adfobj.model_component != "lnd":
        print(
            "\t WARNING: 'global_mean_timeseries_lnd' is for land cases, so the"
            " cases will be averaged as atmosphere data instead."
        )
    # End if
    global_mean_timeseries(adfobj)
