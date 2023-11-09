import numpy as np
import pandas as pd
import scipy.stats.mstats

import matplotlib.pyplot as plt


def analyse( t, v ):
    """
    Calculates analytics over a timeseries t,v
    :param t: Array-like, timestamps
    :param v: Array-like, values
    :return: (dict) A variety of analytics
    """

    # Should be params with defaults
    slice_len = 1000
    distr_bins = 7

    means = []
    stds = []
    distrs = []
    slopes = []
    intercepts = []

    # For consistency, pre-compute bin edges 
    bin_edges = np.histogram_bin_edges( v[np.isfinite(v)], distr_bins )
    
    for i in range(0,len(t),slice_len):
        # Make slices of `slice_len` items each
        if i+slice_len < len(t):
            t_slice = t[i:i+slice_len]
            v_slice = v[i:i+slice_len]
        else:
            t_slice = t[i:]
            v_slice = v[i:]
        t_slice = t_slice[np.isfinite(v_slice)]
        v_slice = v_slice[np.isfinite(v_slice)]

        if len(v_slice) > 2:
            # Distribution of values
            means.append( np.mean(v_slice) )
            stds.append( np.std(v_slice) )
            # Returns histogram, edges tuple. Ignore edges
            (h,_) = np.histogram( v_slice, bins=bin_edges )
            distrs.append( h )
            # Slope and intercept
            theil = scipy.stats.mstats.theilslopes( v_slice, t_slice )
            slopes.append( theil.slope )
            intercepts.append( theil.intercept )
            # Todo: spectral

        else:
            slopes.append( np.nan )
            intercepts.append( np.nan )

    retv = {}
    retv["values"] = {}
    retv["values"]["means"] = np.array(means)
    retv["values"]["stds"] = np.array(stds)
    retv["values"]["distrs"] = np.array(distrs)
    retv["trend"] = {}
    retv["trend"]["slopes"] = np.array(slopes)
    retv["trend"]["intercepts"] = np.array(intercepts)

    return retv
#end def analyse


def similarity( t, v1, v2 ):
    """
    Calculates various metrics that measure how different two
    time-aligned timeseries appear when different analytics is
    applied to them.
    :param t: Array-like, timestamps
    :param v1: Array-like, values
    :param v2: Array-like, values
    :return: dict with different metrics:
       mean absolute error and RMS error when directly comparing values
    """
    retv = {
        "mae": np.mean(np.abs(v2-v1)),
        "rmse":np.sqrt(np.mean((v2-v1)**2))
    }
    anal1 = analyse( t, v1 )
    anal2 = analyse( t, v2 )

    retv["rmse_means"] = np.sqrt(np.mean( (anal2["values"]["means"]-anal1["values"]["means"])**2 ))
    retv["rmse_stds"]  = np.sqrt(np.mean( (anal2["values"]["stds"]-anal1["values"]["stds"])**2 ))
    fig,ax = plt.subplots()
    x1 = np.arange( 0.0, len(anal1["values"]["means"]) )
    x2 = np.arange( 0.3, len(anal2["values"]["means"]) )
    ax.errorbar(x1, anal1["values"]["means"], anal1["values"]["stds"], linestyle='None', marker='x')
    ax.errorbar(x2, anal2["values"]["means"], anal2["values"]["stds"], linestyle='None', marker='^')
    retv["plot_means"]  = fig

    retv["rmse_slopes"] = np.sqrt(np.mean( (anal2["trend"]["slopes"]-anal1["trend"]["slopes"])**2 ))
    retv["rmse_intercepts"] = np.sqrt(np.mean( (anal2["trend"]["intercepts"]-anal1["trend"]["intercepts"])**2 ))
    fig,ax = plt.subplots()
    x1 = np.arange( 0.0, len(anal1["trend"]["slopes"]) )
    ax.errorbar(x1, anal1["trend"]["slopes"] )
    ax.errorbar(x2, anal2["trend"]["slopes"] )
    retv["plot_slopes"]  = fig

    return retv
    
#end def rmse
