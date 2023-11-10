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
    slice_len = 1024
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


def make_errorbars( data ):
    fig,ax = plt.subplots()
    x0 = 0.0
    for d in data:
        x = np.arange( x0, len(d["means"]) )
        ax.errorbar(x, d["means"], d["stds"], linestyle='None', marker='.')
        # Move the starting point slightly to the right, some that the
        # next graph does not fall exactly over this obne
        x0 += 0.3
    #end for
    return fig
#end def make_errorbars



def make_lines( data ):
    fig,ax = plt.subplots()
    x0 = 0.0
    for d in data:
        x = np.arange( x0, len(d) )
        ax.plot( x, d )
        # Move the starting point slightly to the right, some that the
        # next graph does not fall exactly over this obne
        x0 += 0.3
    return fig
    #end for



def similarity( t1, v1, t2, v2 ):
    """
    Calculates various metrics that measure how different two
    timeseries appear when different analytics is applied to them.
    :param t1: Array-like, timestamps
    :param v1: Array-like, values
    :param t2: Array-like, timestamps
    :param v2: Array-like, values
    :return: dict with different metrics:
       mean absolute error and RMS error when directly comparing values
    """
    # Interpolate from t2,v2 onto the times in t1
    v2_aligned = np.interp( t1, t2, v2 )

    retv = {
        "mae": np.mean(np.abs(v2-v1)),
        "rmse":np.sqrt(np.mean((v2-v1)**2))
    }
    anal1 = analyse( t1, v1 )
    anal2 = analyse( t2, v2 )
    anal2a= analyse( t1, v2_aligned )

    retv["rmse_means"] = np.sqrt(np.mean( (anal2a["values"]["means"]-anal1["values"]["means"])**2 ))
    retv["rmse_stds"]  = np.sqrt(np.mean( (anal2a["values"]["stds"]-anal1["values"]["stds"])**2 ))
    retv["plot_means"] = make_errorbars( [anal1["values"],anal2a["values"]] )

    retv["rmse_slopes"] = np.sqrt(np.mean( (anal2a["trend"]["slopes"]-anal1["trend"]["slopes"])**2 ))
    retv["rmse_intercepts"] = np.sqrt(np.mean( (anal2a["trend"]["intercepts"]-anal1["trend"]["intercepts"])**2 ))
    retv["plot_slopes"] = make_lines( [anal1["trend"]["slopes"],anal2a["trend"]["slopes"]] )

    return retv
    
#end def similarity


