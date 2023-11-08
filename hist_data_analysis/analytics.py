import numpy as np
import pandas as pd
import scipy.stats.mstats

def analyse( t, v ):
    """
    Calculates analytics over a timeseries t,v
    :param t: Array-like, timestamps
    :param v: Array-like, values
    :return: (dict) A variety of analytics
    """

    retv = {}
    retv["mean"] = np.mean( v )
    retv["std"]  = np.std( v )
    retv["distr"]= np.histogram( v[np.isfinite(v)], 7 )

    # Make slices of 1000 items each
    slice_len = 1000
    slopes = []
    intercepts = []
    for i in range(0,len(t),slice_len):
        if i+slice_len < len(t):
            t_slice = t[i::i+slice_len]
            v_slice = v[i::i+slice_len]
        else:
            t_slice = t[i::]
            v_slice = v[i::]
        t_slice = t_slice[np.isfinite(v_slice)]
        v_slice = v_slice[np.isfinite(v_slice)]
        if len(v_slice) > 0:
            theil = scipy.stats.mstats.theilslopes( v_slice, t_slice )
            slopes.append( theil.slope )
            intercepts.append( theil.intercept )
        else:
            slopes.append( np.nan )
            intercepts.append( np.nan )
    retv["trend_slopes"] = np.array(slopes)
    retv["trend_intercepts"] = np.array(intercepts)

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

    retv["d_mean"] = np.abs( anal2["mean"] - anal1["mean"] )
    retv["d_std"]  = np.abs( anal2["std"]  - anal1["std"] )
    #retv["rmse_dist"] = np.sqrt(np.mean( (anal2["distr"]-anal1["distr"])**2 ))
    retv["rmse_slopes"] = np.sqrt(np.mean( (anal2["trend_slopes"]-anal1["trend_slopes"])**2 ))
    retv["rmse_intercepts"] = np.sqrt(np.mean( (anal2["trend_intercepts"]-anal1["trend_intercepts"])**2 ))
    return retv
    
#end def rmse
