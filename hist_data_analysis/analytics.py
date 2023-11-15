import numpy as np
import pandas as pd
import scipy.stats.mstats

import matplotlib.pyplot as plt


DISTR_BINS = 7
# CAREFUL: must be a power of 2. See below why.
SLICE_LEN = 1024


def analyse( t, v ):
    """
    Calculates analytics over a timeseries t,v
    :param t: Array-like, timestamps
    :param v: Array-like, values
    :return: (dict) A variety of analytics
    """

    means = []
    stds = []
    distrs = []
    slopes = []
    intercepts = []
    amplitudes = []
    powers = []

    # For consistency, pre-compute bin edges 
    bin_edges = np.histogram_bin_edges( v[np.isfinite(v)], DISTR_BINS )
    
    for i in range(0,len(t),SLICE_LEN):
        # Make slices of `SLICE_LEN` items each
        if i+SLICE_LEN < len(t):
            t_slice = t[i:i+SLICE_LEN]
            v_slice = v[i:i+SLICE_LEN]
        else:
            t_slice = t[i:]
            v_slice = v[i:]
        t_slice = t_slice[np.isfinite(v_slice)]
        v_slice = v_slice[np.isfinite(v_slice)]

        l = len(v_slice)
        if l > 2:
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
            if (l & (l-1) == 0):
                # Only calculate spectral amplitudes if
                # len is a power of 2.
                # CAREFUL: `SLICE_LEN` above must be a power
                # of 2 so we only lose the very last slice.
                # TODO: FFT assumes values are uniformly distributed
                # in time, so we should have checked for gaps and NaNs.
                f = np.fft.fft(v_slice)
                amplitudes.append( f )
                powers.append( (np.abs(f)/SLICE_LEN)**2 )
        else:
            means.append( np.nan )
            stds.append( np.nan )
            distrs.append( np.tile([np.nan],DISTR_BINS) )
            slopes.append( np.nan )
            intercepts.append( np.nan )
            amplitudes.append( np.tile([np.nan],SLICE_LEN) )
            powers.append( np.tile([np.nan],SLICE_LEN) )

    retv = {}
    retv["values"] = {}
    retv["values"]["means"] = np.array(means)
    retv["values"]["stds"] = np.array(stds)
    retv["values"]["distrs"] = np.array(distrs)
    retv["trend"] = {}
    retv["trend"]["slopes"] = np.array(slopes)
    retv["trend"]["intercepts"] = np.array(intercepts)
    retv["spectral"] = {}
    retv["spectral"]["amplitudes"] = np.array(amplitudes)
    retv["spectral"]["powers"] = np.array(powers)

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
    #end for
    return fig
#end def make_lines



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
    retv["rmse_spectr_pow"] = np.sqrt(np.mean( (anal2a["spectral"]["powers"]-anal1["spectral"]["powers"])**2 ))

    return retv
    
#end def similarity


