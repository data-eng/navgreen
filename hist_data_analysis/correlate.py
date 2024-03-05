import logging

import pandas
import matplotlib.pyplot as plt

import scipy.signal
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics

import navgreen_base


hp_cols = ["DATETIME","BTES_TANK","DHW_BUFFER","POWER_HP","Q_CON_HEAT"]
pv_cols = ["DATETIME","OUTDOOR_TEMP","PYRANOMETER","DHW_BOTTOM","POWER_PVT","Q_PV"]

def load_data():
    df = pandas.read_csv( "data/DATA_FROM_PLC.csv", parse_dates=["Date&time"], low_memory=False )
    navgreen_base.process_data( df, hist_data=True )
        
    print( "All data: {} rows".format(len(df)) )
    df = df[(df['DATETIME'] > '2022-08-31') & (df['DATETIME'] < '2023-09-01')]
    print( "12 full months, 2022-09 to 2023-08: {} rows".format(len(df)) )
    df = df.loc[ df["FOUR_WAY_VALVE"] == "HEATING" ]
    print( "HEATING: {} rows".format(len(df)) )

    # Add calculated dolumns
    df["Q_CON_HEAT"] = 4.18 * (998.0/3600.0 * df["FLOW_CONDENSER"]) * (df["WATER_OUT_COND"] - df["WATER_IN_COND"])
    df["Q_PV"] = (3.6014 + 0.004*(df["PVT_IN"] + df["PVT_OUT"])/2.0 - 0.000002*pow((df["PVT_IN"] + df["PVT_OUT"])/2.0,2)) * ((1049.0- 0.475*(df["PVT_IN"] + df["PVT_OUT"])/2.0-0.0018*pow((df["PVT_IN"] + df["PVT_OUT"])/2.0,2))/3600.0*df["FLOW_PVT"])*(df["PVT_OUT"] - df["PVT_IN"])

    # Drop unneeded columns and then drop rows with NaN
    # to avoid dropping rows with NaNs in cols we don't need anyway
    # Cannot be done on df_hp,df_pv views, modifying views is unsafe.
    for c in df.columns:
        if (c not in hp_cols) and (c not in pv_cols):
            df.drop( c, axis="columns", inplace=True )
    df.dropna( inplace=True )
    print( "No NaNs: {} rows".format(len(df)) )

    df_hp = df[ hp_cols ]
    df_hp = df_hp[ df_hp["POWER_HP"] > 1 ]
    print( "HP, POWER > 1: {} rows".format(len(df)) )

    df_pv = df[ pv_cols ]
    df_pv = df_pv[ df_pv["PYRANOMETER"] > 0.15 ]
    print( "PV, PYRAN > 0.15: {} rows".format(len(df)) )

    # Print some monthly stats about the data
    for m in range(9,21):
        df_monthly = df[ df.DATETIME.dt.month == ((m-1)%12)+1 ]
        print( "{} to {}".format(
            df_monthly.DATETIME.dt.date.iloc[0],
            df_monthly.DATETIME.dt.date.iloc[-1]) )
        for col in df_monthly.columns:
            print( "  {}: {} {} {}".format(
                col,df_monthly[col].min(),
                df_monthly[col].mean(),df_monthly[col].max()) )

    return df, df_hp, df_pv
#end def load_data


def normalize( df, cols ):
    newdf = {}
    for col in cols:
        series = df[col]
        series = (series - series.mean()) / series.std()
        series = scipy.signal.detrend( series )
        newdf[col] = series
    return pandas.DataFrame( newdf )
#end def prepreprocess


def prepare_hp( df, with_diff=False ):
    X = df[ ["BTES_TANK","DHW_BUFFER"] ]
    y1 = df[ "POWER_HP" ]
    y2 = df[ "Q_CON_HEAT" ]
    return X, y1, y2
#end def prepare_hp


def prepare_pv( df ):
    X = df[ ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"] ]
    y1 = df[ "POWER_PVT" ]    
    y2 = df[ "Q_PV" ]    
    return X, y1, y2
#end def prepare_pv

# Makes a list of 2nd-order-poly feature names
# in the same order as returned by converter
def make_feature_names( converter ):
    retv = []
    for line in converter.powers_:
        name = ""
        for i,exp in enumerate(line):
            if exp == 2:
                name = "{}^2".format(converter.feature_names_in_[i])
            if exp == 1:
                if name == "":
                    name = converter.feature_names_in_[i]
                else:
                    name = name + "*" + converter.feature_names_in_[i]
        #end for exp
        retv.append(name)
    #end for line
    return retv
#end def

def make_plot( index, label, X, realY, predY ):
    fig,ax = plt.subplots(1,2)
    ax[0].plot( index, realY, 'r-',label="Real values" )
    ax[0].plot( index, predY, 'g-',label="Pred values" )
    xx = X.to_numpy()
    ax[1].plot( index, realY, 'r-',label="Real values" )
    ax[1].plot( index, xx[:,0],'b-',label="X[0]" )
    ax[1].plot( index, xx[:,1],'b-',label="X[1]" )
    fig.supxlabel( label )
    return fig
#end def make_plot


# Trains and applies model and returns
# [val_score,val_rel_err,test_score,test_rel_err] array,
# relative error and score (fraction of variance in
# dependent var that is explained by indep variables.
# val_: validation on training data itself
# test_: on unseen data
# If plot_index is the x axis (time points), it also
# plots predictions, truths over the test data.
def regress( model, X, y, testX, testY, plot_index=None, label=None ):
    model.fit( X, y )
    score = round(100*model.score( X, y ))
    yy = model.predict( X )
    err = round(100*sklearn.metrics.mean_absolute_error(y,yy)/y.mean())
    retv = [score, err]
    yy = model.predict( testX )
    err = round(100*sklearn.metrics.mean_absolute_error(testY,yy)/y.mean())
    retv.extend( [score, err] )
    if plot_index is None:
        return (retv,None)
    else:
        p = make_plot( plot_index, label, testX, testY, yy )
        return (retv,p)
#end def regress


df, df_hp, df_pv = load_data()
grp = "6h"

print( "##### SCENARIO 1 #####" )
print( "Each month, train on 20 days and test on rest" )
print()

if 1==1:
    mydf = df_hp
    prepare = prepare_hp
    name = "PV"
else:
    mydf = df_pv
    prepare = prepare_pv
    name = "HP"

results = []
for mm in range(9,21):
    m = ((mm-1)%12)+1
    mydf1 = mydf[ mydf.DATETIME.dt.month == m ]
    # Drop NaN again, as there might be empty periods
    train = mydf1[mydf1.DATETIME.dt.day<21].set_index( "DATETIME" ).resample( grp ).mean().dropna()
    test = mydf1[mydf1.DATETIME.dt.day>=21].set_index( "DATETIME" ).resample( grp ).mean().dropna()
    X, y1, y2 = prepare( train )
    testX, testY1, testY2 = prepare( test )
    #print( "X: {}, y1: {}, y2: {}, testX: {}, testY1: {}, testY2:{}".format(
    #    X.shape, y1.shape, y2.shape, testX.shape, testY1.shape, testY2.shape) )

    # Prepare features for regressing on squares
    converter = sklearn.preprocessing.PolynomialFeatures( degree=2, include_bias=False )
    sqX = pandas.DataFrame(
        converter.fit_transform(X),
        columns=make_feature_names(converter) )
    testSqX = pandas.DataFrame(
        converter.fit_transform(testX),
        columns=make_feature_names(converter) )

    make_plots = None
    #make_plots = test.index

    row = [ name, "POWER", "LR", m ]
    r,p = regress( sklearn.linear_model.LinearRegression(),
                   X, y1, testX, testY1, make_plots )
    row.extend( r )
    results.append( row )
    if p is not None: p.savefig( "{}_{}.png".format("pvt_LR",m) )
    #The actual model is:
    #print( "LR, result: y = {} * X + {}".format(m1.coef_,m1.intercept_) )

    row = [ name, "Q", "LR", m ]
    r,p = regress( sklearn.linear_model.LinearRegression(),
                   X, y2, testX, testY2, make_plots )
    row.extend( r )
    results.append( row )
    if p is not None: p.savefig( "{}_{}.png".format("qpv_LR",m) )

    row = [ name, "POWER", "SQ", m ]
    r,p = regress( sklearn.linear_model.LinearRegression(),
                   sqX, y1, testSqX, testY1, make_plots )
    row.extend( r )
    results.append( row )
    if p is not None: p.savefig( "{}_{}.png".format("pvt_SQ",m) )

    row = [ name, "Q", "SQ", m ]
    r,p = regress( sklearn.linear_model.LinearRegression(),
                   sqX, y2, testSqX, testY2, make_plots )
    row.extend( r )
    results.append( row )
    if p is not None: p.savefig( "{}_{}.png".format("qpv_SQ",m) )
# end for all months

# results is a list of result rows
df_res = pandas.DataFrame(
    results,
    columns=["name","depvar","method","month",
             "val_score","val_rel_err","test_score","test_rel_err"] )

print( df_res[ (df_res.depvar=="POWER") & (df_res.method=="SQ") ]["test_rel_err"] )
print( df_res[ (df_res.depvar=="POWER") & (df_res.method=="LR") ]["test_rel_err"] )


if 1==0:
    d["PREV_POWER_PVT"] = d["POWER_PVT"].shift(1)
    d["DIFF_POWER_PVT"] = d["POWER_PVT"] - d["PREV_POWER_PVT"]
    d.dropna( inplace=True )
    tzise = int(len(d)*0.8)
    d_train  = d.iloc[:tzise,:]
    d_test   = d.iloc[tzise:,:]
    X, y = prepare_pv_diff( d_train )
    m = sklearn.linear_model.LinearRegression()
    m.fit( X, y )
    print( "Group by 6h, with prev, result: y = {} * X + {}".format(m.coef_,m.intercept_) ) 
    print( "Group by 6h, with prev, on training set: {}".format(m.score(X,y)) ) 
    X, y = prepare_pv_diff( d_test )
    print( "Group by 6h, with prev, on test set: {}".format(m.score(X,y)) ) 

    d = df_pv.rolling( "6h", on="DATETIME" ).mean()
    d.dropna( inplace=True )
    tzise = int(len(d)*0.8)
    d_train  = d.iloc[:tzise,:]
    d_test   = d.iloc[tzise:,:]
    X, y = prepare_pv( d_train )
    m = sklearn.linear_model.LinearRegression()
    m.fit( X, y )
    print( "Roll by 6h, result: y = {} * X + {}".format(m.coef_,m.intercept_) ) 
    print( "Roll by 6h, on training set: {}".format(m.score(X,y)) ) 
    X, y = prepare_pv( d_test )
    print( "Roll by 6h, on test set: {}".format(m.score(X,y)) ) 

    d["PREV_POWER_PVT"] = d["POWER_PVT"].shift(1)
    d["DIFF_POWER_PVT"] = d["POWER_PVT"] - d["PREV_POWER_PVT"]
    d.dropna( inplace=True )
    tzise = int(len(d)*0.8)
    d_train  = d.iloc[:tzise,:]
    d_test   = d.iloc[tzise:,:]
    X, y = prepare_pv_diff( d_train )
    m = sklearn.linear_model.LinearRegression()
    m.fit( X, y )
    print( "Roll 6h, with prev, result: y = {} * X + {}".format(m.coef_,m.intercept_) ) 
    print( "Roll by 6h, with prev, on training set: {}".format(m.score(X,y)) ) 
    X, y = prepare_pv_diff( d_test )
    print( "Roll by 6h, with prev, on test set: {}".format(m.score(X,y)) ) 

    #pivot=dforig.pivot_table(columns=pandas.cut(dforig["PYRANOMETER"], bins), aggfunc='size')
