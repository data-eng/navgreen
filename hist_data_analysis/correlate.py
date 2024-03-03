import logging

import pandas
import matplotlib.pyplot as plt

import scipy.signal
import sklearn.linear_model
import navgreen_base


hp_cols = ["DATETIME","BTES_TANK","DHW_BUFFER","POWER_HP","Q_CON_HEAT"]
pv_cols = ["DATETIME","OUTDOOR_TEMP","PYRANOMETER","DHW_BOTTOM","POWER_PVT","Q_PV"]

def load_data():
    df = pandas.read_csv( "data/DATA_FROM_PLC.csv", parse_dates=["Date&time"], low_memory=False )
    if 1==1:
        navgreen_base.process_data( df, hist_data=True )
    else:
        df.rename(columns={
            "3-WAY_EVAP_OPERATION": "THREE_WAY_EVAP_OPERATION",
            "3-WAY_COND_OPERATION": "THREE_WAY_COND_OPERATION",
            "3-WAY_SOLAR_OPERATION": "THREE_WAY_SOLAR_OPERATION",
            "4_WAY_VALVE": "FOUR_WAY_VALVE",
            "Date&time": "DATETIME",
            "RECEIVER LIQUID_OUT": "RECEIVER_LIQUID_OUT"}, inplace=True)
        
    print( "All data: {} rows".format(len(df)) )
    df = df[(df['DATETIME'] > '2022-05-01') & (df['DATETIME'] < '2023-09-01')]
    print( "16 full months, 2022-05 to 2023-08: {} rows".format(len(df)) )
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
            df.drop( c, axis="columns" )
    df.dropna( inplace=True )
    print( "No NaNs: {} rows".format(len(df)) )

    df_hp = df[ hp_cols ]
    print( "HP: {} rows".format(len(df)) )

    df_pv = df[ pv_cols ]
    df_pv = df_pv.loc[ df_pv["PYRANOMETER"] > 0.15 ]
    print( "PV, PYRAN above 0.15: {} rows".format(len(df)) )

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
    df = df.loc[ df["POWER_HP"] > 1000 ]
    X = df[ ["BTES_TANK","DHW_BUFFER"] ]
    y = df[ "POWER_HP" ]
    return X, y
#end def prepare_hp


def prepare_pv( df ):
    X = df[ ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"] ]
    y1 = df[ "POWER_PVT" ]    
    y2 = df[ "Q_PV" ]    
    return X, y1, y2

def prepare_pv_diff( df ):
    X = df[ ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "PREV_POWER_PVT", "DIFF_POWER_PVT"] ]
    y = df[ "POWER_PVT" ]    
    return X, y


df, df_hp, df_pv = load_data()

train = df_pv[(df_pv.DATETIME > "2023-08-01") & (df_pv.DATETIME < "2023-08-21")]
# Drop NaN again, as there are empty 3h periods
train = train.set_index( "DATETIME" ).resample( "3h" ).mean().dropna()
print( "1-20 Aug 2023, groupby 3h: {} rows".format(len(train)) )
test = df_pv[(df_pv.DATETIME > "2023-08-21") & (df_pv.DATETIME < "2023-09-01")]
test = test.set_index( "DATETIME" ).resample( "3h" ).mean().dropna()
print( "21-31 Aug 2023, groupby 3h: {} rows".format(len(test)) )
X, y1, y2 = prepare_pv( train )
m1 = sklearn.linear_model.LinearRegression()
m1.fit( X, y1 )
print( "LR, result: y = {} * X + {}".format(m1.coef_,m1.intercept_) ) 
print( "LR, training set: {}".format(m1.score(X,y1)) ) 
m2 = sklearn.linear_model.LinearRegression()
m2.fit( X, y2 )
print( "LR, result: y = {} * X + {}".format(m2.coef_,m2.intercept_) ) 
print( "LR, training set: {}".format(m2.score(X,y2)) ) 
X, y1, y2 = prepare_pv( test )
print( "LR, test set: {}".format(m1.score(X,y1)) ) 
print( "LR, test set: {}".format(m2.score(X,y2)) ) 

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
