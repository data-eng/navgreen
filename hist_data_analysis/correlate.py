import pandas as pd
import logging
import matplotlib.pyplot as plt

import scipy.signal
import sklearn.linear_model
import navgreen_base


def load_data():
    df = pd.read_csv( "data/DATA_FROM_PLC.csv", parse_dates=["Date&time"], low_memory=False )
    #df = navgreen_base.process_data( df, hist_data=True )
    #dfQ4=df[(df['DATETIME'] > '2022-10-01') & (df['DATETIME'] < '2023-01-01')]
    df = df.loc[ df["4_WAY_VALVE"] == "HEATING" ]

    df["Q_CON_HEAT"] = 4.18 * (998.0/3600.0 * df["FLOW_CONDENSER"]) * (df["WATER_OUT_COND"] - df["WATER_IN_COND"])
    df_hp = df[ ["Date&time","BTES_TANK","DHW_BUFFER","POWER_HP","Q_CON_HEAT"] ]

    df["Q_PV"] = (3.6014 + 0.004*(df["PVT_IN"] + df["PVT_OUT"])/2.0 - 0.000002*pow((df["PVT_IN"] + df["PVT_OUT"])/2.0,2)) * ((1049.0- 0.475*(df["PVT_IN"] + df["PVT_OUT"])/2.0-0.0018*pow((df["PVT_IN"] + df["PVT_OUT"])/2.0,2))/3600.0*df["FLOW_PVT"])*(df["PVT_OUT"] - df["PVT_IN"])
    df_pv = df[ ["Date&time","OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM","POWER_PVT","Q_PV"] ]

    return df_hp, df_pv
#end def preprocess_data

def prepare_hp( df ):
    df = df.loc[ df["POWER_HP"] > 1000 ]

    # Not on discontinous data
    #scipy.signal.detrend( df["POWER_HP"] )

    Q_CON_HEAT = 4.18 * (998.0/3600.0 * df["FLOW_CONDENSER"]) * (df["WATER_OUT_COND"] - df["WATER_IN_COND"])

    if 1==0:
        d = {}
        series = df["BTES_TANK"]
        all["BTES_TANK"] = (series - series.mean()) / series.std()
        all["BTES_TANK"] = scipy.signal.detrend( all["BTES_TANK"] )

        series = df["DHW_BUFFER"]
        all["DHW_BUFFER"] = (series - series.mean()) / series.std()
        all["DHW_BUFFER"] = scipy.signal.detrend( all["DHW_BUFFER"] )

        series = df["POWER_HP"]
        all["POWER_HP"] = (series - series.mean()) / series.std()
        all["POWER_HP"] = scipy.signal.detrend( all["POWER_HP"] )

        series = Q_CON_HEAT
        all["Q_CON_HEAT"] = (series - series.mean()) / series.std()
        all["Q_CON_HEAT"] = scipy.signal.detrend( all["Q_CON_HEAT"] )

        data = pd.DataFrame( d )

    X = df[ ["BTES_TANK","DHW_BUFFER"] ]
    y = df[ "POWER_HP" ]

    return X, y

def prepare_pv( df ):
    X = df[ ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"] ]
    y = df[ "POWER_PVT" ]    
    return X, y

def prepare_pv_diff( df ):
    X = df[ ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "PREV_POWER_PVT", "DIFF_POWER_PVT"] ]
    y = df[ "POWER_PVT" ]    
    return X, y


df_hp, df_pv = load_data()

df_pv.dropna( inplace=True )
df_pv = df_pv.loc[ df_pv["PYRANOMETER"] > 0.15 ]

d = df_pv.set_index( "Date&time" ).resample( "6h" ).mean()
# there are more NaNs, because some 6h-spans are empty
d.dropna( inplace=True )
tzise = int(len(d)*0.8)
d_train  = d.iloc[:tzise,:]
d_test   = d.iloc[tzise:,:]
X, y = prepare_pv( d_train )
m = sklearn.linear_model.LinearRegression()
m.fit( X, y )
print( "Group by 6h, on training set: {}".format(m.score(X,y)) ) 
X, y = prepare_pv( d_test )
print( "Group by 6h, on test set: {}".format(m.score(X,y)) ) 

d["PREV_POWER_PVT"] = d["POWER_PVT"].shift(1)
d["DIFF_POWER_PVT"] = d["POWER_PVT"] - d["PREV_POWER_PVT"]
d.dropna( inplace=True )
tzise = int(len(d)*0.8)
d_train  = d.iloc[:tzise,:]
d_test   = d.iloc[tzise:,:]
X, y = prepare_pv_diff( d_train )
m = sklearn.linear_model.LinearRegression()
m.fit( X, y )
print( "Group by 6h, with prev, on training set: {}".format(m.score(X,y)) ) 
X, y = prepare_pv_diff( d_test )
print( "Group by 6h, with prev, on test set: {}".format(m.score(X,y)) ) 

d = df_pv.rolling( "6h", on="Date&time" ).mean()
d.dropna( inplace=True )
tzise = int(len(d)*0.8)
d_train  = d.iloc[:tzise,:]
d_test   = d.iloc[tzise:,:]
X, y = prepare_pv( d_train )
m = sklearn.linear_model.LinearRegression()
m.fit( X, y )
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
print( "Roll by 6h, with prev, on training set: {}".format(m.score(X,y)) ) 
X, y = prepare_pv_diff( d_test )
print( "Roll by 6h, with prev, on test set: {}".format(m.score(X,y)) ) 


#pivot=dforig.pivot_table(columns=pd.cut(dforig["PYRANOMETER"], bins), aggfunc='size')
