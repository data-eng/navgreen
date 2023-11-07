import pandas as pd


def load_csv():
    df = pd.read_csv( "data/DATA_FROM_PLC.csv",
                      parse_dates=["Date&time"], low_memory=False )
    return df
#end def load_csv


def estimate_sampling_rate( df ):
    return df.shape[0]
#end def estimate


def runme():
    df = load_csv()
    r = estimate_sampling_rate( df )
    print( r )
