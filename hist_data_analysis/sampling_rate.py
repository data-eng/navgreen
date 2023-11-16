import pandas as pd

from navgreen_base import process_data

from hist_data_analysis import similarity


FIGURES_DIR = "figures"


def undersampling_error( t, v ):
    retv = pd.DataFrame(
        columns=["Undersampling",
                 "rmse_means", "rmse_stds",
                 "rmse_slopes", "rmse_intercepts",
                 "rmse_spectr_pow"] )

    for i in range(3,8):
        step = 2**i
        #print( "  When keeping 1 out of {}".format(step) )
        sim = similarity( t, v, t[::step], v[::step] )
        sim["plot_means"].savefig( FIGURES_DIR + "/means-{}.png".format(step) )
        sim["plot_slopes"].savefig( FIGURES_DIR + "/slopes-{}.png".format(step) )
        retv.loc[len(retv.index)] = [step,
                                     sim["rmse_means"], sim["rmse_stds"],
                                     sim["rmse_slopes"], sim["rmse_intercepts"],
                                     sim["rmse_spectr_pow"]]
    #end for
    return retv
#end def undersampling_error


def runme():
    df = pd.read_csv("data/DATA_FROM_PLC.csv", parse_dates=["Date&time"], low_memory=False)
    df = process_data(df, hist_data=True)

    dfQ4=df[(df['DATETIME'] > '2022-10-01') & (df['DATETIME'] < '2023-01-01')]
    err = None
    for col in ["PVT_IN_TO_DHW"]:
        #print( "{} {}".format(col, df[col].isna().sum()) )
        err = undersampling_error( dfQ4["DATETIME"], dfQ4[col] )
    return err
