import pandas as pd
import logging

from navgreen_base import process_data

from hist_data_analysis import similarity

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
# Configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(stream_handler)

FIGURES_DIR = "figures"


def undersampling_error( t, v ):
    retv = pd.DataFrame(
        columns=["Undersampling",
                 "rmse_means", "rmse_stds",
                 "rmse_slopes", "rmse_intercepts",
                 "rmse_spectr_pow"] )

    for i in range(3,8):
        step = 2**i
        logger.debug( "  When keeping 1 out of {}".format(step) )
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
    df = pd.read_csv("data/DATA_FROM_PLC_CONV.csv", parse_dates=["DATETIME"], low_memory=False)
    df = process_data(df)

    dfQ4=df[(df['DATETIME'] > '2022-10-01') & (df['DATETIME'] < '2023-01-01')]
    err = None
    for col in ["PVT_IN_TO_DHW"]:
        logger.debug( "{} {}".format(col, df[col].isna().sum()) )
        err = undersampling_error( dfQ4["DATETIME"], dfQ4[col] )
    return err
