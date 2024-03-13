import logging
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import scipy.signal
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics

import navgreen_base

# Configure logger and set its level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Configure format
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
# Configure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# Add handler to logger
logger.addHandler(stream_handler)

data = {
    "hp": {
        "cols": ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"],
        "feats": ["BTES_TANK", "DHW_BUFFER"],
        "pred1": "POWER_HP",
        "pred2": "Q_CON_HEAT"
    },
    "pv": {
        "cols": ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"],
        "feats": ["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"],
        "pred1": "POWER_PVT",
        "pred2": "Q_PVT"
    }
}

def load_data(print_stats=False):
    """
    Loads the data from the historical data DataFrame
    :return: whole dataframe, dataframe for hp, dataframe for solar
    """
    df = pd.read_csv("data/DATA_FROM_PLC.csv", parse_dates=["Date&time"], low_memory=False)
    df = navgreen_base.process_data(df, hist_data=True)

    logger.info("All data: {} rows".format(len(df)))
    df = df[(df['DATETIME'] > '2022-08-31') & (df['DATETIME'] < '2023-09-01')]
    logger.info("12 full months, 2022-09 to 2023-08: {} rows".format(len(df)))
    df = df.loc[df["FOUR_WAY_VALVE"] == "HEATING"]
    logger.info("HEATING: {} rows".format(len(df)))

    # Add calculated columns (thermal flows)
    df["Q_CON_HEAT"] = 4.18 * (998.0 / 3600.0 * df["FLOW_CONDENSER"]) * (df["WATER_OUT_COND"] - df["WATER_IN_COND"])
    df["Q_PVT"] = ((3.6014 + 0.004 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.000002 *
                    pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) *
                   ((1049.0 - 0.475 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.0018 *
                     pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) / 3600.0 * df["FLOW_PVT"]) *
                   (df["PVT_OUT"] - df["PVT_IN"]))

    # Drop unneeded columns and then drop rows with NaN
    # to avoid dropping rows with NaNs in cols we don't need anyway
    # Cannot be done on df_hp,df_pv views, modifying views is unsafe.
    for c in df.columns:
        if (c not in data["hp"]["cols"]) and (c not in data["pv"]["cols"]):
            df.drop(c, axis="columns", inplace=True)
    df.dropna(inplace=True)
    logger.info("No NaNs: {} rows".format(len(df)))

    df_hp = df[data["hp"]["cols"]]
    df_hp = df_hp[df_hp["POWER_HP"] > 1.0]
    logger.info("HP, POWER > 1: {} rows".format(len(df_hp)))

    df_pv = df[data["pv"]["cols"]]
    df_pv = df_pv[df_pv["PYRANOMETER"] > 0.15]
    logger.info("PV, PYRAN > 0.15: {} rows".format(len(df_pv)))

    if print_stats:
        # No need to re-run every time
        # Print some monthly stats about the data
        for m in range(9, 21):
            df_monthly = df[df.DATETIME.dt.month == ((m - 1) % 12) + 1]
            logger.info("{} to {}".format(df_monthly.DATETIME.dt.date.iloc[0],
                                          df_monthly.DATETIME.dt.date.iloc[-1]))
            for col in df_monthly.columns:
                logger.info("  {}: {} {} {}".format(col, df_monthly[col].min(),
                                                    df_monthly[col].mean(), df_monthly[col].max()))
                
    return df, df_hp, df_pv

def normalize(df, cols):
    """
    Normalize and de-trend data
    :param df: dataframe
    :param cols: columns to apply the transformation
    :return: processed dataframe
    """
    newdf = {}
    for col in cols:
        series = df[col]
        series = (series - series.mean()) / series.std()
        series = scipy.signal.detrend(series)
        newdf[col] = series
    return pd.DataFrame(newdf)

class Function:
    def __init__(self, terms):
        self.terms = terms

    def exe(self, df):
        result = pd.DataFrame()
        cols = df.columns
        num_params = len(cols)
        for combo in permutations(cols, num_params):
            name = self.format(combo)
            values = [df[param] for param in combo]
            result[name] = self.output(values)
        return result

    def output(self, values):
        return sum(w * (val ** e) for (w, e), val in zip(self.terms, values))

    def format(self, params):
        terms = [f"{('+' if w >= 0 else '')}{w}*{param}^{e}" for (w, e), param in zip(self.terms, params)]
        return ' '.join(terms)

def prepare(mode, df, f):
    """
    Create input and output values for a heatpump PLC related ML problem
    :param df: dataframe
    :return: input, output1, output2
    """
    X = df[mode["feats"]]
    y1 = df[mode["pred1"]]
    y2 = df[mode["pred2"]]
    return f(X), y1, y2

def make_plot(index, label, X, realY, predY):
    """
    Plot ML model results
    :param index: number of plot
    :param label: plot's label
    :param X: input
    :param realY: real y label
    :param predY: predicted y label
    :return: figure
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(index, realY, 'r-', label="Real values")
    ax[0].plot(index, predY, 'g-', label="Pred values")
    xx = X.to_numpy()
    ax[1].plot(index, realY, 'r-', label="Real values")
    ax[1].plot(index, xx[:, 0], 'b-', label="X[0]")
    # ax[1].plot(index, xx[:, 1], 'b-', label="X[1]")
    fig.supxlabel(label)
    return fig
# end def make_plot

def make_scatter(data):
    """
    Create scatter plot of data
    :param data: 3-tuple of input and output values
    :return: figure
    """
    X, Y1, Y2 = data
    if len(X.columns) == 1:
        fig, ax = plt.subplots(2, 1)
        for i, y in enumerate([Y1, Y2]):
            ax[i].set_xlabel(X.columns[0])
            ax[i].set_ylabel(y.name)
            ax[i].scatter(x=X[X.columns[0]], y=y)
    else:
        fig, ax = plt.subplots(2, len(X.columns))
        for i, y in enumerate([Y1, Y2]):
            for j, xlabel in enumerate(X.columns):
                ax[i, j].set_xlabel(xlabel)
                ax[i, j].set_ylabel(y.name)
                ax[i, j].scatter(x=X[xlabel], y=y)
    fig.subplots_adjust(wspace=2, hspace=5)
    plt.tight_layout()
    return fig

def visualize(name, df):
    for mm in range(9, 21):
        m = ((mm - 1) % 12) + 1
        newdf = df[df.DATETIME.dt.month == m]
        f = Function(terms=[(0, 1), (-1, 2), (3, 1)]).exe

        X, y1, y2 = prepare(data[name.lower()], newdf, f)
        ff = make_scatter(data=(X, y1, y2))
        ff.savefig("{}_{}.png".format(name, m))
        plt.close()

def main():
    logger.info("Data visualization for Heat Pump (HP) and Photovoltaic (PV) systems.")

    _, df_hp, df_pv = load_data()

    visualize("HP", df_hp)
    visualize("PV", df_pv)