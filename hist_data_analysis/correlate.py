import pandas as pd
import matplotlib.pyplot as plt

import scipy.signal
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics

import navgreen_base

import logging

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

hp_cols = ["DATETIME", "BTES_TANK", "DHW_BUFFER", "POWER_HP", "Q_CON_HEAT"]
pv_cols = ["DATETIME", "OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM", "POWER_PVT", "Q_PVT"]


def load_data():
    """
    Loads the data from the historical data DataFrame
    :return: whole dataframe, dataframe for hp, dataframe for solar
    """
    df = pd.read_csv("data/DATA_FROM_PLC_CONV.csv", parse_dates=["DATETIME"], low_memory=False)
    df = navgreen_base.process_data(df)

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
        if (c not in hp_cols) and (c not in pv_cols):
            df.drop(c, axis="columns", inplace=True)
    df.dropna(inplace=True)
    logger.info("No NaNs: {} rows".format(len(df)))

    df_hp = df[hp_cols]
    df_hp = df_hp[df_hp["POWER_HP"] > 1.0]
    logger.info("HP, POWER > 1: {} rows".format(len(df_hp)))

    df_pv = df[pv_cols]
    df_pv = df_pv[df_pv["PYRANOMETER"] > 0.15]
    logger.info("PV, PYRAN > 0.15: {} rows".format(len(df_pv)))

    return df_hp, df_pv
# end def load_data


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
# end def normalize


def prepare_hp(df):
    """
    Create input and output values for a heatpump PLC related ML problem
    :param df: dataframe
    :return: input, output1, output2
    """
    X = df[["BTES_TANK", "DHW_BUFFER"]]
    y1 = df["POWER_HP"]
    y2 = df["Q_CON_HEAT"]
    return X, y1, y2
# end def prepare_hp


def prepare_pv(df):
    """
    Create input and output values for a solar PLC related ML problem
    :param df: dataframe
    :return: input, output1, output2
    """
    X = df[["OUTDOOR_TEMP", "PYRANOMETER", "DHW_BOTTOM"]]
    y1 = df["POWER_PVT"]
    y2 = df["Q_PVT"]
    return X, y1, y2
# end def prepare_pv


def make_feature_names(converter):
    """
    Makes a list of 2nd-order-poly feature names in the same order as returned by converter
    :param converter: feature converter
    :return: list of names
    """
    retv = []
    for line in converter.powers_:
        name = ""
        for i, exp in enumerate(line):
            if exp == 2:
                name = "{}^2".format(converter.feature_names_in_[i])
            if exp == 1:
                name = converter.feature_names_in_[i] if name == "" else name + "*" + converter.feature_names_in_[i]
        # end for exp
        retv.append(name)
    # end for line
    return retv
# end def make_feature_names

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
    fig, ax = plt.subplots(2, len(X.columns))
    for i, y in enumerate([Y1, Y2]):
        for j, xlabel in enumerate(X.columns):
            ax[i, j].set_xlabel(xlabel)
            ax[i, j].set_ylabel(y.name)
            ax[i, j].scatter(x=X[xlabel], y=y)
    return fig
# end def make_scatter


def regress(model, X, y, testX, testY, plot_index=None, label=None):
    """
    Trains and applies model
    :param model: ML model
    :param X: input train/valid data
    :param y: output train/valid data
    :param testX: input test data
    :param testY: output test data
    :param plot_index: if plot_index is the x-axis (time points), it also plots predictions, truths over the test data
    :param label: label to show on plot
    :return: [val_score,val_rel_err,test_score,test_rel_err] array,  relative error and score (fraction of variance in
    dependent var that is explained by indep variables) [val_: validation on training data itself & test_: on unseen data]
    """
    model.fit(X, y)
    score = round(100 * model.score(X, y))
    yy = model.predict(X)
    err = round(100 * sklearn.metrics.mean_absolute_error(y, yy) / y.mean())
    retv = [score, err]
    yy = model.predict(testX)
    err = round(100 * sklearn.metrics.mean_absolute_error(testY, yy) / y.mean())
    retv.extend([float('inf'), err])
    if plot_index is None:
        return retv, None
    else:
        p = make_plot(plot_index, label, testX, testY, yy)
        return retv, p
# end def regress


def scenario1(name, mydf, prepare, grp):
    """
    Plot/Training-test scenario 1
    :param name: name if the scenario (atm hp, pv)
    :param mydf: dataframe
    :param prepare: function that prepares I/O fot ML task and plots
    :param grp: frequency to group data by
    :return:
    """
    results = []
    for mm in range(9, 21):
        m = ((mm - 1) % 12) + 1
        mydf1 = mydf[mydf.DATETIME.dt.month == m]

        X, y1, y2 = prepare(mydf1)
        ff = make_scatter((X, y1, y2))
        ff.savefig("{}_{}.png".format(name, m))
        plt.close()

        # Square features
        sqX = pow(X, 2)
        ff = make_scatter((sqX, y1, y2))
        ff.savefig("sq_{}_{}.png".format(name, m))
        plt.close()

        ff = make_scatter(prepare(mydf1))
        ff.savefig("{}_{}.png".format(name, m))
        plt.close()

        # Drop NaN again, as there might be empty periods
        train = mydf1[mydf1.DATETIME.dt.day < 21].set_index("DATETIME").resample(grp).mean().dropna()
        test = mydf1[mydf1.DATETIME.dt.day >= 21].set_index("DATETIME").resample(grp).mean().dropna()
        X, y1, y2 = prepare(train)
        testX, testY1, testY2 = prepare(test)

        # Prepare features for regressing on squares
        converter = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
        try:
            sqX = pd.DataFrame(converter.fit_transform(X), columns=make_feature_names(converter))
            testSqX = pd.DataFrame(converter.fit_transform(testX), columns=make_feature_names(converter))

            make_plots = None
            # make_plots = test.index

            row = [name, "POWER", "LR", m]
            r, p = regress(sklearn.linear_model.LinearRegression(), X, y1, testX, testY1, make_plots)
            row.extend(r)
            results.append(row)
            if p is not None: p.savefig("{}_{}.png".format("pvt_LR", m))

            row = [name, "Q", "LR", m]
            r, p = regress(sklearn.linear_model.LinearRegression(), X, y2, testX, testY2, make_plots)
            row.extend(r)
            results.append(row)
            if p is not None: p.savefig("{}_{}.png".format("qpv_LR", m))

            row = [name, "POWER", "SQ", m]
            r, p = regress(sklearn.linear_model.LinearRegression(), sqX, y1, testSqX, testY1, make_plots)
            row.extend(r)
            results.append(row)
            if p is not None: p.savefig("{}_{}.png".format("pvt_SQ", m))

            row = [name, "Q", "SQ", m]
            r, p = regress(sklearn.linear_model.LinearRegression(), sqX, y2, testSqX, testY2, make_plots)
            row.extend(r)
            results.append(row)
            if p is not None: p.savefig("{}_{}.png".format("qpv_SQ", m))
        except:
            logger.info("Month {} is empty".format(m))
    # end for all months

    # results is a list of result rows
    df_res = pd.DataFrame(
        results,
        columns=["name", "depvar", "method", "month",  "val_score", "val_rel_err", "test_score", "test_rel_err"])

    for method in ["SQ", "LR"]:
        for depvar in ["POWER", "Q"]:
            avgerr = round(df_res[(df_res.depvar == depvar) & (df_res.method == method)]["test_rel_err"].mean())
            maxerr = round(df_res[(df_res.depvar == depvar) & (df_res.method == method)]["test_rel_err"].max())
            logger.info("{} {}: Avg {}% Max {}%".format(depvar, method, avgerr, maxerr))
# end def scenario1


def scenario2(name, mydf, prepare, grp):
    """
    Plot/Training-test scenario 2
    :param name: name if the scenario (atm hp, pv)
    :param mydf: dataframe
    :param prepare: function that prepares I/O fot ML task and plots
    :param grp: frequency to group data by
    :return:
    """
    results = []
    for mm in range(9, 21):
        m = ((mm - 1) % 12) + 1
        mydf1 = mydf[mydf.DATETIME.dt.month == m]

        # Drop NaN again, as there might be empty periods
        train = mydf1[mydf1.DATETIME.dt.day < 21].set_index("DATETIME").resample(grp).mean().dropna()
        test = mydf1[mydf1.DATETIME.dt.day >= 21].set_index("DATETIME").resample(grp).mean().dropna()
        X, y1, _ = prepare(train)
        testX, testY1, _ = prepare(test)

        if X.shape[0]<3 or testX.shape[0]<1:
            logger.info(f'Dataframe not enough values for training and/or evaluating at month {m}. Cannot compute score. Continue.')
            continue

        col_keep = ["DATETIME", "DHW_BUFFER", "POWER_HP", "PYRANOMETER", "POWER_PVT"]
        for c in X.columns:
            if c not in col_keep:
                X = X.drop(c, axis="columns")
                testX = testX.drop(c, axis="columns")


        make_plots = None
        # make_plots = test.index

        row = [name, "POWER", "LR", m]
        r, p = regress(sklearn.linear_model.LinearRegression(),
                       X, y1, testX, testY1, make_plots)
        row.extend(r)
        results.append(row)
        if p is not None: p.savefig("{}_{}.png".format("pvt_LR", m))
    # end for all months

    # results is a list of result rows
    df_res = pd.DataFrame(
        results,
        columns=["name", "depvar", "method", "month",
                 "val_score", "val_rel_err", "test_score", "test_rel_err"])

    avgerr = round(df_res["test_rel_err"].mean())
    maxerr = round(df_res["test_rel_err"].max())
    logger.info("{}: Avg {}% Max {}%".format(name, avgerr, maxerr))
# end def scenario2


def correlate(): # Main callable function
    df_hp, df_pv = load_data()
    grp = "6h"

    logger.info("##### SCENARIO 1 #####")
    logger.info("Each month, train on 20 days and test on rest\n")

    scenario1("HP", df_hp, prepare_hp, grp)
    scenario1("PV", df_pv, prepare_pv, grp)

    logger.info("##### SCENARIO 2 #####")
    logger.info("Each month, train on 20 days and test on rest")
    logger.info("Only try POWER ~ BUFFER and POWER ~ PYRANO\n")

    scenario2("HP", df_hp, prepare_hp, grp)
    scenario2("PV", df_pv, prepare_pv, grp)
# end def correlate
