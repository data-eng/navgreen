from datetime import datetime
import pandas as pd

dataframe_date = datetime.now().strftime("%Y-%m-%d")


csv_file = f'./data/data_from_plc_{dataframe_date}.csv'

df = pd.read_csv(csv_file)

df["QPVT_TRUE"] = ((3.6014 + 0.004 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.000002 *
                pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) *
               ((1049.0 - 0.475 * (df["PVT_IN"] + df["PVT_OUT"]) / 2.0 - 0.0018 *
                 pow((df["PVT_IN"] + df["PVT_OUT"]) / 2.0, 2)) / 3600.0 * df["FLOW_PVT"]) *
               (df["PVT_OUT"] - df["PVT_IN"]))


keep_columns_ = ["QPVT_TRUE", "Date_time_local"]
df = df.drop([c for c in df.columns if c not in keep_columns_], axis=1)
df = df.dropna()
df = df.reset_index()

df.rename(columns={"Date_time_local": "DATETIME"}, inplace=True)
df['DATETIME'] = pd.to_datetime(df['DATETIME']).strftime('%Y-%m-%d %H:%M:%S')

df = df.groupby(pd.Grouper(key='DATETIME', freq="3h")).agg({'QPVT_TRUE': 'mean'})
df = df.reset_index()

ml_control_csv = f'./ml_control/setpoints_{dataframe_date}_v2.csv'
df_ctrl = pd.read_csv(ml_control_csv)

df_ctrl['DATETIME'] = pd.to_datetime(df_ctrl['DATETIME']).strftime('%Y-%m-%d %H:%M:%S')
df_ctrl = df_ctrl.drop(columns=["QPVT_TRUE"])

# Concatenate DataFrames along the 'DATETIME' column
df_ctrl = pd.concat([df.set_index('DATETIME'), df_ctrl.set_index('DATETIME')], axis=1, join='outer')
df_ctrl = df_ctrl.reset_index()
df_ctrl['DATETIME'] = pd.to_datetime(df_ctrl['DATETIME']).strftime('%Y-%m-%d %H:%M:%S')

df_ctrl = df_ctrl['DATETIME', 'SETPOINT_FROM_ML', 'SETPOINT_VALUE', 'DHW', 'QPVT_PRED', 'QPVT_TRUE']

df_ctrl.to_csv(ml_control_csv, index=False)
