import numpy as np
import pandas as pd

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


def make_point(measurement, row, value_columns, tag_columns):
    p = influxdb_client.Point(measurement)
    p.time(row["DATETIME"])

    # Tag with the state of the valves, as context
    for col in tag_columns:
        if row[col] is not np.nan:
            p.tag(col, row[col])
    # Add the sensor data fields
    for col in value_columns:
        if row[col] is not np.nan:
            p.field(col, row[col])
    return p


def write_data(row, url, token, organization, bucket):
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.write_api(write_options=SYNCHRONOUS)

    # Apply reasonable limits
    for col in solar:
        if row[col] > 2.0:  # Error that happens at nighttime
            row[col] = 0.0
    for col in temp_sensors:
        if row[col] < -20.0 or row[col] > 100.0:
            row[col] = np.nan
    for col in pressure:
        if row[col] < 0.0 or row[col] > 35.0:
            row[col] = np.nan

    p = make_point("pressure", row, pressure, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("temperature", row, temp_sensors, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("flow", row, flow, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("power", row, power, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("solar", row, solar, control)
    api.write(bucket=bucket, org=organization, record=p)
    p = make_point("other", row, other, control)
    api.write(bucket=bucket, org=organization, record=p)
    # Also add controls as values, for viz
    p = make_point("control", row, control, [])
    api.write(bucket=bucket, org=organization, record=p)
    # print("Write ok")


def read_data(url, token, organization, bucket):
    """
    Reads data from a specified bucket and stores it in a DataFrame
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: The DataFrame
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.query_api()
    query = f'from(bucket: "{bucket}") |> range(start: 0)'
    print("Started reading...")

    data = api.query_data_frame(org=organization, query=query)

    dfs = []
    for datum in data:
        # Pivot the DataFrame to separate fields into different columns
        df = datum.pivot(index='_time', columns='_field', values='_value')
        # Reset the index to make the '_time' column a regular column
        df.reset_index(inplace=True)
        df.columns.name = None

        dfs += [df]

    df1 = dfs[0]
    df2 = dfs[1]
    df = pd.concat([df1, df2], axis=1, join='outer', sort=False)
    df = df.rename(columns={'_time': 'DATETIME'})
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    print("Ended reading.")

    return df


def delete_data(url, token, organization, bucket):
    """
    Deletes all data from a specified bucket.
    :param url: Url of the InfluxDB
    :param token: Authorization token
    :param organization: Organization where the bucket belongs
    :param bucket: The bucket to delete data from
    :return: None
    """
    influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=organization)
    api = influx_client.delete_api()
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="temperature"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="pressure"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="flow"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="solar"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="other"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="power"')
    api.delete(bucket=bucket, org=organization, start="1970-01-01T00:00:00Z", stop="2030-01-01T00:00:00Z",
               predicate='_measurement="control"')

water_temp = ["PVT_IN_TO_DHW", "PVT_OUT_FROM_DHW", "PVT_IN_TO_SOLAR_BUFFER", "PVT_OUT_FROM_SOLAR_BUFFER",
              "SOLAR_BUFFER_IN", "SOLAR_BUFFER_OUT", "BTES_TANK_IN", "BTES_TANK_OUT", "SOLAR_HEAT_REJECTION_IN",
              "SOLAR_HEAT_REJECTION_OUT", "WATER_IN_EVAP", "WATER_OUT_EVAP", "WATER_IN_COND", "WATER_OUT_COND",
              "SH1_IN", "SH1_RETURN", "AIR_HP_TO_BTES_TANK", "DHW_INLET", "DHW_OUTLET", "DHW_BOTTOM", "SH_INLET",
              "SH_RETURN", "PVT_IN", "PVT_OUT"]

other_temp = ["OUTDOOR_TEMP", "BTES_TANK", "SOLAR_BUFFER_TANK", "SH_BUFFER", "DHW_BUFFER", "INDOOR_TEMP"]

ref_temp = ["RECEIVER_LIQUID_IN", "RECEIVER_LIQUID_OUT", "ECO_LIQUID_OUT", "SUCTION_TEMP",
            "DISCHARGE_TEMP", "ECO_VAPOR_TEMP", "EXPANSION_TEMP", "ECO_EXPANSION_TEMP"]

pressure = ["SUCTION_PRESSURE", "DISCHARGE_PRESSURE", "ECO_PRESSURE"]

flow = ["FLOW_EVAPORATOR", "FLOW_CONDENSER", "FLOW_DHW", "FLOW_SOLAR_HEAT_REJECTION",
        "FLOW_PVT", "FLOW_FAN_COILS_INDOOR"]

power = ["POWER_HP", "POWER_PVT"]

solar = ["PYRANOMETER"]

other = ["Compressor_HZ", "EEV_LOAD1", "EEV_LOAD2"]

control = ["THREE_WAY_EVAP_OPERATION", "THREE_WAY_COND_OPERATION", "THREE_WAY_SOLAR_OPERATION",
           "FOUR_WAY_VALVE", "AIR_COOLED_COMMAND", "Residential_office_mode", "MODBUS_LOCAL"]

temp_sensors = []
temp_sensors.extend(water_temp)
temp_sensors.extend(other_temp)
temp_sensors.extend(ref_temp)
