import pandas as pd
import numpy as np
import random
import logging

from navgreen_base import temp_sensors, pressure, solar

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


def create_sample_input():

    df = pd.read_csv("./data/data_from_plc_2023-10-29.csv")

    temp_outlier = -1
    press_outlier = -1
    sol_outlier = -1
    flow_outlier = -1


    df = df.loc[200: df.shape[0]//6]  # Customized index so that we have both day and night values
    df = df.reset_index()

    for index, row in df.iterrows():

        for temp in temp_sensors:
            if row[temp] < -20.0 or row[temp] > 100.0:
                temp_outlier = index
                break

        for press in pressure:
            if row[press] < 0.0 or row[press] > 30.0:
                press_outlier = index
                break

        for sol in solar:
            if row[sol] > 2.0:
                sol_outlier = index
                break

        if row['FLOW_CONDENSER'] >= 3.27:
            flow_outlier = index

        if temp_outlier != -1 and press_outlier != -1 and sol_outlier != -1 and flow_outlier != -1:
            break

    # No 'Real' outliers expected
    assert sol_outlier == -1 and temp_outlier == -1 and press_outlier == -1 and flow_outlier == -1

    # Add 'Fake' outliers until we obtain real ones from the live data.

    # One fake outlier for a random temperature value
    # Will try and 'catch' different case than pressures
    while True:
        random_index_temp = random.randint(0, df.shape[0]-1)
        i = random.randint(0, len(temp_sensors) - 1)

        if not np.isnan(df.loc[random_index_temp, temp_sensors[i]]):
            break

    df.loc[random_index_temp, temp_sensors[i]] = -21.0

    logger.info(f'Twitched temp sensor {temp_sensors[i]} at index: {random_index_temp}')

    # One fake outlier for a random pressure value.
    # For the time being 'pressure' hasn't got any values at all
    random_index_press = random.randint(0, df.shape[0]-1)
    j = random.randint(0, len(pressure) - 1)
    df.loc[random_index_press, pressure[j]] = 31.0

    logger.info(f'Twitched pressure sensor {pressure[j]} at index: {random_index_press}')

    # One fake outlier for a random solar value
    random_index_solar = random.randint(0, df.shape[0]-1)
    k = random.randint(0, len(solar) - 1)
    df.loc[random_index_solar, solar[k]] = 2.1

    logger.info(f'Twitched solar sensor {solar[k]} at index: {random_index_solar}')

    random_index_flow = random.randint(0, df.shape[0]-1)
    df.at[random_index_flow, 'FLOW_CONDENSER'] = 3.721

    logger.info(f'Twitched flow condenser sensor at index: {random_index_flow}')

    logger.info(df.loc[random_index_temp, temp_sensors[i]])
    logger.info(df.loc[random_index_press, pressure[j]])
    logger.info(df.loc[random_index_solar, solar[k]])
    logger.info(df.at[random_index_flow, 'FLOW_CONDENSER'])

    print(temp_sensors[i], pressure[j], solar[k])

    # Store sample_input
    df.to_csv(f'./testcases/sample_data/sample_input.csv', mode='w', index=False)
