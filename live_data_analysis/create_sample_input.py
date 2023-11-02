import pandas as pd
import numpy as np
import random

from data_and_connection import temp_sensors, pressure, solar

df = pd.read_csv("../data_from_plc_2023-10-29.csv")

temp_outlier = -1
press_outlier = -1
sol_outlier = -1


df = df.loc[200: df.shape[0]//6]  # Customized index so that we have both day and night values
df = df.reset_index()

for index, row in df.iterrows():

    for temp in temp_sensors:
        if row[temp] < -20.0 or row[temp] > 100.0:
            temp_outlier = index
            break

    for press in pressure:
        if row[press] < 0.0 or row[press] > 35.0:
            press_outlier = index
            break

    for sol in solar:
        if row[sol] > 2.0:
            sol_outlier = index
            break

    if temp_outlier != -1 and press_outlier != -1 and sol_outlier != -1:
        break

# No 'Real' outliers expected
assert sol_outlier == -1 and temp_outlier == -1 and press_outlier == -1

# Add 'Fake' outliers until we obtain real ones from the live data.

# One fake outlier for a random temperature value
# Will try and 'catch' different case than pressures
while True:
    random_index_temp = random.randint(0, df.shape[0]-1)
    i = random.randint(0, len(temp_sensors) - 1)

    if not np.isnan(df.loc[random_index_temp, temp_sensors[i]]):
        break

df.loc[random_index_temp, temp_sensors[i]] = -21.0

print(f'Twitched temp sensor {temp_sensors[i]} at index: {random_index_temp}')

# One fake outlier for a random pressure value.
# For the time being 'pressure' hasn't got any values at all
random_index_press = random.randint(0, df.shape[0]-1)
j = random.randint(0, len(pressure) - 1)
df.loc[random_index_press, pressure[j]] = 36.0

print(f'Twitched pressure sensor {pressure[j]} at index: {random_index_press}')

# One fake outlier for a random solar value
random_index_solar = random.randint(0, df.shape[0]-1)
k = random.randint(0, len(solar) - 1)
df.loc[random_index_solar, solar[k]] = 6.0

print(f'Twitched solar sensor {solar[k]} at index: {random_index_solar}')

print(df.loc[random_index_temp, temp_sensors[i]])
print(df.loc[random_index_press, pressure[j]])
print(df.loc[random_index_solar, solar[k]])

# Store sample_input
df.to_csv(f'./test/sample_input.csv', mode='w', index=False)
