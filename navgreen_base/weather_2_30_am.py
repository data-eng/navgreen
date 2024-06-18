import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import os
import dateparser
import csv
import pandas as pd
import numpy as np


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(lineno)d:%(message)s')
file_handler = logging.FileHandler('./logger_2_30.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


mappings = {
    "Άπνοια": 0,
    "Βόρειος": 2, "B": 2, "Β": 2,
    "Βορειοανατολικός": 6, "BA": 6, "ΒΑ": 6,
    "Ανατολικός": 10, "A": 10, "Α": 10,
    "Νοτιοανατολικός": 4, "NA": 4, "ΝΑ": 4,
    "Νότιος": -2, "N": -2, "Ν": -2,
    "Νοτιοδυτικός": -6, "ΝΔ": -6,
    "Δυτικός": -10, "Δ": -10,
    "Βορειοδυτικός": -4, "ΒΔ": -4,

    "ΚΑΘΑΡΟΣ": 0,
    "ΠΕΡΙΟΡΙΣΜΕΝΗ ΟΡΑΤΟΤΗΤΑ": 1,
    "ΑΡΑΙΗ ΣΥΝΝΕΦΙΑ": 2,
    "ΛΙΓΑ ΣΥΝΝΕΦΑ": 3,
    "ΑΡΚΕΤΑ ΣΥΝΝΕΦΑ": 4,
    "ΣΥΝΝΕΦΙΑΣΜΕΝΟΣ": 5,
    "ΑΣΘΕΝΗΣ ΒΡΟΧΗ": 6,
    "ΒΡΟΧΗ": 7,

    "ΠΟΛΥ ΧΑΜΗΛΕΣ ΘΕΡΜΟΚΡΑΣΙΕΣ ΓΙΑ ΤΗΝ ΕΠΟΧΗ": -2,
    "ΧΑΜΗΛΕΣ ΘΕΡΜΟΚΡΑΣΙΕΣ ΓΙΑ ΤΗΝ ΕΠΟΧΗ": -1,
    "ΚΑΝΟΝΙΚΕΣ ΘΕΡΜΟΚΡΑΣΙΕΣ ΓΙΑ ΤΗΝ ΕΠΟΧΗ": 0,
    "ΥΨΗΛΕΣ ΘΕΡΜΟΚΡΑΣΙΕΣ ΓΙΑ ΤΗΝ ΕΠΟΧΗ": 1,
    "ΠΟΛΥ ΥΨΗΛΕΣ ΘΕΡΜΟΚΡΑΣΙΕΣ ΓΙΑ ΤΗΝ ΕΠΟΧΗ": 2,

    'Μεση θερμοκρασία:': 'MEAN_TEMP',
    'Μέση μέγιστη:': 'MEAN_HIGH_TEMP',
    'Μέση ελάχιστη:': 'MEAN_LOW_TEMP',
    'Υψηλότερη μέγιστη θερμοκρασία:': 'MAX_HIGH_TEMP',
    'Χαμηλότερη ελάχιστη θερμοκρασία:': 'MIN_LOW_TEMP',
    'Μέση βροχόπτωση:': 'MEAN_RAINFALL',
    'Υψηλότερη ημερήσια βροχόπτωση:': 'MAX_DAILY_RAINFALL'
}


def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f'Error fetching the URL: {e}')
        return None


def parse_html_content(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup
    except Exception as e:
        logger.error(f'Error parsing the HTML content: {e}')
        return None


def get_date(name, format='%d/%m/%Y', lang='el'):

    date = dateparser.parse(name, languages=[lang])
    date = date.strftime(format)
    return date


def detailed(html):
    """
    Parses the detailed weather data from the HTML (7-days forecast).
    :param html: BeautifulSoup object
    :return: tuple
    """
    fieldnames = ['ACCESS_DATETIME', 'FORECAST_DATETIME', 'SUNRISE', 'SUNSET', 'TEMPERATURE',
                  'HUMIDITY', 'WIND_SPEED', 'BEAUFORT', 'WIND_DIRECTION', 'SKY']
    csv_data = []

    perhours = html.find_all('tr', class_='perhour rowmargin')
    # Get the corresponding values for 24h-span
    hours = 24 // 3
    perhours = perhours[:hours]

    for perhour in perhours:

        data = {}

        forecast_date = perhour.find_previous('td', class_='forecastDate')
        flleft = forecast_date.find('div', class_='flleft')
        dayNumbercf = flleft.find('span', class_='dayNumbercf')
        monthNumbercf = flleft.find('span', class_='monthNumbercf')

        if dayNumbercf:
            day = dayNumbercf.contents[0]
            month = monthNumbercf.contents[0]
            curr_forecast_date = get_date(name=f"{day} {month}")
        else:
            whole_date = monthNumbercf.contents[0]
            day, month, year = whole_date.strip().split('/')
            curr_forecast_date = f"{day:0>2}/{month:0>2}/{year:0>4}"

        data['ACCESS_DATETIME'] = datetime.today().strftime('%d/%m/%Y %H:%M:%S')

        span = flleft.find(['div', 'span'], class_='pull-right forecastright')
        fulltime = perhour.find('td', class_='innerTableCell fulltime')
        temperature = perhour.find('td', class_='innerTableCell temperature tempwidth')
        anemosfull = perhour.find('td', class_='innerTableCell anemosfull')
        phenomeno_name = perhour.find('td', class_='phenomeno-name')

        text = span.get_text(strip=True)
        cycles = text.split('-')
        sunrise = cycles[0].strip().split()[-1]
        sunrise = datetime.strptime(sunrise, '%H:%M').time()
        data['SUNRISE'] = sunrise

        sunset = cycles[1].strip().split()[-1]
        sunset = datetime.strptime(sunset, '%H:%M').time()
        data['SUNSET'] = sunset

        time = datetime.strptime(fulltime.text.strip(), '%H:%M').time() if fulltime else pd.NaT
        data['FORECAST_DATETIME'] = pd.to_datetime(f"{curr_forecast_date} {time}", format='%d/%m/%Y %H:%M:%S')

        temp = temperature.text.split('°')[0] if temperature else np.nan
        data['TEMPERATURE'] = temp

        humidity = perhour.find('td', class_='humidity')
        humidity = humidity.text.strip().split()[0].strip().split('%')[0] if humidity else np.nan
        data['HUMIDITY'] = humidity

        wind_speed = anemosfull.text.split()[3] if anemosfull else np.nan
        data['WIND_SPEED'] = wind_speed

        beaufort = anemosfull.text.split()[0] if anemosfull else np.nan
        data['BEAUFORT'] = beaufort

        wind_dir = anemosfull.text.split()[2] if anemosfull else np.nan
        wind_dir = mappings.get(wind_dir, np.nan)
        data['WIND_DIRECTION'] = wind_dir

        sky = phenomeno_name.find(text=True, recursive=False).strip() if phenomeno_name else np.nan
        sky = mappings.get(sky, np.nan)
        data['SKY'] = sky

        csv_data.append(data)

    return fieldnames, csv_data


def write_csv(filename, fieldnames, data):
    file_exists = os.path.isfile(filename)  # Check if the file exists

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only if the file is newly created

        for row in data:
            row = {key: str(value).strip() for key, value in row.items()}
            writer.writerow(row)


def get_predictions(url, output_file):
    html_content = fetch_website_content(url)

    if not html_content:
        logger.error('Failed to retrieve website content.')
        return False

    soup = parse_html_content(html_content)

    if not soup:
        logger.error('Failed to parse HTML content.')
        return False

    fieldnames, data = detailed(html=soup)

    if not fieldnames or not data:
        logger.error('No weather extracted.')
        return False

    write_csv(output_file, fieldnames, data)

    return True


def meteo_func(func, *args, **kwargs):

    no_data = True

    while no_data:
        try:
            logger.info(f"Running job at {datetime.now()}")

            # Generate the CSV filename with the current month and year
            current_date = datetime.now()
            csv_filename = current_date.strftime('%B_%Y.csv').lower()  # e.g., june_2024.csv

            result = func(*args, output_file=csv_filename, **kwargs)
            while not result:
                logger.error("Function did not return True. Retrying in 1 minute...")
                time.sleep(60)
                result = func(*args, output_file=csv_filename, **kwargs)

            logger.info("Function returned True successfully.")
            no_data = False

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            logger.error("Retrying the job in 1 minute...")
            time.sleep(60)


if __name__ == '__main__':
    url = 'https://meteo.gr/cf.cfm?city_id=88'
    # Call the function daily
    meteo_func(get_predictions, url)
