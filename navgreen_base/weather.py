import os
import time
import csv
from bs4 import BeautifulSoup
from datetime import datetime
from deep_translator import GoogleTranslator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

translator = GoogleTranslator(source='auto', target='en')

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

def erase(text, lst):
    """
    Erases occurrences in a string specified within some list.
    :param text: input str
    :param lst: list of values to erase
    :return: output str
    """
    for elem in lst:
        text = text.replace(elem, '')
    return text

def get_date(day, month_el):
    """
    Gets the date in the format DD/MM from the provided day and month.
    :param day: day value
    :param month_el: month name in greek
    :return: date string
    """
    month_en = erase(text=translator.translate(month_el), lst=['Of ', 'of '])
    month = datetime.strptime(month_en, '%B').month
    date = f"{day:0>2}/{month:0>2}"
    return date

def current(html):
    """
    Parses the current weather data from the HTML (1-day forecast).
    :param html: BeautifulSoup object
    :return: tuple
    """
    fieldnames=['STATION', 'DATE', 'TIME', 'SUNRISE', 'SUNSET', 'DAYLIGHT', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE',
                'WIND_SPEED', 'BEAUFORT', 'WIND_DIRECTION', 'HIGH_TEMP', 'LOW_TEMP', 'RAINFALL', 'PEAK_GUST']
    csv_data = []

    live_panels = html.find_all('div', class_='livepanel')
    sunblockheader = html.find('div', class_='sunblockheader')

    for panel in live_panels:
        data = {}

        station = panel.parent.get('id')
        livetime = panel.parent.find('span', class_='livetime')
        newtemp = panel.find('div', class_='newtemp')
        ygrasia = panel.find('div', class_='ygrasia')
        piesi = panel.find('div', class_='piesi')
        windnr = panel.find_all('div', class_='windnr')
        windtxt2 = panel.find('div', class_='windtxt2')
        hight = panel.find('span', class_='hight')
        lowt = panel.find('span', class_='lowt')
        dailydata = panel.find_all('div', class_='dailydata')

        station = station.split('stations')[1]
        data['STATION'] = station

        date = sunblockheader.text.split()[1:3]
        day, month = date[0], date[1]
        data['DATE'] = get_date(day, month)

        time = datetime.strptime(livetime.text.strip(), '%H:%M').time() if livetime else 'N/A'
        data['TIME'] = time

        sunrise = sunblockheader.text.split()[3:4][0]
        sunrise = datetime.strptime(sunrise, '%H:%M').time() if sunrise else 'N/A'
        data['SUNRISE'] = sunrise

        sunset = sunblockheader.text.split()[4:5][0]
        sunset = datetime.strptime(sunset, '%H:%M').time() if sunset else 'N/A'
        data['SUNSET'] = sunset

        hours = int(sunblockheader.text.split()[8])
        minutes = int(sunblockheader.text.split()[10]) 
        daylight = (hours * 60) + minutes
        data['DAYLIGHT'] = daylight

        temp = newtemp.text.split('°')[0] if newtemp else 'N/A'
        data['TEMPERATURE'] = temp

        humidity = ygrasia.text.split(':')[1].split('%')[0] if ygrasia else 'N/A'
        data['HUMIDITY'] = humidity

        pressure = piesi.text.split(':')[1].split()[0] if piesi else 'N/A'
        data['PRESSURE'] = pressure

        wind_speed = windnr[0].text.split()[0] if windnr else 'N/A'
        data['WIND_SPEED'] = wind_speed

        beaufort = windnr[1].text.split()[0] if windnr else 'N/A'
        data['BEAUFORT'] = beaufort

        wind_dir = windtxt2.text.strip() if windtxt2 else 'N/A'
        data['WIND_DIRECTION'] = mappings.get(wind_dir, 'N/A')

        high_temp = hight.text.split('°')[0] if hight else 'N/A'
        data['HIGH_TEMP'] = high_temp

        low_temp = lowt.text.split('°')[0] if lowt else 'N/A'
        data['LOW_TEMP'] = low_temp

        rainfall = dailydata[2].text.split()[2] if len(dailydata) > 0 else 'N/A'
        data['RAINFALL'] = rainfall

        peak_gust = dailydata[3].text.split()[3].replace('-', 'N/A') if dailydata else 'N/A'
        data['PEAK_GUST'] = peak_gust

        if time != "N/A":
            csv_data.append(data)
    
    return fieldnames, csv_data

def historical(html):
    """
    Parses the historical weather data from the HTML (multi-annual forecast).
    :param html: BeautifulSoup object
    :return: tuple
    """
    fieldnames=['MONTH', 'START_YEAR', 'END_YEAR', 'MEAN_TEMP', 'MEAN_HIGH_TEMP', 'MEAN_LOW_TEMP',
                'MAX_HIGH_TEMP', 'MIN_LOW_TEMP', 'MEAN_RAINFALL', 'MAX_DAILY_RAINFALL']
    csv_data = []
    
    histpanels = html.find_all('div', class_='historicalpanel')

    for histpanel in histpanels:
        items = histpanel.find_all(['div', 'span'], class_='historyitem')
        data = {}

        headernew2 = histpanel.find('div', class_='headernew2')

        header = headernew2.text.strip().split()
        month = erase(text=translator.translate(header[0]), lst=['Of ', 'of '])
        month = datetime.strptime(month, '%B').month
        data['MONTH'] = month

        start_year = header[1].lstrip('(')
        data['START_YEAR'] = start_year

        end_year = header[3].rstrip(')')
        data['END_YEAR'] = end_year

        for item in items:
            title = item.find('span', class_='historytitle').text.strip()
            value = item.find(['div', 'span'], class_='historicaltemp').text.split('\xa0')[0]
            data[mappings.get(title, title)] = value

        csv_data.append(data)
    
    return fieldnames, csv_data

def brief(html):
    """
    Parses the brief weather data from the HTML (4-days forecast).
    :param html: BeautifulSoup object
    :return: tuple
    """
    fieldnames=['DATE', 'FORECAST', 'SUNRISE', 'SUNSET', 'HIGH_TEMP', 'LOW_TEMP', 'INFO_TEMP']
    csv_data = []
    
    sunblockheader = html.find('div', class_='sunblockheader')
    dayblocks = html.find_all('div', class_='dayblockinside')

    for dayblock in dayblocks:
        data = {}

        subheader_calendar = dayblock.find('div', class_='subheader_calendar')
        datenumber_calendar = subheader_calendar.find('div', class_='datenumber_calendar')
        month_calendar = subheader_calendar.find('div', class_='month_calendar')
        infotemp = dayblock.find('div', class_='infotemp')
        sunriseSet_calendar = dayblock.find('div', class_='sunriseSet_calendar')
        minmax = dayblock.find('div', class_='minmax')
        hightemp = minmax.find('div', class_='hightemp')
        lowtemp = minmax.find('div', class_='lowtemp')

        date = sunblockheader.text.split()[1:3]
        day, month = date[0], date[1]
        data['DATE'] = get_date(day, month)

        day, month = datenumber_calendar.text.strip(), month_calendar.text.strip()
        data['FORECAST'] = get_date(day, month)

        info_temp = mappings.get(infotemp.text.strip(), 'N/A')
        data['INFO_TEMP'] = info_temp

        sunrise = sunriseSet_calendar.text.split()[1:2]
        sunrise = datetime.strptime(sunrise[0], '%H:%M').time()
        data['SUNRISE'] = sunrise

        sunset = sunriseSet_calendar.text.split()[4:5]
        sunset = datetime.strptime(sunset[0], '%H:%M').time()
        data['SUNSET'] = sunset

        high_temp = hightemp.text.split()[0]
        data['HIGH_TEMP'] = high_temp

        low_temp = lowtemp.text.split()[0]
        data['LOW_TEMP'] = low_temp

        csv_data.append(data)

    return fieldnames, csv_data

def detailed(html):
    """
    Parses the detailed weather data from the HTML (7-days forecast).
    :param html: BeautifulSoup object
    :return: tuple
    """    
    fieldnames=['DATE', 'FORECAST', 'SUNRISE', 'SUNSET', 'TIME', 'TEMPERATURE',
                'HUMIDITY', 'WIND_SPEED', 'BEAUFORT', 'WIND_DIRECTION', 'SKY']
    csv_data = []

    sunblockheader = html.find('div', class_='sunblockheader')
    perhours = html.find_all('tr', class_='perhour rowmargin')

    for perhour in perhours:
        data = {}

        date = sunblockheader.text.split()[1:3]
        day, month = date[0], date[1]
        data['DATE'] = get_date(day, month)

        forecast_date = perhour.find_previous('td', class_='forecastDate')
        flleft = forecast_date.find('div', class_='flleft')
        dayNumbercf = flleft.find('span', class_='dayNumbercf')
        monthNumbercf = flleft.find('span', class_='monthNumbercf')
        span = flleft.find(['div', 'span'], class_='pull-right forecastright')
        fulltime = perhour.find('td', class_='innerTableCell fulltime')
        temperature = perhour.find('td', class_='innerTableCell temperature tempwidth')
        anemosfull = perhour.find('td', class_='innerTableCell anemosfull')
        phenomeno_name = perhour.find('td', class_='phenomeno-name')

        if dayNumbercf:
            day = dayNumbercf.contents[0]
            month = monthNumbercf.contents[0]
            forecast = get_date(day, month)
        else:
            whole_date = monthNumbercf.contents[0]
            day, month, _ = whole_date.strip().split('/')
            forecast = f"{day:0>2}/{month:0>2}"

        data['FORECAST'] = forecast

        text = span.get_text(strip=True)
        cycles = text.split('-')
        sunrise = cycles[0].strip().split()[-1]
        sunrise = datetime.strptime(sunrise, '%H:%M').time()
        data['SUNRISE'] = sunrise

        sunset = cycles[1].strip().split()[-1]
        sunset = datetime.strptime(sunset, '%H:%M').time()
        data['SUNSET'] = sunset

        time = datetime.strptime(fulltime.text.strip(), '%H:%M').time() if fulltime else 'N/A'
        data['TIME'] = time

        temp = temperature.text.split('°')[0] if temperature else 'N/A'
        data['TEMPERATURE'] = temp

        humidity = perhour.find('td', class_='humidity')
        humidity = humidity.text.strip().split()[0].strip().split('%')[0] if humidity else 'N/A'
        data['HUMIDITY'] = humidity

        wind_speed = anemosfull.text.split()[3] if anemosfull else 'N/A'
        data['WIND_SPEED'] = wind_speed

        beaufort = anemosfull.text.split()[0] if anemosfull else 'N/A'
        data['BEAUFORT'] = beaufort

        wind_dir = anemosfull.text.split()[2] if anemosfull else 'N/A'
        wind_dir = mappings.get(wind_dir, 'N/A')
        data['WIND_DIRECTION'] = wind_dir

        sky = phenomeno_name.find(text=True, recursive=False).strip() if phenomeno_name else 'N/A'
        sky = mappings.get(sky, 'N/A')
        data['SKY'] = sky

        csv_data.append(data)
    
    return fieldnames, csv_data

def write_csv(filename, fieldnames, data):
    """
    Writes data to a CSV file.
    :param filename: str
    :param fieldnames: list
    :param data: list of dictionaries
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()

        for row in data:
            row = {key: str(value).strip() for key, value in row.items()}
            writer.writerow(row)

def main():
    in_path = "meteo"
    out_path = "static"

    prefix_to_func = {
        "C": current,
        "H": historical,
        "B": brief,
        "D": detailed
    }

    start_time = time.time()

    for html_name in os.listdir(in_path):
        with open(os.path.join(in_path, html_name), "r", encoding="utf-8") as file:
            html = file.read()
            soup = BeautifulSoup(html, 'html.parser')
            for prefix, func in prefix_to_func.items():
                csv_name = os.path.join(out_path, f"{prefix}-{html_name}.csv")
                fieldnames, data = func(html=soup)
                write_csv(csv_name, fieldnames, data)

    end_time = time.time()
    logger.info("Total time: {} seconds".format(end_time - start_time))