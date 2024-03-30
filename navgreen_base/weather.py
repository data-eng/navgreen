import os
import csv
from bs4 import BeautifulSoup
from datetime import datetime
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')

directions = {
    "Άπνοια": 0,
    "Βόρειος": 2, "B": 2, "Β": 2,
    "Βορειοανατολικός": 6, "BA": 6, "ΒΑ": 6,
    "Ανατολικός": 10, "A": 10, "Α": 10,
    "Νοτιοανατολικός": 4, "NA": 4, "ΝΑ": 4,
    "Νότιος": -2, "N": -2, "Ν": -2,
    "Νοτιοδυτικός": -6, "ΝΔ": -6,
    "Δυτικός": -10, "Δ": -10,
    "Βορειοδυτικός": -4, "ΒΔ": -4,
}

phenomena = {
    "ΚΑΘΑΡΟΣ": 0,
    "ΠΕΡΙΟΡΙΣΜΕΝΗ ΟΡΑΤΟΤΗΤΑ": 1,
    "ΑΡΑΙΗ ΣΥΝΝΕΦΙΑ": 2,
    "ΛΙΓΑ ΣΥΝΝΕΦΑ": 3,
    "ΑΡΚΕΤΑ ΣΥΝΝΕΦΑ": 4,
    "ΣΥΝΝΕΦΙΑΣΜΕΝΟΣ": 5,
    "ΑΣΘΕΝΗΣ ΒΡΟΧΗ": 6,
    "ΒΡΟΧΗ": 7
}

def current(content, name):
    soup = BeautifulSoup(content, 'html.parser')
    live_panels = soup.find_all('div', class_='livepanel')
    sunblockheader = soup.find('div', class_='sunblockheader')
    csv_data = []

    for panel in live_panels:
        data = {}

        station = panel.parent.get('id').split('stations')[1]
        data['Station'] = station

        date = sunblockheader.text.split()[1:3]
        day, month = date[0], date[1]
        month = translator.translate(month).replace('Of ', '')
        month = datetime.strptime(month, '%B').month
        date = f"{day.strip('&nbsp;'):0>2}/{month:0>2}"
        data['Date'] = date

        time = panel.parent.find('span', class_='livetime')
        time = datetime.strptime(time.text.strip(), '%H:%M').time() if time else 'N/A'
        data['Time'] = time

        sunrise = sunblockheader.text.split()[3:4] # ['06:47']
        sunrise = datetime.strptime(sunrise[0], '%H:%M').time()
        data['Sunrise'] = sunrise

        sunset = sunblockheader.text.split()[4:5] # ['18:25']
        sunset = datetime.strptime(sunset[0], '%H:%M').time()
        data['Sunset'] = sunset

        hours = int(sunblockheader.text.split()[8])
        minutes = int(sunblockheader.text.split()[10]) 
        daylight = (hours * 60) + minutes
        data['Daylight'] = daylight

        temp = panel.find('div', class_='newtemp')
        temp = temp.text.strip().split('°')[0] if temp else 'N/A'
        data['Temperature'] = temp

        humidity = panel.find('div', class_='ygrasia')
        humidity = humidity.text.strip().split(':')[1].strip().split('%')[0] if humidity else 'N/A'
        data['Humidity'] = humidity

        pressure = panel.find('div', class_='piesi')
        pressure = pressure.text.strip().split(':')[1].split()[0] if pressure else 'N/A'
        data['Pressure'] = pressure

        wind = panel.find_all('div', class_='windnr')
        wind_speed = wind[0].text.split()[0] if wind else 'N/A'
        data['Wind_Speed'] = wind_speed

        beaufort = wind[1].text.split()[0] if wind else 'N/A'
        data['Beaufort'] = beaufort

        wind_dir = panel.find('div', class_='windtxt2')
        wind_dir = wind_dir.text.strip() if wind_dir else 'N/A'
        data['Wind_Direction'] = directions.get(wind_dir, 'N/A')

        high_temp = panel.find('span', class_='hight')
        high_temp = high_temp.text.split('°')[0].strip() if high_temp else 'N/A'
        data['High_Temp'] = high_temp

        low_temp = panel.find('span', class_='lowt')
        low_temp = low_temp.text.split('°')[0].strip() if low_temp else 'N/A'
        data['Low_Temp'] = low_temp

        rainfall = panel.find_all('div', class_='dailydata')
        rainfall = rainfall[2].text.split()[2] if len(rainfall) > 0 else 'N/A'
        data['Rainfall'] = rainfall

        peak_gust = panel.find_all('div', class_='dailydata')
        peak_gust = peak_gust[3].text.split()[3] if peak_gust else 'N/A'
        data['Peak_Gust'] = peak_gust.replace('-', 'N/A')

        if time != "N/A":
            csv_data.append(data)

    with open(name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Station', 'Date', 'Time', 'Sunrise', 'Sunset', 'Daylight',
                                                  'Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Beaufort',
                                                  'Wind_Direction', 'High_Temp', 'Low_Temp', 'Rainfall', 'Peak_Gust'])
        writer.writeheader()
        writer.writerows(csv_data)

def historical(content, name):
    pass

def brief(content, name):
    pass

def detailed(content, name):
    soup = BeautifulSoup(content, 'html.parser')
    sunblockheader = soup.find('div', class_='sunblockheader')
    perhours = soup.find_all('tr', class_='perhour rowmargin')
    csv_data = []

    for perhour in perhours:
        data = {}

        date = sunblockheader.text.split()[1:3]
        day, month = date[0], date[1]
        month = translator.translate(month).replace('Of ', '').replace('of ', '')
        month = datetime.strptime(month, '%B').month
        date = f"{day.strip('&nbsp;'):0>2}/{month:0>2}"
        data['Date'] = date

        forecast_date = perhour.find_previous('td', class_='forecastDate')
        flleft = forecast_date.find('div', class_='flleft')
        dayNumbercf = flleft.find('span', class_='dayNumbercf')
        monthNumbercf = dayNumbercf.find('span', class_='monthNumbercf')
        day = dayNumbercf.contents[0]
        month = monthNumbercf.contents[0]
        month = translator.translate(month).replace('Of ', '').replace('of ', '')
        month = datetime.strptime(month, '%B').month
        forecast = f"{day:0>2}/{month:0>2}"
        data['Forecast'] = forecast

        span = dayNumbercf.find('span', class_='pull-right forecastright')
        sunrise = span.contents[0].split('-')[0].strip().split()[1]
        sunrise = datetime.strptime(sunrise, '%H:%M').time()
        data['Sunrise'] = sunrise

        sunset = span.contents[0].split('-')[1].strip().split()[1]
        sunset = datetime.strptime(sunset, '%H:%M').time()
        data['Sunset'] = sunset

        time = perhour.find('td', class_='innerTableCell fulltime')
        time = datetime.strptime(time.text.strip(), '%H:%M').time() if time else 'N/A'
        data['Time'] = time

        temp = perhour.find('td', class_='innerTableCell temperature tempwidth')
        temp = temp.text.strip().split('°')[0] if temp else 'N/A'
        data['Temperature'] = temp

        humidity = perhour.find('td', class_='humidity')
        humidity = humidity.text.strip().split()[0].strip().split('%')[0] if humidity else 'N/A'
        data['Humidity'] = humidity

        wind = perhour.find('td', class_='innerTableCell anemosfull')
        wind_speed = wind.text.split()[3] if wind else 'N/A'
        data['Wind_Speed'] = wind_speed

        beaufort = wind.text.split()[0] if wind else 'N/A'
        data['Beaufort'] = beaufort

        wind_dir = wind.text.split()[2] if wind else 'N/A'
        wind_dir = directions.get(wind_dir, 'N/A')
        data['Wind_Direction'] = wind_dir

        sky = perhour.find('td', class_='phenomeno-name')
        sky = sky.text.strip() if sky else 'N/A'
        sky = phenomena.get(sky, 'N/A')
        data['Sky'] = sky

        csv_data.append(data)

    with open(name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Date', 'Forecast', 'Sunrise', 'Sunset', 'Time', 'Temperature', 
                                                  'Humidity', 'Wind_Speed', 'Beaufort', 'Wind_Direction', 'Sky'])
        writer.writeheader()
        writer.writerows(csv_data)

def main():
    in_path = "meteo"
    out_path = "static"

    prefix_to_func = {
        "C": current,
        #"H": historical,
        #"B": brief,
        #"D": detailed
    }

    for filename in os.listdir(in_path):
        with open(os.path.join(in_path, filename), "r", encoding="utf-8") as file:
            html = file.read()
            for prefix, func in prefix_to_func.items():
                name = os.path.join(out_path, f"{prefix}-{filename}.csv")
                func(content=html, name=name)
                #break
            #break