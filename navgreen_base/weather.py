import os
import csv
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')

def current(content, name):
    soup = BeautifulSoup(content, 'html.parser')
    live_panels = soup.find_all('div', class_='livepanel')
    csv_data = []

    for panel in live_panels:
        data = {}

        station = panel.parent.get('id').split('stations')[1]
        data['Station'] = station

        temp = panel.find('div', class_='newtemp')
        temp = temp.text.strip().split('°')[0] if temp else 'N/A'
        data['Temperature'] = temp

        humidity = panel.find('div', class_='ygrasia')
        humidity = humidity.text.strip().split(':')[1].strip().split('%')[0] if humidity else 'N/A'
        data['Humidity'] = humidity

        pressure = panel.find('div', class_='piesi')
        pressure = pressure.text.strip().split(':')[1].split()[0] if pressure else 'N/A'
        data['Pressure'] = pressure

        wind_speed = panel.find('div', class_='windnr')
        wind_speed = wind_speed.text.split()[0] if wind_speed else 'N/A'
        data['Wind_Speed'] = wind_speed

        wind_dir = panel.find('div', class_='windtxt2')
        wind_dir = wind_dir.text.strip() if wind_dir else 'N/A'
        data['Wind_Direction'] = translator.translate(wind_dir)

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
        data['Peak_Gust'] = peak_gust

        csv_data.append(data)

    with open(name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Station', 'Temperature', 'Humidity', 'Pressure',
                                                  'Wind_Speed', 'Wind_Direction', 'High_Temp',
                                                  'Low_Temp', 'Rainfall', 'Peak_Gust'])
        writer.writeheader()
        writer.writerows(csv_data)

def historical(content, name):
    pass

def brief(content, name):
    pass

def detailed(content, name):
    pass

def main():
    path = "meteo"

    prefix_to_func = {
        "C": current,
        "H": historical,
        "B": brief,
        "D": detailed
    }

    for filename in os.listdir(path):
        with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
            html = file.read()
            for prefix, func in prefix_to_func.items():
                name = f"{prefix}-{filename}.csv"
                func(content=html, name=name)
                break
            break