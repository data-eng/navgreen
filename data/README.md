# navgreen data

The code assumes that the following (uncommitted) files are found
in this directory:

* `DATA_FROM_PLC.csv`: The original historical data

* `meteo/`: weather forecasts, one file per day, scraped by executing
  daily the following:

  ``curl -s https://meteo.gr/cf.cfm?city_id=88 -o /home/user/meteo/$(date +%Y%m%d)``

* `obs_weather.txt`: actual weather, obtained by filtering for
  `provider=open_weather_map` or `provider=rain` or `provider=snow`
  the influxDB lp lines obtained from the OpenWeatherMap API
  (cf. `https://gitlab.com/dataeng/gigacampus`)
