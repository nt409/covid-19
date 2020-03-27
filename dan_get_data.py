from urllib.request import urlopen
from collections import Counter

from dan_constants import COUNTRY_NAME_TO_CODE, MONTHS_DICT

try:
    import COVID19Py
    covid19 = COVID19Py.COVID19()
    USE_API = True
except Exception as e:
    print("Failed to use COVID19 Python API", e)
    USE_API = False

COUNTRY_LIST_WORLDOMETER = ['australia', 'uk', 'us', 'italy', 'spain', 'germany', 'iran',
                            'france', 'ireland', 'china', 'south-korea', 'switzerland', 'netherlands',
                            'austria', 'belgium', 'norway', 'sweden', 'portugal', 'brazil', 'canada',
                            'denmark', 'malaysia', 'poland', 'greece', 'indonesia', 'philippines',
                            'china-hong-kong-sar', 'iraq', 'algeria', 'world']


def get_data(country_name):
    if country_name in COUNTRY_LIST_WORLDOMETER:
        try:
            data = get_data_from_worldometer(country_name)
            return data
        except Exception as e:
            print("Could not retrieve data from Worldometer, trying JHU...", e)

    if not USE_API:
        return None
    data = get_data_from_api(country_name)

    if not data['Cases']['data']:
        return None
    return data


def get_data_from_api(country_name):
    country_code = COUNTRY_NAME_TO_CODE[country_name]
    locations = covid19.getLocationByCountryCode(country_code, timelines=True)

    confirmed_dict = Counter()
    deaths_dict = Counter()
    for i, location in enumerate(locations):
        confirmed_dict.update(location['timelines']['confirmed']['timeline'])
        deaths_dict.update(location['timelines']['deaths']['timeline'])
    confirmed_dict = dict(confirmed_dict)
    deaths_dict = dict(deaths_dict)

    data = {}
    for title, title_dict in [('Cases', confirmed_dict), ('Deaths', deaths_dict)]:
        data[title] = {'dates': [], 'data': []}
        for date, value in title_dict.items():
            data[title]['dates'].append(date.split('T')[0])
            data[title]['data'].append(value)

    return data


def get_data_from_worldometer(country_name):
    base_url = 'https://www.worldometers.info/coronavirus/country/'
    if country_name == 'world':
        url = 'https://www.worldometers.info/coronavirus/'
    else:
        url = base_url + country_name
    f = urlopen(url)
    webpage = f.read()
    dates = str(webpage).split('categories: ')[1].split('\\n')[0].replace('[', '').replace(']', '').replace('"','').replace('},', '').replace('},', '').replace('  ', '').split(',')

    # Convert dates
    for i, date in enumerate(dates):
        month, day = date.split()
        dates[i] = f"2020-{MONTHS_DICT[month]}-{day}"

    titles = ['Cases', 'Deaths', 'Currently Infected']
    data = {}
    for line in str(webpage).split('series: ')[1:]:
        keys = line.split(': ')

        done = 0
        for k, key in enumerate(keys):
            if 'name' in key:
                name = keys[k + 1].replace("\\'", "").split(',')[0]
                if name not in titles:
                    break
                done += 1
            if 'data' in key:
                datum = keys[k + 1].replace('[', '').split(']')[0].split(',')
                data[name] = {'dates': dates, 'data': datum }
                done += 1
            if done == 2:
                break

    return data