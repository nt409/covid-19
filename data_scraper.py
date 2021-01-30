from urllib.request import urlopen, Request
from collections import Counter

from data_constants import COUNTRY_NAME_TO_CODE, MONTHS_DICT, WORLDOMETER_NAME

def get_USE_API():
    try:
        import COVID19Py
        covid19 = COVID19Py.COVID19()
        USE_API = True
        return USE_API, covid19

    except Exception as e:
        print("Failed to use COVID19 Python API", e)
        USE_API = False
        return USE_API, None



COUNTRY_LIST_WORLDOMETER = ['world', 'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'anguilla',
                            'antigua-and-barbuda', 'argentina', 'armenia', 'aruba', 'australia', 'austria',
                            'azerbaijan', 'bahamas', 'bahrain','bangladesh', 'barbados', 'belarus', 'belgium', 'belize',
                            'benin', 'bermuda', 'bhutan','bolivia', 'bosnia-and-herzegovina', 'botswana', 'brazil',
                            'british-virgin-islands', 'brunei-darussalam', 'bulgaria', 'burkina-faso', 'burundi',
                            'cabo-verde', 'cambodia', 'cameroon', 'canada', 'caribbean-netherlands', 'cayman-islands',
                            'central-african-republic', 'chad', 'channel-islands', 'chile', 'china-hong-kong-sar',
                            'china-macao-sar', 'china', 'colombia', 'congo', 'costa-rica', 'cote-d-ivoire', 'croatia',
                            'cuba', 'curacao', 'cyprus', 'czech-republic', 'democratic-republic-of-the-congo',
                            'denmark', 'djibouti', 'dominica', 'dominican-republic', 'ecuador', 'egypt', 'el-salvador',
                            'equatorial-guinea', 'eritrea', 'estonia', 'ethiopia', 'faeroe-islands',
                            'falkland-islands-malvinas', 'fiji', 'finland', 'france', 'french-guiana',
                            'french-polynesia', 'gabon', 'gambia', 'georgia', 'germany', 'ghana', 'gibraltar', 'greece',
                            'greenland', 'grenada', 'guadeloupe', 'guatemala', 'guinea-bissau', 'guinea', 'guyana',
                            'haiti', 'holy-see', 'honduras', 'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq',
                            'ireland', 'isle-of-man', 'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan',
                            'kenya', 'kuwait', 'kyrgyzstan', 'laos', 'latvia', 'lebanon', 'liberia', 'libya',
                            'liechtenstein', 'lithuania', 'luxembourg', 'macedonia', 'madagascar', 'malawi', 'malaysia',
                            'maldives', 'mali', 'malta', 'martinique', 'mauritania', 'mauritius', 'mayotte', 'mexico',
                            'moldova', 'monaco', 'mongolia', 'montenegro', 'montserrat', 'morocco', 'mozambique',
                            'myanmar', 'namibia', 'nepal', 'netherlands', 'new-caledonia', 'new-zealand', 'nicaragua',
                            'niger', 'nigeria', 'norway', 'oman', 'pakistan', 'panama', 'papua-new-guinea', 'paraguay',
                            'peru', 'philippines', 'poland', 'portugal', 'qatar', 'reunion', 'romania', 'russia',
                            'rwanda', 'saint-barthelemy', 'saint-kitts-and-nevis', 'saint-lucia', 'saint-martin',
                            'saint-pierre-and-miquelon', 'saint-vincent-and-the-grenadines', 'san-marino',
                            'sao-tome-and-principe', 'saudi-arabia', 'senegal', 'serbia', 'seychelles', 'sierra-leone',
                            'singapore', 'sint-maarten', 'slovakia', 'slovenia', 'somalia', 'south-africa',
                            'south-korea', 'south-sudan', 'spain', 'sri-lanka', 'state-of-palestine', 'sudan',
                            'suriname', 'swaziland', 'sweden', 'switzerland', 'syria', 'taiwan', 'tanzania', 'thailand',
                            'timor-leste', 'togo', 'trinidad-and-tobago', 'tunisia', 'turkey',
                            'turks-and-caicos-islands', 'uganda', 'uk', 'ukraine', 'united-arab-emirates', 'uruguay',
                            'us', 'uzbekistan', 'venezuela', 'viet-nam', 'western-sahara', 'zambia', 'zimbabwe']


def get_data(country_name):
    worldometer_cname = country_name.replace(' ', '-') if country_name not in WORLDOMETER_NAME \
        else WORLDOMETER_NAME[country_name]

    if worldometer_cname in COUNTRY_LIST_WORLDOMETER:
        try:
            data = get_data_from_worldometer(worldometer_cname)
            return data
        except Exception as e:
            print(f"Could not retrieve data from Worldometer, trying JHU...{e}")
    
    USE_API = get_USE_API()[0]
    if not USE_API:
        return None
    data = get_data_from_api(country_name)

    if not data['Cases']['data']:
        return None
    return data


def get_data_from_api(country_name):
    country_code = COUNTRY_NAME_TO_CODE[country_name]
    
    covid19 = get_USE_API()[1]
    
    locations = covid19.getLocationByCountryCode(country_code, timelines=True)

    confirmed_dict = Counter()
    deaths_dict = Counter()
    for location in locations:
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


def convert_dates(dates):
    """
    Convert array of dates in format "MMM DD" OR "yyyy" to "YYYY-M-DD"
    """
    out = []

    for i, date in enumerate(dates):
        if len(date.split())==1:
            yr = dates[i].strip()
            month, day = dates[i-1].split()
            out.append(f"{yr}-{MONTHS_DICT[month]}-{day}")

    return out


def get_data_from_worldometer(country_name):
    base_url = 'https://www.worldometers.info/coronavirus/country/'
    if country_name == 'world':
        url = 'https://www.worldometers.info/coronavirus/'
    else:
        url = base_url + country_name

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()

    titles = ['Cases', 'Deaths', 'Currently Infected']
    data = {}
    for line in str(webpage).replace('\\n', '').split('Highcharts.chart')[1:]:
        keys = line.split(': ')

        done = 0
        for k, key in enumerate(keys):
            if 'categories' in key and 'xAxis' in keys[k-1]:
                plot_dates = keys[k + 1].replace('[', '').split(']')[0].replace('"', '').split(',')
                plot_dates = convert_dates(plot_dates)
            if 'name' in key and 'series' in keys[k-1]:
                name = keys[k + 1].replace("\\'", "").split(',')[0]
                if name not in titles:
                    break
                done += 1
            if 'data' in key:
                datum = keys[k + 1].replace('[', '').split(']')[0].split(',')
                data[name] = {'dates': plot_dates, 'data': datum }
                done += 1
            if done == 2:
                break
    
    return data