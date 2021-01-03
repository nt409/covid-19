from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


def get_vaccine_data():

    url = 'https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv'

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    webpage = urlopen(req).read()
    
    soup = BeautifulSoup(webpage, 'html.parser')
    
    headings=soup.find_all('th')

    cols = {}
    categories = ['location', 'date', 'total_vaccinations']

    for category in categories:
        is_this_col = [True if category in hh else False for hh in headings]
        cols[category] = int(np.where(is_this_col)[0])
    
    lists = {}
    for category in categories:
        lists[category] = []


    for row in soup.find_all('tr')[1:]:
        col = row.find_all('td')

        for category in categories:
            try:
                from_table = col[cols[category]+1]
                from_table = str(from_table)
                without_open_tag = from_table.split('<td>')[1]
                without_tags = without_open_tag.split('</td>')[0]
                new_row = without_tags
            except Exception as e:
                print(e)
                new_row='NA'
            lists[category].append(new_row)

    df = pd.DataFrame({})
    for category in categories:
        df[category] = lists[category]
    
    return df



VACCINE_COUNTRY_LIST = [
    'United Kingdom',
    'England',
    'Northern Ireland',
    'Scotland',
    'Wales',
    'United States',
    'World',
    'Argentina',
    'Austria',
    'Bahrain',
    'Bulgaria',
    'Canada',
    'Chile',
    'China',
    'Costa Rica',
    'Croatia',
    'Denmark',
    'Estonia',
    'Finland',
    'France',
    'Germany',
    'Greece',
    'Hungary',
    'Iceland',
    'Ireland',
    'Israel',
    'Italy',
    'Kuwait',
    'Latvia',
    'Lithuania',
    'Luxembourg',
    'Mexico',
    'Oman',
    'Poland',
    'Portugal',
    'Romania',
    'Russia',
    ]