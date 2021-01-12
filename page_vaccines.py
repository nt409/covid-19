import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import datetime

from vaccine_scraper import VACCINE_COUNTRY_LIST





layout_vaccine = html.Div(className="data-page-container",
                    #   id='output-layout', 
                      children=[
    html.Div(
        html.H1('Coronavirus vaccinations by country')
    ,className="data-page-title", id="v-data-title-container"),

    html.Div([
        dbc.Card([

                html.Div(html.I("Select countries of interest, then click the Plot button."), className="select-text"),

                html.Div([
                    dbc.Checklist(
                        id=f"{c_name}-v-data",
                        options=[{'label': c_name.title(),
                                'value': c_name}],
                        value=[c_name] if c_name in ['United Kingdom', 'United States'] else [],
                        )
                    for i, c_name in enumerate(VACCINE_COUNTRY_LIST)]),
        
        ],
        className="inter-card country-picker",
        ),
    ],className="data-country-picker"),





        html.Div([

                    
            html.Div([
                dbc.Button([
                    html.Div([
                        html.Img(src='/assets/images/plot.svg')
                    ],className="plot-button-logo"),
                    html.Div('Plot',className='plot-button-text')
                    ],
                    color='primary',
                    className='plot-button data',
                    id="button-plot-vd"),
            ]),

            html.Div(["This section enables you to compare different countries' vaccination data in real-time. Use the checkboxes on the left to select the countries to plot. Data will automatically update as they are published. The data are doses administered, which is not the same as number of people vaccinated, which depends on the dose regime. Data source: ",
            html.A('OWID', href='https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv '),
                    "."],
            className="this-section"),

            dcc.Loading(id="loading-icon-vd", children=[html.Div(id="loading-output-1-vd")], type="default"),

            html.Hr(),

            html.H3(children='Total vaccinations', className="plot-title"),

            
            html.Div(dcc.Graph(id='vaccine-plot',
                config = {'modeBarButtonsToRemove': [
                    'pan2d',
                    'toImage',
                    'select2d',
                    'toggleSpikelines',
                    'hoverCompareCartesian',
                    'hoverClosestCartesian',
                    'lasso2d',

                ]}
                ),className='data-fig'),
        

])
])


