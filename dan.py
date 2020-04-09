import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from matplotlib import colors as mcolors
import datetime
import json
import copy

from dan_get_data import get_data
from dan_constants import POPULATIONS

colours = ['#1f77b4','#ff7f0e', '#2ca02c','#9467bd', '#8c564b', '#e377c2', '#d62728', '#7f7f7f', '#bcbd22', '#17becf',
           'blue', 'purple', 'pink', 'cyan', '#FF1493', 'navy', '#aaffc3', '#228B22', '#aa6e28', '#FFA07A',
           ] + list(mcolors.CSS4_COLORS.keys())

COUNTRY_LIST = ['world',
                'uk',
                'us',
                'italy',
                'spain',
                'germany',
                'iran',
                'france',
                'australia',
                'albania',
                'algeria',
                'andorra',
                'argentina',
                'armenia',
                'austria',
                'bahrain',
                'belgium',
                'bosnia and herzegovina',
                'brazil',
                'brunei',
                'bulgaria',
                'burkina faso',
                'canada',
                'chile',
                'china',
                'colombia',
                'costa rica',
                'croatia',
                'cyprus',
                'czechia',
                'denmark',
                'dominican republic',
                'ecuador',
                'egypt',
                'estonia',
                'finland',
                'greece',
                'hong kong',
                'hungary',
                'iceland',
                'india',
                'indonesia',
                'iraq',
                'ireland',
                'israel',
                'japan',
                'jordan',
                'kuwait',
                'latvia',
                'lebanon',
                'lithuania',
                'luxembourg',
                'malaysia',
                'mexico',
                'moldova',
                'morocco',
                'netherlands',
                'new zealand',
                'north macedonia',
                'norway',
                'pakistan',
                'palestine',
                'panama',
                'peru',
                'philippines',
                'poland',
                'portugal',
                'qatar',
                'romania',
                'russia',
                'san marino',
                'saudi arabia',
                'serbia',
                'singapore',
                'slovakia',
                'slovenia',
                'south africa',
                'south korea',
                'sri lanka',
                'sweden',
                'switzerland',
                'taiwan',
                'thailand',
                'tunisia',
                'turkey',
                'united arab emirates',
                'ukraine',
                'uruguay',
                ]

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
server = app.server

colors = {
    'background': '#f4f6f7',
    'text': '#111111'
}


layout_dan = html.Div(style={'backgroundColor': colors['background'],'backgroundColor': '#f4f6f7'}, # id='cont-data',
                      children=[

    html.Div(style={'height': '3vh'}),

    html.Div([
        html.Div([
            html.Button(
                children='Plot',
                id='button-plot',
                type='submit',
                style={"margin": "15px", 'background-color': '#446E9B', 'color': 'white', 'width': '80%',
                       'height': '30px', 'font-size': '20px', 'border': 'None', 'border-radius': '10px'}
            ),
            html.I("Select countries of interest, then click the Plot button above.",
                   style={'textAlign': 'center', 'color': colors['text'],
                          "marginLeft": "5px", "marginRight": "15px"}),
            html.Div(style={'marginTop': '10px'}),
            html.Div([
                dbc.Checklist(
                    id=c_name,
                    options=[{'label': c_name.title() if c_name not in ['us', 'uk'] else c_name.upper(),
                              'value': c_name}],
                    value=[c_name] if c_name in ('us', 'uk', 'italy') else [],
                    style={"marginLeft": "15px", 'textAlign': 'left'},
                    inputStyle={"marginRight": "5px"})
                for i, c_name in enumerate(COUNTRY_LIST)]),
        ], style={'width': '17%', 'display': 'inline-block', 'vertical-align': 'top',
                  'background-color': '#d1d4d7', 'horizontal-align': 'left', 'textAlign': 'center'}),
        html.Div(style={'width': '5%', 'display': 'inline-block'}),
        html.Div([
            html.Div([
                html.I("This section enables you to compare different countries' reported cases and deaths in real-time, and predict future numbers assuming exponential growth.")],
                style={'marginBottom': '3vh','marginTop': '1vh'}
            ),

            html.Div([

                html.I("Fit exponential from: ",
                       style={'textAlign': 'center', 'color': colors['text'], "marginLeft": "15px",}),
                dcc.DatePickerSingle(
                    id='start-date',
                    min_date_allowed=datetime.date(2020, 1, 22),
                    max_date_allowed=datetime.date(2022, 1, 1),
                    initial_visible_month=datetime.date.today() - datetime.timedelta(days=7),
                    date=datetime.date.today() - datetime.timedelta(days=7),
                    display_format='D-MMM-YYYY',
                    style={'textAlign': 'center'}
                ),
            ], style={'display': 'inline-block', 'horizontal-align': 'center', 'textAlign': 'center'}),
            html.Div([
                html.I("Predict until: ",
                       style={'textAlign': 'center', 'color': colors['text'], "marginLeft": "15px", }),
                dcc.DatePickerSingle(
                    id='end-date',
                    min_date_allowed=datetime.date(2020, 1, 22),
                    max_date_allowed=datetime.date(2022, 1, 1),
                    initial_visible_month=datetime.date.today(),
                    date=datetime.date.today(),
                    display_format='D-MMM-YYYY',
                    style={'textAlign': 'center'}
                ),
            ], style={'display': 'inline-block', 'horizontal-align': 'center', 'textAlign': 'center',
                      "marginBottom": "15px",}),
            dbc.Checklist(
                id='show-exponential-check',
                options=[{'label': "Show exponential fits?", 'value': 'exponential'}],
                value=['exponential'],
                style={'textAlign': 'center', "marginBottom": "0px"},
                inputStyle={"marginRight": "5px"}
            ),
            dbc.Checklist(
                id='normalise-check',
                options=[{'label': "Plot as percentage of population?", 'value': 'normalise'}],
                value=[],
                style={'textAlign': 'center', "marginBottom": "20px"},
                inputStyle={"marginRight": "5px"}
            ),
            # dcc.Loading(id="loading-icon", children=[html.Div(id="loading-output-1")], type="default"),
            dbc.Spinner(html.Div(id="loading-icon"),color='primary',type='grow'),

            


            html.Hr(),
            html.H3(children='Total Cases', style={'textAlign': 'center', 'color': colors['text'],
                                                   'marginTop': '30px'}),

            html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                dbc.Checklist(
                    id='align-cases-check',
                    options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                              'value': 'align'}],
                    value=[],
                    style={'textAlign': 'left', "marginRight": "4px", 'display': 'inline-block'},
                    inputStyle={"marginRight": "5px"}
                ),
                dcc.Input(
                    id="align-cases-input",
                    type="number",
                    placeholder='Number of cases',
                    value=1000,
                    min=0,
                    debounce=True,
                    style={'width': 80},
                ),
                html.Div(id='display_percentage_text_cases', style={'display': 'none'}, children=[
                    html.P("% of population")
                ]),
            ]),
            dcc.Graph(id='infections-plot',style={'width': '100%', 'height': '50vh'}),
            html.H3(children='Total Deaths', style={'textAlign': 'center', 'color': colors['text'],
                                                    'marginTop': '10px'}),
            html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                dbc.Checklist(
                    id='align-deaths-check',
                    options=[{'label': "Align countries by the date when the number of confirmed deaths was ",
                              'value': 'align'}],
                    value=[],
                    style={'textAlign': 'left', "marginRight": "4px", 'display': 'inline-block'},
                    inputStyle={"marginRight": "5px"}
                ),
                dcc.Input(
                    id="align-deaths-input",
                    type="number",
                    placeholder='Number of deaths',
                    value=20,
                    min=0,
                    debounce=True,
                    style={'width': 80},
                ),
                html.Div(id='display_percentage_text_deaths', style={'display': 'none'}, children=[
                    html.P("% of population")
                ]),
            ]),
            dcc.Graph(id='deaths-plot',style={'width': '100%', 'height': '50vh'}),
            html.Div(id='active-cases-container', style={'display': 'block'}, children=[
                html.H3(children='Active Cases', style={'textAlign': 'center', 'color': colors['text']}),
                html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                    dbc.Checklist(
                        id='align-active-cases-check',
                        options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                                  'value': 'align'}],
                        value=[],
                        style={'textAlign': 'left', "marginRight": "4px", 'display': 'inline-block'},
                        inputStyle={"marginRight": "5px"}
                    ),
                    dcc.Input(
                        id="align-active-cases-input",
                        type="number",
                        placeholder='Number of cases',
                        value=1000,
                        min=0,
                        debounce=True,
                        style={'width': 80},
                    ),
                ]),
                html.Div(id='display_percentage_text_active', style={'display': 'none'}, children=[
                    html.P("% of population")
                ]),
                dcc.Graph(id='active-plot',style={'width': '100%', 'height': '50vh'}),
            ]),

            html.Div(id='daily-cases-container', children=[
                html.H3(children='Daily New Cases', style={'textAlign': 'center', 'color': colors['text'],
                                                           'marginTop': '10px'}),
                html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                    dcc.Checklist(
                        id='align-daily-cases-check',
                        options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                                  'value': 'align'}],
                        value=[],
                        style={'textAlign': 'left', "marginRight": "4px", 'display': 'inline-block'},
                        inputStyle={"marginRight": "5px"}
                    ),
                    dcc.Input(
                        id="align-daily-cases-input",
                        type="number",
                        placeholder='Number of cases',
                        value=1000,
                        min=0,
                        debounce=True,
                        style={'width': 80},
                    ),
                    html.Div(id='display_percentage_text_daily_cases', style={'display': 'none'}, children=[
                        html.P("% of population")
                    ]),
                ]),
                dcc.Graph(id='daily-cases-plot',style={'width': '100%', 'height': '50vh'}),
            ]),

            html.Div(id='daily-deaths-container', children=[
                html.H3(children='Daily New Deaths', style={'textAlign': 'center', 'color': colors['text'],
                                                            'marginTop': '10px'}),
                html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                    dcc.Checklist(
                        id='align-daily-deaths-check',
                        options=[{'label': "Align countries by the date when the number of confirmed deaths was ",
                                  'value': 'align'}],
                        value=[],
                        style={'textAlign': 'left', "marginRight": "4px", 'display': 'inline-block'},
                        inputStyle={"marginRight": "5px"}
                    ),
                    dcc.Input(
                        id="align-daily-deaths-input",
                        type="number",
                        placeholder='Number of deaths',
                        value=1000,
                        min=0,
                        debounce=True,
                        style={'width': 80},
                    ),
                    html.Div(id='display_percentage_text_daily_deaths', style={'display': 'none'}, children=[
                        html.P("% of population")
                    ]),
                ]),
                dcc.Graph(id='daily-deaths-plot',style={'width': '100%', 'height': '50vh'}),
            ]),

            html.H3(children='New Cases vs Total Cases', style={'textAlign': 'center', 'color': colors['text'],
                                                    'marginTop': '10px'}),
            dcc.Graph(id='new-vs-total-cases',style={'width': '100%', 'height': '50vh'}),

            html.H3(children='New Deaths vs Total Deaths', style={'textAlign': 'center', 'color': colors['text'],
                                                                  'marginTop': '10px'}),
            dcc.Graph(id='new-vs-total-deaths',style={'width': '100%', 'height': '50vh'}),

            html.Li(html.I(
                "Caution should be applied when directly comparing the number of confirmed cases of each country. "
                "Different countries have different testing rates, and may underestimate the number of cases "
                "by varying amounts."),
                style={'textAlign': 'justify', 'color': colors['text']}),
            html.Li(html.I(
                "The models assume exponential growth - social distancing, quarantining, herd immunity, "
                "and other factors will slow down the predicted trajectories. "
                "Thus, predicting too far in the future is not recommended."),
                style={'textAlign': 'justify', 'color': colors['text']}),
            html.Li(html.I(
                "Some countries do not have available data for the number of Active Cases. "),
                style={'textAlign': 'justify', 'color': colors['text']}),
            html.Li(html.I(
                "The last plot is an informative way to compare how each country was increasing when they had "
                "different numbers of total cases (each point is a different day); countries that fall below "
                "the general linear line on the log-log plot are reducing their growth rate of COVID-19 cases."),
                style={'textAlign': 'justify', 'color': colors['text']}),
        ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top', 'horizontal-align': 'center',
                  'textAlign': 'center', "marginLeft": "0px"}),
        html.Hr(),
        html.Div(id='hidden-stored-data', style={'display': 'none'}),
        
    ], style={'horizontal-align': 'center', 'textAlign': 'center'}),
])


if __name__ == '__main__':
    app.layout = layout_dan
    app.run_server(debug=True)
