import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import datetime

from data_constants import COUNTRY_LIST









colors = {
    'background': 'white', # '#f4f6f7',
    'text': '#111111'
}







layout_data = html.Div(className="data-page-container",
                      id='output-layout', children=[
    html.Div(
        html.H1('Coronavirus cases and deaths by country')
    ,className="data-page-title", id="data-title-container"),



    # html.Div([
    
    html.Div([
        dbc.Card([

                html.Div(html.I("Select countries of interest, then click the Plot button."), className="select-text"),

                html.Div([
                    dbc.Checklist(
                        id=c_name,
                        options=[{'label': c_name.title() if c_name not in ['us', 'uk'] else c_name.upper(),
                                'value': c_name}],
                        value=[c_name] if c_name in ('uk', 'sweden', 'germany') else [],
                        )
                    for i, c_name in enumerate(COUNTRY_LIST)]),
        
        ],
        className="inter-card country-picker",
        ),
    ],className="data-country-picker"),





        html.Div([

                    
            html.Div([
                dbc.Button([
                    # html.I(className="fas fa-chart-area"),
                    html.Div([
                        html.Img(src='/assets/images/plot.svg')
                    ],className="plot-button-logo"),
                    html.Div('Plot',className='plot-button-text')
                    ],
                    color='primary',
                    className='plot-button data',
                    id="button-plot"),
            ]),

            html.Div("This section enables you to compare different countries' reported cases and deaths in real-time, and predict future numbers assuming exponential growth.", 
            className="this-section"),

            html.Div([
                html.Div("Fit exponential from: "),
                dcc.DatePickerSingle(
                    id='start-date',
                    min_date_allowed=datetime.date(2020, 1, 22),
                    max_date_allowed=datetime.date(2022, 1, 1),
                    initial_visible_month=datetime.date.today() - datetime.timedelta(days=7),
                    date=datetime.date.today() - datetime.timedelta(days=7),
                    display_format='D-MMM-YYYY',
                ),
            ],className="date-and-text checklist"),

            html.Div([
                html.Div("Predict until: "),
                dcc.DatePickerSingle(
                    id='end-date',
                    min_date_allowed=datetime.date(2020, 1, 22),
                    max_date_allowed=datetime.date(2022, 1, 1),
                    initial_visible_month=datetime.date.today(),
                    date=datetime.date.today(),
                    display_format='D-MMM-YYYY',
                ),
            ],className="date-and-text checklist"),
            
            
            html.Div([
                # html.Div("Show exponential fits?"),
                dbc.Checklist(
                    id='show-exponential-check',
                    options=[{'label': "Show exponential fits?", 'value': 'exponential'}],
                    value=['exponential'],
                    className="checklist-top",
                ),
            ],className="date-and-text"),
            
            html.Div([
                # html.Div("Plot as percentage of population?"),
                dbc.Checklist(
                    id='normalise-check',
                    options=[{'label': "Plot as percentage of population?", 'value': 'normalise'}],
                    value=[],
                    className="checklist-top",
                ),
            ],className='date-and-text'),
            
            
            dcc.Loading(id="loading-icon", children=[html.Div(id="loading-output-1")], type="default"),

            html.Hr(),

            html.H3(children='Total Cases', className="plot-title"),

            html.Div(className="align-countries", children=[
                html.Div([
                    dbc.Checklist(
                        id='align-cases-check',
                        options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                                'value': 'align'}],
                        value=[],
                    ),
                ],className="checkpoint-container"), 

                html.Div(className='align-container',children=[
                    dcc.Input(
                        id="align-cases-input",
                        type="number",
                        placeholder='Number of cases',
                        value=1000,
                        min=0,
                        debounce=True,
                    ),
                ]),
                html.Div(className="percent-container",children=[
                    html.Div(id='display_percentage_text_cases', style={'display': 'none'}, children=[
                        html.P("% of population")
                    ]),
                ]),
            ]),
            
            html.Div(dcc.Graph(id='infections-plot'),className='data-fig'),


            html.H3(children='Total Deaths', className="plot-title"),
            html.Div(className="align-countries", children=[
                html.Div([
                    dbc.Checklist(
                        id='align-deaths-check',
                        options=[{'label': "Align countries by the date when the number of confirmed deaths was ",
                                'value': 'align'}],
                        value=[],
                    ),
                ],className="checkpoint-container"), 

                html.Div(className='align-container',children=[
                    dcc.Input(
                        id="align-deaths-input",
                        type="number",
                        placeholder='Number of deaths',
                        value=20,
                        min=0,
                        debounce=True,
                        
                    ),
                ]),
                html.Div(className="percent-container",children=[
                    html.Div(id='display_percentage_text_deaths', style={'display': 'none'}, children=[
                        html.P("% of population")
                    ]),
                ]),
            ]),
            html.Div(dcc.Graph(id='deaths-plot'),className='data-fig'),

            html.Div(id='active-cases-container', children=[

                html.H3(children='Active Cases', className="plot-title"),
                html.Div(className="align-countries", children=[
                    html.Div([
                        dbc.Checklist(
                            id='align-active-cases-check',
                            options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                                    'value': 'align'}],
                            value=[],
                        ),
                    ],className="checkpoint-container"), 

                    html.Div(className='align-container',children=[
                        dcc.Input(
                            id="align-active-cases-input",
                            type="number",
                            placeholder='Number of cases',
                            value=1000,
                            min=0,
                            debounce=True,
                            
                        ),
                    ]),
                ]),
                html.Div(className="percent-container",children=[
                    html.Div(id='display_percentage_text_active', style={'display': 'none'}, children=[
                        html.P("% of population")
                    ]),
                ]),
                html.Div(dcc.Graph(id='active-plot'),className='data-fig'),
            ]),

            html.Div(id='daily-cases-container', children=[
                html.H3(children='Daily New Cases', className="plot-title"),
                html.Div(className="align-countries", children=[
                    html.Div([
                        dcc.Checklist(
                            id='align-daily-cases-check',
                            options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                                        'value': 'align'}],
                            value=[],
                        ),
                    ],className="checkpoint-container"),

                    html.Div(className='align-container',children=[
                        dcc.Input(
                            id="align-daily-cases-input",
                            type="number",
                            placeholder='Number of cases',
                            value=1000,
                            min=0,
                            debounce=True,
                            
                        ),
                    ]),
                    html.Div(className="percent-container",children=[
                        html.Div(id='display_percentage_text_daily_cases', style={'display': 'none'}, children=[
                            html.P("% of population")
                        ]),
                    ]),
                ]),
                html.Div(dcc.Graph(id='daily-cases-plot'),className='data-fig'),
            ]),

            html.Div(id='daily-deaths-container', children=[
                html.H3(children='Daily New Deaths', className="plot-title"),
                html.Div(className="align-countries", children=[
                    html.Div([
                        dcc.Checklist(
                            id='align-daily-deaths-check',
                            options=[{'label': "Align countries by the date when the number of confirmed deaths was ",
                                    'value': 'align'}],
                            value=[],
                        ),
                    ],className="checkpoint-container"), 

                    html.Div(className='align-container',children=[
                        dcc.Input(
                            id="align-daily-deaths-input",
                            type="number",
                            placeholder='Number of deaths',
                            value=1000,
                            min=0,
                            debounce=True,
                            
                        ),
                    ]),
                    html.Div(className="percent-container",children=[
                        html.Div(id='display_percentage_text_daily_deaths', style={'display': 'none'}, children=[
                            html.P("% of population")
                        ]),
                    ]),
                ]),
                html.Div(dcc.Graph(id='daily-deaths-plot'),className='data-fig'),
            ]),

            html.H3(children='New Cases vs Total Cases', className="plot-title"),
            html.Div(dcc.Graph(id='new-vs-total-cases'),className='data-fig'),

            html.H3(children='New Deaths vs Total Deaths', className="plot-title"),
            html.Div(dcc.Graph(id='new-vs-total-deaths'),className='data-fig'),

            html.Div([
                html.Li(html.I(
                    "Caution should be applied when directly comparing the number of confirmed cases of each country. "
                    "Different countries have different testing rates, and may underestimate the number of cases "
                    "by varying amounts."),
                    ),
                html.Li(html.I(
                    "The models assume exponential growth - social distancing, quarantining, herd immunity, "
                    "and other factors will slow down the predicted trajectories. "
                    "Thus, predicting too far in the future is not recommended."),
                    ),
                html.Li(html.I(
                    "Some countries do not have available data for the number of Active Cases. "),
                    ),
                html.Li(html.I(
                    "The last plot is an informative way to compare how each country was increasing when they had "
                    "different numbers of total cases (each point is a different day); countries that fall below "
                    "the general linear line on the log-log plot are reducing their growth rate of COVID-19 cases."),
                    ),
            ], className='data-text')
        ], className="data-graphs"),


        # ],className="data-page-container2"),

        
        
        
        html.Hr(),

        html.Div(id='hidden-stored-data', style={'display': 'none'}),
        

])


