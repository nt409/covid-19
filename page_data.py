import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import datetime

from data_constants import COUNTRY_LIST









colors = {
    'background': 'white', # '#f4f6f7',
    'text': '#111111'
}







layout_data = html.Div(style={'backgroundColor': colors['background']},
                      id='output-layout', children=[


    html.Div(style={'height': '20px'}),

    html.Div([
        dbc.Card([

                html.Div([
                    dbc.Button([html.I(className="fas fa-chart-area"),' Plot'],
                                    color='primary',
                                    className='mb-3',
                                    id="button-plot",
                                    size='lg',
                                    style = {'cursor': 'pointer', 'marginTop': '10px'}),

                ]),

                html.I("Select countries of interest, then click the Plot button above.",
                    style={'textAlign': 'center', 'color': colors['text'],
                            "margin-left": "5px", "margin-right": "5px"}),

                html.Div([
                    dbc.Checklist(
                        id=c_name,
                        options=[{'label': c_name.title() if c_name not in ['us', 'uk'] else c_name.upper(),
                                'value': c_name}],
                        value=[c_name] if c_name in ('us', 'uk', 'italy') else [],
                        style={"margin-left": "15px", 'horizontal-align': 'left', 'textAlign': 'left'},
                        inputStyle={"margin-right": "5px"})
                    for i, c_name in enumerate(COUNTRY_LIST)]),
        
        ],
        className="inter-card",
        style={'display': 'inline-block',
                'vertical-align': 'top',
                'horizontal-align': 'left', 
                'width': '17%',
                'textAlign': 'center'}),



        html.Div(style={'width': '5%', 'display': 'inline-block'}),

        html.Div([
            html.Div([
                html.I("This section enables you to compare different countries' reported cases and deaths in real-time, and predict future numbers assuming exponential growth.")],
                style={'marginBottom': '20px','marginTop': '20px'}
            ),

            html.Div([

                html.I("Fit exponential from: ",
                       style={'textAlign': 'center', 'color': colors['text'], "margin-left": "15px",}),
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
                       style={'textAlign': 'center', 'color': colors['text'], "margin-left": "15px", }),
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
                      "margin-bottom": "15px",}),
            dbc.Checklist(
                id='show-exponential-check',
                options=[{'label': "Show exponential fits?", 'value': 'exponential'}],
                value=['exponential'],
                style={'textAlign': 'center', "margin-bottom": "0px"},
                inputStyle={"margin-right": "5px"}
            ),
            dbc.Checklist(
                id='normalise-check',
                options=[{'label': "Plot as percentage of population?", 'value': 'normalise'}],
                value=[],
                style={'textAlign': 'center', "margin-bottom": "20px"},
                inputStyle={"margin-right": "5px"}
            ),
            dcc.Loading(id="loading-icon", children=[html.Div(id="loading-output-1")], type="default"),

            html.Hr(),
            html.H3(children='Total Cases', style={'textAlign': 'center', 'color': colors['text'],
                                                   'marginTop': '30px'}),

            html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                dbc.Checklist(
                    id='align-cases-check',
                    options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                              'value': 'align'}],
                    value=[],
                    style={'textAlign': 'left', "margin-right": "4px", 'display': 'inline-block'},
                    inputStyle={"margin-right": "5px"}
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
            dcc.Graph(id='infections-plot'),
            html.H3(children='Total Deaths', style={'textAlign': 'center', 'color': colors['text'],
                                                    'marginTop': '10px'}),
            html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                dbc.Checklist(
                    id='align-deaths-check',
                    options=[{'label': "Align countries by the date when the number of confirmed deaths was ",
                              'value': 'align'}],
                    value=[],
                    style={'textAlign': 'left', "margin-right": "4px", 'display': 'inline-block'},
                    inputStyle={"margin-right": "5px"}
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
            dcc.Graph(id='deaths-plot'),
            html.Div(id='active-cases-container', style={'display': 'block'}, children=[
                html.H3(children='Active Cases', style={'textAlign': 'center', 'color': colors['text']}),
                html.Div(style={'display': 'inline-block', 'textAlign': 'left'}, children=[
                    dbc.Checklist(
                        id='align-active-cases-check',
                        options=[{'label': "Align countries by the date when the number of confirmed cases was ",
                                  'value': 'align'}],
                        value=[],
                        style={'textAlign': 'left', "margin-right": "4px", 'display': 'inline-block'},
                        inputStyle={"margin-right": "5px"}
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
                dcc.Graph(id='active-plot'),
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
                        style={'textAlign': 'left', "margin-right": "4px", 'display': 'inline-block'},
                        inputStyle={"margin-right": "5px"}
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
                dcc.Graph(id='daily-cases-plot'),
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
                        style={'textAlign': 'left', "margin-right": "4px", 'display': 'inline-block'},
                        inputStyle={"margin-right": "5px"}
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
                dcc.Graph(id='daily-deaths-plot'),
            ]),

            html.H3(children='New Cases vs Total Cases', style={'textAlign': 'center', 'color': colors['text'],
                                                    'marginTop': '10px'}),
            dcc.Graph(id='new-vs-total-cases'),

            html.H3(children='New Deaths vs Total Deaths', style={'textAlign': 'center', 'color': colors['text'],
                                                                  'marginTop': '10px'}),
            dcc.Graph(id='new-vs-total-deaths'),

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
                  'textAlign': 'center', "margin-left": "0px"}),
        html.Hr(),
        html.Div(id='hidden-stored-data', style={'display': 'none'}),
        
    ], style={'horizontal-align': 'center', 'textAlign': 'center'}),
])


