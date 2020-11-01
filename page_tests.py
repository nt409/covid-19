import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from config import dummy_figure

test_content = html.Div([

        dbc.Card(
            html.Div([
            dcc.Graph(figure=dummy_figure,id='tests-plot'),
            html.Hr(),
            dcc.Graph(figure=dummy_figure,id='tests-plot-1'),
            ],
            style={'margin': '10px'}
            ),
        color='light'
        ),


    ])

test_controls = html.Div([

        dbc.Card([
            dbc.FormGroup([
                dbc.Row([ # R2205
                    dbc.Col([ # C2206

                html.P(id="tests-text",style={'marginTop': '15px'}),
                html.P(id="tests-text-1"),
                html.P(id="tests-text-2",style={'marginBottom': '15px'}),

                dbc.Spinner(html.Div(id="loading-tests"),color='primary'),

                dbc.Row([ # R2210

                    dbc.Button([html.I(className="fas fa-chart-area"),' Plot'],
                                    color='primary',
                                    className='mb-3',
                                    id="plot-button-tests",
                                    size='lg',
                                    style = {'cursor': 'pointer'}),
                        ],

                justify='center'), # R2210
                
                html.Hr(),
                
                dbc.Row([ # R2235

                    dbc.Col([ # C2241
                        dbc.Row(html.H5("Prior"), justify='center'),
                        dbc.Row(html.P("Probability of having covid based on their characteristics", style={'fontSize': '12px'}),justify='center'),

                        dbc.Row([
                        dbc.Input(id="prior", placeholder="Enter a number between 0 and 1", type="number", 
                            style={'textAlign':'center'}, min=0, max=1, step=10**(-4),
                            value=0.60
                            ),
                        ]),
                    ],
                    width=5,
                    lg=3,
                    align="center",
                    style={'margin': '10px'},
                    ), # C2241



                    dbc.Col([ # C2251
                        dbc.Row(html.H5("Sensitivity"), justify='center'),
                        dbc.Row(html.P("Probability test correctly identifies positive", style={'fontSize': '12px'}),justify='center'),

                        dbc.Row([
                        dbc.Input(id="sens", placeholder="Enter a number between 0 and 1", type="number", 
                            style={'textAlign':'center'}, min=0, max=1, step=10**(-4),
                            value=0.7
                            ),
                        ]),
                    ],
                    width=5,
                    lg=3,
                    align="center",
                    style={'margin': '10px'},
                    ), # C2251




                    dbc.Col([ # C2261
                        dbc.Row(html.H5("Specificity"), justify='center'),
                        dbc.Row(html.P("Probability test correctly identifies negative", style={'fontSize': '12px'}),justify='center'),

                        dbc.Row([
                        dbc.Input(id="spec", placeholder="Enter a number between 0 and 1", type="number",
                            style={'textAlign':'center'}, min=0, max=1, step=10**(-4),
                            value=0.95
                            ),
                        ]),
                    ],
                    width=5,
                    lg=3,
                    align="center",
                    style={'margin': '10px'},
                    ), # C2261
                


                ],
                justify='center'), # R2235

                    ],
                    width=10,
                    align="center",
                    style={'margin': '10px'},
                    ), # C2206
                ],
                justify='center'), # R2205
            ])
        ],
        color='light'
        ),


    ])


layout_tests = html.Div([
                                        dbc.Row([ # R2191
                                            dbc.Col([
                                                    html.Div(test_content,id='test-content', style={'margin': '10px'}),                                                                                        

                                                    html.Div(test_controls,id='test-controls', style={'margin': '10px'}),

                                            ],
                                            width=True,
                                            ),

                                        ],
                                        justify='center',
                                        style={'margin': '15px'}
                                        ),  # R2191

    ],
    style={'fontSize': '11', 'marginBottom': '200px'},
    )

