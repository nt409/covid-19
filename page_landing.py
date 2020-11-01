import dash
import dash_bootstrap_components as dbc
import dash_html_components as html



card_inter  = dbc.Card(
    [
        html.A([
            dbc.CardImg(src="https://res.cloudinary.com/hefjzc2gb/image/upload/c_fill,h_465,w_1100/v1588428400/inter_ggjecx.png", top=True),
            ],
            href = '/inter',
            style = {'cursor': 'pointer'}
        ),
        dbc.CardBody(
            [
                html.H4("Interactive Model",style={'textAlign': 'center'},  className="card-title"),
                html.Div(
                    "See how different control measures implemented today could impact on infections, hospitalisations and deaths.",
                    className="card-text",
                    style={'textAlign': 'justify', 'marginTop': '10px', 'marginBottom': '10px'}
                ),
        dbc.Row([
                dbc.Button("Start predicting", href='/inter', color="primary"),
        ],
        justify='center'),
        ]
        ),
    ],
)




card_intro  = dbc.Card(
    [
        html.A([
            dbc.CardImg(src="https://res.cloudinary.com/hefjzc2gb/image/upload/c_fill,g_face,h_465,w_1100,x_595,y_51/v1588428998/Capture_ljzq0a.png", top=True),
            ],
            href = '/intro',
            style = {'cursor': 'pointer'}
        ),
        dbc.CardBody(
            [
                html.H4("Background", style={'textAlign': 'center'},  className="card-title"),
                html.Div(
                    "An introduction to mathematical modelling presented by experts in epidemiology from the University of Cambridge.",
                    className="card-text",
                    style={'textAlign': 'justify', 'marginTop': '10px', 'marginBottom': '10px'}
                ),
            dbc.Row([
                dbc.Button("Learn more", href = '/intro', color="primary"),
            ],
            justify='center'),
            ],
            ),
    ]
    )



card_int_tests  = dbc.Card(
    [
        html.A([
            dbc.CardImg(src="https://res.cloudinary.com/hefjzc2gb/image/upload/c_fill,g_face,h_465,w_1100,x_595,y_51/v1602007670/tests_wkwc2u.png", top=True),
            ],
            href = '/tests',
            style = {'cursor': 'pointer'}
        ),
        dbc.CardBody(
            [
                html.H4("Interpreting tests", style={'textAlign': 'center'},  className="card-title"),
                html.Div(
                    "Use conditional probability to consider how likely different tests are to give correct results.",
                    className="card-text",
                    style={'textAlign': 'justify', 'marginTop': '10px', 'marginBottom': '10px'}
                ),
            dbc.Row([
                dbc.Button("Learn more", href = '/tests', color="primary"),
            ],
            justify='center'),
            ],
            ),
    ]
    )

card_data  = dbc.Card(
    [
        html.A([
            dbc.CardImg(src="https://res.cloudinary.com/hefjzc2gb/image/upload/c_fill,h_465,w_1100/v1588428400/data_dd04fu.png", top=True),
            ],
            href = '/data',
            style = {'cursor': 'pointer'}
        ),
        dbc.CardBody(
            [
                html.H4("Global Data Feed", style={'textAlign': 'center'}, className="card-title"),
                html.Div(
                    "Real-time data on coronavirus cases and deaths from hundreds of countries around the world.",
                    className="card-text",
                    style={'textAlign': 'justify', 'marginTop': '10px', 'marginBottom': '10px'}
                ),
                dbc.Row([
                    dbc.Button("Explore", href = '/data',color="primary"),
                ],
                justify='center'),
            ]
        ),
    ],
    # style={"width": "18rem"},
)



layout_enter = html.Div(
        [
        dbc.Row([
        dbc.Col([


            dbc.Row([
                dbc.Col([
                card_intro,
                ],
                width=5,
                md = 3,
                style = {'marginRight': '20px', 'marginLeft': '20px', 'marginTop': '20px', 'marginBottom': '20px', 'textAlign': 'center'}
                ),
                
                dbc.Col([
                card_inter,
                ],
                width=5,
                md = 3,
                style = {'marginRight': '20px', 'marginLeft': '20px', 'marginTop': '20px', 'marginBottom': '20px', 'textAlign': 'center'}
                ),

                dbc.Col([
                card_int_tests,
                ],
                width=5,
                md = 3,
                style = {'marginRight': '20px', 'marginLeft': '20px', 'marginTop': '20px', 'marginBottom': '20px', 'textAlign': 'center'}
                ),

                dbc.Col([
                card_data,
                ],
                width=5,
                md = 3,
                style = {'marginRight': '20px', 'marginLeft': '20px', 'marginTop': '20px', 'marginBottom': '20px', 'textAlign': 'center'}
                ),

            ],
            justify='center',
            style = {'marginTop': '30px', 'marginBottom': '30px'}
            ),

        

        
        ],
        width = 12,
        ),
        ],
        justify = 'center',
        style = {'marginTop': '30px', 'fontSize': '80%'}),


        ])
