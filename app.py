import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from gevent.pywsgi import WSGIServer
import pandas as pd
from math import floor, ceil, exp
from parameters_cov import params, age_risk_df_out
import numpy as np
import plotly.graph_objects as go
import copy
from flask import Flask
from flask_caching import Cache
import datetime
import json
from json import JSONEncoder

from config import preset_dict_high, preset_dict_low, \
    presets_dict_dropdown, initial_hr, initial_lr, \
    initial_strat

from cov_functions import run_model, test_probs, begin_date, \
    outcome_fn

from plotting import Bar_chart_generator, MultiFigureGenerator, \
    longname, month_len, extract_info, test_bar_plot, test_bar_plot2

from dan import layout_dan, COUNTRY_LIST, colours
from dan_get_data import get_data, COUNTRY_LIST_WORLDOMETER
from dan_constants import POPULATIONS

FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"


# min_date = '2020-2-15' # first day of data
# min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )

# max_date = datetime.datetime.today() - datetime.timedelta(days=1)
# max_date = str(max_date).split(' ')[0]

try:
    gd = get_data('uk')
    min_date = gd['Cases']['dates'][0]
    max_date = gd['Cases']['dates'][-1]
except:
    print("Cannot get dates from Worldometer")



min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )
max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d' )


COUNTRY_LIST_NICK = COUNTRY_LIST
COUNTRY_LIST_NICK = sorted(COUNTRY_LIST_NICK)
COUNTRY_LIST_NICK.remove('world')


initial_country = COUNTRY_LIST_NICK.index('uk')

backgroundColor = None # 'white' # '#f4f6f7'
disclaimerColor = '#e9ecef'





########################################################################################################################

# external_stylesheets = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
tab_label_color = 'black' # "#00AEF9"
external_stylesheets = [dbc.themes.LITERA, FA]
# Cerulean
# COSMO
# JOURNAL
# Litera
# MINTY
# SIMPLEX - not red danger
# spacelab good too
# UNITED

# spacelab

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.config.suppress_callback_exceptions = True

########################################################################################################################
# setup







annotz = [dict(x  = 0.5,
                    y  = 0.6,
                    text="Press the<br>'Plot' button!",
                    showarrow=False,
                    font=dict(
                        size=20,
                        color='black'
                    ),
                    xref = 'paper',
                    yref = 'paper',
        )]

scatter = go.Scatter(
                    x = [0,1],
                    y = [0,1],
                    mode = 'lines',
                    showlegend=False,
                    line = {'width': 0},
                    # legendgroup = listings[key]['host'],
                    # opacity=1,
                    )

dummy_figure = go.Figure(data=[scatter], layout= {'template': 'simple_white', 'annotations': annotz, 'xaxis': {'fixedrange': True}, 'yaxis': {'fixedrange': True}})




bar_height = '100'
bar_width  =  '100'

bar_non_crit_style = {'height': bar_height, 'width': bar_width, 'display': 'block' }





# cache = Cache(app.server, config={
#     # try 'filesystem' if you don't want to setup redis
#     'CACHE_TYPE': 'redis',
#     'CACHE_REDIS_URL': 'redis://h:paa75aa4b983ba337eb43b831e6833be6b6887e56023aa417e392dd2bf337e8b8@ec2-18-213-184-148.compute-1.amazonaws.com:31119'
# })













########################################################################################################################










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








########################################################################################################################
layout_model = html.Div([
    # dbc.Row([
        # html.Div([
            # dbc.Row([
                # dbc.Col([
                    # html.Div([

                                                                                                                                                                # dbc.Col([

                                                                                                                                                                    html.H3('Model Explanation',
                                                                                                                                                                    className = 'display-4',
                                                                                                                                                                    style = {'marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center', 'fontSize': '250%'}),

                                                                                                                                                                    html.Hr(),
                                                                                                                                                                    dcc.Markdown(
                                                                                                                                                                    '''
                                                                                                                                                                    *Underlying all of the predictions is a mathematical model. In this Section we explain how the mathematical model works.*

                                                                                                                                                                    We present a compartmental model for COVID-19, split by risk categories. That is to say that everyone in the population is **categorised** based on **disease status** (susceptible/ infected/ recovered/ hospitalised/ critical care/ dead) and based on **COVID risk**.
                                                                                                                                                                    
                                                                                                                                                                    The model is very simplistic but still captures the basic spread mechanism. It is far simpler than the [**Imperial College model**](https://spiral.imperial.ac.uk/handle/10044/1/77482), but it uses similar parameter values and can capture much of the relevant information in terms of how effective control will be.

                                                                                                                                                                    It is intended solely as an illustrative, rather than predictive, tool. We plan to increase the sophistication of the model and to update parameters as more (and better) data become available to us.
                                                                                                                                                                    
                                                                                                                                                                    We have **two risk categories**: high and low. **Susceptible** people get **infected** after contact with an infected person (from either risk category). A fraction of infected people (*h*) are **hospitalised** and the rest **recover**. Of these hospitalised cases, a fraction (*c*) require **critical care** and the rest recover. Of those in critical care, a fraction (*d*) **die** and the rest recover.

                                                                                                                                                                    The recovery fractions depend on which risk category the individual is in.
                                                                                                                                                                

                                                                                                                                                                    ''',
                                                                                                                                                                    style = {'textAlign': 'justify', 'font-family': 'sans-serif'}

                                                                                                                                                                    ),



                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/Capture_lomery.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto', 'display': 'block','marginTop': '10px','marginBottom': '10px'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    dcc.Markdown('''

                                                                                                                                                                    The selection of risk categories is done in the crudest way possible - an age split at 60 years (based on the age structure data below). A more nuanced split would give a more effective control result, since there are older people who are at low risk and younger people who are at high risk. In many cases, these people will have a good idea of which risk category they belong to.

                                                                                                                                                                    *For the more mathematically inclined reader, a translation of the above into a mathematical system is described below.*

                                                                                                                                                                    ''',style={'textAlign': 'justify', 'font-family': 'sans-serif', 'marginTop' : '20px','marginBottom' : '20px'}),
                                                                                                                                                                    
                                                                                                                                                                    

                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/eqs_f3esyu.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '10px','marginBottom': '10px'})
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),


                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/text_toshav.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '10px','marginBottom': '10px'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    
                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    Of those requiring critical care, we assume that if they get treatment, a fraction *1-d* recover. If they do not receive it they die, taking 2 days. The number able to get treatment must be lower than the number of ICU beds available.
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify', 'font-family': 'sans-serif'}),



                                                                                                                                                                    html.Hr(),

                                                                                                                                                                    html.H4('Parameter Values',style={'fontSize': '150%', 'textAlign': 'center'}),

                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    The model uses a weighted average across the age classes below and above 60 to calculate the probability of a member of each class getting hospitalised or needing critical care. Our initial conditions are updated to match new data every day (meaning the model output is updated every day, although in '**Custom Options**' there is the choice to start from any given day).

                                                                                                                                                                    We assume a 10 day delay on hospitalisations, so we use the number infected 10 days ago to inform the number hospitalised (0.044 of infected) and in critical care (0.3 of hospitalised). We calculate the approximate number recovered based on the number dead, assuming that 0.009 infections cause death. All these estimates are as per the Imperial College paper ([**Ferguson et al**](https://spiral.imperial.ac.uk/handle/10044/1/77482)).

                                                                                                                                                                    The number of people infected, hospitalised and in critical care are calculated from the recorded data. We assume that only half the infections are reported ([**Fraser et al.**](https://science.sciencemag.org/content/early/2020/03/30/science.abb6936)), so we double the recorded number of current infections. The estimates for the initial conditions are then distributed amongst the risk groups. These proportions are calculated using conditional probability, according to risk (so that the initial number of infections is split proportionally by size of the risk categories, whereas the initially proportion of high risk deaths is much higher than low risk deaths).

                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify', 'font-family': 'sans-serif'}),



                                                                                                                                                                    

                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1586345773/table_fhy8sf.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '10px','marginBottom': '10px'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),



                                                                                                                                                                    html.Div('** the Imperial paper uses 8 days in hospital if critical care is not required (as do we). It uses 16 days (with 10 in ICU) if critical care is required. Instead, if critical care is required we use 8 days in hospital (non-ICU) and then either recovery or a further 8 in intensive care (leading to either recovery or death).',
                                                                                                                                                                    style={'fontSize':'70%'}),

                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    Please use the following links: [**Ferguson et al**](https://spiral.imperial.ac.uk/handle/10044/1/77482), [**Anderson et al**](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30567-5/fulltext) and [**Zhao et al**](https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0006158&type=printable)
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify', 'font-family': 'sans-serif'}),


                                                                                                                                                                    html.H4('Age Structure',style={'fontSize': '150%', 'textAlign': 'center'}),
                                                                                                                                                                    
                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    The age data is taken from [**GOV.UK**](https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest) and the hospitalisation and critical care data is from the [**Imperial College Paper**](https://spiral.imperial.ac.uk/handle/10044/1/77482) (Ferguson et al.). This means that the age structure will not be accurate when modelling other countries.

                                                                                                                                                                    To find the probability of a low risk case getting hospitalised (or subsequently put in critical care), we take a weighted average by proportion of population. Note that the figures below are proportion of *symptomatic* cases that are hospitalised, which we estimate to be 55% of cases ([**Ferguson et al.**](https://spiral.imperial.ac.uk/handle/10044/1/77482)). The number requiring critical care is a proportion of this hospitalised number.

                                                                                                                                                                    *The table below shows the age structure data that was used to calculate these weighted averages across the low risk category (under 60) and high risk (over 60) category.*
                                                                                                                                                                    
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify', 'font-family': 'sans-serif', 'marginTop': '20px','marginBottom': '20px'}
                                                                                                                                                                    
                                                                                                                                                                    ),

                                                                                                                                                                    dbc.Table.from_dataframe(age_risk_df_out, striped=True, bordered = True, hover=True)




                        ],style={'fontSize': '11'})














########################################################################################################################
layout_intro = html.Div([
    dbc.Row([
                            # dbc.Jumbotron([##
                            dbc.Row([
    
    dbc.Col([
                            html.Div([
                html.Div([


    dbc.Tabs(id='intro-tabs',
             active_tab='tab_0',
             children = [
                
        
        
        
        dbc.Tab(label='Introduction to modelling',  tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_0', children=[

            
            html.H3('Introduction to mathematical modelling',className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            html.Hr(),

            html.Div('''
            Watch this video from Dr Cerian Webb, an expert in epidemiology and modelling from the University of Cambridge.
            ''',
            style={'textAlign': 'justify'}),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1586536825/WhatIsModellingv2_172141_jwpplb.mp4', #vc_h264
                    controls=True,
                    style={'maxWidth':'100%','height': 'auto','marginTop': '10px','marginBottom': '30px'}),
                    
                    ],
                    justify='center'
                    ),
            
                    

            
            html.Hr(),


            html.H3('Introducing SIR models',className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            html.Hr(),

            html.Div('''
            Watch this explanation from Dr Cerian Webb, to find out more about basic epidemiological models.
            ''',
            style={'textAlign': 'justify'}),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1585814499/StandardSIRModel_hu5ztn.mp4', # 
                    controls=True,
                    style={'maxWidth':'100%','height': 'auto','marginTop': '10px','marginBottom': '30px'}),
                    
                    ],
                    justify='center'
                    ),
            
                    
            
            html.Hr(),


            html.H3('Introducing the basic reproductive number',className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            html.Hr(),

            html.Div('''
            Watch Dr Cerian Webb introduce the basic reproductive number.
            ''',
            style={'textAlign': 'justify'}),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1586536823/AllAboutR_173637_poxzmb.mp4',
                    controls=True,
                    style={'maxWidth':'100%','height': 'auto','marginTop': '10px','marginBottom': '30px'}),
                    
                    ],
                    justify='center'
                    ),
            
            html.Hr(),

            html.H3('Introducing Herd Immunity',className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            html.Hr(),


            html.Div('''
            Watch Dr Cerian Webb introduce the concept of herd immunity.
            ''',
            style={'textAlign': 'justify'}),

            dbc.Row([
                    
                    html.Video(src="https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1588167893/HerdImmunity_144205_dyhaiy.mp4",
                    controls=True,
                    style={'maxWidth':'100%','height': 'auto','marginTop': '10px','marginBottom': '30px'}),
                    
                    ],
                    justify='center'
                    ),
            
            # html.Hr(),
            
            
        ]),

        dbc.Tab(label='Definitions', tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_defs',
            children=[
                
            html.H3('Definitions',className='display-4',
            style = { 'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            html.Hr(),

            html.Div('''            

            There are two key concepts that you need to understand before we can fully explore how the control measures work.
            '''),
            
            html.H4('1. Basic Reproduction Number',
            style = {'marginTop': '10px', 'fontSize': '150%'}),

            dcc.Markdown('''
            Any infectious disease requires both infectious individuals and susceptible individuals to be present in a population to spread. The higher the number of susceptible individuals, the faster it can spread since an infectious person can spread the disease to more susceptible people before recovering.

            The average number of infections caused by a single infected person is known as the '**effective reproduction number**' (*R*). If this number is less than 1 (each infected person infects less than one other on average) then the disease will not continue to spread. If it is greater than 1 then the disease will spread. For COVID-19 most estimates for *R* are between 2 and 3. We use the value *R*=2.4.
            ''',
            style={'textAlign': 'justify'}),

            html.H4('2. Herd Immunity',
            style = {'marginTop': '10px', 'fontSize': '150%'}),
            
            dcc.Markdown('''            


            Once the number of susceptible people drops below a certain threshold (which is different for every disease, and in simpler models depends on the basic reproduction number), the population is no longer at risk of an epidemic (so any new infection introduced will not cause infection to spread through an entire population).

            Once the number of susceptible people has dropped below this threshold, the population is termed to have '**herd immunity**'. Herd immunity is either obtained through sufficiently many individuals catching the disease and developing personal immunity to it, or by vaccination.

            For COVID-19, there is a safe herd immunity threshold of around 60% (=1-1/*R*), meaning that if 60% of the population develop immunity then the population is **safe** (no longer at risk of an epidemic).

            Coronavirus is particularly dangerous because most countries have almost 0% immunity since the virus is so novel. Experts are still uncertain whether you can build immunity to the virus, but the drop in cases in China would suggest that you can. Without immunity it would be expected that people in populated areas get reinfected, which doesn't seem to have happened.
            
            A further concern arises over whether the virus is likely to mutate. However it is still useful to consider the best way to managing each strain.
            ''',
            style={'textAlign': 'justify', 'font-family': 'sans-serif'}),

            ]
        ),

        dbc.Tab(label='COVID-19 Control Strategies',tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_control', children=[

            html.H3('Keys to a successful control strategy',
            className = 'display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            html.Hr(),


            dcc.Markdown('''            
            There are three main goals a control strategy sets out to achieve:

            1. Reduce the number of deaths caused by the pandemic,

            2. Reduce the load on the healthcare system,

            3. Ensure the safety of the population in future.

            An ideal strategy achieves all of the above whilst also minimally disrupting the daily lives of the population.

            However, controlling COVID-19 is a difficult task, so there is no perfect strategy. We will explore the advantages and disadvantages of each strategy.
            ''',
            style={'textAlign': 'justify', 'font-family': 'sans-serif'}),
            
            html.Hr(),

            html.H3('Strategies',
            className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            html.Hr(),

            
            html.H4('Reducing the infection rate',
            style = {'fontSize': '150%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),


            dcc.Markdown('''            

            Social distancing, self isolation and quarantine strategies slow the rate of spread of the infection (termed the 'infection rate'). In doing so, we can reduce the load on the healthcare system (goal 2) and (in the short term) reduce the number of deaths.

            This has been widely referred to as 'flattening the curve'; buying nations enough time to bulk out their healthcare capacity. The stricter quarantines are the best way to minimise the death rate whilst they're in place. A vaccine can then be used to generate sufficient immunity.

            However, in the absence of a vaccine these strategies do not ensure the safety of the population in future (goal 3), meaning that the population is still highly susceptible and greatly at risk of a future epidemic. This is because these strategies do not lead to any significant level of immunity within the population, so as soon as the measures are lifted the epidemic restarts. Further, strict quarantines carry a serious economic penalty.

            COVID-19 spreads so rapidly that it is capable of quickly generating enough seriously ill patients to overwhelm the intensive care unit (ICU) capacity of most healthcase systems in the world. This is why most countries have opted for strategies that slow the infection rate. It is essential that the ICU capacity is vastly increased to ensure it can cope with the number of people that may require it.
            ''',
            style={'textAlign': 'justify', 'font-family': 'sans-serif'}),


            html.H4('Protecting the high risk',
            style = {'fontSize': '150%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

            


            dcc.Markdown('''            
            One notable feature of COVID-19 is that it puts particular demographics within society at greater risk. The elderly and the immunosuppressed are particularly at risk of serious illness caused by coronavirus.

            The **interactive model** presented here is designed to show the value is protecting the high risk members of society. It is critically important that the high risk do not catch the disease.

            If 60% of the population catch the disease, but all are classified as low risk, then very few people will get seriously ill through the course of the epidemic. However, if a mixture of high and low risk individuals catch the disease, then many of the high risk individuals will develop serious illness as a result.


            ''',
            style={'textAlign': 'justify', 'font-family': 'sans-serif'}
            ),

        ]),


        dbc.Tab(label='How to use', tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_1', children=[
                    



                    # dbc.Container(html.Div(style={'height':'5px'})),
                    html.H3('How to use the interactive model',className='display-4',
                    style = { 'fontSize': '250%', 'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '10px'}),

                    # html.H4('How to use the interactive model',className='display-4'),
                    
                    html.Hr(),
                    
                    dbc.Col([
                    dcc.Markdown('''


                    We present a model parameterised for COVID-19. The interactive element allows you to predict the effect of different **control measures**.

                    We use **control** to mean an action taken to try to reduce the severity of the epidemic. In this case, control measures (e.g. social distancing and quarantine/lockdown) will affect the '**infection rate**' (the rate at which the disease spreads through the population).

                    Stricter measures (e.g. lockdown) have a more dramatic effect on the infection rate than less stringent measures.
                    
                    To start predicting the outcome of different strategies, press the button below!

                    ''',
                    style={'textAlign': 'justify', 'font-family': 'sans-serif'}
                    # style={'fontSize':20}
                    ),
                    
                    dbc.Row([
                    dbc.Button('Start Calculating', href='/inter', size='lg', color='primary',
                    style={'marginTop': '10px', 'textAlign': 'center', 'fontSize': '100%'}
                    ),
                    ],
                    justify='center'),
                    # ],width={'size':3,'offset':1},
                    # ),
                    ],
                    style={'marginTop': '10px'},
                    width = True),


                        #end of tab 1
                    ]),
                    
        dbc.Tab(label='Model Explanation', tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_explan', children=[
                    layout_model
                    ]),

            
    #end of tabs
    ])
],
style={'fontSize': '11'}
)


],
style= {'marginLeft': '50px', 'marginRight': '50px', 'marginBottom': '50px', 'marginTop': '50px',}
),

],
width=12
),

],
justify = 'center',
style= {'width': '90%','backgroundColor': backgroundColor}
),
    
],
justify='center')

]),



#########################################################################################################################################################






############################################################################################################################################################################################################################
Control_text = html.Div(
    html.I('Change the options and press plot.'),
style = {'fontSize': '85%', 'marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center'}),



control_choices_main = html.Div([
    dbc.Row([ # R1662

    dbc.Col([ #C1666
    
    # html.I(className="far fa-calendar-check"),
    # 
    # html.I(className="fas fa-clock"),
    # html.I(className="far fa-hospital"),


    html.H2(className="far fa-hospital",style={'marginTop': '20px'}),

    html.H6([
    ' Control Type ',
    dbc.Button(' ? ',
        color='primary',
        size='sm',
        id='popover-control-target',
        style={'cursor': 'pointer','marginBottom': '2px'}
        ),
    ],
    style={'fontSize': '80%', 'marginBottom': '10px','textAlign': 'center'}),


    dbc.Popover(
    [
    dbc.PopoverHeader('Control'),
    dbc.PopoverBody(dcc.Markdown(
    '''

    The type of **control** determines how much we can reduce the **infection rate** of the disease (how quickly the disease is transmitted between people).
    
    We consider control of **two risk groups**; high risk and low risk. High risk groups are more likely to get seriously ill if they catch the disease.

    *For further explanation, read the [**Background**](/intro)*.

    ''',
    style={'textAlign': 'justify', 'font-family': 'sans-serif'}

    ),),
    ],
    id = "popover-control",
    is_open=False,
    target="popover-control-target",
    placement='right',
    ),

        

    html.Div([
    dcc.Dropdown(
        id = 'preset',
        options=[{'label': presets_dict_dropdown[key],
        'value': key} for key in presets_dict_dropdown],
        value= initial_strat,
        clearable = False,
        searchable=False,
        style={'white-space':'nowrap'}
    ),],
    style={'cursor': 'pointer', 'fontSize': '70%', 'marginTop': '10px', 'marginBottom': '10px','textAlign': 'center'}),
        



    html.H2(style={'marginTop': '20px'},className="fas fa-calendar-check"),

    html.H6([
    ' Months of Control ',
    dbc.Button(' ? ',
    color='primary',
    size='sm',
    id='popover-months-control-target',
    style= {'cursor': 'pointer','marginBottom': '2px'}
    ),
    ],
    style={'fontSize': '80%', 'marginBottom': '10px','textAlign': 'center'}),


    
    html.Div([
    dcc.RangeSlider(
                id='month-slider',
                min=0,
                max=floor(params.max_months_controlling),
                step=1,
                # pushable=0,
                marks={i: str(i) for i in range(0,floor(params.max_months_controlling)+1,3)},
                value=[0,1],
    ),
    ],
    style={'fontSize': '70%'},
    ),


    dbc.Popover(
        [
        dbc.PopoverHeader('Control Timing'),
        dbc.PopoverBody(dcc.Markdown(
        '''

        Use this slider to determine when control **starts** and **finishes**.

        When control is in place the infection rate is reduced by an amount depending on the strategy choice.

        When control is not in place the infection rate returns to the baseline level (100%).
        
        ''',
        style={'textAlign': 'justify', 'font-family': 'sans-serif'}
        ),),
        ],
        id = "popover-months-control",
        is_open=False,
        target="popover-months-control-target",
        placement='right',
    ),


    ],
    width=True,
    style={'textAlign': 'center'},
    ),### C1666

    ],
    justify='center',
    style =  {'margin': '2px'}
    ), # R1662

    ],
    # style={'width': '100px'}
    )
########################################################################################################################

                                                                                                                                                                    


control_choices_other =  html.Div([
    dbc.Row([ # R1871
        dbc.Col([ # C1872

html.H2(style={'marginTop': '20px'},className="fas fa-globe-europe"),
html.H6([
    ' Country'
    ],
    style={'fontSize': '80%', 'marginBottom': '10px','textAlign': 'center'}),
                                        
html.Div([
dcc.Dropdown(
    id = 'model-country-choice',
    options=[{'label': c_name.title() if c_name not in ['us', 'uk'] else c_name.upper(), 'value': num} for num, c_name in enumerate(COUNTRY_LIST_NICK)],
    value= initial_country,
    clearable = False,
    searchable=False,
    style={'white-space':'nowrap'}
),],
style={'cursor': 'pointer', 'fontSize': '70%', 'marginTop': '10px', 'marginBottom': '10px','textAlign': 'center'}),
                                            


html.H2(style={'marginTop': '20px'},className="fas fa-calendar-check"),
html.H6([
    ' Model Start Date'],
    style={'fontSize': '80%', 'textAlign': 'center', 'marginBottom': '10px'}
    ),

dbc.Row([ # R1943
dcc.DatePickerSingle(
id='model-start-date',
min_date_allowed = min_date + datetime.timedelta(days=26),
max_date_allowed = max_date,
initial_visible_month =  max_date,
date = max_date,
display_format='D-MMM-YYYY',
style={'textAlign': 'center', 'fontSize': '70%'}
),
],justify='center'), # R1943



                                        
html.H2(style={'marginTop': '20px'},className="fas fa-syringe"),
html.H6([
' Vaccination starts ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-vaccination-target',
style= {'cursor': 'pointer'}),
],
style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}),


html.Div([
dcc.Slider(
id='vaccine-slider',
min   = 0,
max   = 9,
step  = 3,
marks = {i: 'Never' if i==0 else f'Month {str(i)}' if i==3 else str(i) for i in range(0,10,3)},
value = 3,
),
],
),



dbc.Popover(
[
dbc.PopoverHeader('Vaccination'),
dbc.PopoverBody(dcc.Markdown(
'''

We assume a vaccine will not be available for 6 months.

See how the introduction of a vaccine can drastically reduce the death toll if a sufficiently small proportion of the population have been infected.

''',
style={'textAlign': 'justify', 'font-family': 'sans-serif'}
),),
],
id = "popover-vaccination",
is_open=False,
target="popover-vaccination-target",
placement='left',
),



html.H2(style={'marginTop': '20px'},className="fas fa-user-md"),
html.H6([
' Critical Care Capacity Increase ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-cc-care-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
],
style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}),

dcc.Slider(
id='ICU-slider',
min = 0,
max = 5,
step = 1,
marks={i: str(i)+'x' for i in range(6)},
value=1,
),

dbc.Popover(
[
dbc.PopoverHeader('ICU Capacity'),
dbc.PopoverBody(dcc.Markdown(
'''

Select '0x' to assume that ICU capacity stays constant, or pick an amount by which capacity is bulked up each year (relative to the amount initially).

''',
style={'textAlign': 'justify', 'font-family': 'sans-serif'}
),),
],
id = "popover-cc-care",
is_open=False,
target="popover-cc-care-target",
placement='top',
),




html.H2(style={'marginTop': '20px'},className="fas fa-chart-area"),
html.H6([
' Results Type ',
],
style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}),

# style={'fontSize': '150%', 'marginTop': '30px', 'marginBottom': '30px','textAlign': 'center'}),

html.Div([
dcc.Dropdown(
id = 'dropdown',
options=[{'label': 'Disease Progress Curves','value': 'DPC_dd'},
{'label': 'Bar Charts','value': 'BC_dd'},
{'label': 'Strategy Overview','value': 'SO_dd'},
],
value= 'DPC_dd',
clearable = False,
searchable=False,
style={'white-space':'nowrap'}
),],
style={'cursor': 'pointer', 'textAlign': 'center', 'marginBottom': '30px'}),







],
width=True,
style =  {'textAlign': 'center'}
), # C1872

],
justify='center',
style =  {'margin': '2px'}
), # R1871

])



########################################################################################################################

                                                                                                                                                                    


control_choices_custom =  html.Div([
    dbc.Row([ # 3R1871
        dbc.Col([ # 3C1872





html.H4("Custom Options ",
style={'marginBottom': '10px', 'textAlign': 'center', 'marginTop': '20px','fontSize': '120%'}),




html.H6('Number Of Strategies',
            style={'fontSize': '80%', 'marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center'}),

dbc.Row([ #R2225
dbc.RadioItems(
id = 'number-strats-radio',
options=[
{'label': 'One', 'value': 'one'},
{'label': 'Two', 'value': 'two'},
],
value= 'one',
inline=True,
labelStyle = {'fontSize': '70%'}
),
],justify='center'), #R2225



html.H6(
[
'Infection Rate ',
dbc.Button(' ? ',
color='primary',
size='sm',
id = 'popover-inf-rate-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
],
style={'fontSize': '80%','marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center'}),


dbc.Popover(
[
dbc.PopoverHeader('Infection Rate'),
dbc.PopoverBody(dcc.Markdown(
'''

The *infection rate* relates to how quickly the disease is transmitted. **Control** measures affect transmission/infection rates (typically lowering them).

Adjust by choosing a preset strategy  or making your own custom choice ('**Control Type**').

''',
style={'textAlign': 'justify', 'font-family': 'sans-serif'}
),),
],
id = "popover-inf-rate",
is_open=False,
target="popover-inf-rate-target",
placement='top',
),




html.H2(style={'marginTop': '10px'},className="fas fa-skull-crossbones"),
html.Div(id='strat-lr-infection',style = {'textAlign': 'center','fontSize': '80%'}),

dcc.Slider(
id='low-risk-slider',
min=0,
max=len(params.fact_v)-1,
step = 1,
marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
value=initial_lr,
),

html.H2(style={'marginTop': '20px'},className="fas fa-skull-crossbones"),
html.Div(id='strat-hr-infection',style = {'textAlign': 'center','fontSize': '80%'}),

dcc.Slider(
id='high-risk-slider',
min=0,
max=len(params.fact_v)-1,
step = 1,
marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
value=initial_hr,
),





dbc.Col([ # 3C2258

html.H6([
    ' Strategy Two: Low Risk Infection Rate (%)'],style={'fontSize': '80%','textAlign': 'center'}),

dcc.Slider(
id='low-risk-slider-2',
min=0,
max=len(params.fact_v)-1,
step = 1,
marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
value=5,
),

html.H6([
        ' Strategy Two: High Risk Infection Rate (%)'], style = {'fontSize': '80%','textAlign': 'center'}),

dcc.Slider(
id='high-risk-slider-2',
min=0,
max=len(params.fact_v)-1,
step = 1,
marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
value=8,
),
],width=True, # 3C2258
id='strat-2-id'),










],
width=True,
style={'textAlign': 'center'},
), # 3C1872

],
justify='center',
style =  {'margin': '2px'}
), # 3R1871

])











control_choices_lockdown =  html.Div([
    dbc.Row([ # 2R1871
        dbc.Col([ # 2C1872


html.H4("Lockdown Cycle Options ",style={'marginBottom': '10px', 'textAlign': 'center' ,'marginTop': '20px','fontSize': '120%'}),




html.H6(['Groups allowed out of lockdown ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-groups-allowed-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
],
style={'fontSize': '80%','marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center'}),


dbc.Popover(
[
dbc.PopoverHeader('Lockdown Cycles: Groups'),
dbc.PopoverBody(dcc.Markdown(
'''

In a strategy where lockdowns are 'switched on and off', you may choose to continue to protect the high risk by continuing their lockdown.

Choose whether to keep high risk in lockdown or allow all groups to leave lockdown (this is the default setting).

''',
style={'textAlign': 'justify', 'font-family': 'sans-serif'}
),),
],
id = "popover-groups-allowed",
is_open=False,
target="popover-groups-allowed-target",
placement='top',
),


dbc.Row([ # 2R2088
dbc.RadioItems(
id = 'hr-ld',
options=[
{'label': 'Low Risk Only', 'value': 0},
{'label': 'Both Groups', 'value': 1},
],
value= 1,
inline=True,
labelStyle = {'fontSize': '70%'}
),
],justify='center'),  # 2R2088



                                                        
html.H2(style={'marginTop': '20px'},className="fas fa-clock"),
html.H6([
    ' Cycle: Time On ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-cycles-on-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
],
style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}),

dcc.Slider(
id='cycle-on',
min = 1,
max = 8,
step = 1,
marks={i: 'Weeks: ' + str(i) if i==1 else str(i) for i in range(1,9)},
value=2,
),

dbc.Popover(
[
dbc.PopoverHeader('Lockdown Cycles: Time On'),
dbc.PopoverBody(dcc.Markdown(
'''

Use this slider to adjust the amount of time that the country is in lockdown under the strategy 'Lockdown cycles'.

This allows us to consider a system where the country is in lockdown for 3 weeks say, followed by a week of normality.

''',
style={'textAlign': 'justify', 'font-family': 'sans-serif'}
),),
],
id = "popover-cycles-on",
is_open=False,
target="popover-cycles-on-target",
placement='top',
),


html.H2(style={'marginTop': '20px'},className="fas fa-clock"),
html.H6([
    ' Cycle: Time Off ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-cycles-off-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
],
style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}),

dcc.Slider(
id='cycle-off',
min = 1,
max = 8,
step = 1,
marks={i: 'Weeks: ' + str(i) if i==1 else str(i) for i in range(1,9)},
value=4,
),

dbc.Popover(
[
dbc.PopoverHeader('Lockdown Cycles: Time Off'),
dbc.PopoverBody(dcc.Markdown(
'''

Use this slider to adjust the amount of time that the country is out of lockdown under the strategy 'Lockdown cycles'.

This allows us to consider a system where the country is in lockdown for 3 weeks say, followed by a week of normality.

''',
style={'textAlign': 'justify', 'font-family': 'sans-serif'}
),),
],
id = "popover-cycles-off",
is_open=False,
target="popover-cycles-off-target",
placement='top',
),



],
width=True,
style={'textAlign': 'center'},
), # 2C1872

],
justify='center',
style =  {'margin': '2px'}
), # 2R1871

])


#########################################################################################################################################################

dpc_content = html.Div([

        dbc.Card(
            html.Div([
            dcc.Graph(figure=dummy_figure,id='line-plot-2'), # ,style={'height': '100px', 'width': '100%'}),
            ],
            style={'margin': '10px'}
            ),
        color='light'
        ),


    ])





barChart_content =  dbc.Col([

        dbc.Col([

            
            html.Div(
                        [
                                dbc.Row([
                                    html.H4(style={'fontSize': '150%', 'textAlign': 'center'}, children = [
                                    
                                    html.Div(['Total Deaths (Percentage) ',
                                    
                                    ],
                                    style= {'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px'}),

                                    ]),
                                ]
                                ,justify='center'),
                                ],
                                id='bar-plot-1-title',style={ 'display':'block', 'textAlign': 'left'}),

                                dcc.Graph(id='bar-plot-1',style=bar_non_crit_style),
                        
                                dcc.Markdown('''

                                    This plot shows a prediction for the number of deaths caused by the epidemic.
                                    
                                    Most outcomes result in a much higher proportion of high risk deaths, so it is critical that any strategy should protect the high risk.

                                    Quarantine/lockdown strategies are very effective at slowing the death rate, but only work whilst they're in place (or until a vaccine is produced).

                                    ''',style={'fontSize': '100%', 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '60px'}),
                        

                        
                html.Hr(),




                                            html.Div(
                                                [dbc.Row([##
                                                    html.H4(style={'fontSize': '150%', 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px'}, children = [

                                                        html.Div(['Peak ICU Bed Capacity Requirement ',
                                                        ],style= {'textAlign': 'center'}),


                                                    ]),
                                                ],
                                                justify='center'),##
                                                ],
                                                id='bar-plot-3-title', style={'display':'block'}),

                                                dcc.Graph(id='bar-plot-3',style=bar_non_crit_style),
                                            
                                                dcc.Markdown('''

                                                    This plot shows the maximum ICU capacity needed.
                                                    
                                                    Better strategies reduce the load on the healthcare system by reducing the numbers requiring Intensive Care at any one time.

                                                    ''',style={'fontSize': '100%' , 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '60px' }),
                                

                            html.Hr(),


                                            html.Div(
                                                    [dbc.Row([##
                                                        html.H4(style={'fontSize': '150%', 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px'}, children = [

                                                            html.Div(['Time ICU (Current) Bed Capacity Exceeded ',

                                                            ],style= {'textAlign': 'center'}),

                                                        ]),
                                                        
                                                    ],
                                                    justify='center'),##
                                                    ],
                                            id='bar-plot-4-title',style={'display':'block'}),

                                            dcc.Graph(id='bar-plot-4',style=bar_non_crit_style),
                            
                                            dcc.Markdown('''

                                                This plot shows the length of time for which ICU capacity is exceeded, over the calculated number of years.

                                                Better strategies will exceed the ICU capacity for shorter lengths of time.

                                                ''',style={'fontSize': '100%' , 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '60px' }),
                    html.Hr(),



                        


                                html.Div(
                                        [dbc.Row([##
                                            html.H4(style={'fontSize': '150%', 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px'}, children = [

                                                html.Div(['Herd Immunity Threshold ',
                                                ],
                                                style= {'textAlign': 'center'}),
                                            

                                            ]),

                                        ],
                                    justify='center'),##
                                        ],
                                id='bar-plot-2-title',style={ 'display':'block'}),

                                dcc.Graph(id='bar-plot-2',style=bar_non_crit_style),
                                

                                dcc.Markdown('''

                                    This plot shows how close to the 60% population immunity the strategy gets.
                                    
                                    Strategies with a lower *infection rate* can delay the course of the epidemic but once the strategies are lifted there is no protection through herd immunity. Strategies with a high infection rate can risk overwhelming healthcare capacity.

                                    The optimal outcome is obtained by making sure the 60% that do get the infection are from the low risk group.

                                    ''',style={'fontSize': '100%' , 'textAlign': 'center' , 'marginTop': '30px', 'marginBottom': '60px'}),
                    

                    html.Hr(),


                            
                            html.Div(
                                    [dbc.Row([##

                                            html.H4(style={'fontSize': '150%', 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px'}, children = [

                                            html.Div(['Time Until Herd Immunity Threshold Reached ',
                                            ],style= {'textAlign': 'center'}),

                                            ]),
                                    ],
                                    justify='center'
                                    ),##
                                    ],
                            id='bar-plot-5-title',style={ 'display':'block'}),

                            dcc.Graph(id='bar-plot-5',style=bar_non_crit_style),
                            
                            dcc.Markdown(
                                """
                                This plot shows the length of time until the safe threshold for population immunity is 95% reached.
                                
                                We allow within 5% of the safe threshold, since some strategies get very close to full safety very quickly and then asymptotically approach it (but in practical terms this means the population is safe).

                                The longer it takes to reach this safety threshold, the longer the population must continue control measures because it is at risk of a further epidemic.

                                """
                                ,style={'fontSize': '100%' , 'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '30px' }),

                    
                    ],
                    align='center',
                    width=12,
                    ),



                
                



],width=True),







controls = dbc.Card([
        
dbc.Row([
    dbc.Col([

        html.Div(Control_text),

        dbc.Row([
            dbc.Spinner(html.Div(id="loading-sol-1"),color='primary'),
            
            dbc.Button([html.I(className="fas fa-chart-area"),' Plot'],
                            color='primary',
                            className='mb-3',
                            id="plot-button",
                            size='lg',
                            style = {'cursor': 'pointer'}),
            
            dbc.Spinner(html.Div(id="loading-line-output-1"),color='primary'),
        
        ],
        justify='center'),
                                                                
        dbc.Tabs(
            active_tab='tab_main',
            children = [
                
            dbc.Tab(label='Main controls',
                        tab_style = { 'textAlign': 'center', 'cursor': 'pointer'},
                        label_style={"color": tab_label_color, 'fontSize':'120%'}, 
                        tab_id='tab_main',
                        children=control_choices_main
                        ),
            dbc.Tab(label='Custom options',
                        tab_style = {'display': 'none'},
                        label_style={"color": tab_label_color, 'fontSize':'120%'}, 
                        id='tab-custom',
                        children=[control_choices_custom]
                        ),
            dbc.Tab(label='Lockdown cycle options',
                        tab_style = {'display': 'none'},
                        label_style={"color": tab_label_color, 'fontSize':'120%'}, 
                        id='tab-ld-cycle',
                        children=[control_choices_lockdown],
                        ),
            dbc.Tab(label='Other',
                        tab_style = { 'textAlign': 'center', 'cursor': 'pointer'},
                        label_style={"color": tab_label_color, 'fontSize':'120%'}, 
                        # tab_id='tab_main',
                        children=control_choices_other
                        ),
            ]) # end of tabs

        ],
        width=12,
        style={'margin': '5px'}
        )
],
justify='center'
),
                                                                
],
color='light',
)



textCard = dbc.Card([
dbc.Row([
    dbc.Col([
    html.H4(
        "Interactive interface; explore potential effects of different control strategies.",
        style={'marginBottom': '20px', 'textAlign': 'center' ,'marginTop': '20px','fontSize': '120%'}),
        

    html.Div(
        'The model predicts how the epidemic will progress, depending on the disease status of members of the population within each country.',
        style={'fontSize': '85%', 'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),
            
    html.H4(
        "Use the sliders and dropdowns below to change the control strategy.",
        style={'marginBottom': '20px', 'textAlign': 'center' ,'marginTop': '20px','fontSize': '120%'}),
        
    html.Div(
        'Compare the effect of different strategies on the predicted number of infections, hospitalisations, and deaths.',
        style={'fontSize': '85%', 'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

    html.Div(id='worked-div'),

    ],
    width=True,
    style={'margin': '10px'}
    )

],
justify='center'
),

],
color='light'
)




layout_inter = html.Div([
                    # store results
                    # dcc.Store(id='sol-calculated-cache'),
                    dcc.Store(id='sol-calculated-do-nothing-cache'),
                    dcc.Store(id='prev-deaths-cache'),
                    dcc.Store(id='store-initial-conds-cache'),
                    dcc.Store(id='store-get-data-worked-cache'),
                    # dcc.Store(id='store-upper-lower-cache'),

                    # replace with Cache soon!

                    # dcc.Store(id='sol-calculated'),
                    # dcc.Store(id='sol-calculated-do-nothing'),
                    # dcc.Store(id='prev-deaths'),
                    # dcc.Store(id='store-initial-conds'),
                    # dcc.Store(id='store-get-data-worked'),
                    # dcc.Store(id='store-upper-lower'),
                    


                    # dbc.Row([ # R2507
                    #     dbc.Col([
                    #             html.Div([
                    #             html.Span('Disclaimer: ',style={'color': '#C37C10'}), # orange
                    #             'This work is for educational purposes only and not for accurate prediction of the pandemic.'],
                    #             style = {'fontSize': '80%', 'color': '#446E9B', 'fontWeight': 'bold'}
                    #             ),
                    #         ],width=True,
                    #         style={'textAlign': 'center'}
                    #         ),
                    # ],
                    # align="center",
                    # # style={'backgroundColor': disclaimerColor}
                    # ), # R2507









                                    


                                        dbc.Row([ # R2599
                                            dbc.Col([
                                                    html.Div(textCard,
                                                    style={'marginBottom': '10px'}
                                                    ),

                                                    dbc.Row([
                                                        dbc.Col([
                                                        html.Div(controls,
                                                        style={'marginTop': '10px'}
                                                        )
                                                        ],
                                                        width=12,
                                                        lg=3
                                                        ),

                                                        dbc.Col([
                                                            html.Div(dpc_content,id='DPC-content',style={'marginTop': '10px'}),  
                                                            html.Div(barChart_content,id='bc-content',style={'display': 'none', 'marginTop': '10px'}),
                                                            html.Div(id='strategy-outcome-content',style={'display': 'none', 'marginTop': '10px'}),
                                                        ],
                                                        width=12,
                                                        lg=9
                                                        )
                                                    ],
                                                    )
                                            ],
                                            width=True,
                                            ),


                                        ],
                                        justify='center',
                                        style={'margin': '20px'}
                                        ),  # R2599


                                        

#########################################################################################################################################################
                                                                                                                                                             


    ],
    style={'fontSize': '11', 'marginBottom': '40px'},
    )












##############################################################################################

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













# app.layout
        
page_layout = html.Div([
    
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Background", href="/intro")),
                dbc.NavItem(dbc.NavLink("Interactive model", href="/inter")),
                dbc.NavItem(dbc.NavLink("Interpreting tests", href="/tests")),
                dbc.NavItem(dbc.NavLink("Real-time data", href="/data")),

            ],
            brand="Modelling COVID-19 Control",
            brand_href="/",
            brand_style = {'fontSize': '120%'},
            color="primary",
            # sticky = 'top',
            expand = 'lg',
            # className = 'navbar-collapse',
            style= {'fontSize': '120%'},
            dark=True,
        ),





        ##

        # # page content
        dcc.Store(id='saved-url',data='/'),
        dcc.Location(id='page-url', refresh=False),

        dbc.Spinner(html.Div(id="loading-page"),color='primary',size='lg'),
        html.Div(id='page-content',children=layout_enter),


        html.Hr(),

        html.Footer([
                    "Contact us at: ",
                     html.A('covid.at.plants@gmail.com',href='')
                     ],
        style={'textAlign': 'center', 'fontSize': '90%', 'marginBottom': '20px', 'marginTop': '20px'}),

        html.Footer('This page is intended for illustrative/educational purposes only, and not for accurate prediction of the pandemic.',
                    style={'textAlign': 'center', 'fontSize': '90%', 'color': '#446E9B', 'fontWeight': 'bold'}),
        html.Footer([
                    "Authors: ",
                     html.A('Nick P. Taylor', href='https://twitter.com/TaylorNickP'),", ",
                     html.A('Daniel Muthukrishna', href='https://twitter.com/DanMuthukrishna'),
                     " and Dr Cerian Webb. ",
                     ],
        style={'textAlign': 'center', 'fontSize': '90%'}),
        html.Footer([
                     html.A('Source code', href='https://github.com/nt409/covid-19'), ". ",
                     "Data is taken from ",
                     html.A("Worldometer", href='https://www.worldometers.info/coronavirus/'), " if available or otherwise ",
                     html.A("Johns Hopkins University (JHU) CSSE", href="https://github.com/ExpDev07/coronavirus-tracker-api"), "."
                    ],
                    style={'textAlign': 'center', 'fontSize': '90%'}),

        

        ],
        style={'fontSize': '11', 'font-family': 'sans-serif'}
        # 
        )
##
########################################################################################################################





app.layout = page_layout

app.title = 'Modelling COVID-19 Control'

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-163339118-1"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'UA-163339118-1');
        </script>

        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""



########################################################################################################################
# callbacks









@app.callback(Output('saved-url', 'data'),            
            [Input('page-url', 'pathname')],
            [State('saved-url', 'data')],
            )
def change_pathname(pathname,saved_pathname):

    # print('change pathname', pathname, saved_pathname)

    if pathname==saved_pathname:
        raise PreventUpdate
    else:
        return pathname




@app.callback([Output('page-content', 'children'),
            Output('loading-page','children')],
            [Input('saved-url', 'data')])
def display_page(pathname):

    if pathname == '/data':
        return [layout_dan, None]
    elif pathname == '/intro':
        return [layout_intro, None]
    elif pathname == '/inter':
        return [layout_inter, None]
    elif pathname == '/tests':
        return [layout_tests, None]
    else:
        return [layout_enter, None]

########################################################################################################################
# collapse
def toggle(n, is_open):
    if n:
        return not is_open
    return is_open



########################################################################################################################
# popovers


for p in [ "control", "months-control", "vaccination",  "cc-care" , "inf-rate", "inf-tab", "cont-tab", "example","red-deaths","ICU","herd", 'cycles-off', 'cycles-on', 'groups-allowed']: # "res-type" , "pick-strat",
    app.callback(
        Output(f"popover-{p}", "is_open"),
        [Input(f"popover-{p}-target", "n_clicks")
        ],
        [State(f"popover-{p}", "is_open")],
    )(toggle)




##############################################################################################################################



@app.callback(
    [
    
    Output('strat-2-id', 'style'),


    Output('strat-hr-infection','children'),
    Output('strat-lr-infection','children'),

    Output('tab-custom', 'tab_style'),
    Output('tab-ld-cycle', 'tab_style'),

    ],
    [
    Input('number-strats-radio', 'value'),
    Input('preset', 'value'),
    ])
def invisible_or_not(num,preset):




    if num=='two':
        strat_H = [html.H6([
            ' Strategy One: High Risk Infection Rate (%)'],style={'fontSize': '100%'}),]
        strat_L = [html.H6([
            ' Strategy One: Low Risk Infection Rate (%)'],style={'fontSize': '100%'}),]
        says_strat_2 = None

    else:

        strat_H = [html.H6([
            ' High Risk Infection Rate (%)'],style={'fontSize': '100%'}),]

        strat_L = [html.H6([
            ' Low Risk Infection Rate (%)'],style={'fontSize': '100%'}),]

        says_strat_2 = {'display': 'none'}

    if preset!='C':
        says_strat_2 = {'display': 'none'}
    
    custom_tab_style = {'display': 'none'}
    ld_cycle_tab_style = {'display': 'none'}
    if preset=='C':
        custom_tab_style = {'textAlign': 'center', 'cursor': 'pointer'}
    
    if preset=='LC':
        ld_cycle_tab_style = {'textAlign': 'center', 'cursor': 'pointer'}

    
    return [says_strat_2, strat_H, strat_L, custom_tab_style, ld_cycle_tab_style]

########################################################################################################################




@app.callback(
            [Output('low-risk-slider', 'value'),
            Output('high-risk-slider', 'value'),
            Output('low-risk-slider', 'disabled'), 
            Output('high-risk-slider', 'disabled'),
            Output('number-strats-radio','options'),
            Output('number-strats-radio','value'),
            Output('cycle-off', 'disabled'),
            Output('cycle-on', 'disabled'),
            Output('hr-ld', 'options'),
            ],
            [
            Input('preset', 'value'),
            ],
            [State('number-strats-radio','value')])
def preset_sliders(preset,number_strs):

    # print('preset sliders')

    lockdown_cycles_dis = True
    options_lockdown_cycles = [
                                {'label': 'Low Risk Only', 'value': 0, 'disabled': True},
                                {'label': 'Both Groups', 'value': 1, 'disabled': True},
                            ]
    if preset=='LC':
        lockdown_cycles_dis = False
        options_lockdown_cycles = [
                                {'label': 'Low Risk Only', 'value': 0},
                                {'label': 'Both Groups', 'value': 1},
                            ]

    if preset == 'C':
        dis = False
        options=[
                {'label': 'One', 'value': 'one'},
                {'label': 'Two', 'value': 'two'},
            ]
        number_strats = number_strs
    else:
        dis = True
        options=[
                {'label': 'One', 'value': 'one','disabled': True},
                {'label': 'Two', 'value': 'two','disabled': True},
            ]
        number_strats = 'one'
    if preset in preset_dict_low:
        return [preset_dict_low[preset], preset_dict_high[preset], dis, dis, options, number_strats, lockdown_cycles_dis, lockdown_cycles_dis, options_lockdown_cycles]
    else:
        return [preset_dict_low['N'], preset_dict_high['N'], dis, dis, options, number_strats, lockdown_cycles_dis, lockdown_cycles_dis, options_lockdown_cycles]
    



def find_sol(preset,month,lr_in,hr_in,lr2_in,hr2_in,num_strat,vaccine,ICU_grow,date,country_num,t_off,t_on,hr_ld,init_stored,worked):

    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    if vaccine==0:
        vaccine = None

    if init_stored is None or date!=init_stored[5] or country!=init_stored[6]:
        I0, R0, H0, C0, D0, worked, _ = begin_date(date,country)
        initial_conds = [I0, R0, H0, C0, D0, date, country]
    else:
        initial_conds = init_stored
        I0 = init_stored[0]
        R0 = init_stored[1]
        H0 = init_stored[2]
        C0 = init_stored[3]
        D0 = init_stored[4]
        date=init_stored[5]
        country=init_stored[6]

    if worked is None:
        worked = False
    
    if not worked:
        worked_div = html.Div('Getting data for this country/date combination failed... try another. UK 8th April data used instead.',
                                style={'textAlign': 'center', 'color': 'red', 'fontWeight': 'bold'})
    else:
        worked_div = None
    
    if preset=='C':
        lr = params.fact_v[int(lr_in)]
        hr = params.fact_v[int(hr_in)]
    else:
        lr = params.fact_v[preset_dict_low[preset]]
        hr = params.fact_v[preset_dict_high[preset]]

    if preset=='N':
        month=[0,0]
    
    let_HR_out = True
    if preset=='LC':
        time_start = month[0]
        time_end   = month[1]
        time_on = t_on*7/month_len
        time_off = t_off*7/month_len
        if hr_ld==0:
            let_HR_out = False
        month = [time_start,time_on]
        mm = month[1]
        while mm + time_off+time_on < time_end:
            month.append(mm+time_off)
            month.append(mm+time_off+time_on)
            mm = mm + time_off + time_on
    
    

    months_controlled = [month_len*i for i in month]


    if month[0]==month[1]:
        months_controlled= None
    
    sols = []
    sols.append(run_model(beta_L_factor=lr,beta_H_factor=hr,
                            t_control=months_controlled,
                            vaccine_time=vaccine,
                            I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,
                            ICU_grow=ICU_grow,let_HR_out=let_HR_out))
    if num_strat=='two':
        lr2 = params.fact_v[int(lr2_in)]
        hr2 = params.fact_v[int(hr2_in)]
        sols.append(run_model(beta_L_factor=lr2,beta_H_factor=hr2,
                                t_control=months_controlled,
                                vaccine_time=vaccine,
                                I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,
                                ICU_grow=ICU_grow,let_HR_out=let_HR_out))
    
    sols_upper_lower = []
    jj = 0
    control_factor = 0.75
    for kk in [0.25,2]: # baseline assumption: overestimate by 4x
        if jj ==0:
            lr_new = control_factor*lr # makes control less effective
            hr_new = control_factor*hr # makes control less effective
        else:
            lr_new = lr + (1 - control_factor)*(1 - lr) # makes control more effective
            hr_new = hr + (1 - control_factor)*(1 - hr) # makes control more effective

        jj = jj+1


        H0_new = H0*kk
        C0_new = C0*kk
        
        I0_new = I0 + H0 + C0 - H0_new - C0_new

        R0_new = R0
        D0_new = D0

        upp_low = run_model(beta_L_factor=lr_new,beta_H_factor=hr_new,
                                t_control=months_controlled,
                                vaccine_time=vaccine,
                                I0=I0_new,R0=R0_new,H0=H0_new,C0=C0_new,D0=D0_new,
                                ICU_grow=ICU_grow,let_HR_out=let_HR_out)

        sols_upper_lower.append(upp_low)

    return sols, initial_conds, worked, worked_div, sols_upper_lower



@app.callback(
    [Output('sol-calculated-do-nothing-cache', 'data'),
    Output('prev-deaths-cache', 'data')],
    [
    Input('ICU-slider', 'value'),
    Input('model-start-date', 'date'),
    Input('model-country-choice', 'value'),
    ])
def find_sol_do_noth(ICU_grow,date,country_num):
    # @cache.memoize()  # in seconds
    
    # print(dash.callback_context.triggered,'do nothing')

    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    I0, R0, H0, C0, D0, _, prev_deaths = begin_date(date,country)

    sol_do_nothing = run_model(I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,
                                    beta_L_factor=1,beta_H_factor=1,
                                    t_control=None,
                                    ICU_grow=ICU_grow)
    
    return sol_do_nothing, prev_deaths











###################################################################################################################################################################################


@app.callback([
                Output('loading-sol-1','children'),
                Output('store-initial-conds-cache', 'data'),
                Output('store-get-data-worked-cache', 'data'),
                Output('worked-div', 'children'),
                # find sol

                Output('DPC-content', 'style'),
                Output('bc-content', 'style'),
                Output('strategy-outcome-content', 'style'),
                

                Output('strategy-outcome-content', 'children'),


                Output('bar-plot-1', 'figure'),
                Output('bar-plot-2', 'figure'),
                Output('bar-plot-3', 'figure'),
                Output('bar-plot-4', 'figure'),
                Output('bar-plot-5', 'figure'),


                Output('line-plot-2', 'figure'),





                
                Output('loading-line-output-1','children'),
                
                

               ],
               [
                Input('plot-button', 'n_clicks'),
               ],
               [
                State('saved-url', 'data'),

                State('sol-calculated-do-nothing-cache', 'data'),

                State('dropdown', 'value'),
                State('model-country-choice', 'value'),
                State('model-start-date','date'),
                State('prev-deaths-cache','data'),

                State('cycle-off', 'value'),
                State('cycle-on', 'value'),
                
                State('preset', 'value'),
                State('month-slider', 'value'),
                
                State('low-risk-slider', 'value'),
                State('high-risk-slider', 'value'),
                State('low-risk-slider-2', 'value'),
                State('high-risk-slider-2', 'value'),

                State('hr-ld', 'value'),

                State('store-initial-conds-cache','data'),
                State('store-get-data-worked-cache','data'),

                State('number-strats-radio', 'value'),
                State('vaccine-slider', 'value'),
                State('ICU-slider','value'),

                ])
def render_interactive_content(plot_button,
                                pathname,

                                sol_do_nothing,

                                results_type,
                                country_num,
                                date,
                                prev_deaths,

                                t_off,
                                t_on,
                                
                                preset,
                                month,

                                lr_in,
                                hr_in,
                                lr2_in,
                                hr2_in,

                                hr_ld,

                                init_stored,
                                worked,

                                num_strat,
                                vaccine_time,
                                ICU_grow):

    # print(f'render {pathname}')

    sols, initial_conds, worked, worked_div, upper_lower_sol = find_sol(preset,
                    month,lr_in,hr_in,lr2_in,hr2_in,
                    num_strat,vaccine_time,ICU_grow,date,country_num,
                    t_off,t_on,hr_ld,init_stored,worked)




    if pathname in ['/data','/intro','/home','/tests','/']:
        raise PreventUpdate




    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    

    
    DPC_style = {'display' : 'none'}
    BC_style  = {'display': 'none'}
    SO_style  = {'display' : 'none'}

    if results_type == 'BC_dd':
        BC_style = {'display': 'block', 'marginTop': '10px'}

    elif results_type == 'SO_dd':
        SO_style = {'display': 'block', 'marginTop': '10px'}

    else: # 'DPC
        DPC_style = {'display': 'block', 'marginTop': '10px'}
        results_type = 'DPC_dd' # in case wasn't


########################################################################################################################


    strategy_outcome_text = ['']


    if results_type!='BC_dd': # tab2!='interactive' # pathname!='inter' or 
        bar1 = dummy_figure
        bar2 = dummy_figure
        bar3 = dummy_figure
        bar4 = dummy_figure
        bar5 = dummy_figure

    if results_type!='DPC_dd': # tab2 'interactive' # pathname!='inter' or 
        fig_out = dummy_figure






    if preset!='C':
        num_strat = 'one'
        

    sols.append(sol_do_nothing)

    # bar plot data
    time_reached_data = []
    time_exceeded_data = []

    crit_cap_data_L_1yr = []
    crit_cap_data_H_1yr = []
    crit_cap_quoted_1yr = []
    ICU_data_1yr = []
    herd_list_1yr = []

    crit_cap_data_L_2yr = []
    crit_cap_data_H_2yr = []
    crit_cap_quoted_2yr = []
    ICU_data_2yr = []
    herd_list_2yr = []

    crit_cap_data_L_3yr = []
    crit_cap_data_H_3yr = []
    crit_cap_quoted_3yr = []
    ICU_data_3yr = []
    herd_list_3yr = []
    for ii in range(len(sols)):
        if sols[ii] is not None and ii<len(sols)-1:
            sol = sols[ii]

########################################################################################################################
    if results_type!='DPC_dd': # tab != DPC

        #loop start
        for ii in range(len(sols)):
            # 
            if sols[ii] is not None:
                sol = sols[ii]
                
                yy = np.asarray(sol['y'])
                tt = np.asarray(sol['t'])

                
                num_t_points = yy.shape[1]

                metric_val_L_3yr, metric_val_H_3yr, ICU_val_3yr, herd_fraction_out, time_exc, time_reached = extract_info(yy,tt,num_t_points,ICU_grow)
                
                crit_cap_data_L_3yr.append(metric_val_L_3yr) #
                crit_cap_data_H_3yr.append(metric_val_H_3yr) #
                ICU_data_3yr.append(ICU_val_3yr)
                herd_list_3yr.append(herd_fraction_out) ##
                time_exceeded_data.append(time_exc) ##
                time_reached_data.append(time_reached) ##


                num_t_2yr = ceil(2*num_t_points/3)
                metric_val_L_2yr, metric_val_H_2yr, ICU_val_2yr, herd_fraction_out = extract_info(yy,tt,num_t_2yr,ICU_grow)[:4]

                crit_cap_data_L_2yr.append(metric_val_L_2yr) #
                crit_cap_data_H_2yr.append(metric_val_H_2yr) #
                ICU_data_2yr.append(ICU_val_2yr)
                herd_list_2yr.append(herd_fraction_out) ##


                num_t_1yr = ceil(num_t_points/3)
                metric_val_L_1yr, metric_val_H_1yr, ICU_val_1yr, herd_fraction_out = extract_info(yy,tt,num_t_1yr,ICU_grow)[:4]

                crit_cap_data_L_1yr.append(metric_val_L_1yr) #
                crit_cap_data_H_1yr.append(metric_val_H_1yr) #
                ICU_data_1yr.append(ICU_val_1yr)
                herd_list_1yr.append(herd_fraction_out) ##


        
        # loop end

        for jj in range(len(crit_cap_data_H_3yr)):
            crit_cap_quoted_1yr.append( (1 - (crit_cap_data_L_1yr[jj] + crit_cap_data_H_1yr[jj])/(crit_cap_data_L_1yr[-1] + crit_cap_data_H_1yr[-1]) ))

            crit_cap_quoted_2yr.append( (1 - (crit_cap_data_L_2yr[jj] + crit_cap_data_H_2yr[jj])/(crit_cap_data_L_2yr[-1] + crit_cap_data_H_2yr[-1]) ))

            crit_cap_quoted_3yr.append( (1 - (crit_cap_data_L_3yr[jj] + crit_cap_data_H_3yr[jj])/(crit_cap_data_L_3yr[-1] + crit_cap_data_H_3yr[-1]) ))

        ########################################################################################################################
        # SO results
        if  results_type=='SO_dd':

            strategy_outcome_text = html.Div([

                outcome_fn(month,sols[0]['beta_L'],sols[0]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[0],herd_list_2yr[0],ICU_data_2yr[0],preset,number_strategies = num_strat,which_strat=1), # hosp,
                html.Hr(),
                outcome_fn(month,sols[1]['beta_L'],sols[1]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[1],herd_list_2yr[1],ICU_data_2yr[1],preset,number_strategies = num_strat,which_strat=2), # hosp,
                ],
                style = {'fontSize': '20px'}
                )

        
        ########################################################################################################################
        # BC results

        if results_type=='BC_dd':

            crit_cap_bar_1yr = [crit_cap_data_L_1yr[i] + crit_cap_data_H_1yr[i] for i in range(len(crit_cap_data_H_1yr))]
            crit_cap_bar_3yr = [crit_cap_data_L_3yr[i] + crit_cap_data_H_3yr[i] for i in range(len(crit_cap_data_H_3yr))]


            bar1 = Bar_chart_generator(crit_cap_bar_1yr      ,text_addition='%'         , y_title='Population'                    , hover_form = '%{x}, %{y:.3%}'                         ,color = 'LightSkyBlue' ,data_group=crit_cap_bar_3yr, yax_tick_form='.1%')
            bar2 = Bar_chart_generator(herd_list_1yr         ,text_addition='%'         , y_title='Percentage of Safe Threshold'  , hover_form = '%{x}, %{y:.1%}<extra></extra>'          ,color = 'LightSkyBlue' ,data_group=herd_list_3yr,yax_tick_form='.1%',maxi=False,yax_font_size_multiplier=0.8)
            bar3 = Bar_chart_generator(ICU_data_1yr          ,text_addition='x current' , y_title='Multiple of Capacity'  , hover_form = '%{x}, %{y:.1f}x Current<extra></extra>' ,color = 'LightSkyBlue'     ,data_group=ICU_data_3yr  )
            bar4 = Bar_chart_generator(time_exceeded_data    ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'LightSkyBlue'   )
            bar5 = Bar_chart_generator(time_reached_data     ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'LightSkyBlue')



########################################################################################################################
    # DPC results

    if results_type=='DPC_dd':

        date = datetime.datetime.strptime(date.split('T')[0], '%Y-%m-%d')

        startdate = copy.copy(date)




        if vaccine_time==9:
            vaccine_time = None

        if False: # plot_with_do_nothing==[1] and num_strat=='one' and preset!='N':
            sols_to_plot = sols
            comp_dn = True
        else:
            sols_to_plot = sols[:-1]
            comp_dn = False

        if True: # plot_ICU_cap!=[1]:
            ICU_plot = False
        else:
            ICU_plot = True


        
        month_cycle = None
        if preset=='LC':
            time_start = month[0]
            time_end   = month[1]
            time_on = t_on*7/month_len
            time_off = t_off*7/month_len

            month_cycle = [time_start,time_start+time_on]
            mm = month_cycle[1]
            while mm + time_off+time_on < time_end:
                month_cycle.append(mm+time_off)
                month_cycle.append(mm+time_off+time_on)
                mm = mm + time_off + time_on

        fig_out = MultiFigureGenerator(upper_lower_sol,sols_to_plot,month,num_strat,vaccine_time=vaccine_time,ICU_grow=ICU_grow, ICU_to_plot=ICU_plot ,comp_dn=comp_dn, country = country,month_cycle=month_cycle,preset=preset,startdate=startdate, previous_deaths=prev_deaths)

    

        
########################################################################################################################

    return [
    None,
    initial_conds,
    worked,
    worked_div,



    DPC_style,
    BC_style,
    SO_style,

    strategy_outcome_text,

    bar1,
    bar2,
    bar3,
    bar4,
    bar5,

    fig_out,

    None
    ]

########################################################################################################################








# dan's callbacks


@app.callback([Output('align-cases-check', 'options'),
               Output('align-cases-input', 'value'),
               Output('display_percentage_text_cases', 'style'),
               Output('align-deaths-check', 'options'),
               Output('align-deaths-input', 'value'),
               Output('display_percentage_text_deaths', 'style'),
               Output('align-active-cases-check', 'options'),
               Output('align-active-cases-input', 'value'),
               Output('display_percentage_text_active', 'style'),
               Output('align-daily-cases-check', 'options'),
               Output('align-daily-cases-input', 'value'),
               Output('display_percentage_text_daily_cases', 'style'),
               Output('align-daily-deaths-check', 'options'),
               Output('align-daily-deaths-input', 'value'),
               Output('display_percentage_text_daily_deaths', 'style')],
              [Input('normalise-check', 'value')])
def update_align_options(normalise_by_pop):

    # print('dan 1')

    if normalise_by_pop:
        options_cases = [{'label': "Align countries by the date when the percentage of confirmed cases was ",
                    'value': 'align'}]
        options_deaths = [{'label': "Align countries by the date when the number of confirmed deaths was ",
                    'value': 'align'}]
        hidden_text = {'display': 'inline-block'}
        return [options_cases, 0.0015, hidden_text,
                options_deaths, 0.000034, hidden_text,
                options_cases, 0.0015, hidden_text,
                options_cases, 0.0015, hidden_text,
                options_deaths, 0.000034, hidden_text]
    else:
        options_cases = [{'label': "Align countries by the date when the number of confirmed cases was ",
                    'value': 'align'}]
        options_deaths = [{'label': "Align countries by the date when the number of confirmed deaths was ",
                    'value': 'align'}]
        hidden_text = {'display': 'none'}
        return[options_cases, 1000, hidden_text,
               options_deaths, 20, hidden_text,
               options_cases, 1000, hidden_text,
               options_cases, 1000, hidden_text,
               options_deaths, 20, hidden_text]


@app.callback([Output('infections-plot', 'figure'),
               Output('deaths-plot', 'figure'),
               Output('active-plot', 'figure'),
               Output('daily-cases-plot', 'figure'),
               Output('daily-deaths-plot', 'figure'),
               Output('new-vs-total-cases', 'figure'),
               Output('new-vs-total-deaths', 'figure'),
               Output('hidden-stored-data', 'children'),
               Output("loading-icon", "children")],
              [
               Input('button-plot', 'n_clicks'),
               Input('start-date', 'date'),
               Input('end-date', 'date'),
               Input('show-exponential-check', 'value'),
               Input('normalise-check', 'value'),
               Input('align-cases-check', 'value'),
               Input('align-cases-input', 'value'),
               Input('align-deaths-check', 'value'),
               Input('align-deaths-input', 'value'),
               Input('align-active-cases-check', 'value'),
               Input('align-active-cases-input', 'value'),
               Input('align-daily-cases-check', 'value'),
               Input('align-daily-cases-input', 'value'),
               Input('align-daily-deaths-check', 'value'),
               Input('align-daily-deaths-input', 'value')],
              [State('page-url', 'pathname'),
                State('hidden-stored-data', 'children')] +
              [State(c_name, 'value') for c_name in COUNTRY_LIST])
def update_plots(n_clicks, start_date, end_date, show_exponential, normalise_by_pop,
                 align_cases_check, align_cases_input, align_deaths_check, align_deaths_input, align_active_cases_check,
                 align_active_cases_input, align_daily_cases_check, align_daily_cases_input,
                 align_daily_deaths_check, align_daily_deaths_input, pathname, saved_json_data, *args):

    # print('dan 2',dash.callback_context.triggered)

    if pathname in ['/inter','/intro','/home','/tests','/']:
        raise PreventUpdate

    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    country_names = []
    for country in args:
        country_names.extend(country)

    if saved_json_data is None:
        country_data = {}
    else:
        country_data = json.loads(saved_json_data)

    for i, country in enumerate(country_names):
        if country not in country_data.keys():
            try:
                data = get_data(country)
                country_data[country] = data
            except Exception as e:
                print(e)
                country_names.remove(country)
                continue

    out = []
    for title in ['Cases', 'Deaths', 'Currently Infected', 'Daily New Cases', 'Daily New Deaths']:
        if normalise_by_pop:
            axis_title = f"{title} (% of population)"
        else:
            axis_title = title

        if title == 'Cases':
            align_countries = align_cases_check
            align_input = align_cases_input
        elif title == 'Deaths':
            align_countries = align_deaths_check
            align_input = align_deaths_input
        elif title == 'Currently Infected':
            align_countries = align_active_cases_check
            align_input = align_active_cases_input
        elif title == 'Daily New Cases':
            align_countries = align_daily_cases_check
            align_input = align_daily_cases_input
        elif title == 'Daily New Deaths':
            align_countries = align_daily_deaths_check
            align_input = align_daily_deaths_input

        figs = []

        if align_countries:
            xaxis_title = f'Days since the total confirmed cases reached {align_input}'
            if normalise_by_pop:
                xaxis_title += '% of the population'
        else:
            xaxis_title = ''

        layout_normal = {
            'yaxis': {'title': axis_title, 'type': 'linear', 'showgrid': True},
            'xaxis': {'title': xaxis_title, 'showgrid': True},
            'showlegend': True,
            'margin': {'l': 70, 'b': 100, 't': 0, 'r': 0},
            'updatemenus': [
                dict(
                    buttons=list([
                        dict(
                            args=["yaxis", {'title': axis_title, 'type': 'linear', 'showgrid': True}],
                            label="Linear",
                            method="relayout"
                        ),
                        dict(
                            args=["yaxis", {'title': axis_title, 'type': 'log', 'showgrid': True}],
                            label="Logarithmic",
                            method="relayout"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10, "b": 10},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        }

        layout_daily_plot = copy.copy(layout_normal)
        layout_daily_plot['updatemenus'].append(
            dict(
                buttons=list([
                    dict(
                        args=[{"visible": [False, False] + [False, False, True]*len(country_names) if show_exponential else [False] + [False, True]*len(country_names)}],
                        label="Bar",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, True] + [True, True, False]*len(country_names) if show_exponential else [True] + [True, False]*len(country_names)}],
                        label="Scatter",
                        method="update"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10, "b": 10},
                showactive=True,
                x=0.2,
                xanchor="left",
                y=1.2,
                yanchor="top"
                ),
            )

        if show_exponential:
            figs.append(go.Scatter(x=[datetime.date(2020, 2, 20)] if not align_countries else [0],
                                y=[0],
                                mode='lines',
                                line={'color': 'black', 'dash': 'dash'},
                                showlegend=True,
                                visible=False if title in ['Daily New Cases', 'Daily New Deaths'] else 'legendonly',
                                name=fr'Best exponential fits',
                                yaxis='y1',
                                legendgroup='group2', ))
            label = fr'COUNTRY : best fit (doubling time)'
        else:
            label = fr'COUNTRY'
        figs.append(go.Scatter(x=[datetime.date(2020, 2, 20)] if not align_countries else [0],
                            y=[0],
                            mode='lines+markers',
                            line={'color': 'black'},
                            showlegend=True,
                            visible=False if title in ['Daily New Cases', 'Daily New Deaths'] else 'legendonly',
                            name=label,
                            yaxis='y1',
                            legendgroup='group2', ))

        for i, c in enumerate(country_names):
            if country_data[c] is None:
                print("Cannot retrieve data from country:", c)
                continue
            if title == 'Daily New Cases':
                dates = country_data[c]['Cases']['dates'][1:]
                xdata = np.arange(len(dates))
                ydata = np.diff(np.array(country_data[c]['Cases']['data']).astype('float'))
            elif title == 'Daily New Deaths':
                dates = country_data[c]['Deaths']['dates'][1:]
                xdata = np.arange(len(dates))
                ydata = np.diff(np.array(country_data[c]['Deaths']['data']).astype('float'))
            elif title not in country_data[c]:
                continue
            else:
                dates = country_data[c][title]['dates']
                xdata = np.arange(len(dates))
                ydata = country_data[c][title]['data']
                ydata = np.array(ydata).astype('float')

            date_objects = []
            for date in dates:
                date_objects.append(datetime.datetime.strptime(date, '%Y-%m-%d').date())
            date_objects = np.asarray(date_objects)

            if normalise_by_pop:
                ydata = ydata/POPULATIONS[c] * 100

            if align_countries:
                if title in ['Cases', 'Deaths']:
                    idx_when_n_cases = np.abs(ydata - align_input).argmin()
                elif title in ['Currently Infected', 'Daily New Cases']:
                    ydata_cases = np.array(country_data[c]['Cases']['data']).astype('float')
                    ydata_cases = ydata_cases / POPULATIONS[c] * 100 if normalise_by_pop else ydata_cases
                    idx_when_n_cases = np.abs(ydata_cases - align_input).argmin()
                elif title in ['Daily New Deaths']:
                    ydata_cases = np.array(country_data[c]['Deaths']['data']).astype('float')
                    ydata_cases = ydata_cases / POPULATIONS[c] * 100 if normalise_by_pop else ydata_cases
                    idx_when_n_cases = np.abs(ydata_cases - align_input).argmin()
                if title in ['Daily New Cases', 'Daily New Deaths']:
                    idx_when_n_cases -= 1

                xdata = xdata - idx_when_n_cases

            model_date_mask = (date_objects <= end_date) & (date_objects >= start_date)

            model_dates = []
            model_xdata = []
            date = start_date
            d_idx = min(xdata[model_date_mask])
            while date <= end_date:
                model_dates.append(date)
                model_xdata.append(d_idx)
                date += datetime.timedelta(days=1)
                d_idx += 1
            model_xdata = np.array(model_xdata)

            b, logA = np.polyfit(xdata[model_date_mask], np.log(ydata[model_date_mask]), 1)
            lin_yfit = np.exp(logA) * np.exp(b * model_xdata)

            if show_exponential:
                if np.log(2) / b > 1000 or np.log(2) / b < 0:
                    double_time = 'no growth'
                else:
                    double_time = fr'{np.log(2) / b:.1f} days to double'
                label = fr'{c.upper():<10s}: 2^(t/{np.log(2)/b:.1f}) ({double_time})'
            else:
                label = fr'{c.upper():<10s}'

            figs.append(go.Scatter(x=date_objects if not align_countries else xdata,
                                y=ydata,
                                hovertext=[f"Date: {d.strftime('%d-%b-%Y')}" for d in date_objects] if align_countries else '',
                                mode='lines+markers',
                                marker={'color': colours[i]},
                                line={'color': colours[i]},
                                showlegend=True,
                                visible=False if title in ['Daily New Cases', 'Daily New Deaths'] else True,
                                name=label,
                                yaxis='y1',
                                legendgroup='group1', ))

            if show_exponential:
                if np.log(2) / b < 0:
                    show_plot = False
                else:
                    show_plot = True
                figs.append(go.Scatter(x=model_dates if not align_countries else model_xdata,
                                    y=lin_yfit,
                                    hovertext=[f"Date: {d.strftime('%d-%b-%Y')}" for d in model_dates] if align_countries else '',
                                    mode='lines',
                                    line={'color': colours[i], 'dash': 'dash'},
                                    showlegend=False,
                                    visible=False if title in ['Daily New Cases', 'Daily New Deaths'] else show_plot,
                                    name=fr'Model {c.upper():<10s}',
                                    yaxis='y1',
                                    legendgroup='group1', ))

            if title in ['Daily New Cases', 'Daily New Deaths']:
                figs.append(go.Bar(x=date_objects if not align_countries else xdata,
                                y=ydata,
                                hovertext=[f"Date: {d.strftime('%d-%b-%Y')}" for d in date_objects] if align_countries else '',
                                showlegend=True,
                                visible=True,
                                name=label,
                                marker={'color': colours[i]},
                                yaxis='y1',
                                legendgroup='group1'))
                layout_out = copy.copy(layout_daily_plot)
            else:
                layout_out = copy.copy(layout_normal)

        out.append({'data': figs, 'layout': layout_out})

    # Plot 'New Cases vs Total Cases' and 'New Deaths vs Total Deaths'
    for title in ['Cases', 'Deaths']:
        fig_new_vs_total = []
        for i, c in enumerate(country_names):
            l = 7  # Number of days to look back
            cases = np.array(country_data[c][title]['data']).astype('float')
            xdata = np.copy(cases[l:])
            ydata = np.diff(cases)
            len_ydata = len(ydata)

            # Compute new cases over the past l days
            ydata = np.sum([np.array(ydata[i:i + l]) for i in range(len_ydata) if i <= (len_ydata - l)], axis=1)

            dates = country_data[c][title]['dates'][l:]
            date_objects = []
            for date in dates:
                date_objects.append(datetime.datetime.strptime(date, '%Y-%m-%d').date())
            date_objects = np.asarray(date_objects)

            mask = xdata > 100 if title == 'Cases' else xdata > 10
            xdata = xdata[mask]
            ydata = ydata[mask]
            date_objects = date_objects[mask]

            if normalise_by_pop:
                xdata = xdata / POPULATIONS[c] * 100
                ydata = ydata / POPULATIONS[c] * 100

            fig_new_vs_total.append(go.Scatter(x=xdata,
                                            y=ydata,
                                            hovertext=[f"Date: {d.strftime('%d-%b-%Y')}" for d in date_objects],
                                            mode='lines+markers',
                                            marker={'color': colours[i]},
                                            line={'color': colours[i]},
                                            showlegend=True,
                                            name=fr'{c.upper():<10s}',
                                            yaxis='y1',
                                            legendgroup='group1', ))
        if normalise_by_pop:
            yaxis_title = f'New {title} (% of population) per week (log scale)'  # {l} days'
            xaxis_title = f'Total {title} (% of population) (log scale)'
        else:
            yaxis_title = f'New {title} per week'  # {l} days)'
            xaxis_title = f'Total {title}'
        layout_new_vs_total = {
            'yaxis': {'title': yaxis_title, 'type': 'log', 'showgrid': True},
            'xaxis': {'title': xaxis_title, 'type': 'log', 'showgrid': True},
            'showlegend': True,
            'margin': {'l': 70, 'b': 100, 't': 50, 'r': 0},
        }
        out.append({'data': fig_new_vs_total, 'layout': layout_new_vs_total})

    out.append(json.dumps(country_data))
    out.append(None)

    return out


##################################################################################################################################

@app.callback([
                Output('tests-plot', 'figure'),
                Output('tests-plot-1', 'figure'),
                Output('tests-text', 'children'),
                Output('tests-text-1', 'children'),
                Output('tests-text-2', 'children'),
                Output('loading-tests', 'children')
            ],
            [
                Input('plot-button-tests', 'n_clicks'),
            ],
            [
                State('prior', 'value'),
                State('sens', 'value'),
                State('spec', 'value'),
            ],
            )
def calculate_test_probs(plot_button,prior,sens,spec):

    true_pos, false_pos, true_neg, false_neg = test_probs(prior,sens,spec)
    
    outputs = [true_pos, false_pos, true_neg, false_neg]

    # have covid given negative test
    negative = false_neg/(false_neg+true_neg)

    # have covid given positive test
    positive = true_pos/(true_pos+false_pos)

    title1 = f"Proportion of test results positive: <b>{round(true_pos+false_pos,3)}</b>,<br>Proportion of test results negative: <b>{round(true_neg+false_neg,3)}</b>."
    
    text = (f"If {prior} of the population that get tested are actually infected, " +
              f"then with a test sensitivity of {sens} and specificity of {spec}, for " +
              f"a large testing population we would expect {round(100*false_pos,4)}% to be false positives and {round(100*false_neg,4)}% to be false negatives.") 

    text2 = (f"So {round(100*false_pos,4)}% of people tested get told they have covid when they don't, and " + 
              f"{round(100*false_neg,4)}% get told they don't have covid when they do.")

    text3 = (f"Under these circumstances, if you receive a positive test, you have a {round(100*positive,2)}% chance of having covid. " + 
                f"If you receive a negative test, you have a {round(100*negative,2)}% chance of having covid.")

    title2 = (f"Probability have covid given positive test: <b>{round(positive,3)}</b> (=True Pos/[sum of red])," +
                f"<br>Probability have covid given negative test: <b>{round(negative,3)}</b> (=False Neg/[sum of blue]).")

    fig = test_bar_plot2(outputs, title2)
    fig2 = test_bar_plot(outputs, title1)


    return [fig, fig2, text, text2, text3, None]



########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
