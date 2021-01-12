import time
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from gevent.pywsgi import WSGIServer

import pandas as pd
from math import ceil, exp
import numpy as np
import plotly.graph_objects as go
import copy
from flask import Flask
# from flask_caching import Cache
import datetime
import json

from parameters_cov import params


from config import preset_dict_high, preset_dict_low, \
    presets_dict_dropdown, dummy_figure

from cov_functions import run_model, test_probs, begin_date, \
    outcome_fn

from plotting import Bar_chart_generator, MultiFigureGenerator, \
    month_len, extract_info, test_bar_plot, test_bar_plot2, \
    vaccine_plot

from data_scraper import get_data
from data_constants import POPULATIONS, COUNTRY_LIST, \
    COUNTRY_LIST_NICK, COLOURS
from vaccine_scraper import VACCINE_COUNTRY_LIST, get_vaccine_data

from page_landing import layout_enter
from page_intro import layout_intro
from page_interactive import layout_inter
from page_data import layout_data
from page_tests import layout_tests
from page_vaccines import layout_vaccine

# start_time = time.time()
# print(f"7 {time.time()-start_time} seconds")



# layout_enter = render_template('layout_enter.html', title='/')

# activate venv
# Environments\CovidWebsite\Scripts\activate

vaccine_df = get_vaccine_data()



########################################################################################################################
FA = "https://use.fontawesome.com/releases/v5.12.1/css/all.css"
# dash_ss = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# , FA]
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
# external_stylesheets=external_stylesheets,

app = Dash(__name__, 
        external_stylesheets=external_stylesheets,
        assets_folder='assets')

server = app.server

app.config.suppress_callback_exceptions = True

########################################################################################################################
# setup



# cache = Cache(app.server, config={
#     # try 'filesystem' if you don't want to setup redis
#     'CACHE_TYPE': 'redis',
#     'CACHE_REDIS_URL': 'redis://h:paa75aa4b983ba337eb43b831e6833be6b6887e56023aa417e392dd2bf337e8b8@ec2-18-213-184-148.compute-1.amazonaws.com:31119'
# })






#########################################################################################################################################################


app.layout = html.Div([
        
        html.Header([
        
            html.Div([
                html.A("LowHighCovid", href="/", className="navLinkTitle"),
            ],
            className="my-title",
            ),

            html.Div(
                html.Img(src='/assets/images/menu.svg',id="hamburger"),
                id="menu-button",className="hide-desktop"),


            html.Div([
                html.Nav(
                    children=[
                            html.Div([
                                dbc.Collapse([
                                    
                                    html.Div([html.A("Data", className="navLink nv-lk-mb pad-five grey-text"), #  greyed-link"
                                        html.I( className="fa fa-caret-down grey-col")
                                    ],
                                    id="down-arrow-mb",
                                    className="full-width pad-ten down-arrow-cls"), # backgd-grey

                                    dbc.Collapse([
                                        html.Div(html.A("Cases and deaths", href="/covid-data", className="navLink nv-lk-mb pad-five"), className="white-backgd pad-sides"),
                                        html.Div(html.A("Vaccinations", href="/vaccine-data", className="navLink nv-lk-mb pad-five"), className="white-backgd pad-sides"),
                                    ],
                                    id="data-menu-mb",
                                    is_open=False),



                                    html.Div([html.A("Modelling", className="navLink nv-lk-mb pad-five grey-text"), #  greyed-link"
                                        html.I( className="fa fa-caret-down grey-col")
                                    ],
                                    id="down-arrow-model-mb",
                                    className="full-width pad-ten down-arrow-cls"), # backgd-grey
                                    
                                    dbc.Collapse([
                                        html.Div(html.A("Background", href="/intro", className="navLink nv-lk-mb pad-five"), className="white-backgd pad-sides"),
                                        html.Div(html.A("Interpreting tests", href="/covid-tests", className="navLink nv-lk-mb pad-five"), className="white-backgd pad-sides"),
                                        html.Div(html.A("Interactive model", href="/interactive-model", className="navLink nv-lk-mb pad-five"), className="white-backgd pad-sides"),
                                        ],
                                    id="modelling-menu-mb",
                                    is_open=False),

                                ],
                                id="nav-menu",
                                is_open=False),
                            ],
                            className="pad-five hide-desktop",
                            id="mobile-navs"
                            ),
                            
                            html.Div([
                                
                                html.Div([
                                html.Div([html.A("Data", className="navLink pad-five"),
                                        html.I(className="fa fa-caret-down grey-col")
                                    ],
                                id="down-arrow-data-dt",
                                className="down-arrow-cls arrow-cont"),

                                dbc.Collapse([
                                    html.Div(html.A("Cases and deaths", href="/covid-data", className="navLink nv-lk-dt pad-five"), className="full-width grey-hover pad-ten"),
                                    html.Div(html.A("Vaccinations", href="/vaccine-data", className="navLink nv-lk-dt pad-five"), className="full-width grey-hover pad-ten"),
                                    ],
                                id="data-menu-dt",
                                className="navbar-dropdown-desktop",
                                is_open=False),
                                ]),

                                html.Div([
                                html.Div([html.A("Modelling", className="navLink pad-five"),
                                        html.I(className="fa fa-caret-down grey-col")
                                    ],
                                id="down-arrow-model-dt",
                                className="down-arrow-cls arrow-cont"),
                                    
                                dbc.Collapse([
                                    html.Div(html.A("Background", href="/intro", className="navLink nv-lk-dt pad-five"), className="full-width grey-hover pad-ten"),
                                    html.Div(html.A("Interpreting tests", href="/covid-tests", className="navLink nv-lk-dt pad-five"), className="full-width grey-hover pad-ten"),
                                    html.Div(html.A("Interactive model", href="/interactive-model", className="navLink nv-lk-dt pad-five"), className="full-width grey-hover pad-ten"),
                                    ],
                                id="modelling-menu-dt",
                                className="navbar-dropdown-desktop",
                                is_open=False),
                                ]),

                            ],
                            className="other-links show-desktop hide-mobile",
                            ),
                ]),
            ],
            id="main-nav-links"
            ),
        
        ],
        ),





        ##
        dcc.Location(id='page-url', refresh=False),
        
        html.Div(id='page-content',children=html.Div('Loading...',style={'height': '100vh'})),

        html.Footer([

        html.Div([
                html.A("Case/death data", href="/covid-data", className="footer-navlink", id="footer-data"),
                html.A("Vaccination data", href="/vaccine-data", className="footer-navlink", id="footer-vaccine-data"),
                html.A("Modelling background", href="/intro", className="footer-navlink", id="footer-intro"),
                html.A("Interpreting tests", href="/covid-tests", className="footer-navlink", id="footer-tests"),
                html.A("Interactive model", href="/interactive-model", className="footer-navlink", id="footer-inter"),
                ],
                className="footer-links",
                ),

        html.Div([
                        html.Div(["Contact us at: ", html.A('covid.at.plants@gmail.com',href='')]),
                        
                        html.Div("The modelling section is intended for illustrative/educational purposes only, and not for accurate prediction of the pandemic.",
                        className="disclaimer"),

                        html.Div(["Authors: ",
                                html.A('Nick P. Taylor', href='https://twitter.com/TaylorNickP'),", ",
                                html.A('Daniel Muthukrishna', href='https://twitter.com/DanMuthukrishna'),
                                " and Dr Cerian Webb. ",
                                ],
                        ),
                        
                        html.Div([
                        html.A('Source code', href='https://github.com/nt409/covid-19'),
                        ". Data is taken from ",
                        html.A("Worldometer", href='https://www.worldometers.info/coronavirus/'),
                        " if available or otherwise ",
                        html.A("Johns Hopkins University (JHU) CSSE", href="https://github.com/ExpDev07/coronavirus-tracker-api"),
                        ".",
                        ],className="source-code"
                        ),

                    ],
        className="foot-container",
        ),
        
        ],className="footer-container")



        ],
        )
# end of layout






########################################################################################################################
# app.title = 'LowHighCovid'


# added in viewport tag for media queries
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <meta name="description" 
        content="LowHighCovid - coronavirus data HQ. Track cases, deaths and vaccine data in hundreds of countries, or predict the progression of the global pandemic.">
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=UA-163339118-1"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'UA-163339118-1');
        </script>

        {%metas%}
        <title>Covid data and modelling HQ - LowHighCovid</title>
        <link rel="icon" href="assets/favicon.ico">
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


# <link rel="stylesheet" href="assets/data.css">
#         <link rel="stylesheet" href="assets/home.css">
#         <link rel="stylesheet" href="assets/inter.css">
#         <link rel="stylesheet" href="assets/intro.css">
#         <link rel="stylesheet" href="assets/navbar.css">
#         <link rel="stylesheet" href="assets/main.css">





########################################################################################################################
# callbacks



@app.callback(Output('page-content', 'children'),
            [Input('page-url', 'pathname')])
def display_page(pathname):
    if pathname == '/covid-data':
        return layout_data
    elif pathname == '/intro':
        return layout_intro
    elif pathname == '/interactive-model':
        return layout_inter
    elif pathname == '/covid-tests':
        return layout_tests
    elif pathname == '/vaccine-data':
        return layout_vaccine
    elif pathname == '/':
        return layout_enter
    else:
        return html.Div('404: Page not found',style={'height': '100vh'})


########################################################################################################################
# collapse
def toggle(n, is_open):
    if n:
        return not is_open
    return is_open



########################################################################################################################
# popovers


for p in ["control", "months-control", "vaccination",  "cc-care" , "inf-rate", "inf-tab", "cont-tab", "example","red-deaths","ICU","herd", 'cycles-off', 'cycles-on', 'groups-allowed']: # "res-type" , "pick-strat",
    app.callback(
        Output(f"popover-{p}", "is_open"),
        [Input(f"popover-{p}-target", "n_clicks")
        ],
        [State(f"popover-{p}", "is_open")],
    )(toggle)

for id_name, activator in zip(["modelling-menu-mb",
            "data-menu-mb",
            "modelling-menu-dt",
            "data-menu-dt",
            "nav-menu"],
            ["down-arrow-model-mb",
            "down-arrow-mb",
            "down-arrow-model-dt",
            "down-arrow-data-dt",
            "menu-button"]):

    app.callback(Output(id_name, "is_open"),
        [Input(activator, "n_clicks")
        ],
        [State(id_name, "is_open")],
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
    ],
    [State('page-url', 'pathname')]
    )
def invisible_or_not(num,preset,pathname):

    if not pathname in ['interactive-model','/interactive-model']:
        raise PreventUpdate


    if num=='two':
        strat_H = 'Strategy one: high risk infection rate (%)'
        strat_L = 'Strategy one: low risk infection rate (%)'
        says_strat_2 = None

    else:

        strat_H = 'High risk infection rate (%)'

        strat_L = 'Low risk infection rate (%)'

        says_strat_2 = {'display': 'none'}

    if preset!='C':
        says_strat_2 = {'display': 'none'}
    
    custom_tab_style = {'display': 'none'}
    ld_cycle_tab_style = {'display': 'none'}
    if preset=='C':
        custom_tab_style = {'display': 'block'}
    
    if preset=='LC':
        ld_cycle_tab_style = {'display': 'block'}

    
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
            [State('number-strats-radio','value'),
            State('page-url', 'pathname')])
def preset_sliders(preset,number_strs,pathname):

    # print('preset sliders')
    if not pathname in ['interactive-model','/interactive-model']:
        raise PreventUpdate

    lockdown_cycles_dis = True
    options_lockdown_cycles = [
                                {'label': 'Low risk only', 'value': 0, 'disabled': True},
                                {'label': 'Both groups', 'value': 1, 'disabled': True},
                            ]
    if preset=='LC':
        lockdown_cycles_dis = False
        options_lockdown_cycles = [
                                {'label': 'Low risk only', 'value': 0},
                                {'label': 'Both groups', 'value': 1},
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

    # if vaccine==0:
        # vaccine = None

    if init_stored is None or date!=init_stored[5] or country!=init_stored[6]:
        I0, R0, H0, C0, D0, worked, min_date, max_date, _ = begin_date(date,country)
        initial_conds = [I0, R0, H0, C0, D0, date, country, min_date, max_date]
    else:
        initial_conds = init_stored
        I0 = init_stored[0]
        R0 = init_stored[1]
        H0 = init_stored[2]
        C0 = init_stored[3]
        D0 = init_stored[4]
        date=init_stored[5]
        country=init_stored[6]
        min_date = init_stored[7]
        max_date = init_stored[8]

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
                            date=date,
                            I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,
                            ICU_grow=ICU_grow,let_HR_out=let_HR_out))
    if num_strat=='two':
        lr2 = params.fact_v[int(lr2_in)]
        hr2 = params.fact_v[int(hr2_in)]
        sols.append(run_model(beta_L_factor=lr2,beta_H_factor=hr2,
                                t_control=months_controlled,
                                vaccine_time=vaccine,
                                date=date,
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
                                date=date,
                                I0=I0_new,R0=R0_new,H0=H0_new,C0=C0_new,D0=D0_new,
                                ICU_grow=ICU_grow,let_HR_out=let_HR_out)

        sols_upper_lower.append(upp_low)

    return [sols,
        initial_conds,
        worked,
        worked_div, 
        sols_upper_lower,
        min_date,
        max_date]


def find_sol_do_noth(ICU_grow,date,country_num):
    # @cache.memoize()  # in seconds
    # print(dash.callback_context.triggered,'do nothing')

    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    I0, R0, H0, C0, D0, _, _, _, prev_deaths = begin_date(date,country)

    sol_do_nothing = run_model(I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,
                                    beta_L_factor=1,beta_H_factor=1,
                                    t_control=None,
                                    date=date,
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

                Output('model-start-date','min_date_allowed'),
                Output('model-start-date','max_date_allowed'),

                Output('graph-title', 'children')
                
                

               ],
               [
                Input('plot-button', 'n_clicks'),
               ],
               [
                State('page-url', 'pathname'),

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
    

    if not pathname in ['interactive-model','/interactive-model'] or plot_button is None:
        raise PreventUpdate

    # could cache these
    sol_do_nothing, prev_deaths = find_sol_do_noth(ICU_grow,date,country_num)
    # print(f'render {pathname}')
    sols, initial_conds, worked, worked_div, upper_lower_sol, min_date, max_date = find_sol(preset,
                    month,lr_in,hr_in,lr2_in,hr2_in,
                    num_strat,vaccine_time,ICU_grow,date,country_num,
                    t_off,t_on,hr_ld,init_stored,worked)
    
    min_date = str(min_date).split(' ')[0]
    max_date = str(max_date).split(' ')[0]
    
    min_date = min_date.split('T')[0]
    max_date = max_date.split('T')[0]
    
    min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )
    max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d' )









    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    country_name = country.title() if country not in ['us', 'uk'] else 'the '+country.upper()

    strategy_name = presets_dict_dropdown[preset]

    try:
        m_diff = month[1] - month[0]
        if m_diff!=1:
            months_string = f' for {str(m_diff)} months'
        else:
            months_string = f' for {str(m_diff)} month'
    except:
        months_string = ''

    graph_title = f'What might happen if strategy "{strategy_name}" were used{months_string} in {country_name}?'

    
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




        # if vaccine_time==9:
            # vaccine_time = None

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

    None,

    min_date + datetime.timedelta(days=26),
    max_date,

    graph_title
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
        options_deaths = [{'label': "Align countries by the date when the percentage of confirmed deaths was ",
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
    if not pathname in ['/covid-data','covid-data']:
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
    layout_out = []
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
            'yaxis': {'title': axis_title, 'type': 'linear', 'showgrid': True}, # , 'fixedrange': True
            'xaxis': {'title': xaxis_title, 'showgrid': True}, # , 'fixedrange': True
            'showlegend': True,
            'legend': dict(
                x = 0.5,
                font=dict(size=11),
                y = 1.03,
                xanchor= 'center',
                yanchor= 'bottom'
            ),
            'height': 450,
            'margin': {'l': 50, 'b': 10, 't': 10, 'r': 10, 'pad': 0},

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
                    direction="up",
                    pad={"r": 10, "t": 10, "b": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="center",
                    y=-0.1,
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
                direction="up",
                pad={"r": 10, "t": 10, "b": 10},
                showactive=True,
                x=0.9,
                xanchor="center",
                y=-0.1,
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

            # if show_exponential:
            #     if np.log(2) / b > 1000 or np.log(2) / b < 0:
            #         double_time = 'no growth'
            #     else:
            #         double_time = fr'{np.log(2) / b:.1f} days to double'
            #     label = fr'{c.upper():<10s}: 2^(t/{np.log(2)/b:.1f}) ({double_time})'
            # else:
            label = fr'{c.upper():<10s}'

            figs.append(go.Scatter(x=date_objects if not align_countries else xdata,
                                y=ydata,
                                hovertext=[f"Date: {d.strftime('%d-%b-%Y')}" for d in date_objects] if align_countries else '',
                                mode='lines+markers',
                                marker={'color': COLOURS[i]},
                                line={'color': COLOURS[i]},
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
                                    line={'color': COLOURS[i], 'dash': 'dash'},
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
                                marker={'color': COLOURS[i]},
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
                                            marker={'color': COLOURS[i]},
                                            line={'color': COLOURS[i]},
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
            'yaxis': {'title': yaxis_title, 'type': 'log', 'showgrid': True, 'automargin': True}, # , 'fixedrange': True
            'xaxis': {'title': xaxis_title, 'type': 'log', 'showgrid': True, 'automargin': True}, # , 'fixedrange': True
            'showlegend': True,
            'height': 450,
            'legend': dict(
                        x = 0.5,
                        font=dict(size=11),
                        y = 1.03,
                        xanchor= 'center',
                        yanchor= 'bottom'
                    ),
            'margin': {'l': 50, 'b': 10, 't': 10, 'r': 10, 'pad': 0},



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
                State('page-url', 'pathname'),
            ],
            )
def calculate_test_probs(plot_button,prior,sens,spec,pathname):
    
    if not pathname in ['/covid-tests','covid-tests']:
        raise PreventUpdate

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

##################################################################################################################################

@app.callback([
                Output('vaccine-plot', 'figure'),
                Output('loading-icon-vd', 'children')
            ],
            [
                Input('button-plot-vd', 'n_clicks'),
                Input('vd-normalise-check', 'value'),
            ],
            [State(f"{c_name}-v-data", 'value') for c_name in VACCINE_COUNTRY_LIST]
            )
def vaccine_callback(plot_button, normalise_by_pop, *c_names):
    country_names = []
    for country in c_names:
        country_names.extend(country)

    fig = vaccine_plot(vaccine_df, country_names, normalise_by_pop)
    return [fig, None]

########################################################################################################################
if __name__ == '__main__':
    # app.run_server(debug=False)
    app.run_server(debug=True)