import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import datetime
from math import floor

from parameters_cov import params

from config import presets_dict_dropdown, initial_hr, \
    initial_lr, initial_strat, \
    dummy_figure, bar_non_crit_style
    

from data_constants import COUNTRY_LIST_NICK, initial_country

min_date = '2020-2-15' # first day of data
min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )

max_date = datetime.datetime.today() - datetime.timedelta(days=2)
max_date = str(max_date).split(' ')[0]

# try:
#     gd = get_data('uk')
#     min_date = gd['Cases']['dates'][0]
#     max_date = gd['Cases']['dates'][-1]
#     min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )
#     max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d' )
# except:
#     print("Cannot get dates from Worldometer")





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

    html.H6(className="control button", children=[
    ' Control Type ',
    dbc.Button(' ? ',
        color='primary',
        size='sm',
        id='popover-control-target',
        style={'cursor': 'pointer','marginBottom': '2px'}
        ),
    ]),
    # style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}


    dbc.Popover(
    [
    dbc.PopoverHeader('Control'),
    dbc.PopoverBody(dcc.Markdown(
    '''

    The type of **control** determines how much we can reduce the **infection rate** of the disease (how quickly the disease is transmitted between people).
    
    We consider control of **two risk groups**; high risk and low risk. High risk groups are more likely to get seriously ill if they catch the disease.

    *For further explanation, read the [**Background**](/intro)*.

    ''',
    style={'textAlign': 'justify'}

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

    html.H6(className="controls button", children=[
    ' Months of Control ',
    dbc.Button(' ? ',
    color='primary',
    size='sm',
    id='popover-months-control-target',
    style= {'cursor': 'pointer','marginBottom': '2px'}
    ),
    ],
    # style={'fontSize': '80%', 'marginBottom': '10px','textAlign': 'center'}
    ),


    
    html.Div([
    dcc.RangeSlider(
                id='month-slider',
                min=0,
                max=floor(params.max_months_controlling),
                step=1,
                # pushable=0,
                marks={i: str(i) for i in range(0,floor(params.max_months_controlling)+1,3)},
                value=[0,2],
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
        style={'textAlign': 'justify'}
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
html.H6(className="controls", children=' Country ', # style={'fontSize': '80%', 'marginBottom': '10px','textAlign': 'center'}
    ),
                                        
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
html.H6(className="controls",children=' Model Start Date ',
    # style={'fontSize': '80%', 'textAlign': 'center', 'marginBottom': '10px'}
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
html.H6(className="controls button", children=[
' Vaccination starts ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-vaccination-target',
style= {'cursor': 'pointer'}),
],
# style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}
),


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
style={'textAlign': 'justify'}
),),
],
id = "popover-vaccination",
is_open=False,
target="popover-vaccination-target",
placement='left',
),



html.H2(style={'marginTop': '20px'},className="fas fa-user-md"),
html.H6(className="controls button", children=[
' Critical Care Capacity Increase ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-cc-care-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
],
# style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}
),

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
style={'textAlign': 'justify'}
),),
],
id = "popover-cc-care",
is_open=False,
target="popover-cc-care-target",
placement='top',
),




html.H2(style={'marginTop': '20px'},className="fas fa-chart-area"),
html.H6(className="controls",children=' Results Type ',
# style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}
),

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




html.H6(className="controls", children='Number Of Strategies',
            # style={'fontSize': '80%', 'marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center'}
            ),

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



html.H6(className="controls button", children=[
'Infection Rate ',
dbc.Button(' ? ',
color='primary',
size='sm',
id = 'popover-inf-rate-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
],
# style={'fontSize': '80%','marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center'}
),


dbc.Popover(
[
dbc.PopoverHeader('Infection Rate'),
dbc.PopoverBody(dcc.Markdown(
'''

The *infection rate* relates to how quickly the disease is transmitted. **Control** measures affect transmission/infection rates (typically lowering them).

Adjust by choosing a preset strategy  or making your own custom choice ('**Control Type**').

''',
style={'textAlign': 'justify'}
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

html.H6(className="controls",children=' Strategy Two: Low Risk Infection Rate (%) '),
# style={'fontSize': '80%','textAlign': 'center'}),

dcc.Slider(
id='low-risk-slider-2',
min=0,
max=len(params.fact_v)-1,
step = 1,
marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
value=5,
),

html.H6(className="controls",children=' Strategy Two: High Risk Infection Rate (%) ', style = {'fontSize': '80%','textAlign': 'center'}),

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




html.H6(className="controls button", children=['Groups allowed out of lockdown ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-groups-allowed-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
]),
# style={'fontSize': '80%','marginTop': '10px', 'marginBottom': '10px', 'textAlign': 'center'}),


dbc.Popover(
[
dbc.PopoverHeader('Lockdown Cycles: Groups'),
dbc.PopoverBody(dcc.Markdown(
'''

In a strategy where lockdowns are 'switched on and off', you may choose to continue to protect the high risk by continuing their lockdown.

Choose whether to keep high risk in lockdown or allow all groups to leave lockdown (this is the default setting).

''',
style={'textAlign': 'justify'}
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
html.H6(className="controls button", children=[
    ' Cycle: Time On ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-cycles-on-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
]),
# style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}),

dcc.Slider(
id='cycle-on',
min = 1,
max = 8,
step = 1,
marks={i: 'Weeks: ' + str(i) if i==1 else str(i) for i in range(1,9)},
value=3,
),

dbc.Popover(
[
dbc.PopoverHeader('Lockdown Cycles: Time On'),
dbc.PopoverBody(dcc.Markdown(
'''

Use this slider to adjust the amount of time that the country is in lockdown under the strategy 'Lockdown cycles'.

This allows us to consider a system where the country is in lockdown for 3 weeks say, followed by a week of normality.

''',
style={'textAlign': 'justify'}
),),
],
id = "popover-cycles-on",
is_open=False,
target="popover-cycles-on-target",
placement='top',
),


html.H2(style={'marginTop': '20px'},className="fas fa-clock"),
html.H6(className="controls button", children=[
    ' Cycle: Time Off ',
dbc.Button(' ? ',
color='primary',
size='sm',
id='popover-cycles-off-target',
style= {'cursor': 'pointer','marginBottom': '2px'}),
]),
# style={'fontSize': '80%', 'marginBottom': '10px', 'textAlign': 'center'}),

dcc.Slider(
id='cycle-off',
min = 1,
max = 8,
step = 1,
marks={i: 'Weeks: ' + str(i) if i==1 else str(i) for i in range(1,9)},
value=3,
),

dbc.Popover(
[
dbc.PopoverHeader('Lockdown Cycles: Time Off'),
dbc.PopoverBody(dcc.Markdown(
'''

Use this slider to adjust the amount of time that the country is out of lockdown under the strategy 'Lockdown cycles'.

This allows us to consider a system where the country is in lockdown for 3 weeks say, followed by a week of normality.

''',
style={'textAlign': 'justify'}
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
            html.H4(id='graph-title', style={'fontSize': '120%'}),
            dcc.Graph(figure=dummy_figure,id='line-plot-2'), # ,style={'height': '100px', 'width': '100%'}),
            ],
            style={'margin': '10px'}
            ),
        className="inter-card"
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
                
            dbc.Tab(labelClassName='tab', label='Main controls',
                        tab_style = {'textAlign': 'center', 'cursor': 'pointer'},
                        tab_id='tab_main',
                        children=control_choices_main
                        ),
            dbc.Tab(labelClassName='tab', label='Custom options',
                        tab_style = {'display': 'none'},
                        id='tab-custom',
                        children=[control_choices_custom]
                        ),
            dbc.Tab(labelClassName='tab', label='Lockdown cycle options',
                        tab_style = {'display': 'none'},
                        id='tab-ld-cycle',
                        children=[control_choices_lockdown],
                        ),
            dbc.Tab(labelClassName='tab', label='Other',
                        tab_style = {'textAlign': 'center', 'cursor': 'pointer'},
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
className="inter-card",
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
className="inter-card"
)





















layout_inter = html.Div([
                    # store results
                    dcc.Store(id='sol-calculated-do-nothing-cache'),
                    dcc.Store(id='prev-deaths-cache'),
                    dcc.Store(id='store-initial-conds-cache'),
                    dcc.Store(id='store-get-data-worked-cache'),

                    # replace with Cache soon!









                                    


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

