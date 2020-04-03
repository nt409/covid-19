import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from flask import Flask
from gevent.pywsgi import WSGIServer
import pandas as pd
from math import floor, ceil, exp
from parameters_cov import params, df2
import numpy as np
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
import copy
from cov_functions import simulator
import flask


from dan import layout_dan, COUNTRY_LIST, colours
from dan_get_data import get_data
from dan_constants import POPULATIONS
import datetime
import json



########################################################################################################################

# external_stylesheets = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
external_stylesheets = dbc.themes.CERULEAN
# JOURNAL
# MINTY
# SIMPLEX - not red danger
# UNITED
# Cerulean is ok

app = dash.Dash(__name__, external_stylesheets=[external_stylesheets])

server = app.server

app.config.suppress_callback_exceptions = True
########################################################################################################################
# setup

initial_lr = 8
initial_hr = 4
initial_month = 8

df = copy.deepcopy(df2)
df = df.loc[:,'Age':'Pop']
df2 = df.loc[:,['Pop','Hosp','Crit']].astype(str) + '%'
df = pd.concat([df.loc[:,'Age'],df2],axis=1)
df = df.rename(columns={"Hosp": "Hospitalised", "Crit": "Requiring Critical Care", "Pop": "Population"})

init_lr = params.fact_v[initial_lr]
init_hr = params.fact_v[initial_hr]


def generate_table(dataframe, max_rows=10):
    return dbc.Table.from_dataframe(df, striped=True, bordered = True, hover=True)


dummy_figure = {'data': [], 'layout': {'template': 'simple_white'}}

bar_height = '100'

bar_width  =  '100'

bar_non_crit_style = {'height': bar_height, 'width': bar_width, 'display': 'block' }

presets_dict = {'N': 'Do Nothing',
                'MSD': 'Social Distancing',
                'HL': 'Quarantine High Risk, Mild Social Distancing For Low Risk',
                'Q': 'Quarantine All',
                'H': 'Quarantine High Risk, No Social Distancing For Low Risk',
                'C': 'Custom'}

preset_dict_high = {'Q': 0, 'MSD': 4, 'HL': 0, 'H': 0, 'N':6}
preset_dict_low  = {'Q': 0, 'MSD': 4, 'HL': 5, 'H': 6, 'N':6}

month_len = 365/12


group_vec = ['BR','HR','LR']

longname = {'S': 'Susceptible',
        'I': 'Infected',
        'R': 'Recovered',
        'H': 'Hospitalised',
        'C': 'Critical',
        'D': 'Deaths',
}

linestyle = {'BR': 'solid',
        'HR': 'dot',
        'LR': 'dash'}
group_strings = {'BR': ' Sum of Risk Groups',
        'HR': ' High Risk',
        'LR': ' Low Risk'}

factor_L = {'BR': 1,
        'HR': 0,
        'LR': 1}

factor_H = {'BR': 1,
        'HR': 1,
        'LR': 0}

colors = {'S': 'blue',
        'I': 'orange',
        'R': 'green',
        'H': 'red',
        'C': 'black',
        'D': 'purple',
        }

index = {'S': params.S_L_ind,
        'I': params.I_L_ind,
        'R': params.R_L_ind,
        'H': params.H_L_ind,
        'C': params.C_L_ind,
        'D': params.D_L_ind,
        }










########################################################################################################################

txt1 = dcc.Markdown('''

    This plot shows a prediction for the number of deaths (*or critical care depending on hospital category settings*) caused by the epidemic in the absence of a vaccine. It also shows the split between the deaths in the high and low risk groups.
    
    Most outcomes result in a much higher proportion of high risk deaths, so it is critical that any strategy should protect the high risk.

    Quarantine/lockdown strategies are very effective at slowing the death rate, but only work whilst they're in place (or until a vaccine is produced).

    ''',style={'fontSize': '1.4vh' })
txt2 = dcc.Markdown('''

    This plot shows how close to the 60% population immunity the strategy gets.
    
    Strategies with a lower *infection rate* can delay the course of the epidemic but once the strategies are lifted there is no protection through herd immunity. Strategies with a high infection rate can risk overwhelming healthcare capacity.

    The optimal outcome is obtained by making sure the 60% that do get the infection are from the low risk group.

    ''',style={'fontSize': '1.4vh' })
txt3 = dcc.Markdown('''

    This plot shows the maximum ICU capacity needed.
    
    Better strategies reduce the load on the healthcare system by reducing the numbers requiring Intensive Care at any one time.

    ''',style={'fontSize': '1.4vh' })
txt4 = dcc.Markdown('''

    This plot shows the length of time for which ICU capacity is exceeded, over the calculated number of years.

    Better strategies will exceed the ICU capacity for shorter lengths of time.

    ''',style={'fontSize': '1.4vh' })
txt5 = dcc.Markdown('''

    This plot shows the length of time until the safe threshold for population immunity is 95% reached.
    
    We allow within 5% of the safe threshold, since some strategies get very close to full safety very quickly and then asymptotically approach it (but in practical terms this means the ppulation is safe).

    The longer it takes to reach this safety threshold, the longer the population must continue control measures because it is at risk of a further epidemic.

    ''',style={'fontSize': '1.4vh' })

########################################################################################################################


def Bar_chart_generator(data,data2 = None, data_group = None,name1=None,name2=None,preset=None,text_addition=None,color=None,y_title=None,yax_tick_form=None,maxi=True,yax_font_size_multiplier=None,hover_form=None): # ,title_font_size=None): #title
    
    font_size = 10

    if yax_font_size_multiplier is None:
        yax_font_size = font_size
    else:
        yax_font_size = yax_font_size_multiplier*font_size

    ledge = None
    show_ledge = False
    
    if len(data)==2:
        cats = ['Strategy Choice','Do Nothing']
    else:
        cats = ['Strategy One','Strategy Two','Do Nothing']
    
    order_vec = [len(data)-1,0,1]
    order_vec = order_vec[:(len(data))]

    data1 = [data[i] for i in order_vec]
    cats  = [cats[i] for i in order_vec]
    if data2 is not None:
        data2 = [data2[i] for i in order_vec]
    
    if data_group is not None:
        name1 = 'End of Year 1'

    trace0 = go.Bar(
        x = cats,
        y = data1,
        marker=dict(color=color),
        name = name1,
        hovertemplate=hover_form
    )

    traces = [trace0]
    barmode = None
    if data_group is not None:
        data_group = [data_group[i] for i in order_vec]
        traces.append( go.Bar(
            x = cats,
            y = data_group,
            # marker=dict(color=color),
            name = 'End of Year 3',
            hovertemplate=hover_form
        ))
        barmode='group'
        show_ledge = True


    if data2 is not None:
        traces.append(go.Bar(
        x = cats,
        y = data2,
        hovertemplate=hover_form,
        name = name2)
        )
        show_ledge = True

    if show_ledge:
        ledge = dict(
                       font=dict(size=font_size),
                       x = 0.5,
                       y = 1.02,
                       xanchor= 'center',
                       yanchor= 'bottom',
                   )
    
        
    

    # cross
    if data_group is not None:
        data_use = data_group
    elif data2 is not None:
        data_use = [data1[i] + data2[i] for i in range(len(data1))] 
    else:
        data_use = data1
    counter_bad = 0
    counter_good = 0
    if len(data_use)>1:
        for i, dd in enumerate(data_use):
            if maxi and dd == max(data_use):
                worst_cat = cats[i]
                worst_cat_y = dd
                counter_bad += 1
            if maxi and dd == min(data_use):
                best_cat = cats[i]
                best_cat_y = dd
                counter_good += 1
            if not maxi and dd == min(data_use):
                worst_cat = cats[i]
                worst_cat_y = dd
                counter_bad += 1
            if not maxi and dd == max(data_use):
                best_cat = cats[i]
                best_cat_y = dd
                counter_good += 1
        
        if counter_bad<2:
            traces.append(go.Scatter(
                x= [worst_cat],
                y= [worst_cat_y/2],
                mode='markers',
                marker_symbol = 'x',
                marker_size = (30/20)*font_size,
                marker_line_width=1,
                opacity=0.5,
                marker_color = 'red',
                marker_line_color = 'black',
                hovertemplate='Worst Strategy',
                showlegend=False,
                name = worst_cat
            ))
        if counter_good<2:
            traces.append(go.Scatter(
                x= [best_cat],
                y= [best_cat_y/2],
                opacity=0.5,
                # mode='markers',
                # marker_symbol = 'star', # 'star # U+2705
                # marker_line_width=1,
                # marker_color = 'green',
                # marker_line_color = 'black',
                
                mode = 'text',
                text = [r'✅'],

                textfont= dict(size= (30/20)*font_size),

                hovertemplate='Best Strategy',
                showlegend=False,
                name = best_cat
            ))

    layout = go.Layout(
                    # autosize=False,
                    font = dict(size=font_size),
                    barmode = barmode,
                    template="simple_white", #plotly_white",
                    yaxis_tickformat = yax_tick_form,
                    height=450,
                    legend = ledge,
                    xaxis=dict(showline=False),
                    yaxis = dict(
                        automargin = True,
                        showline=False,
                        title = y_title,
                        title_font = dict(size=yax_font_size),
                    ),
                    showlegend = show_ledge,

                    transition = {'duration': 500},
                   )



    return {'data': traces, 'layout': layout}


########################################################################################################################
def time_exceeded_function(yy,tt):
    True_vec = [ (yy[params.C_H_ind,i]+yy[params.C_L_ind,i]) > params.ICU_capacity for i in range(len(tt))]
    Crit_vals = [ (yy[params.C_H_ind,i]+yy[params.C_L_ind,i])  for i in range(len(tt))]

    c_low = [-2]
    c_high = [-1]
    ICU = False
    if max(Crit_vals)>params.ICU_capacity:
        ICU = True

        for i in range(len(True_vec)-1):
            if not True_vec[i] and True_vec[i+1]: # entering
                y1 = 100*(yy[params.C_H_ind,i]+yy[params.C_L_ind,i])
                y2 = 100*(yy[params.C_H_ind,i+1]+yy[params.C_L_ind,i+1])
                t1 = tt[i]
                t2 = tt[i+1]
                t_int = t1 + (t2- t1)* abs((100*params.ICU_capacity - y1)/(y2-y1)) 
                c_low.append(t_int) # 0.5 * ( tt[i] + tt[i+1]))
            if True_vec[i] and not True_vec[i+1]: # leaving
                y1 = 100*(yy[params.C_H_ind,i]+yy[params.C_L_ind,i])
                y2 = 100*(yy[params.C_H_ind,i+1]+yy[params.C_L_ind,i+1])
                t1 = tt[i]
                t2 = tt[i+1]
                t_int = t1 + (t2- t1)* abs((100*params.ICU_capacity - y1)/(y2-y1)) 
                c_high.append(t_int) # 0.5 * ( tt[i] + tt[i+1]))
        


    if len(c_low)>len(c_high):
        c_high.append(tt[-1]+1)
    return c_low, c_high, ICU



########################################################################################################################
def preset_strat(preset):
    if preset in preset_dict_high:
        lr = params.fact_v[preset_dict_low[preset]]
        hr = params.fact_v[preset_dict_high[preset]]
    else:
        lr = 1
        hr = 1
    

    return lr, hr



# def strat_table(month,beta_H,beta_L,N_sols,i):
#         if N_sols==2:
#             if i==1:
#                 Value_string = 'Value, Strategy One'
#             else:
#                 Value_string = 'Value, Strategy Two'

#         else:
#             Value_string = 'Value'

        
#         return html.Div([html.Hr(),
#                         dbc.Row([dbc.Button('Infection Rate 🛈',
#                                     color='info',
#                                     className='mb-3',
#                                     size='sm',
#                                     id='popover-inf-tab-target',
#                                     style= {'cursor': 'pointer'}),
#                                     dbc.Button('Control Timings 🛈',
#                                     color='info',
#                                     className='mb-3',
#                                     size='sm',
#                                     id='popover-cont-tab-target',
#                                     style= {'cursor': 'pointer'})],justify='around'),
            
#                         dbc.Table(
#                         [
#                             html.Thead(
#                                 html.Tr([ 
#                                 html.Th("Strategy variable"),
#                                 html.Th(Value_string)
#                                 ])
#                                 ),
#                         ]
#                         +
#                         [ 
#                         html.Tbody([
#                                 html.Tr([ 
#                                     html.Td(html.H5(["High Risk Infection Rate "
#                                         ],
#                                         id="tooltip-hr",
#                                         style={'color': 'white', 'fontSize': '100%'})), # , "cursor": "pointer"
#                                     html.Td(html.H5('{0:,.0f}'.format(100*beta_H) + '%',style={'color': 'white', 'fontSize': '100%'}))
#                                     ]),
#                                 html.Tr([ 
#                                     html.Td(html.H5(["Low Risk Infection Rate",
#                                         ],style={'color': 'white', 'fontSize': '100%'})),
#                                     html.Td(html.H5('{0:,.0f}'.format(100*beta_L) + '%',style={'color': 'white', 'fontSize': '100%'}))
#                                 ]),
#                                 html.Tr([ 
#                                     html.Td(html.H5(['Control Starts',
#                                         ],style={'color': 'white', 'fontSize': '100%'})),
#                                     html.Td(html.H5('Month ' + str(month[0]),style={'color': 'white', 'fontSize': '100%'}))
#                                 ]),
#                                 html.Tr([ 
#                                     html.Td(html.H5(['Control Ends '
#                                     ],
#                                     id="tooltip-month",
#                                     style={'color': 'white', 'fontSize': '100%'})), # , "cursor": "pointer"
#                                     html.Td(html.H5('Month ' + str(month[1]),style={'color': 'white', 'fontSize': '100%'}))
#                                 ]),
#                         ]),
#                         ],
#                         bordered=True,
#                         dark=True,
#                         hover=True,
#                         responsive=True,
#                         striped=True,
#                     style={'margin-bottom': '2vh'}
#                     ),

#                     dbc.Popover(
#                         [
#                         dbc.PopoverHeader('Infection Rate'),
#                         dbc.PopoverBody(dcc.Markdown(
#                         '''

#                         The Infection Rate relates to how quickly the disease is transmitted. Control measures reduce transmission rates (typically lowering them).
                    
#                         *Use the '**Pick Your Strategy**' bar on the left to adjust by choosing a preset strategy or making your own custom choice.*

#                         '''
#                         ),),
#                         ],
#                         id = "popover-inf-tab",
#                         is_open=False,
#                         target="popover-inf-tab-target",
#                         placement='right',
#                     ),

#                     dbc.Popover(
#                         [
#                         dbc.PopoverHeader('Control Timings'),
#                         dbc.PopoverBody(dcc.Markdown(
#                         '''

#                         Use the '**Months of Control**' option in the '**Pick Your Strategy**' bar to adjust when we start controlling the epidemic.
                    
#                         When control is in place the infection rates are modified (by an amount depending on the choice of control).
                        
#                         *When control is not in place the infection rates remain at a baseline level (100%).*
#                         '''
#                         ),),
#                         ],
#                         id = "popover-cont-tab",
#                         is_open=False,
#                         target="popover-cont-tab-target",
#                         placement='right',
#                     ),



#                     ],
#                     style={'fontSize': '1.6vh'}
#                     )


########################################################################################################################
def extract_info(yy,tt,t_index):
###################################################################
    # find percentage deaths/critical care
    # if deaths:
    metric_val_L_3yr = yy[params.D_L_ind,t_index-1]
    metric_val_H_3yr = yy[params.D_H_ind,t_index-1]
    # else:
    #     metric_val_L_3yr = yy[params.C_L_ind,t_index-1]
    #     metric_val_H_3yr = yy[params.C_H_ind,t_index-1]

###################################################################
    ICU_val_3yr = [yy[params.C_H_ind,i] + yy[params.C_L_ind,i] for i in range(t_index)]
    ICU_val_3yr = max(ICU_val_3yr)/params.ICU_capacity

###################################################################
    # find what fraction of herd immunity safe threshold reached
    herd_val_3yr = [yy[params.S_H_ind,i] + yy[params.S_L_ind,i] for i in range(t_index)]
    
    herd_lim = 1/(params.R_0)

    herd_fraction_out = min((1-herd_val_3yr[-1])/(1-herd_lim),1)

###################################################################
    # find time ICU capacity exceeded

    time_exc = 0

    # if True:
    c_low, c_high, ICU = time_exceeded_function(yy,tt)

    time_exc = [c_high[jj] - c_low[jj] for jj in range(1,len(c_high)-1)]
    time_exc = sum(time_exc)
    
    if c_high[-1]>0:
        if c_high[-1]<=tt[-1]:
            time_exc = time_exc + c_high[-1] - c_low[-1]
        else:
            time_exc = time_exc + tt[-1] - c_low[-1]
    time_exc = time_exc/month_len




###################################################################
# find herd immunity time till reached

    multiplier_95 = 0.95
    threshold_herd_95 = (1-multiplier_95) + multiplier_95*herd_lim

    time_reached = 50 # i.e never reached unless below satisfied
    if herd_val_3yr[-1] < threshold_herd_95:
        herd_time_vec = [tt[i] if herd_val_3yr[i] < threshold_herd_95 else 0 for i in range(len(herd_val_3yr))]
        herd_time_vec = np.asarray(herd_time_vec)
        time_reached  = min(herd_time_vec[herd_time_vec>0])/month_len
        
    return metric_val_L_3yr, metric_val_H_3yr, ICU_val_3yr, herd_fraction_out, time_exc, time_reached





########################################################################################################################
def human_format(num):
    if num<1 and num>=0.1:
        return '%.1f' % num
    elif num<0.1:
        return '%.3f' % num
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G'][magnitude])


########################################################################################################################
def figure_generator(sols,month,output,groups,num_strat,groups2,ICU_to_plot=False,vaccine_time=None): # hosp

    font_size = 14
    
    lines_to_plot = []
    ymax = 0

    names = ['S','I','R','H','C','D']
    
    
    if num_strat=='one':
        group_use = groups
    if num_strat=='two':
        group_use = groups2


    group_string = str()
    for group in group_vec:
        if group in group_use:
            group_string = group_string + ',' + group_strings[group]
    

    linestyle_numst = ['solid','dash','dot','dashdot','longdash','longdashdot']
    
    len_data_points = len(sols[0]['t'])
    len_to_plot = ceil(len_data_points) # *years/3)
    if len(sols)>1:
        strat_list = [': Strategy;',': Do Nothing;']
    else:
        strat_list = [':']

    ii = -1
    for sol in sols:
        ii += 1
        for name in names:
            if name in output:
                for group in group_vec:
                    
                    if group in group_use:
                        sol['y'] = np.asarray(sol['y'])
                        if num_strat=='one':
                            name_string = strat_list[ii] + group_strings[group] # ':' +
                            line_style_use = linestyle[group]
                        if num_strat=='two':
                            name_string = ': Strategy ' + str(ii+1) + '; ' + group_strings[group]
                            line_style_use = linestyle_numst[ii]
                        
                        xx = [i/month_len for i in sol['t']]
                        if group=='BR':
                            group_hover_str = ''
                        elif group=='HR':
                            group_hover_str = 'High Risk' + '<br>'
                        else:
                            group_hover_str = 'Low Risk' + '<br>'

                        line =  {'x': xx[:len_to_plot], 'y': (100*factor_L[group]*sol['y'][index[name],:len_to_plot] + 100*factor_H[group]*sol['y'][index[name] + params.number_compartments,:len_to_plot]),
                                'hovertemplate': group_hover_str +
                                                 longname[name] + ': %{y:.2f}%<br>' +
                                                 'Time: %{x:.1f} Months<extra></extra>',
                                'line': {'color': str(colors[name]), 'dash': line_style_use }, 'legendgroup': name ,'name': longname[name] + name_string}
                        lines_to_plot.append(line)


        # setting up pink boxes
        ICU = False
        # print(ii,num_strat,group_use,output)
        if ii==0 and num_strat=='one' and len(group_use)>0 and len(output)>0: # 'True_deaths' in hosp 
            yyy = sol['y']
            ttt = sol['t']
            c_low, c_high, ICU = time_exceeded_function(yyy,ttt)

    for line in lines_to_plot:
        ymax = max(ymax,max(line['y']))

    

    if ymax<100:
        ymax2 = min(1.1*ymax,100)
    else:
        ymax2 = 100

    yax = dict(range= [0,ymax2])
    ##

    annotz = []
    shapez = []

    if month[0]!=month[1]:
        shapez.append(dict(
                # filled Blue Control Rectangle
                type="rect",
                x0= month[0], #month_len*
                y0=-0.01,
                x1= month[1], #month_len*
                y1=yax['range'][1],
                line=dict(
                    color="LightSkyBlue",
                    width=0,
                ),
                fillcolor="LightSkyBlue",
                opacity=0.25
            ))
            
    if ICU:
        # if which_plots=='two':
        control_font_size = font_size*(16/24) # '10em'
        ICU_font_size = font_size*(16/24) # '10em'

        yval_pink = 0.3
        yval_blue = 0.82


        for c_min, c_max in zip(c_low, c_high):
            if c_min>0 and c_max>0:
                shapez.append(dict(
                        # filled Pink ICU Rectangle
                        type="rect",
                        x0=c_min/month_len,
                        y0=-0.01,
                        x1=c_max/month_len,
                        y1=yax['range'][1],
                        line=dict(
                            color="pink",
                            width=0,
                        ),
                        fillcolor="pink",
                        opacity=0.5,
                        xref = 'x',
                        yref = 'y'
                    ))
                annotz.append(dict(
                        x  = 0.5*(c_min+c_max)/month_len,
                        y  = yval_pink,
                        text="ICU<br>" + " Capacity<br>" + " Exceeded",
                        # hoverinfo='ICU Capacity Exceeded',
                        showarrow=False,
                        textangle= 0,
                        font=dict(
                            size= ICU_font_size,
                            color="purple"
                        ),
                        opacity=0.7,
                        xref = 'x',
                        yref = 'paper',
                ))

    else:
        control_font_size = font_size*(30/24) #'11em'
        yval_blue = 0.4




    if month[0]!=month[1]:
        annotz.append(dict(
                x  = max(0.5*(month[0]+month[1]), 0.5),
                y  = yval_blue,
                text="Control<br>" + " In <br>" + " Place",
                # hoverinfo='Control In Place',
                textangle=0,
                font=dict(
                    size= control_font_size,
                    color="blue"
                ),
                showarrow=False,
                opacity=0.5,
                xshift= 0,
                xref = 'x',
                yref = 'paper',
        ))
    

            
    if ICU_to_plot:
        lines_to_plot.append(
        dict(
        type='scatter',
            x=[-0.01,max(sol['t'])+1], y=[params.ICU_capacity,params.ICU_capacity],
            mode='lines',
            opacity=0.8,
            legendgroup='thresholds',
            line=dict(
            color= 'maroon',
            dash = 'dash'
            ),
            hovertemplate= 'ICU Capacity: %{y}',
            name= 'ICU Capacity'))

    if vaccine_time is not None:
        lines_to_plot.append(
        dict(
        type='scatter',
            x=[vaccine_time,vaccine_time], y=[yax['range'][0],yax['range'][1]],
            mode='lines',
            opacity=0.9,
            legendgroup='thresholds',
            line=dict(
            color= 'royalblue',
            dash = 'dash'
            ),
            hovertemplate= 'Vaccination starts',
            name= 'Vaccination starts'))

    

    lines_to_plot.append(
    dict(
        type='scatter',
        x = [0,sol['t'][-1]],
        y = [ 0, params.UK_population],
        yaxis="y2",
        opacity=0,
        showlegend=False
    ))
    

    
    yy2 = [0, 10**(-6), 2*10**(-6), 5*10**(-6), 10**(-5), 2*10**(-5), 5*10**(-5), 10**(-4), 2*10**(-4), 5*10**(-4), 10**(-3), 2*10**(-3), 5*10**(-3), 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 200]
    yy = [0.95*i for i in yy2]


    for i in range(len(yy)-1):
        if yax['range'][1]>=yy[i] and yax['range'][1] < yy[i+1]:
            yax2_vec = np.linspace(0,yy2[i+1],11)

    vec = [i*(params.UK_population) for i in yax2_vec]


    # yax_form_log = '.4%'
    # yax2_vec_log = np.linspace(-100,100,20)
    #[100*np.log10(i) for i in np.linspace(10**(-10), yax['range'][1] ,6)] # np.log10(i)
    # print(log_range,yax2_vec_log)
    # if yax['range'][1]>100:
    #     yax['range'][1] = 100
    log_bottom = -8
    log_range = [log_bottom,np.log10(yax['range'][1])]
    
    pop_log_vec = [10**(i) for i in np.linspace(log_bottom,np.log10(yax['range'][1]),6)]

    vec2 = [i*(params.UK_population) for i in pop_log_vec]
    # print(vec2)


    start = int(datetime.date.today().strftime('%m'))
    month_labels = []
    for j in range(4):
        for i in range(1,13):
            month_labels.append(datetime.date(2020+j, i, 1).strftime('%b %y'))
    
    month_labels = month_labels[(start-1):(start-1+37)]

    xtext = [str(month_labels[2*i]) for i in range(1+floor(max(sol['t'])/(2*month_len)))]
    xvals = [2*i for i in range(1+floor(max(sol['t'])/(2*month_len)))]
    # if max(sol['t'])>370:
    # else:
    #     xtext = [str(month_labels(2*i)) for i in range(1+floor(max(sol['t'])/month_len))]
    #     xvals = [ i for i in range(1+floor(max(sol['t'])/month_len))]
    
    tick_form = ['%','.1%','.2%','.3%','.4%','.5%','.6%','.7%','.8%']
    upper_lim = [2,0.1,10**(-2),10**(-3),10**(-4),10**(-5),10**(-6),10**(-7),10**(-8)]
    yax_form = None
    for i in range(len(tick_form)):
        if yax['range'][1]<upper_lim[i]:
            yax_form = tick_form[i]
    if yax_form is None:
        yax_form = '.9%'




    layout = go.Layout(
                    annotations=annotz,
                    shapes=shapez,
                    template="simple_white",
                    font = dict(size= font_size), #'12em'),
                    legend = dict(
                       font=dict(size=font_size*(16/24)),#'8em'),
                       x = 0.5,
                       y = 1.02,
                       xanchor= 'center',
                       yanchor= 'bottom',
                   ),
                   legend_orientation = 'h',
                   legend_title='<b> Key <b>',

                   yaxis= dict(mirror= True,
                        title='Proportion of Population',
                        range= yax['range'],
                        showline=False,
                        # linewidth=0,
                        # tickformat=yax_form
                   ),
                #    yaxis_tickformat = yax_form,

                   xaxis= dict(
                        title='Time (Months)',
                        range= [0, (2/3)*max(sol['t'])/month_len],
                        showline=False,
                        # rangeslider_visible=True,
                        # linewidth=0,
                        ticktext = xtext,
                        tickvals = xvals
                       ),
                    
                    yaxis2 = dict(
                        title = 'UK Population',
                        overlaying='y1',
                        showline=False,
                        # linewidth=0,
                        # showgrid=True,
                        range = yax['range'],
                        side='right',
                        ticktext = [human_format(0.01*vec[i]) for i in range(len(yax2_vec))],
                        tickvals = [i for i in  yax2_vec],
                   ),
                    updatemenus= [
                    dict(
                        buttons=list([
                            dict(
                                args=[{"yaxis": {'title': 'Percentage of Population', 'type': 'linear', 'range': yax['range'], 'showline':False},
                                "yaxis2": {'title': 'UK Population','type': 'linear', 'overlaying': 'y1', 'range': yax['range'], 'ticktext': [human_format(0.01*vec[i]) for i in range(len(yax2_vec))], 'tickvals': [i for i in  yax2_vec],'showline':False,'side':'right'}
                                }], # tickformat
                                label="Linear",
                                method="relayout"
                            ),
                            dict(
                                args=[{"yaxis": {'title': 'Percentage of Population', 'type': 'log', 'range': log_range,'showline':False},
                                "yaxis2": {'title': 'UK Population','type': 'log', 'overlaying': 'y1', 'range': log_range, 'ticktext': [human_format(0.01*vec2[i]) for i in range(len(pop_log_vec))], 'tickvals': [i for i in  pop_log_vec],'showline':False,'side':'right'}
                                }], # 'tickformat': yax_form_log,
                                label="Logarithmic",
                                method="relayout"
                            )
                    ]),
                    x=0,
                    xanchor="right",
                    active=0,
                    y=1.2,
                    showactive=True,
                    yanchor="top"
                    ),
                    dict(
                        buttons=list([
                            dict(
                                args = ["xaxis", {'range': [0, (1/3)*max(sol['t'])/month_len] , 'ticktext': [str(month_labels[i]) for i in range(1+floor(max(sol['t'])/(month_len)))], 'tickvals': [i for i in range(1+floor(max(sol['t'])/(month_len)))], 'title': 'Time (Months)', 'showline':False ,}],
                                label="Years: 1",
                                method="relayout"
                            ),
                            dict(
                                args = ["xaxis", {'range': [0, (2/3)*max(sol['t'])/month_len] , 'ticktext': [str(month_labels[2*i]) for i in range(1+floor(max(sol['t'])/(2*month_len)))], 'tickvals': [2*i for i in range(1+floor(max(sol['t'])/(2*month_len)))], 'title': 'Time (Months)', 'showline':False ,}], #  
                                label="Years: 2",
                                method="relayout"
                            ),
                            dict(
                                args = ["xaxis", {'range': [0, (3/3)*max(sol['t'])/month_len] , 'ticktext': [str(month_labels[3*i]) for i in range(1+floor(max(sol['t'])/(3*month_len)))], 'tickvals': [3*i for i in range(1+floor(max(sol['t'])/(3*month_len)))], 'title': 'Time (Months)', 'showline':False ,}],
                                label="Years: 3",
                                method="relayout"
                            )
                    ]),
                    x=1,
                    xanchor="right",
                    showactive=True,
                    active=1,
                    direction='up',
                    y=-0.2,
                    yanchor="top"
                    ),
                    ],
                   )
                   

    return {'data': lines_to_plot, 'layout': layout}




########################################################################################################################


def cards_fn(death_stat_1st,dat3_1st,herd_stat_1st,color_1st_death,color_1st_herd,color_1st_ICU):
    return html.Div([

                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                        [
                            dbc.CardHeader(
                                        ['Reduction in deaths:']
                                ),
                            dbc.CardBody([html.H1(str(round(death_stat_1st,1))+'%',className='card-title',style={'fontSize': '150%'})]),
                            dbc.CardFooter('compared to doing nothing'),

                        ],color=color_1st_death,inverse=True
                    )
                    ],width=4,style={'textAlign': 'center'}),
    

                    dbc.Col([
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                        ['ICU requirement:']
                                        # style={"textDecoration": "underline"},
                                # ),
                                # id="tooltip-ICU",
                                # style= {"cursor": "pointer"}
                                ),
                            dbc.CardBody([html.H1(str(round(dat3_1st,1)) + 'x',className='card-title',style={'fontSize': '150%'})],),
                            dbc.CardFooter('multiple of current capacity'),

                        ],color=color_1st_ICU,inverse=True
                    )
                    ],width=4,style={'textAlign': 'center'}),


                    dbc.Col([
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                # html.Span(
                                        ['Herd immunity:']

                                ),
                            dbc.CardBody([html.H1(str(round(herd_stat_1st,1))+'%',className='card-title',style={'fontSize': '150%'})]),
                            dbc.CardFooter('of safe threshold'),

                        ],color=color_1st_herd,inverse=True
                    )
                    ],width=4,style={'textAlign': 'center'}),
                    
        ],
        no_gutters=True),
    
    # ],
    # width=True)

    ],style={'margin-top': '2vh', 'margin-bottom': '2vh','fontSize':'75%'})




def outcome_fn(month,beta_L,beta_H,death_stat_1st,herd_stat_1st,dat3_1st,death_stat_2nd,herd_stat_2nd,dat3_2nd,preset,number_strategies,which_strat): # hosp
    
    death_stat_1st = 100*death_stat_1st
    herd_stat_1st = 100*herd_stat_1st

    death_stat_2nd = 100*death_stat_2nd
    herd_stat_2nd = 100*herd_stat_2nd


    on_or_off = {'display': 'block','textAlign': 'center'}
    if number_strategies=='one':
        num_st = ''
        if which_strat==2:
            on_or_off = {'display': 'none'}
    else:
        num_st = 'One '
    strat_name = presets_dict[preset]

    if which_strat==1:
        Outcome_title = strat_name + ' Strategy ' + num_st
    else:
        Outcome_title = strat_name + ' Strategy Two'
    



    death_thresh1 = 66
    death_thresh2 = 33

    herd_thresh1 = 66
    herd_thresh2 = 33

    ICU_thresh1 = 5
    ICU_thresh2 = 10

    color_1st_death = 'success'
    if death_stat_1st<death_thresh1:
        color_1st_death = 'warning'
    if death_stat_1st<death_thresh2:
        color_1st_death = 'danger'

    color_1st_herd = 'success'
    if herd_stat_1st<herd_thresh1:
        color_1st_herd = 'warning'
    if herd_stat_1st<herd_thresh2:
        color_1st_herd = 'danger'

    color_1st_ICU = 'success'
    if dat3_1st>ICU_thresh1:
        color_1st_ICU = 'warning'
    if dat3_1st>ICU_thresh2:
        color_1st_ICU = 'danger'

    
    color_2nd_death = 'success'
    if death_stat_2nd<death_thresh2:
        color_2nd_death = 'warning'
    if death_stat_2nd<death_thresh1:
        color_2nd_death = 'danger'

    color_2nd_herd = 'success'
    if herd_stat_2nd<herd_thresh1:
        color_2nd_herd = 'warning'
    if herd_stat_2nd<herd_thresh2:
        color_2nd_herd = 'danger'

    color_2nd_ICU = 'success'
    if dat3_2nd>ICU_thresh1:
        color_2nd_ICU = 'warning'
    if dat3_2nd>ICU_thresh2:
        color_2nd_ICU = 'danger'




    if on_or_off['display']=='none':
        return None
    else:
        return html.Div([


                
                    dbc.Row([
                        html.H3(Outcome_title,style={'fontSize':'200%'}),
                    ],
                    justify='center'
                    ),
                    html.Hr(),


                    dbc.Row([
                        html.P('In the absence of a vaccine, when compared to doing nothing.', style={'fontSize': '100%'}),
                    ],
                    justify='center', style={'margin-top': '1vh', 'margin-bottom': '1vh'}
                    ),

            
            # dbc
            dbc.Row([

            
                dbc.Col([


                                html.H3('After 1 year:',style={'fontSize': '180%'}),

                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button('Reduction in deaths 🛈',
                                                    color='info',
                                                    className='mb-3',
                                                    id="popover-red-deaths-target",
                                                    size='sm',
                                                    style = {'cursor': 'pointer', 'margin': '0px'}),
                                                    dbc.Popover(
                                                        [
                                                        dbc.PopoverHeader('Reduction in deaths'),
                                                        dbc.PopoverBody(dcc.Markdown(
                                                        '''

                                                        This box shows the reduction in deaths due to the control strategy choice.

                                                        '''
                                                        ),),
                                                        ],
                                                        id = "popover-red-deaths",
                                                        is_open=False,
                                                        target="popover-red-deaths-target",
                                                        placement='top',
                                                    ),
                                    ],width=4,style={'textAlign': 'center'}),

                                    dbc.Col([

                                                    dbc.Button('ICU requirement 🛈',
                                                    color='info',
                                                    className='mb-3',
                                                    size='sm',
                                                    id='popover-ICU-target',
                                                    style={'cursor': 'pointer', 'margin': '0px'}
                                                    ),

                                                    
                                                    dbc.Popover(
                                                        [
                                                        dbc.PopoverHeader('ICU requirement'),
                                                        dbc.PopoverBody(dcc.Markdown(
                                                        '''

                                                        COVID-19 can cause a large number of serious illnesses very quickly. This box shows the extent to which the NHS capacity would be overwhelmed by the strategy choice (if nothing was done to increase capacity).

                                                        '''
                                                        ),),
                                                        ],
                                                        id = "popover-ICU",
                                                        is_open=False,
                                                        target="popover-ICU-target",
                                                        placement='top',
                                                    ),
                                    ],width=4,style={'textAlign': 'center'}),
                                    
                                    dbc.Col([

                                                    dbc.Button('Herd immunity 🛈',
                                                    color='info',
                                                    className='mb-3',
                                                    size='sm',
                                                    id='popover-herd-target',
                                                    style={'cursor': 'pointer', 'margin': '0px'}
                                                    ),               
                                                                        
                                                    dbc.Popover(
                                                        [
                                                        dbc.PopoverHeader('Herd immunity'),
                                                        dbc.PopoverBody(dcc.Markdown(
                                                        '''

                                                        This box shows how close to the safety threshold for herd immunity we got. If we reached (or exceeded) the threshold it will say 100%.
                                                        
                                                        However, this is the least important goal since an uncontrolled pandemic will reach safe levels of immunity very quickly, but cause lots of serious illness in doing so.
                                                        '''
                                                        ),),
                                                        ],
                                                        id = "popover-herd",
                                                        is_open=False,
                                                        target="popover-herd-target",
                                                        placement='top',
                                                    ),
                                ],width=4,style={'textAlign': 'center'}),

                                ],no_gutters=True),
                    
                                cards_fn(death_stat_1st,dat3_1st,herd_stat_1st,color_1st_death,color_1st_herd,color_1st_ICU),

                                html.H3('After 2 years:',style={'fontSize': '180%'}),

                                cards_fn(death_stat_2nd,dat3_2nd,herd_stat_2nd,color_2nd_death,color_2nd_herd,color_2nd_ICU),


                ],
                width=12,
                ),


            ],
            align='center',
            ),

            ],style=on_or_off)






########################################################################################################################
layout_intro = html.Div([
    dbc.Row([
    dbc.Col([
                            dbc.Jumbotron([##
                                html.Div( [

    dbc.Tabs(id='intro-tabs',
             active_tab='tab_start',
             children = [
                
        
        dbc.Tab(label='Start Here', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_start', children=[
        html.H4('Introduction',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),
        dcc.Markdown('''
        The aim of this website is to demystify modelling of infectious diseases through short videos and interactive models which let you explore how different control strategies will affect the rate that covid-19 spreads. The website has been developed by experts in epidemiology and modelling from the University of Cambridge.
        '''),
        #  Nick and Cerian are based in the Department of Plant Sciences and Daniel is based at the Institute of Astronomy.
        
        html.H4('Who is this website for?',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),
        dcc.Markdown('''
        The content is targeted at people with little or no experience of modelling and might be used as a resource for anyone wishing to understand more about the standstill.
        '''),
        
        html.H4('How to use the website',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),
        dcc.Markdown('''
        It’s up to you how you explore our website. If you already know something about modelling you may want to jump straight to the interactive model. But if you’d like to know a little more about modelling and a detailed explanation of the output then click on the next tabs in this section.
        '''),
        


        ]),
        
        dbc.Tab(label='Introduction to modelling', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_0', children=[

            html.H3('Introduction to mathematical modelling',className='display-4',
            style = {'margin-top': '1vh', 'fontSize': '300%'}),

            html.Hr(),

            dcc.Markdown('''
            Watch this video from Dr Cerian Webb, an expert in epidemiology and modelling from the University of Cambridge.
            '''),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/v1585755877/WhatIsModellingv2_hhqe2h.mp4',
                    controls=True,
                    style={'max-width':'90%','height': 'auto','margin-top': '1vh','margin-bottom': '1vh'}),
                    
                    ],
                    justify='center'
                    ),
            
                    

            
            html.Hr(),


            html.H3('Introducing SIR models',className='display-4',
            style = {'margin-top': '1vh', 'fontSize': '300%'}),

            html.Hr(),

            dcc.Markdown('''
            Watch this explanation from Dr Cerian Webb, to find out more about basic epidemiological models.
            '''),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/v1585814499/StandardSIRModel_hu5ztn.mp4',
                    controls=True,
                    style={'max-width':'90%','height': 'auto','margin-top': '1vh','margin-bottom': '1vh'}),
                    
                    ],
                    justify='center'
                    ),
            
                    

            
            html.Hr(),
            
            html.H3('Definitions',className='display-4',
            style = {'margin-top': '1vh', 'fontSize': '300%'}),

            dbc.Col([
            dcc.Markdown('''            

            There are **two vital concepts** that you need to understand before we can fully explore how the control measures work.
            '''),
            
            html.H4('1. Basic Reproduction Number',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),

            dcc.Markdown('''            
            Any infectious disease requires both infectious individuals and susceptible individuals to be present in a population to spread. The higher the number of susceptible individuals, the faster it can spread since an infectious person can spread the disease to more susceptible people before recovering.

            The average number of infections caused by a single infected person is known as the '**basic reproduction number**' (*R*). If this number is less than 1 (each infected person infects less than one other on average) then the disease won't spread. If it is greater than 1 then the disease will spread. For COVID-19 most estimates for *R* are between 2 and 3. We use the value *R*=2.4.
            '''),

            html.H4('2. Herd Immunity',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),
            
            dcc.Markdown('''            


            Once the number of susceptible people drops below a certain threshold (which is different for every disease, and in simpler models depends on the basic reproduction number), the population is no longer at risk of an epidemic (so any new infection introduced won't cause infection to spread through an entire population).

            Once the number of susceptible people has dropped below this threshold, the population is termed to have '**herd immunity**'. Herd immunity is either obtained through sufficiently many individuals catching the disease and developing personal immunity to it, or by vaccination.

            For COVID-19, there is a safe herd immunity threshold of around 60% (=1-1/*R*), meaning that if 60% of the population develop immunity then the population is **safe** (no longer at risk of an epidemic).

            Coronavirus is particularly dangerous because most countries have almost 0% immunity since the virus is so novel. Experts are still uncertain whether you can build immunity to the virus, but the drop in cases in China would suggest that you can. Without immunity it would be expected that people in populated areas get reinfected, which doesn't seem to have happened.
            
            A further concern arises over whether the virus is likely to mutate. However it is still useful to consider the best way to managing each strain.
            '''),

            
            ],
            width = True), # {'size': 10}),

    #end of tab 2
        ]),

        dbc.Tab(label='COVID-19 Control Strategies', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_control', children=[
            html.H4('Keys to a successful control strategy',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),

            dcc.Markdown('''            
            There are three main goals a control strategy sets out to achieve:

            1. Reduce the number of deaths caused by the pandemic,

            2. Reduce the load on the healthcare system,

            3. Ensure the safety of the population in future.

            An ideal strategy achieves all of the above whilst also minimally disrupting the daily lives of the population.

            However, controlling COVID-19 is a difficult task, so there is no perfect strategy. We will explore the advantages and disadvantages of each strategy.
            '''),
            
            html.Hr(),

            html.H4('Strategies',
            className='display-4',
            style = {'margin-top': '1vh', 'fontSize': '210%'}),
            
            html.H4('Reducing the infection rate',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),


            dcc.Markdown('''            

            Social distancing, self isolation and quarantine strategies slow the rate of spread of the infection (termed the 'infection rate'). In doing so, we can reduce the load on the healthcare system (goal 2) and (in the short term) reduce the number of deaths.

            This has been widely referred to as 'flattening the curve'; buying nations enough time to bulk out their healthcare capacity. The stricter quarantines are the best way to minimise the death rate whilst they're in place. A vaccine can then be used to generate sufficient immunity.

            However, in the absence of a vaccine these strategies do not ensure the safety of the population in future (goal 3), meaning that the population is still highly susceptible and greatly at risk of a future epidemic. This is because these strategies do not lead to any significant level of immunity within the population, so as soon as the measures are lifted the epidemic restarts. Further, strict quarantines carry a serious economic penalty.

            COVID-19 spreads so rapidly that it is capable of quickly generating enough seriously ill patients to overwhelm the intensive care unit (ICU) capacity of most healthcase systems in the world. This is why most countries have opted for strategies that slow the infection rate. It is essential that the ICU capacity is vastly increased to ensure it can cope with the number of people that may require it.
            '''),


            html.H4('Protecting the high risk',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),

            


            dcc.Markdown('''            
            One notable feature of COVID-19 is that it puts particular demographics within society at greater risk. The elderly and the immunosuppressed are particularly at risk of serious illness caused by coronavirus.

            The **interactive model** presented here is designed to show the value is protecting the high risk members of society. It is critically important that the high risk don't catch the disease.

            If 60% of the population catch the disease, but all are classified as low risk, then very few people will get seriously ill through the course of the epidemic. However, if a mixture of high and low risk individuals catch the disease, then many of the high risk individuals will develop serious illness as a result.


            ''',
            ),

        ]),


        dbc.Tab(label='Control Case Study', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_3', children=[
            



            html.Div([
                
            html.H3('Example control strategy',className='display-4',
            style = {'margin-top': '1vh', 'fontSize': '300%'}),

            html.Hr(className='my-2'),

            dbc.Col([

            html.H4("Protecting high risk (quarantine), no change for low risk vs 'do nothing'",
            style = {'margin-top': '1vh', 'fontSize': '180%'}),

            dcc.Markdown('''

            In the absence of a vaccine, the most effective way to protect high risk people is to let the low risk people get the disease and recover, increasing the levels of [**herd immunity**](/intro) within the population (and therefore ensuring future safety for the population) but minimally affecting the hospitalisation rate.

            However, this is a risky strategy. There are very few ICU beds per 100,000 population in most healthcare systems. This means that any level of infection in the system increases the risk of an epidemic that could cause too many patients to need critical care.

            A full quarantine would lead to a lower hospitalisation rate in the short term but more deaths in the long term without a vaccine. But with a vaccine the long term death rate can be minimised with a full quarantine followed by a widespread vaccination programme.

            ''',
            # style={'fontSize':'2vh'}
            ),
            
            dbc.Col([
            dcc.Graph(id='line-plot-intro',style={'height': '50vh', 'display': 'block', 'width': '100%'}),
            html.Div(style={'height': '1vh'}),
            
            # dbc.Container([

            dcc.Graph(id='line-plot-intro-2',
            style={'height': '50vh', 'display': 'block', 'width': '100%'}),
            ],
            width=True
            ),

            dcc.Markdown('''

            The dotted lines in these plots represents the outcome if not control is implemented. The solid lines represent the outcome if we protect the high risk (quarantine) and allow low risk to social distance. The result is a 92% reduction in deaths.

            The control is in place for 9 months, starting after a month.

            To simulate more strategies, press the button below!

            ''',
            # style={'fontSize':'2vh'}
            ),

            # dbc.Col([
            dbc.Button('Start Calculating', href='/inter', size='lg', color='success'
            ,style={'margin-top': '1vh','margin-left': '2vw', 'fontSize': '100%'}
            ),
            ],
            # width={'size':3,'offset':1},
            width = True, # {'size': 12},
            # style = {'margin-left':'1vh','margin-right':'1vh'}

            ),
            ],
            style={'margin-top': '1vh', 'margin-bottom': '1vh'}
            ),
            
    #end of tab 3
        ]),
        # dbc.Tab(label='Explanatory Videos', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_vids', children=[
                    
                    

        # ]),

        dbc.Tab(label='How to use', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_1', children=[
                    



                    # dbc.Container(html.Div(style={'height':'5px'})),
                    html.H3('How to use the interactive model',className='display-4',
                    style = {'margin-top': '1vh', 'fontSize': '300%'}),

                    # html.H4('How to use the interactive model',className='display-4'),
                    
                    html.Hr(className='my-2'),
                    
                    dbc.Col([
                    dcc.Markdown('''


                    We present a model parameterised for COVID-19. The interactive element allows you to predict the effect of different **control measures**.

                    We use **control** to mean an action taken to try to reduce the severity of the epidemic. In this case, control measures (e.g. social distancing and quarantine/lockdown) will affect the '**infection rate**' (the rate at which the disease spreads through the population).

                    Stricter measures (e.g. lockdown) have a more dramatic effect on the infection rate than less stringent measures.
                    
                    There are **two** ways to adjust control:

                    1. Adjust the infection rates from their baseline level,
                    
                    2. Adjust the time for which control is applied.

                    To start predicting the outcome of different strategies, press the button below!

                    ''',
                    # style={'fontSize':20}
                    ),
                    
                    # dbc.Col([
                    dbc.Button('Start Calculating', href='/inter', size='lg', color='success',
                    style={'margin-top': '1vh','margin-left': '2vw', 'fontSize': '100%'}
                    ),
                    # ],width={'size':3,'offset':1},
                    # ),
                    ],
                    style={'margin-top': '1vh'},
                    width = True),

                        #end of tab 1
                    ]),

            
    #end of tabs
    ])
],
style={'fontSize': '2vh'}
)

    ])##  # end of jumbotron
],
width=12,
xl=8
),],
justify='center')
])


Results_interpretation =  html.Div([
    
    dcc.Markdown('''

    The plots will show a prediction for how coronavirus will affect the population. It is assumed that control measures are in place for a **maximum of 15 months**.
    
    We consider the effect of control in the **absence** of a vaccine. Of course, if a vaccine were introduced this would greatly help reduce the damage caused by COVID-19, and would further promote the use of the quarantine strategy before relying on the vaccine to generate [**herd immunity**](/intro).

    You can see how quickly the ICU capacity (relating to the number of intensive care beds available) could be overwhelmed. You can also see how important it is to **protect the high risk group** (potentially by specifically reducing their transmission rate whilst allowing infection to spread more freely through lower risk groups).

    *In the custom settings, there is an option to increase infection rates. **Usually this is a bad idea**, but sometimes increasing rates in the low risk category can be beneficial by increasing the chance that the 'right' people get the infection before the population reaches herd immunity, and also reducing the length of time that control needs to be applied.*
                                                                                                
    *Increasing the infection rate could correspond to increased mixing or (more extreme) actively exposing people to the disease. This can only ever be a good idea if a vaccine is not forthcoming.*

    For further explanation, read the [**Background**](/intro).
    
    '''
    ,style={'fontSize': '100%'}
    ),
])



########################################################################################################################

 # column_1 = 
# inputs_col = 



# results_col = 






#########################################################################################################################################################

Instructions_layout = html.Div([html.H4("Instructions", className="display-4",style={'fontSize': '300%','textAlign': 'center','margin-top': '1vh'}),
                                             
                                                    html.Hr(),

                                                    dcc.Markdown('''

                                                    *In this Section we find a **prediction** for the outcome of your choice of strategy. **Strategy choice** involves choosing a means of **controlling** the disease.*

                                                    1. **Pick your strategy** (bar below)
                                                    
                                                    2. Choose which **results** to display (button below).

                                                    '''
                                                    ,style = {'margin-top': '2vh', 'textAlign': 'left'}
                                                    ),





                                                    
                                    ])





layout_inter = html.Div([
    dbc.Row([
        # column_1,
        




                        # end of first col



########################################################################################################################
## start row 1
# main_page = dbc.Row([
#                     column_1,

########################################################################################################################
                        # col 2
# column_2 =              
                        # dbc.Col([
                        html.Div([
                        html.Div([


                                    # store results
                                    dcc.Store(id='sol-calculated'),# storage_type='session'),
                                    dcc.Store(id='sol-calculated-do-nothing'),# storage_type='session'),
            
                                    # dbc.Col([

                                    # dbc.Jumbotron([
                                    # tabs
                                    dbc.Tabs(id="interactive-tabs", active_tab='tab_0', 
                                        children=[

                                        # tab 0
                                        dbc.Tab(label='Model Output',
                                         label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_0', 
                                         children = [
                                                    # html.Div([



                                                    # Instructions_layout,

                                                    # html.Hr(),

                                                    html.H4('Strategy Outcome',id='line_page_title',className="display-4",style={'fontSize': '300%','textAlign': 'center', 'margin-top': '2vh'}),

                                                    html.Hr(),

                                                    dcc.Markdown('''
                                                    *In this Section we find a **prediction** for the outcome of your choice of **COVID-19 control**. Pick a **strategy** and a **results type** below.*

                                                    '''
                                                    ,style = {'margin-top': '2vh', 'margin-bottom': '2vh', 'textAlign': 'center'}
                                                    ),
                                                    # 1. **Pick your strategy** (bar below)
                                                    
                                                    # 2. Choose which **results** to display (button below).
                                             
                                             
                                                    # html.Hr(),

                                                    # html.Hr(),

                                                dbc.Row([
            
                                                        dbc.Col([
                                                            dbc.Jumbotron([
                                                                



############################################################################################################################################################################################################################
                                                                                            html.Div([

                                                                                                        ##################################

                                                                                                                        # form group 1
                                                                                                                        dbc.FormGroup([

                                                                                ########################################################################################################################

                                                                                                                                                    dbc.Col([
                                                                                                                                                            


                                                                                                                                                            


                                                                                ########################################################################################################################


                                                                                                                                                            
                                                                                                                                                        dbc.Row([
                                                                                                                                                        dbc.Col([

                                                                                                                                                            dbc.Modal(
                                                                                                                                                                [
                                                                                                                                                                    dbc.ModalHeader("Interactive Model",style={'display': 'block','textAlign': 'center'}),
                                                                                                                                                                    dbc.ModalBody(
                                                                                                                                                                        dcc.Markdown(
                                                                                                                                                                        '''
                                                                                                                                                                        This page illustrates the outcome of different **COVID-19 control strategies**.

                                                                                                                                                                        **Pick a strategy** and explore the **results**.
                                                                                                                                                                        ''',style={'textAlign': 'center'}),
                                                                                                                                                                        ),
                                                                                                                                                                    dbc.ModalFooter(
                                                                                                                                                                        dbc.Button("Close", color='success', id='close-modal', className="ml-auto")
                                                                                                                                                                        ,style={'display': 'block','textAlign': 'center'}
                                                                                                                                                                    ),
                                                                                                                                                                ],
                                                                                                                                                                id="modal",
                                                                                                                                                                size="md",
                                                                                                                                                                # backdrop='static',
                                                                                                                                                                is_open=True,
                                                                                                                                                                centered=True
                                                                                                                                                            ),

                                                                                                                                                            html.H4('1. Pick Your Strategy ',style={'fontSize': '180%', 'color': 'blue' ,'margin-top': "3vh"}),

                                                                                                                                                            dbc.ButtonGroup([


                                                                                                                                                            dbc.Button('Instructions 🛈',
                                                                                                                                                            color='info',
                                                                                                                                                            # className='mb-3',
                                                                                                                                                            id="popover-pick-strat-target",
                                                                                                                                                            size='md',
                                                                                                                                                            style = {'cursor': 'pointer', 'margin-bottom': '0.5vh'}),


                                                                                                                                                            dbc.Button('1a. Control type 🛈',
                                                                                                                                                                color='info',
                                                                                                                                                                # className='mb-3',
                                                                                                                                                                size='sm',
                                                                                                                                                                id='popover-control-target',
                                                                                                                                                                style={'cursor': 'pointer','margin-bottom': '0.5vh'}
                                                                                                                                                                ),
                                                                                                                                                            
                                                                                                                                                            dbc.Button('1b. Months of control 🛈',
                                                                                                                                                            color='info',
                                                                                                                                                            # className='mb-3',
                                                                                                                                                            size='sm',
                                                                                                                                                            id='popover-months-control-target',
                                                                                                                                                            style= {'cursor': 'pointer','margin-bottom': '0.5vh'}),

                                                                                                                                                            dbc.Button('1c. Vaccination 🛈',
                                                                                                                                                            color='info',
                                                                                                                                                            # className='mb-3',
                                                                                                                                                            size='sm',
                                                                                                                                                            id='popover-vaccination-target',
                                                                                                                                                            style= {'cursor': 'pointer','margin-bottom': '0.5vh'})

                                                                                                                                                            ],
                                                                                                                                                            vertical=True),


                                                                                                                                                            dcc.Markdown('''*Choose the type of control and when to implement it.*''', style = {'fontSize': '80%'}), # 'textAlign': 'left', 
                                                                                                                                                            


                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Pick Your Strategy'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                1a. Pick the **type of control**.

                                                                                                                                                                1b. Pick the **control timings** (how long control is applied for and when it starts).

                                                                                                                                                                1c. Introduce a **vaccine** if you would like.

                                                                                                                                                                *The other options below are optional custom choices that you may choose to investigate further or ignore altogether*.

                                                                                                                                                                *Click the button to dismiss*.

                                                                                                                                                                '''
                                                                                                                                                                ),),
                                                                                                                                                                ],
                                                                                                                                                                id = "popover-pick-strat",
                                                                                                                                                                target="popover-pick-strat-target",
                                                                                                                                                                is_open=False,
                                                                                                                                                                placement='right',
                                                                                                                                                            ),

                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Control'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                The type of **control** determines how much we can reduce the **infection rate** of the disease (how quickly the disease is transmitted between people).
                                                                                                                                                                
                                                                                                                                                                We consider control of **two risk groups**; high risk and low risk. High risk groups are more likely to get seriously ill if they catch the disease.

                                                                                                                                                                *For further explanation, read the [**Background**](/intro)*.

                                                                                                                                                                '''
                                                                                                                                                                ),),
                                                                                                                                                                ],
                                                                                                                                                                id = "popover-control",
                                                                                                                                                                is_open=False,
                                                                                                                                                                target="popover-control-target",
                                                                                                                                                                placement='right',
                                                                                                                                                            ),

                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Control Timing'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                Use this slider to determine when control **starts** and **finishes**.

                                                                                                                                                                When control is in place the infection rate is reduced by an amount depending on the strategy choice.

                                                                                                                                                                When control is not in place the infection rate returns to the baseline level (100%).
                                                                                                                                                                
                                                                                                                                                                '''
                                                                                                                                                                ),),
                                                                                                                                                                ],
                                                                                                                                                                id = "popover-months-control",
                                                                                                                                                                is_open=False,
                                                                                                                                                                target="popover-months-control-target",
                                                                                                                                                                placement='right',
                                                                                                                                                            ),
                                                                                                                                                            


                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Vaccination'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                We assume a vaccine will not be available for 12 months.
                                                                                                                                                                
                                                                                                                                                                See how the introduction of a vaccine can drastically reduce the death toll if a sufficiently small proportion of the population have been infected.

                                                                                                                                                                '''
                                                                                                                                                                ),),
                                                                                                                                                                ],
                                                                                                                                                                id = "popover-vaccination",
                                                                                                                                                                is_open=False,
                                                                                                                                                                target="popover-vaccination-target",
                                                                                                                                                                placement='right',
                                                                                                                                                            ),



                                                                                                                                                            html.H6([
                                                                                                                                                                '1a. Control Type ',
                                                                                                                                                                ],
                                                                                                                                                                style={'fontSize': '120%','margin-top': '1vh', 'margin-bottom': '1vh'}),
                                                                                                                                                            

                                                                                                                                                            

                                                                                                                                                            html.Div([
                                                                                                                                                                dbc.RadioItems(
                                                                                                                                                                    id = 'preset',
                                                                                                                                                                    options=[{'label': presets_dict[key],
                                                                                                                                                                    'value': key} for key in presets_dict],
                                                                                                                                                                    value= 'MSD',
                                                                                                                                                                ),
                                                                                                                                                            ],
                                                                                                                                                            style={'fontSize': '80%'},
                                                                                                                                                            ),

                                                                                                                                                            html.H6([
                                                                                                                                                                '1b. Months of Control ',
                                                                                                                                                                ],
                                                                                                                                                                style={'fontSize': '120%','margin-top': '1vh', 'margin-bottom': '1vh'}),


                                                                                                                                                            
                                                                                                                                                            html.Div([
                                                                                                                                                            dcc.RangeSlider(
                                                                                                                                                                        id='month-slider',
                                                                                                                                                                        min=0,
                                                                                                                                                                        max=floor(params.max_months_controlling),
                                                                                                                                                                        step=1,
                                                                                                                                                                        # pushable=0,
                                                                                                                                                                        marks={i: str(i) for i in range(0,floor(params.max_months_controlling)+1,3)},
                                                                                                                                                                        value=[1,initial_month],
                                                                                                                                                            ),
                                                                                                                                                            ],
                                                                                                                                                            style={'fontSize': '180%'},
                                                                                                                                                            ),

                                                                                                                                                            html.H6([
                                                                                                                                                                '1c. Vaccination starts',
                                                                                                                                                                ],
                                                                                                                                                                style={'fontSize': '120%','margin-top': '1vh', 'margin-bottom': '1vh'}),

                                                                                                                                                                                                                                                                                                                        html.Div([
                                                                                                                                                            dcc.Slider(
                                                                                                                                                                        id='vaccine-slider',
                                                                                                                                                                        min   = 9,
                                                                                                                                                                        max   = 18,
                                                                                                                                                                        step  = 3,
                                                                                                                                                                        marks = {i: 'Never' if i==9 else 'Month {}'.format(i) if i==12 else str(i) for i in range(9,19,3)},
                                                                                                                                                                        value = 9,
                                                                                                                                                            ),
                                                                                                                                                            ],
                                                                                                                                                            style={'fontSize': '180%'},
                                                                                                                                                            ),



                                                                                                                                                        ],width=True),
                                                                                                                                                        #end of PYS row
                                                                                                                                                        ]),

                                                                                                                                                                    
                                                                                                                                                        html.Hr(),
                                                                                        ###################################



                                                                                                                                                        dbc.Row([
                                                                                                                                                        dbc.Col([

                                                                                                                                                            dbc.ButtonGroup([
                                                                                                                                                            dbc.Button('Custom Options',
                                                                                                                                                            color='warning',
                                                                                                                                                            outline=True,
                                                                                                                                                            className='mb-3',
                                                                                                                                                            id="collapse-button-custom",
                                                                                                                                                            style={'fontSize': '110%', 'cursor': 'pointer'}
                                                                                                                                                            ),



                                                                                                                                                            dbc.Button('🛈',
                                                                                                                                                            color='info',
                                                                                                                                                            className='mb-3',
                                                                                                                                                            id="popover-custom-options-target",
                                                                                                                                                            style={'cursor': 'pointer'}
                                                                                                                                                            )
                                                                                                                                                            ]),


                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Custom Options'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                Use this to choose your own custom strategy (you must first select 'custom' in the 'Control Type' selector above). You can compare two strategies directly or consider one only.
                                                                                                                                                                
                                                                                                                                                                The choice consists of selecting whether to accelerate or decelerate spread of COVID-19 (using the 'infection rate' sliders).
                                                                                                                                                                
                                                                                                                                                                You can choose different infection rates for the different risk groups.
                                                                                                                                                                '''
                                                                                                                                                                ),),
                                                                                                                                                                ],
                                                                                                                                                                id = "popover-custom-options",
                                                                                                                                                                is_open=False,
                                                                                                                                                                target="popover-custom-options-target",
                                                                                                                                                                placement='right',
                                                                                                                                                            ),
                                                                                                                                                            

                                                                                                                                                            
                                                                                                                                                            dbc.Collapse(
                                                                                                                                                                [

                                                                                                                                                                            # html.Hr(),

                                                                                                                                                                            dcc.Markdown('''*To adjust the following, make sure '**1a. Control Type**' is set to 'Custom'.*''', style = {'fontSize': '80%'}), # 'textAlign': 'left', 

                                                                                                                                                                            html.H6('Number Of Strategies',style={'fontSize': '100%'}),

                                                                                                                                                                            # html.Div([
                                                                                                                                                                            dbc.RadioItems(
                                                                                                                                                                            id = 'number-strats-radio',
                                                                                                                                                                            options=[
                                                                                                                                                                                {'label': 'One', 'value': 'one'},
                                                                                                                                                                                {'label': 'Two', 'value': 'two'},
                                                                                                                                                                            ],
                                                                                                                                                                            value= 'one',
                                                                                                                                                                            inline=True,
                                                                                                                                                                            ),

                                                                                                                                                                            html.Hr(),

                                                                                                                                                                            
                                                                                                                                                                            # dbc.Row([
                                                                                                                                                                            dbc.Button('Infection rate 🛈',
                                                                                                                                                                                    size='sm',
                                                                                                                                                                                    color='info',
                                                                                                                                                                                    className='mb-3',
                                                                                                                                                                                    # id="popover-custom-options-target",
                                                                                                                                                                                    id = 'popover-inf-rate-target',
                                                                                                                                                                                    style={'cursor': 'pointer'}
                                                                                                                                                                                    ),
                                                                                                                                                                            # ]),
                                                                                                                                                                            html.Div(id='strat-lr-infection'),
                                                                                                                                                                            
                                                                                                                                                                            
                                                                                                                                                                            dcc.Slider(
                                                                                                                                                                                id='low-risk-slider',
                                                                                                                                                                                min=0,
                                                                                                                                                                                max=len(params.fact_v)-1,
                                                                                                                                                                                step = 1,
                                                                                                                                                                                marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                value=initial_lr,
                                                                                                                                                                            ),


                                                                                                                                                                            html.Div(id='strat-hr-infection'),
                                                                                                                                                                            dcc.Slider(
                                                                                                                                                                                    id='high-risk-slider',
                                                                                                                                                                                    min=0,
                                                                                                                                                                                    max=len(params.fact_v)-1,
                                                                                                                                                                                    step = 1,
                                                                                                                                                                                    marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                    value=initial_hr,
                                                                                                                                                                                    ),


                                                                                                                                                                            dbc.Popover(
                                                                                                                                                                                [
                                                                                                                                                                                dbc.PopoverHeader('Infection Rate'),
                                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                                '''

                                                                                                                                                                                The *infection rate* relates to how quickly the disease is transmitted. **Control** measures affect transmission/infection rates (typically lowering them).
                                                                                                                                                                            
                                                                                                                                                                                Adjust by choosing a preset strategy  or making your own custom choice ('**1a. Control Type**').

                                                                                                                                                                                *You may choose to increase infection rates when using custom control. For more on this, see the 'Interpretation' Section below*
                                                                                                                                                                                

                                                                                                                                                                                '''
                                                                                                                                                                                ),),
                                                                                                                                                                                ],
                                                                                                                                                                                id = "popover-inf-rate",
                                                                                                                                                                                is_open=False,
                                                                                                                                                                                target="popover-inf-rate-target",
                                                                                                                                                                                placement='top',
                                                                                                                                                                            ),
                                                                                                                                                                                    



                                                                                                                                                                            html.Div([
                                                                                                                                                                                    html.H6('Strategy Two: Low Risk Infection Rate (%)',style={'fontSize': '100%'}),

                                                                                                                                                                                    dcc.Slider(
                                                                                                                                                                                        id='low-risk-slider-2',
                                                                                                                                                                                        min=0,
                                                                                                                                                                                        max=len(params.fact_v)-1,
                                                                                                                                                                                        step = 1,
                                                                                                                                                                                        marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                        value=6,
                                                                                                                                                                                    ),

                                                                                                                                                                                    html.H6('Strategy Two: High Risk Infection Rate (%)',style={'fontSize': '100%'}),
                                                                                                                                                                                
                                                                                                                                                                                    dcc.Slider(
                                                                                                                                                                                        id='high-risk-slider-2',
                                                                                                                                                                                        min=0,
                                                                                                                                                                                        max=len(params.fact_v)-1,
                                                                                                                                                                                        step = 1,
                                                                                                                                                                                        marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                        value=6,
                                                                                                                                                                                        ),
                                                                                                                                                                            ],id='strat-2-id'),

                                                                                                                                                                            html.Hr(),
                                                                                                                                                
                                                                                                                                                                ],
                                                                                                                                                                id="collapse-custom",
                                                                                                                                                                is_open=False,
                                                                                                                                                            ),

                                                                                                                                                        ],width=True),
                                                                                                                                                        # end of custom row
                                                                                                                                                        ]),

                                                                                                                                                        dbc.Row([
                                                                                                                                                        dbc.Col([

                                                                                                                                                            dbc.ButtonGroup([
                                                                                                                                                            dbc.Button('Plot Settings',
                                                                                                                                                            color='warning',
                                                                                                                                                            outline=True,
                                                                                                                                                            className='mb-3',
                                                                                                                                                            id="collapse-button-plots",
                                                                                                                                                            style={'fontSize': '110%', 'cursor': 'pointer'}
                                                                                                                                                            ),

                                                                                                                                                            dbc.Button('🛈',
                                                                                                                                                            color='info',
                                                                                                                                                            className='mb-3',
                                                                                                                                                            id="popover-plot-settings-target",
                                                                                                                                                            style = {'cursor': 'pointer'}
                                                                                                                                                            )
                                                                                                                                                            
                                                                                                                                                            ],
                                                                                                                                                            id='plot-settings-collapse',
                                                                                                                                                            ),

                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Plot Settings'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                Press this button to allow you to change the plot settings.
                                                                                                                                                                
                                                                                                                                                                You may plot different risk groups ('Groups To Plot'), and different disease progress categories ('Categories To Plot').

                                                                                                                                                                '''
                                                                                                                                                                ),),
                                                                                                                                                                ],
                                                                                                                                                                id = "popover-plot-settings",
                                                                                                                                                                is_open=False,
                                                                                                                                                                target="popover-plot-settings-target",
                                                                                                                                                                placement='right',
                                                                                                                                                            ),
                                                                                                                                                            
                                                                                                                                                            dbc.Collapse(
                                                                                                                                                                [
                                                                                                        
                                                                                                                                                                                        html.Div([





                                                                                                                                                                                                                                    # html.H6('Years To Plot',style={'fontSize': '120%'}),
                                                                                                                                                                                                                                    # dcc.Slider(
                                                                                                                                                                                                                                    #     id='years-slider',
                                                                                                                                                                                                                                    #     min=1,
                                                                                                                                                                                                                                    #     max=3,
                                                                                                                                                                                                                                    #     marks={i: str(i) for i in range(1,4)},
                                                                                                                                                                                                                                    #     value=2,
                                                                                                                                                                                                                                    # ),

                                                                                                                                                                                                                                    # html.H6('How Many Plots',style={'fontSize': '120%'}),
                                                                                                                                                                                                                                    # dbc.RadioItems(
                                                                                                                                                                                                                                    #     id='how-many-plots-slider',
                                                                                                                                                                                                                                    #     options=[
                                                                                                                                                                                                                                    #         {'label': 'One Plot', 'value': 'all'},
                                                                                                                                                                                                                                    #         # {'label': 'One Plot: Hospital Categories', 'value': 'hosp'},
                                                                                                                                                                                                                                    #         {'label': 'Two Plots (Different Axis Scales)', 'value': 'two'},
                                                                                                                                                                                                                                    #     ],
                                                                                                                                                                                                                                    #     value= 'two',
                                                                                                                                                                                                                                    #     labelStyle = {'display': 'inline-block'}
                                                                                                                                                                                                                                    # ),


                                                                                                                                                                                                                                        html.H6('Groups To Plot',style={'fontSize': '120%'}),
                                                                                                                                                                                                                                        dbc.Checklist(
                                                                                                                                                                                                                                            id = 'groups-checklist-to-plot',
                                                                                                                                                                                                                                            options=[
                                                                                                                                                                                                                                                {'label': 'Both Risk Groups (Sum Of)', 'value': 'BR'},
                                                                                                                                                                                                                                                {'label': 'High Risk Group', 'value': 'HR'},
                                                                                                                                                                                                                                                {'label': 'Low Risk Group', 'value': 'LR'},
                                                                                                                                                                                                                                            ],
                                                                                                                                                                                                                                            value= ['BR'],
                                                                                                                                                                                                                                            labelStyle = {'display': 'inline-block'}
                                                                                                                                                                                                                                        ),


                                                                                                                                                                                                                                        dbc.RadioItems(
                                                                                                                                                                                                                                            id = 'groups-to-plot-radio',
                                                                                                                                                                                                                                            options=[
                                                                                                                                                                                                                                                {'label': 'Both Risk Groups (Sum Of)', 'value': 'BR'},
                                                                                                                                                                                                                                                {'label': 'High Risk Group', 'value': 'HR'},
                                                                                                                                                                                                                                                {'label': 'Low Risk Group', 'value': 'LR'},
                                                                                                                                                                                                                                            ],
                                                                                                                                                                                                                                            value= 'BR',
                                                                                                                                                                                                                                            labelStyle = {'display': 'inline-block'}
                                                                                                                                                                                                                                        ),


                                                                                                                                                                                                                                        html.H6('Categories To Plot',style={'fontSize': '120%'}),

                                                                                                                                                                                                                                        dbc.Checklist(id='categories-to-plot-checklist',
                                                                                                                                                                                                                                                        options=[
                                                                                                                                                                                                                                                            {'label': 'Susceptible', 'value': 'S'},
                                                                                                                                                                                                                                                            {'label': 'Infected', 'value': 'I'},
                                                                                                                                                                                                                                                            {'label': 'Recovered', 'value': 'R'},
                                                                                                                                                                                                                                                            {'label': 'Hospitalised', 'value': 'H'},
                                                                                                                                                                                                                                                            {'label': 'Critical Care', 'value': 'C'},
                                                                                                                                                                                                                                                            {'label': 'Deaths', 'value': 'D'},
                                                                                                                                                                                                                                                        ],
                                                                                                                                                                                                                                                        value= ['S','I','R','H','C','D'],
                                                                                                                                                                                                                                                        labelStyle = {'display': 'inline-block'}
                                                                                                                                                                                                                                                    ),
                                                                                                                                                                                                                                                    

                                                                                                                                                                                                    ],id='outputs-div',
                                                                                                                                                                                                    
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    html.Hr(),
                                                                                                                                                                    ],
                                                                                                                                                                    id="collapse-plots",
                                                                                                                                                                    is_open=False,
                                                                                                                                                                    ),

                                                                                                                                                            ],width=True),
                                                                                                                                                            # end of plot settings row
                                                                                                                                                            ],
                                                                                                                                                            # justify='center',
                                                                                                                                                            # style={'textAlign': 'center'}

                                                                                                                                                            ),


                                                                                                
                                                                                                
                                                                                                
                                                                                                
                                                                                                
                                                                                                


                                                                                #########################################################################################################################################################


                                                                                                                                                    ],
                                                                                                                                                    width = True
                                                                                                                                                    ),

                                                                                                                                                    # ],
                                                                                                                                                    # style={'backgroundColor': "#FFFFFF", 'margin-left': '1vw','margin-right': '1vw','margin-bottom': '2vh','margin-top': '2vh'}
                                                                                                                                                    # ),        
                                                                                                                                        

                                                                                                                                                    # ],
                                                                                                                                                    # style={'backgroundColor': "#FFFFFF",'height': '100%','width': '100%'}),

                                                                                                                                                    # ],
                                                                                                                                                    # ),        
                                                                                                                                                    
                                                                                ########################################################################################################################

                                                                                                                        # end of form group 1
                                                                                                                        ],
                                                                                                                        row=True,
                                                                                                                        ),
                                                                                ########################################################################################################################

                                                                                                        ],
                                                                                                        style={'margin-left':'2vh', 'margin-right':'2vh'}
                                                                                                        ),










############################################################################################################################################################################################################################    

                                                            ]),



                                                        ],
                                                        width = 12,
                                                        xl = 3,
                                                        style={'height': '100%'}
                                                        ),

                                                        dbc.Col([

                                                        dbc.Jumbotron([
                                    ##############################################################################################################################################################################################################################
                                            # start of results col

                                                    html.Div([
                                             
                                                        html.H4('2. Choose Results Type',
                                                        # className='display-4',
                                                        style={'fontSize': '180%', 'color': 'blue' , 'textAlign': 'center' ,'margin-top': "3vh"}),
                                                        
                                                        dcc.Markdown('''*Choose between disease progress curves, bar charts and strategy overviews to explore the outcome of your strategy choice.*''', style = {'textAlign': 'center', 'fontSize': '90%'}),
                                             
                                                        dbc.Row([
                                                        dbc.ButtonGroup(
                                                            children=[
                                                                dbc.Button("Disease Progress Curves",color='success',outline=True,style={'min-width': '17vw'},id='DPC_dd',active=True),
                                                                dbc.Button("Bar Charts",             color='success',outline=True,style={'min-width': '17vw'},id='BC_dd'),
                                                                dbc.Button("Strategy Overview",      color='success',outline=True,style={'min-width': '17vw'},id='SO_dd'),
                                                            ],
                                                            className="mb-3",
                                                            size='lg',
                                                            # outline=True,
                                                            style = {'margin-top': '2vh', 'margin-bottom': '2vh', 'textAlign': 'center'}
                                                        ),
                                                        ],
                                                        justify='center'),
                                                        

                                             
                                                        html.Div([





                                                                        
                                                                        dcc.Markdown('''
                                                                        *Click on the info buttons for explanations*.
                                                                        ''',style = {'textAlign': 'center'}),



                                                                        dbc.Col([

                                                                                                dbc.Col([
                                                                                                    html.Div(
                                                                                                        [
                                                                                                        dbc.Row([
                                                                                                            html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [
                                                                                                            
                                                                                                            html.Div('Plot: Total Deaths (Percentage)',style= {'textAlign': 'center'}), # ,id='bar-plot-1-out'),
                                                                                                            
                                                                                                            dbc.Button('🛈',
                                                                                                            color='info',
                                                                                                            className='mb-3',
                                                                                                            size='lg',
                                                                                                            id="popover-bp1-target",
                                                                                                            style={'cursor': 'pointer'}
                                                                                                            )

                                                                                                            ]),
                                                                                                            dbc.Spinner(html.Div(id="loading-bar-output-1")),
                                                                                                        ]
                                                                                                        ,justify='center'),
                                                                                                        ],
                                                                                                        id='bar-plot-1-title',style={ 'display':'block', 'textAlign': 'left'}),

                                                                                                        dcc.Graph(id='bar-plot-1',style=bar_non_crit_style),
                                                                                                
                                                                                                ],
                                                                                                align='center',
                                                                                                width = 12,
                                                                                                ),
                                                                                                

                                                                                                
                                                                                        html.Hr(),



                                                                                        # html.Div([
                                                                                                    dbc.Col([

                                                                                                                    html.Div(
                                                                                                                        [dbc.Row([##
                                                                                                                            html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                                html.Div('Plot: Peak ICU Bed Capacity Requirement',style= {'textAlign': 'center'}), # id='bar-plot-3-out'),

                                                                                                                            dbc.Button('🛈',
                                                                                                                            color='info',
                                                                                                                            className='mb-3',
                                                                                                                            size='lg',
                                                                                                                            id="popover-bp3-target",
                                                                                                                            style={'cursor': 'pointer'}
                                                                                                                            )

                                                                                                                            ]),
                                                                                                                            dbc.Spinner(html.Div(id="loading-bar-output-3")),
                                                                                                                        ],
                                                                                                                        justify='center'),##
                                                                                                                        ],
                                                                                                                        id='bar-plot-3-title', style={'display':'block'}),

                                                                                                                        dcc.Graph(id='bar-plot-3',style=bar_non_crit_style),
                                                                                                                    
                                                                                                        
                                                                                                        ],
                                                                                                        align='center',
                                                                                                        
                                                                                                        width = 12,
                                                                                                        # xl = 6,
                                                                                                        ),

                                                                                                    html.Hr(),


                                                                                                        dbc.Col([
                                                                                                                    html.Div(
                                                                                                                            [dbc.Row([##
                                                                                                                                html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                                    html.Div('Plot: Time ICU Bed Capacity Exceeded',style= {'textAlign': 'center'}), # id='bar-plot-4-out'),

                                                                                                                                dbc.Button('🛈',
                                                                                                                                color='info',
                                                                                                                                className='mb-3',
                                                                                                                                size='lg',
                                                                                                                                id='popover-bp4-target',
                                                                                                                                style={'cursor': 'pointer'}
                                                                                                                                )

                                                                                                                                ]),
                                                                                                                                
                                                                                                                                dbc.Spinner(html.Div(id="loading-bar-output-4")),
                                                                                                                            ],
                                                                                                                            justify='center'),##
                                                                                                                            ],
                                                                                                                    id='bar-plot-4-title',style={'display':'block'}),

                                                                                                                    dcc.Graph(id='bar-plot-4',style=bar_non_crit_style),
                                                                                                        ],
                                                                                                        align='center',
                                                                                                        width = 12,
                                                                                                        ),
                                                                                                    
                                                                                            html.Hr(),

                                                                                            # ],
                                                                                            # id = 'bar-plots-crit'
                                                                                            # ),


                                                                                                


                                                                                            dbc.Col([
                                                                                                        html.Div(
                                                                                                                [dbc.Row([##
                                                                                                                    html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                        html.Div('Plot: Herd Immunity Threshold',style= {'textAlign': 'center'}), # id='bar-plot-2-out'),
                                                                                                                    
                                                                                                                    dbc.Button('🛈',
                                                                                                                    color='info',
                                                                                                                    className='mb-3',
                                                                                                                    size='lg',
                                                                                                                    id='popover-bp2-target',
                                                                                                                    style={'cursor': 'pointer'}
                                                                                                                    )

                                                                                                                    ]),

                                                                                                                    dbc.Spinner(html.Div(id="loading-bar-output-2")),
                                                                                                                ],
                                                                                                            justify='center'),##
                                                                                                                ],
                                                                                                        id='bar-plot-2-title',style={ 'display':'block'}),

                                                                                                        dcc.Graph(id='bar-plot-2',style=bar_non_crit_style),
                                                                                                        

                                                                                            
                                                                                            ],
                                                                                            align='center',
                                                                                            width = 12,
                                                                                            ),

                                                                                            html.Hr(),


                                                                                            dbc.Col([
                                                                                                    
                                                                                                    html.Div(
                                                                                                            [dbc.Row([##

                                                                                                                    html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                    html.Div('Plot: Time Until Herd Immunity Threshold Reached',style= {'textAlign': 'center'}), # id='bar-plot-5-out'),

                                                                                                                    dbc.Button('🛈',
                                                                                                                    color='info',
                                                                                                                    className='mb-3',
                                                                                                                    size='lg',
                                                                                                                    id='popover-bp5-target',
                                                                                                                    style={'cursor': 'pointer'}
                                                                                                                    )

                                                                                                                    ]),
                                                                                                                    dbc.Spinner(html.Div(id="loading-bar-output-5")),
                                                                                                            ],
                                                                                                            justify='center'
                                                                                                            ),##
                                                                                                            ],
                                                                                                    id='bar-plot-5-title',style={ 'display':'block'}),

                                                                                                    dcc.Graph(id='bar-plot-5',style=bar_non_crit_style),
                                                                                                    

                                                                                            
                                                                                            ],
                                                                                            align='center',
                                                                                            width=12,
                                                                                            ),


                                                                                        
                                                                                        dbc.Popover(
                                                                                            [
                                                                                            dbc.PopoverHeader('Total Deaths'), # Critical Care
                                                                                            dbc.PopoverBody(txt1),
                                                                                            ],
                                                                                            id = "popover-bp1",
                                                                                            is_open=False,
                                                                                            target="popover-bp1-target",
                                                                                            placement='left',
                                                                                        ),
                                                                                        dbc.Popover(
                                                                                            [
                                                                                            dbc.PopoverHeader('Herd immunity threshold'),
                                                                                            dbc.PopoverBody(txt2),
                                                                                            ],
                                                                                            id = "popover-bp2",
                                                                                            is_open=False,
                                                                                            target="popover-bp2-target",
                                                                                            placement='left',
                                                                                        ),

                                                                                        dbc.Popover(
                                                                                            [
                                                                                            dbc.PopoverHeader('ICU capacity'),
                                                                                            dbc.PopoverBody(txt3),
                                                                                            ],
                                                                                            id = "popover-bp3",
                                                                                            is_open=False,
                                                                                            target="popover-bp3-target",
                                                                                            placement='left',
                                                                                        ),

                                                                                        dbc.Popover(
                                                                                            [
                                                                                            dbc.PopoverHeader('ICU capacity (time exceeded)'),
                                                                                            dbc.PopoverBody(txt4),
                                                                                            ],
                                                                                            id = "popover-bp4",
                                                                                            is_open=False,
                                                                                            target="popover-bp4-target",
                                                                                            placement='left',
                                                                                        ),

                                                                                        dbc.Popover(
                                                                                            [
                                                                                            dbc.PopoverHeader('Time to herd immunity'),
                                                                                            dbc.PopoverBody(txt5),
                                                                                            ],
                                                                                            id = "popover-bp5",
                                                                                            is_open=False,
                                                                                            target="popover-bp5-target",
                                                                                            placement='left',
                                                                                        )
                                                                                        
                                                                                        



                                                                        ],width=True),
                                                                    ],id='bc-content',
                                                                    style={'display': 'none'}),

                                                                                        
                                                    html.Div(id='DPC-content',children=[

                                                                dbc.Row([
                                                                        html.H4("All Categories",
                                                                        style={'margin-bottom': '3vh', 'textAlign': 'center' ,'margin-top': '1vh','fontSize': '200%'} # 'margin-left': '2vw', 
                                                                        ),

                                                                        dbc.Spinner(html.Div(id="loading-line-output-1")),
                                                                        ],
                                                                        justify='center',
                                                                ),

                                                                
                                                                dcc.Graph(id='line-plot-1',style={'height': '70vh', 'width': '95%'}), # figure=dummy_figure,

                                                                dbc.Container([html.Div([],style={'height': '3vh'})]),

                                                                html.H4("Hospital Categories",
                                                                style={'margin-bottom': '3vh', 'textAlign': 'center' ,'margin-top': '1vh','fontSize': '200%'} # 'margin-left': '2vw', 
                                                                ),

                                                                dcc.Graph(id='line-plot-2',style={'height': '70vh', 'width': '95%'}), # figure=dummy_figure,

                                                                html.H4("Intensive Care",
                                                                style={'margin-bottom': '3vh', 'textAlign': 'center' ,'margin-top': '1vh','fontSize': '200%'} # 'margin-left': '2vw', 
                                                                ),

                                                                dcc.Graph(id='line-plot-3',style={'height': '70vh', 'width': '95%'}), # figure=dummy_figure,


                                                    ]),
                                             
                                                    html.Div(id = 'strategy-outcome-content',style={'display': 'none'}),

                                                    html.Div(style= {'height': '2vh'}),

                                                    # dbc.Col([
                                                    #             html.Div(id='strategy-table'),
                                                    #         ],
                                                    #         width={'size': 8, 'offset': 2},
                                                    # ),
                                                    
                                                    
                                                ]),


# end of results col
#########################################################################################################################################################











                                                        ]),
                                                        ],
                                                        width = 12,
                                                        xl = 9
                                                        ),


                                                    # end of row
                                            ],
                                            # no_gutters=True
                                            ),
                                                
                                                    
                                             ##################################################################################################
                                                    
                                                    html.Hr(style={'margin-top': '3vh', 'margin-bottom': '2vh'}),
    
                                                    html.H4("Interpretation", className="display-4",style={'fontSize': '300%','textAlign': 'center'}),
                                                    html.Hr(),

                                                    # dbc.Jumbotron([
                                                    Results_interpretation,
                                                    # ]),

                                             
                                             
                                             


                                         ],
                                         
                                         ),
#########################################################################################################################################################
                                                                                                                dbc.Tab(label='Model Explanation', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='model_s',children=[
                                                                                                        
                                                                                                                                                html.Div([
                                                                                                                                                                # dbc.Col([

                                                                                                                                                                    html.H4('Model Explanation',
                                                                                                                                                                    className = 'display-4',
                                                                                                                                                                    style = {'margin-top': '1vh', 'textAlign': 'center', 'fontSize': '300%'}),

                                                                                                                                                                    html.Hr(),
                                                                                                                                                                    dcc.Markdown(
                                                                                                                                                                    '''
                                                                                                                                                                    *Underlying all of the predictions is a mathematical model. In this Section we explain how the mathematical model works.*

                                                                                                                                                                    We present a compartmental model for COVID-19, split by risk categories. That is to say that everyone in the population is **categorised** based on **disease status** (susceptible/ infected/ recovered/ hospitalised/ critical care/ dead) and based on **COVID risk**.
                                                                                                                                                                    
                                                                                                                                                                    The model is very simplistic but still captures the basic spread mechanism. It is far simpler than the [**Imperial College model**](https://spiral.imperial.ac.uk/handle/10044/1/77482), but it uses similar parameter values and can capture much of the relevant information in terms of how effective control will be.

                                                                                                                                                                    It is intended solely as an illustrative, rather than predictive, tool. We plan to increase the sophistication of the model and to update parameters as more (and better) data become available to us. In particular we will shortly be adding the real time global data feed as an input into the model, so that the simulation initial conditions will be based on current data.

                                                                                                                                                                    We have **two risk categories**: high and low. **Susceptible** people get **infected** after contact with an infected person (from either risk category). A fraction of infected people (*h*) are **hospitalised** and the rest **recover**. Of these hospitalised cases, a fraction (*c*) require **critical care** and the rest recover. Of those in critical care, a fraction (*d*) **die** and the rest recover.

                                                                                                                                                                    The recovery fractions depend on which risk category the individual is in.
                                                                                                                                                                

                                                                                                                                                                    '''

                                                                                                                                                                    ),



                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/Capture_lomery.png',
                                                                                                                                                                    style={'max-width':'90%','height': 'auto', 'display': 'block','margin-top': '1vh','margin-bottom': '1vh'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    dcc.Markdown('''

                                                                                                                                                                    The selection of risk categories is done in the crudest way possible - an age split at 60 years (based on the age structure data below). A more nuanced split would give a more effective control result, since there are older people who are at low risk and younger people who are at high risk. In many cases, these people will have a good idea of which risk category they belong to.

                                                                                                                                                                    *For the more mathematically inclined reader, a translation of the above into a mathematical system is described below.*

                                                                                                                                                                    ''',style={'margin-top' : '2vh','margin-bottom' : '2vh'}),
                                                                                                                                                                    
                                                                                                                                                                    

                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/eqs_f3esyu.png',
                                                                                                                                                                    style={'max-width':'90%','height': 'auto','display': 'block','margin-top': '1vh','margin-bottom': '1vh'})
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),


                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/text_toshav.png',
                                                                                                                                                                    style={'max-width':'90%','height': 'auto','display': 'block','margin-top': '1vh','margin-bottom': '1vh'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    
                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    Of those requiring critical care, we assume that if they get treatment, a fraction *1-d* recover. However, if they don't receive it all die. The number able to get treatment must be lower than the number of ICU beds available.
                                                                                                                                                                    '''),



                                                                                                                                                                    html.Hr(),

                                                                                                                                                                    html.H4('Parameter Values',style={'fontSize': '200%'}),

                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    The model uses a weighted average across the age classes below and above 60 to calculate the probability of a member of each class getting hospitalised or needing critical care.
                                                                                                                                                                    '''),



                                                                                                                                                                    

                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/params_w7tebv.png',
                                                                                                                                                                    style={'max-width':'90%','height': 'auto','display': 'block','margin-top': '1vh','margin-bottom': '1vh'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),



                                                                                                                                                                    html.P('** the Imperial paper uses 8 days in hospital if critical care is not required (as do we). It uses 16 days (with 10 in ICU) if critical care is required. Instead, if critical care is required we use 8 days in hospital (non-ICU) and then either recovery or a further 8 in intensive care (leading to either recovery or death).',
                                                                                                                                                                    style={'fontSize':'80%'}),

                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    Please use the following links: [**Ferguson et al**](https://spiral.imperial.ac.uk/handle/10044/1/77482), [**Anderson et al**](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30567-5/fulltext) and [**Zhao et al**](https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0006158&type=printable)
                                                                                                                                                                    '''),


                                                                                                                                                                    html.H4('Age Structure',style={'fontSize': '150%'}),
                                                                                                                                                                    
                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    The age data is taken from [**GOV.UK**](https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest) and the hospitalisation and critical care data is from the [**Imperial College Paper**](https://spiral.imperial.ac.uk/handle/10044/1/77482) (Ferguson et al.).

                                                                                                                                                                    To find the probability of a low risk case getting hospitalised (or subsequently put in critical care), we take a weighted average by proportion of population.

                                                                                                                                                                    *The table below shows the age structure data that was used to calculate these weighted averages across the low risk category (under 60) and high risk (over 60) category.*
                                                                                                                                                                    
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'margin-top': '2vh','margin-bottom': '2vh'}
                                                                                                                                                                    ),

                                                                                                                                                                    generate_table(df),








                                                                                                                                            ],style={'fontSize': '100%'})
                                                                                                                        ]),

                                                                                                ]),

                                                                                    # ]),

                                        
                                    # ],
                                    # width=12
                                    # ),

                        ],
                        style= {'width': '90%', 'margin-left': '5vw', 'margin-right': '5vw', 'margin-top': '10vh', 'margin-bottom': '5vh'}
                        ),


                        ],
                        style= {'width': '90%', 'backgroundColor': '#f4f6f7', 'margin-left': '5vw', 'margin-right': '5vw', 'margin-bottom': '5vh'}
                        ),

                        # ],
                        # width=12)
                        # end of col 2


########################################################################################################################
        # end of row 1
########################################################################################################################


    ],
    # no_gutters=True,
    justify='center'
    )],
    style={'fontSize' : '2vh'},
    id='main-page-id'
    )

















layout_data = html.Div([
                #    dbc.Row([
                   dbc.Col([
                        dbc.Jumbotron([
                        html.H1('Data feed'),
                        ]),
                   ],width={'size':8,'offset':2}
                   ), 
                #    ]),
                ])







navbar = html.Nav([
        html.Div([
            dcc.Tabs([
                dcc.Tab(children=
                        layout_intro,
                        label='Background',value='intro',
                        style={'fontSize':'2vh'}
                        ), #
                dcc.Tab(children=
                        layout_inter,
                        label='Interactive Model',value='interactive',
                        style={'fontSize':'2vh'}
                        ),
                dcc.Tab(children=
                        layout_dan,
                        label='Real Time Global Data Feed',value='data',
                        style={'fontSize':'2vh'}
                        ), #disabled=True),
            ], id='main-tabs', value='inter'),
        ], style={'width': '100vw'}, # , 'display': 'flex', 'justifyContent': 'center'},
        ),
    ],)














# app.layout
        
page_layout = html.Div([
    
    # dbc.Jumbotron([

            
            dbc.Row([
                dbc.Col([
                    # dbc.Container([
                    html.H3(children='Modelling control of COVID-19',
                    className="display-4",
                    style={'margin-top': '1vh','fontSize': '6vh'}
                    ),
                    # dcc.Markdown(
                    # '''*Best viewed in landscape mode. Click on the info buttons for more details*.''',
                    # ),

                    dcc.Markdown('''

                    *Best viewed in **landscape mode**. Click on the info buttons to find out more (and click again to dismiss) *
                    '''
                    ,style={'margin-top': '0.5vh','margin-bottom': '0.2vh','fontSize': '1.8vh'} # 'margin-bottom': '0.5vh'
                    ),
                    # dbc.Row([

                    # html.Div(style={'width': '2vh'}),
                    # dbc.Button('🛈',
                    # color='info',
                    # className='mb-3',
                    # id="popover-example-target",
                    # size='lg',
                    # style = {'cursor': 'pointer'})
                    
                    # ]),

                    # dbc.Popover(
                    #     [
                    #     dbc.PopoverHeader('Example Button'),
                    #     dbc.PopoverBody(dcc.Markdown(
                    #     '''

                    #     Click on any of these for explanations.
                    
                    #     '''
                    #     ),),
                    #     ],
                    #     id = "popover-example",
                    #     is_open=False,
                    #     target="popover-example-target",
                    #     placement='right',
                    # ),

                    dcc.Markdown('''
                    **Disclaimer**: this work is intended for **educational purposes only and not decision making**. There are many uncertainties in the COVID debate. The model is intended solely as an **illustrative rather than predictive tool**.
                    ''',style={'margin-top': '0.2vh','margin-bottom': '1vh','fontSize': '1.6vh'}), # 

                ],width=True,
                style={'margin-left': '10vh'}
                ),
            ],
            align="start",
            style={'backgroundColor': '#e9ecef'}
            ),

        # ]),
        ##
        # navbar
        html.Div([navbar]),
        ##
        # # page content
        dcc.Location(id='url', refresh=False),

        html.Footer(["Authors: ",
                     html.A('Nick P. Taylor', href='https://twitter.com/TaylorNickP'),", ",
                     html.A('Daniel Muthukrishna', href='https://twitter.com/DanMuthukrishna'),
                     " and Dr Cerian Webb. ",
                     html.A('Source code', href='https://github.com/nt409/covid-19'), ". ",
                     "Data is taken from ",
                     html.A("Worldometer", href='https://www.worldometers.info/coronavirus/'), " if available or otherwise ",
                     html.A("Johns Hopkins University (JHU) CSSE", href="https://github.com/ExpDev07/coronavirus-tracker-api"), "."
                    ],
                    style={'textAlign': 'center', 'fontSize': '1.6vh'}),

        

        ],
        # style={'padding': '5%'}
        )
##
########################################################################################################################





app.layout = page_layout




########################################################################################################################
# callbacks







@app.callback(Output('main-tabs', 'value'),
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/inter':
        return 'interactive'
    elif pathname == '/data':
        return 'data'
    elif pathname == '/intro':
        return 'intro'
    else:
        return 'interactive'




@app.callback(
            [Output('low-risk-slider', 'value'),
            Output('high-risk-slider', 'value'),
            Output('low-risk-slider', 'disabled'), 
            Output('high-risk-slider', 'disabled'),
            Output('number-strats-radio','options')
            ],
            [
            Input('preset', 'value'),
            ])
def preset_sliders(preset):
    if preset == 'C':
        dis = False
        options=[
                {'label': 'One', 'value': 'one'},
                {'label': 'Two', 'value': 'two'},
            ]
    else:
        dis = True
        options=[
                {'label': 'One', 'value': 'one','disabled': True},
                {'label': 'Two', 'value': 'two','disabled': True},
            ]
    if preset in preset_dict_low:
        return preset_dict_low[preset], preset_dict_high[preset], dis, dis, options
    else:
        return preset_dict_low['N'], preset_dict_high['N'], dis, dis, options # shouldn't ever be needed



########################################################################################################################
# collapse
def toggle_collapse(n, is_open):
    color = 'warning'

    if n is not None and not is_open: 
        color = 'success'
    else:
        color = 'warning'
    if n:
        return [not is_open, color]
    return [is_open, color]


for p in ["plots", "custom"]: # , "hospital"]:
    app.callback(
        [Output(f"collapse-{p}", "is_open"),
        Output(f"collapse-button-{p}", "color")],
        [Input(f"collapse-button-{p}", "n_clicks")
        ],
        [State(f"collapse-{p}", "is_open")],
    )(toggle_collapse)


########################################################################################################################
# popovers
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

for p in ["pick-strat","control", "months-control", "vaccination","custom-options", "plot-settings", "inf-rate", "inf-tab", "cont-tab", "example","red-deaths","ICU","herd","bp1","bp2","bp3","bp4","bp5"]:
    app.callback(
        Output(f"popover-{p}", "is_open"),
        [Input(f"popover-{p}-target", "n_clicks")
        ],
        [State(f"popover-{p}", "is_open")],
    )(toggle_popover)


# modal
app.callback(
    Output("modal", "is_open"),
    [Input("close-modal", "n_clicks")],
    [State("modal", "is_open")],
)(toggle_popover)



##############################################################################################################################


@app.callback(
    [
    
    Output('strat-2-id', 'style'),

    Output('strat-hr-infection','children'),
    Output('strat-lr-infection','children'),

    Output('groups-to-plot-radio','style'),
    Output('groups-checklist-to-plot','style'),

    ],
    [
    Input('number-strats-radio', 'value'),
    Input('preset', 'value'),
    ])
def invisible_or_not(num,preset):
    
    if num=='two':
        strat_H = [html.H6('Strategy One: High Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        strat_L = [html.H6('Strategy One: Low Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        groups_checklist = {'display': 'none'}
        groups_radio = None
        says_strat_2 = None
    else:
        strat_H = [html.H6('High Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        strat_L = [html.H6('Low Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        groups_checklist = None
        groups_radio = {'display': 'none'}
        says_strat_2 = {'display': 'none'}

    if preset!='C':
        says_strat_2 = {'display': 'none'}

    

    return [says_strat_2,strat_H, strat_L ,groups_radio,groups_checklist]

########################################################################################################################

@app.callback(
    Output('sol-calculated', 'data'),
    [
    Input('preset', 'value'),
    Input('month-slider', 'value'),
    Input('low-risk-slider', 'value'),
    Input('high-risk-slider', 'value'),
    Input('low-risk-slider-2', 'value'),
    Input('high-risk-slider-2', 'value'),
    Input('number-strats-radio', 'value'),
    Input('vaccine-slider', 'value'),
    ])
def find_sol(preset,month,lr,hr,lr2,hr2,num_strat,vaccine): # years , ,hosp
    if vaccine==9:
        vaccine = None
    
    # print(dash.callback_context.triggered)
    if preset=='C':
        lr = params.fact_v[int(lr)]
        hr = params.fact_v[int(hr)]
    else:
        lr, hr = preset_strat(preset)
        num_strat='one'

    if preset=='N':
        month=[0,0]

    
    lr2 = params.fact_v[int(lr2)]
    hr2 = params.fact_v[int(hr2)]
    
    t_stop = 365*3


    months_controlled = [month_len*i for i in month]
    if month[0]==month[1]:
        months_controlled= None

    sols = []
    sols.append(simulator().run_model(beta_L_factor=lr,beta_H_factor=hr,t_control=months_controlled,T_stop=t_stop,vaccine_time=vaccine))
    if num_strat=='two':
        sols.append(simulator().run_model(beta_L_factor=lr2,beta_H_factor=hr2,t_control=months_controlled,T_stop=t_stop,vaccine_time=vaccine))
    
    return sols # {'sols': sols}


@app.callback(
    Output('sol-calculated-do-nothing', 'data'),
    [
    Input('sol-calculated-do-nothing', 'children'),
    ])
def find_sol_do_noth(hosp): # years

    t_stop = 365*3
    
    sol_do_nothing = simulator().run_model(beta_L_factor=1,beta_H_factor=1,t_control=None,T_stop=t_stop)
    
    return sol_do_nothing






########################################################################################################################


@app.callback(
            [Output('line-plot-intro', 'figure'),
            Output('line-plot-intro-2', 'figure')],
            [
            Input('intro-tabs', 'active_tab'),
            ],
            [
            State('sol-calculated-do-nothing', 'data'),
            ])
def intro_content(tab,sol_do_n): #hosp,
        fig1 = dummy_figure
        fig2 = dummy_figure


        if tab=='tab_3':
            lr, hr = preset_strat('H')
            output_use = ['S','I','R']
            output_use_2 = ['C','H','D']
            sols = []
            month = [1,10]
            months_controlled = [month_len*i for i in month]
            year_to_run = 3
            
            sols.append(simulator().run_model(beta_L_factor=lr,beta_H_factor=hr,t_control=months_controlled,T_stop=365*year_to_run))
            sols.append(sol_do_n)
            fig1 = figure_generator(sols,month,output_use,['BR'],'two',['BR'])
            fig2 = figure_generator(sols,month,output_use_2,['BR'],'two',['BR'])

        
        return fig1, fig2





###################################################################################################################################################################################


@app.callback([ 

                Output('DPC-content', 'style'),
                Output('bc-content', 'style'),
                Output('strategy-outcome-content', 'style'),
                
                Output('DPC_dd', 'active'),
                Output('BC_dd', 'active'),
                Output('SO_dd', 'active'),

                Output('line_page_title', 'children'),
                
                # Output('strategy-table', 'children'),

                Output('strategy-outcome-content', 'children'),


                Output('bar-plot-1', 'figure'),
                Output('bar-plot-2', 'figure'),
                Output('bar-plot-3', 'figure'),
                Output('bar-plot-4', 'figure'),
                Output('bar-plot-5', 'figure'),

                Output('loading-bar-output-1','children'),
                Output('loading-bar-output-2','children'),
                Output('loading-bar-output-3','children'),
                Output('loading-bar-output-4','children'),
                Output('loading-bar-output-5','children'),



                Output('plot-settings-collapse', 'style'),

                Output('line-plot-1', 'figure'),
                Output('line-plot-2', 'figure'),
                Output('line-plot-3', 'figure'),


                # Output('line-plot-1', 'style'),
                # Output('line-plot-2', 'style'),
                
                Output('loading-line-output-1','children'),
                
                

                ],
                [
                Input('interactive-tabs', 'active_tab'),



                Input('main-tabs', 'value'),

                
                Input('sol-calculated', 'data'),

                # or any of the plot categories
                Input('groups-checklist-to-plot', 'value'),
                Input('groups-to-plot-radio','value'),                                      
                # Input('how-many-plots-slider','value'),
                Input('categories-to-plot-checklist', 'value'),
                # Input('years-slider', 'value'),

                Input('DPC_dd', 'n_clicks'),
                Input('BC_dd', 'n_clicks'),
                Input('SO_dd', 'n_clicks'),

                ],
               [
                State('DPC_dd', 'active'),
                State('BC_dd', 'active'),
                State('SO_dd', 'active'),
                State('sol-calculated-do-nothing', 'data'),
                State('preset', 'value'),
                State('month-slider', 'value'),
                # State('hosp-cats', 'value'),
                State('number-strats-radio', 'value'),
                State('vaccine-slider', 'value'),

                ])
def render_interactive_content(tab,tab2,sols,groups,groups2,output,DPC_dropdown,BC_dropdown,SO_dropdown,DPC_active,BC_active,SO_active,sol_do_nothing,preset,month,num_strat,vaccine_time): # pathname, tab_intro pathname, hosp

    ctx = dash.callback_context
    
    DPC_style = {'display' : 'none'}
    BC_style  = {'display': 'none'}
    SO_style  = {'display' : 'none'}

    if not ctx.triggered:
        button_id = "DPC_dd"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id in ['BC_dd','SO_dd','DPC_dd']:
        DPC_active = False
        BC_active  = False
        SO_active  = False
        if button_id == 'BC_dd':
            BC_style = {'display': 'block'}
            BC_active = True
        elif button_id == 'SO_dd':
            SO_style = {'display': 'block'}
            SO_active = True
        else: # 'DPC
            DPC_style = {'display': 'block'}
            button_id = 'DPC_dd' # in case wasn't
            DPC_active = True
    else:
        if BC_active:
            button_id = 'BC_dd'
            BC_style = {'display': 'block'}
        elif SO_active:
            button_id = 'SO_dd'
            SO_style = {'display': 'block'}
        else:
            button_id = 'DPC_dd'
            DPC_style = {'display': 'block'}


########################################################################################################################



    Strat_outcome_title = presets_dict[preset] + ' Strategy Outcome'
    strategy_outcome_text = ['']

    plot_settings_on_or_off = {'display': 'none'}

    if sols is None or tab2!='interactive' or button_id!='BC_dd':
        bar1 = dummy_figure
        bar2 = dummy_figure
        bar3 = dummy_figure
        bar4 = dummy_figure
        bar5 = dummy_figure

    if sols is None or tab2!='interactive' or button_id!='DPC_dd':
        fig1 = dummy_figure
        fig2 = dummy_figure
        fig3 = dummy_figure



    if tab2=='interactive':
   
   
        if preset!='C':
            num_strat = 'one'
            

        if sols is not None:
            sols.append(sol_do_nothing)
        
            # bar plot data
            time_reached_data = []
            time_exceeded_data = []
            crit_cap_data_L_3yr = []
            crit_cap_data_H_3yr = []
            crit_cap_quoted_3yr = []
            ICU_data_3yr = []
            herd_list_3yr = []


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
                    # table_out = strat_table(month,sol['beta_H'],sol['beta_L'],len(sols)-1,ii+1)
                    # tables.append(table_out)
########################################################################################################################
            if button_id!='DPC_dd': # tab != DPC

                #loop start
                for ii in range(len(sols)):
                    # print(len(sols),ii)
                    if sols[ii] is not None:
                        sol = sols[ii]
                        
                        yy = np.asarray(sol['y'])
                        tt = np.asarray(sol['t'])

                        
                        num_t_points = yy.shape[1]

                        metric_val_L_3yr, metric_val_H_3yr, ICU_val_3yr, herd_fraction_out, time_exc, time_reached = extract_info(yy,tt,num_t_points)
                        
                        crit_cap_data_L_3yr.append(metric_val_L_3yr) #
                        crit_cap_data_H_3yr.append(metric_val_H_3yr) #
                        ICU_data_3yr.append(ICU_val_3yr)
                        herd_list_3yr.append(herd_fraction_out) ##
                        time_exceeded_data.append(time_exc) ##
                        time_reached_data.append(time_reached) ##


                        num_t_2yr = ceil(2*num_t_points/3)
                        metric_val_L_2yr, metric_val_H_2yr, ICU_val_2yr, herd_fraction_out = extract_info(yy,tt,num_t_2yr)[:4]

                        crit_cap_data_L_2yr.append(metric_val_L_2yr) #
                        crit_cap_data_H_2yr.append(metric_val_H_2yr) #
                        ICU_data_2yr.append(ICU_val_2yr)
                        herd_list_2yr.append(herd_fraction_out) ##


                        num_t_1yr = ceil(num_t_points/3)
                        metric_val_L_1yr, metric_val_H_1yr, ICU_val_1yr, herd_fraction_out = extract_info(yy,tt,num_t_1yr)[:4]

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
                if  button_id=='SO_dd': # sols is not None and

                    strategy_outcome_text = html.Div([

                        dcc.Markdown('''
                            *Click on the info buttons for explanations*.
                            ''',style = {'textAlign': 'center'}),
                        
                        outcome_fn(month,sols[0]['beta_L'],sols[0]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[0],herd_list_2yr[0],ICU_data_2yr[0],preset,number_strategies = num_strat,which_strat=1), # hosp,
                        html.Hr(),
                        outcome_fn(month,sols[1]['beta_L'],sols[1]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[1],herd_list_2yr[1],ICU_data_2yr[1],preset,number_strategies = num_strat,which_strat=2), # hosp,
                        ],
                        style = {'fontSize': '2vh'}
                        )

                
                ########################################################################################################################
                # BC results

                if button_id=='BC_dd': # sols is not None and 

                    crit_cap_bar_1yr = [crit_cap_data_L_1yr[i] + crit_cap_data_H_1yr[i] for i in range(len(crit_cap_data_H_1yr))]
                    crit_cap_bar_3yr = [crit_cap_data_L_3yr[i] + crit_cap_data_H_3yr[i] for i in range(len(crit_cap_data_H_3yr))]


                    bar1 = Bar_chart_generator(crit_cap_bar_1yr      ,text_addition='%'         , y_title='Population'                    , hover_form = '%{x}, %{y:.3%}'                                                   ,data_group=crit_cap_bar_3yr, yax_tick_form='.1%') # name1='Low Risk',name2='High Risk'
                    bar2 = Bar_chart_generator(herd_list_1yr         ,text_addition='%'         , y_title='Percentage of Safe Threshold'  , hover_form = '%{x}, %{y:.1%}<extra></extra>'          ,color = 'mediumseagreen' ,data_group=herd_list_3yr,yax_tick_form='.1%',maxi=False,yax_font_size_multiplier=0.8) # preset = preset,
                    bar3 = Bar_chart_generator(ICU_data_1yr          ,text_addition='x current' , y_title='Multiple of Current Capacity'  , hover_form = '%{x}, %{y:.1f}x Current<extra></extra>' ,color = 'powderblue'     ,data_group=ICU_data_3yr  ) # preset = preset,
                    bar4 = Bar_chart_generator(time_exceeded_data    ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'peachpuff'   ) # preset = preset,
                    bar5 = Bar_chart_generator(time_reached_data     ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'lemonchiffon') # preset = preset,



        ########################################################################################################################
            # DPC results

            if button_id=='DPC_dd': # sols is not None and
                output_2 = [i for i in output if i in ['C','H','D']]
                plot_settings_on_or_off = None

                if vaccine_time==9:
                    vaccine_time = None


                if len(output)>0:
                    fig1 = figure_generator(sols[:-1],month,output,groups,num_strat,groups2,vaccine_time=vaccine_time) # hosp,
                else:
                    fig1 = dummy_figure

                if len(output_2)>0:
                    fig2 = figure_generator(sols[:-1],month,output_2,groups,num_strat,groups2,vaccine_time=vaccine_time) # hosp,
                else:
                    fig2 = dummy_figure
                fig3 = figure_generator(sols[:-1],month,['C'],groups,num_strat,groups2,ICU_to_plot=True,vaccine_time=vaccine_time) # hosp,
            
        ##############

                

########################################################################################################################

    return [
    DPC_style,
    BC_style,
    SO_style,
    DPC_active,
    BC_active,
    SO_active,
    Strat_outcome_title,
    # tables,
    strategy_outcome_text,
    bar1,
    bar2,
    bar3,
    bar4,
    bar5,
    html.Div(),
    html.Div(),
    html.Div(),
    html.Div(),
    html.Div(),
    plot_settings_on_or_off,
    fig1,
    fig2,
    fig3,
    # line_plot_style_1,
    # line_plot_style_2,
    html.Div()
    ]

########################################################################################################################








# dan's callback


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
               Output("loading-icon", "children"),],
              [Input('button-plot', 'n_clicks'),
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
              [State('hidden-stored-data', 'children')] +
              [State(c_name, 'value') for c_name in COUNTRY_LIST])
def update_plots(n_clicks, start_date, end_date, show_exponential, normalise_by_pop,
                 align_cases_check, align_cases_input, align_deaths_check, align_deaths_input, align_active_cases_check,
                 align_active_cases_input, align_daily_cases_check, align_daily_cases_input,
                 align_daily_deaths_check, align_daily_deaths_input, saved_json_data, *args):
    # print(n_clicks, start_date, end_date, args)
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
            data = get_data(country)
            country_data[country] = data

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
                    x=0.,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        }

        layout_daily_plot = copy.deepcopy(layout_normal)
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
                layout_out = copy.deepcopy(layout_daily_plot)
            else:
                layout_out = copy.deepcopy(layout_normal)

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





########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
    # app2.run_server(debug=True)










