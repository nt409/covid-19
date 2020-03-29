import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from flask import Flask
from gevent.pywsgi import WSGIServer
import pandas as pd
from math import floor, ceil
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
# SIMPLEX
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

presets_dict = {'MSD': 'Social Distancing',
                'N': 'Do Nothing',
                'Q': 'Quarantine All',
                'H': 'Quarantine High Risk Only',
                'HL': 'Quarantine High Risk, Mild Social Distancing For Low Risk',
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


def Bar_chart_generator(data,data2 = None, data_group = None,name1=None,name2=None,preset=None,text_addition=None,color=None,y_title=None,yax_tick_form=None,maxi=True,yax_font_size_multiplier=None,hover_form=None): # ,title_font_size=None): #title
    # if preset is None:
    #     preset = 'N'
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
                mode='markers',
                marker_symbol = 'star',
                marker_size = (30/20)*font_size,
                marker_line_width=1,
                opacity=0.5,
                marker_color = 'green',
                marker_line_color = 'black',
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




########################################################################################################################
def extract_info(yy,tt,t_index,deaths):
###################################################################
    # find percentage deaths/critical care
    if deaths:
        metric_val_L_3yr = yy[params.D_L_ind,t_index-1]
        metric_val_H_3yr = yy[params.D_H_ind,t_index-1]
    else:
        metric_val_L_3yr = yy[params.C_L_ind,t_index-1]
        metric_val_H_3yr = yy[params.C_H_ind,t_index-1]

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

    if deaths:
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
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G'][magnitude])


########################################################################################################################
def figure_generator(sols,month,output,groups,hosp,num_strat,groups2,which_plots,years=2):

    font_size = 14
    
    lines_to_plot = []
    ymax = 0

    names = ['S','I','R','H','C']
    
    if 'True_deaths' in hosp:
        names.append('D')

    
    
    if num_strat=='one':
        group_use = groups
    if num_strat=='two':
        group_use = groups2


    group_string = str()
    for group in group_vec:
        if group in group_use:
            group_string = group_string + ',' + group_strings[group]
    

    linestyle_numst = ['solid','dash']
    
    len_data_points = len(sols[0]['t'])
    len_to_plot = ceil(len_data_points*years/3)
    

    ii = -1
    for sol in sols:
        ii += 1
        for name in names:
            if name in output:
                for group in group_vec:
                    if group in group_use:
                        sol['y'] = np.asarray(sol['y'])
                        if num_strat=='one':
                            name_string = ':' + group_strings[group]
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

                        line =  {'x': xx[:len_to_plot], 'y': (factor_L[group]*sol['y'][index[name],:len_to_plot] + factor_H[group]*sol['y'][index[name] + params.number_compartments,:len_to_plot]),
                                'hovertemplate': group_hover_str +
                                                 longname[name] + ': %{y}<br>' +
                                                 'Time: %{x:.1f} Months<extra></extra>',
                                'line': {'color': str(colors[name]), 'dash': line_style_use }, 'legendgroup': name ,'name': longname[name] + name_string}
                        lines_to_plot.append(line)


        # setting up pink boxes
        ICU = False
        if num_strat=='one' and 'True_deaths' in hosp and len(output)>0:
            yyy = sol['y']
            ttt = sol['t']
            c_low, c_high, ICU = time_exceeded_function(yyy,ttt)

    for line in lines_to_plot:
        ymax = max(ymax,max(line['y']))

    

    if ymax<1:
        ymax2 = min(1.1*ymax,1.01)
    else:
        ymax2 = 1.01

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
                opacity=0.15
            ))
            
    if ICU:
        if which_plots=='two':
            control_font_size = font_size*(20/24) # '10em'
            ICU_font_size = font_size*(16/24) # '10em'

            yval_pink = 0.3
            yval_blue = 0.82
        else:
            control_font_size = font_size*(25/24) #'11em'
            ICU_font_size = font_size*(20/24) # '10em'

            yval_pink = 0.35
            yval_blue = 0.86
        text_angle_blue = None
        xshift_use = None


        for c_min, c_max in zip(c_low, c_high):
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
                    opacity=0.3,
                    xref = 'x',
                    yref = 'y'
                ))
            # annotz.append(dict(
            lines_to_plot.append(dict(
                    type='scatter',
                    x  = 0.5*(c_min+c_max)/month_len,
                    y  = params.ICU_capacity, #yval_pink,
                    mode='text',
                    text="ICU",
                    hoverinfo='ICU Capacity Exceeded',
                    # showarrow=False,
                    textangle= 0,
                    textfont=dict(
                        size= ICU_font_size,
                        color="purple"
                    ),
                    opacity=1,
                    # xshift= +8,
                    # xref = 'x',
                    # yref = 'paper',
            ))
        

    else:
        if which_plots=='two':
            control_font_size = font_size*(30/24) #'11em'
            yval_blue = 0.4
        else:
            yval_blue = 0.5
            control_font_size = font_size*(35/24) #'12em'
        text_angle_blue = -90
        xshift_use = 8




    if month[0]!=month[1]:
        annotz.append(dict(
                x  = max(0.5*(month[0]+month[1]), 0.5),
                y  = yval_blue,
                text="Control",
                # hoverinfo='Control In Place',
                textangle=text_angle_blue,
                font=dict(
                    size= control_font_size,
                    color="blue"
                ),
                showarrow=False,
                opacity=0.2,
                xshift= xshift_use,
                xref = 'x',
                yref = 'paper',
        ))
    



    ICU_to_plot = False
    if 'C' in output and ymax<0.05:
        ICU_to_plot = True

            
    if ICU_to_plot:
        lines_to_plot.append(
        dict(
        type='scatter',
            x=[-0.01,max(sol['t'])+1], y=[params.ICU_capacity,params.ICU_capacity],
            mode='lines',
            opacity=0.5,
            legendgroup='thresholds',
            line=dict(
            color= 'black',
            dash = 'dash'
            ),
            hovertemplate= 'ICU Capacity: %{y}',
            name= 'ICU Capacity'))
    
    if 'S' in output and ymax>0.40:
        lines_to_plot.append(
            dict(
            type='scatter',
            x=[-0.01,max(sol['t'])+1], y=[1/params.R_0,1/params.R_0],
            mode='lines',
            opacity=0.4,
            legendgroup='thresholds',
            line=dict(
            color= 'blue',
            dash = 'dash'
            ),
            hovertemplate= 'Safe Threshold; Susceptible Population<br>' +
                            'Less Than: %{y}',
            name= 'Herd Immunity Safe Threshold'))

    lines_to_plot.append(
    dict(
        type='scatter',
        x = [0,sol['t'][-1]],
        y = [ 0, params.UK_population],
        yaxis="y2",
        opacity=0,
        showlegend=False
    ))
    

    
    yy2 = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 200]
    yy = [0.95*i for i in yy2]


    for i in range(len(yy)-1):
        if yax['range'][1]>=yy[i] and yax['range'][1] < yy[i+1]:
            yax2_vec = np.linspace(0,yy2[i+1],11)

    vec = [i*(params.UK_population) for i in yax2_vec]


    # if True: # Months
    if max(sol['t'])>370:
        xtext = [str(2*i) for i in range(1+floor(max(sol['t'])/(2*month_len)))]
        xvals = [2*i for i in range(1+floor(max(sol['t'])/(2*month_len)))]
    else:
        xtext = [str(i) for i in range(1+floor(max(sol['t'])/month_len))]
        xvals = [ i for i in range(1+floor(max(sol['t'])/month_len))]
    
    if yax['range'][1]>0.5:
        yax_form = '%'
    elif yax['range'][1]>0.1:
        yax_form = '.1%'
    else:
        yax_form = '.2%'

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
                   xaxis_title='Time (Months)',
                   yaxis= dict(mirror= True,
                       range= yax['range'],
                       showline=False,
                       linewidth=0,
                   ),
                #    yaxis_type="log",
                   yaxis_tickformat = yax_form,
                   xaxis= dict(
                        range= [0, (years/3)*max(sol['t'])/month_len],
                        showline=False,
                        linewidth=0,
                        # linecolor='red',
                        ticktext = xtext,
                        tickvals = xvals
                       ),
                   yaxis_title='Proportion of Population',
                   yaxis2 = dict(
                        title = 'UK Population',
                        overlaying='y1',
                        showline=False,
                        linewidth=0,
                        showgrid=True,
                        range = yax['range'],
                        side='right',
                        ticktext = [human_format(vec[i]) for i in range(len(yax2_vec))],
                        tickvals = [i for i in  yax2_vec],
                   )
                   )

    return {'data': lines_to_plot, 'layout': layout}










########################################################################################################################
layout_intro = html.Div([dbc.Col([
                            dbc.Jumbotron([##
                                html.Div( [

    dbc.Tabs(id='intro-tabs',
             active_tab='tab_0',
             children = [
                

        dbc.Tab(label='Definitions and Strategies', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_0', children=[

            html.H3('Overview',className='display-4',
            style = {'margin-top': '1vh', 'fontSize': '300%'}),
            html.Hr(className='my-2'),

            dbc.Col([
            
            dbc.Row([
            dbc.Col([
            html.P('Click below to use the interactive model:',
            style={'fontSize':'110%'}
            ),
            dbc.Button('Start Calculating', href='/inter', size='lg', color='success',
            style={'fontSize':'110%'}
            ),
            ],width=6
            ),
            dbc.Col([
            html.P('Click below to view the global data feed:',
            style={'fontSize':'110%'}
            ),
            dbc.Button('Real Time Data', href='/data', size='lg', color='warning',
            style={'fontSize':'110%'}
            ),
            ],width=6
            ),
            ],
            justify='center',
            style = {"margin-top": "25px", "margin-bottom": "15px"},
            ),



            dcc.Markdown('''            

            To find out more about control of coronavirus, read on!

            ''',
            style={'fontSize':'120%'}
            ),

            html.Hr(className='my-2'),
            
            html.H4('Introduction to the modelling study',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),

            dcc.Markdown('''            
            
            This page is intended to help you understand:

            1. How the different ways to control COVID-19 work,

            2. Why it is essential that you follow the control measures.

            But first, there are **two vital concepts** that you need to understand before we can fully explore how the control measures work.
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

            html.H4('Strategies',
            style = {'margin-top': '1vh', 'fontSize': '200%'}),
            
            html.H4('Reducing the infection rate',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),


            dcc.Markdown('''            

            Social distancing, self isolation and quarantine strategies slow the rate of spread of the infection (termed the 'infection rate'). In doing so, we can reduce the load on the healthcare system (goal 2) and (in the short term) reduce the number of deaths.

            This has been widely referred to as 'flattening the curve'; buying nations enough time to bulk out their healthcare capacity. The stricter quarantines are the best way to minimise the death rate whilst they're in place. A vaccine can then be used to generate sufficient immunity.

            However, in the absence of a vaccine these strategies do not ensure the safety of the population in future (goal 3), meaning that the population is still highly susceptible and greatly at risk of a future epidemic. This is because these strategies do not lead to any significant level of immunity within the population, so as soon as the measures are lifted the epidemic restarts. Further, strict quarantines carry a serious economic penalty.

            COVID-19 spreads so rapidly that it is capable of quickly generating enough seriously ill patients to overwhelm the intensive care unit (ICU) capacity of most healthcase systems in the world. This is why most countries have opted for strategies that slow the infection rate. It is essential that the ICU capacity is vastly increased to ensure it can cope with the number of people that may require it.
            '''),
            # dcc.Markdown('''            


            html.H4('Protecting the high risk',
            style = {'margin-top': '1vh', 'fontSize': '180%'}),

            


            dcc.Markdown('''            
            One notable feature of COVID-19 is that it puts particular demographics within society at greater risk. The elderly and the immunosuppressed are particularly at risk of serious illness caused by coronavirus.

            The **interactive model** presented here is designed to show the value is protecting the high risk members of society. It is critically important that the high risk don't catch the disease.

            If 60% of the population catch the disease, but all are classified as low risk, then very few people will get seriously ill through the course of the epidemic. However, if a mixture of high and low risk individuals catch the disease, then many of the high risk individuals will develop serious illness as a result.


            ''',
            ),
            # style={'fontSize':'100%'}

            ],
            width = True), # {'size': 10}),
            # ,width = {'size': 10}),

    #end of tab 2
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
            # style={'height': '100%', 'display': 'block', 'width': '100%'}),
            # ],
            # style={'height': '55vh', 'width': '100%'}
            # )

            ],
            # width = {'size': 11,'offset':1}
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
],width={'size':8,'offset':2})])


Results_explanation =  html.Div([
    html.H4("Instructions and Interpretation", className="display-4",style={'fontSize': '300%'}),
    # html.Hr(className='my-2'), # didn't work??
    html.Hr(className='my-2'),


    dcc.Markdown(
    '''
    ## Instructions

    Use the '**Inputs**' on the left hand bar to adjust choice of control measures. You can adjust the control measures and the length of time that they are implemented.

    Use the '**Plot Settings**' button on the left if you'd like to adjust what the plot shows. You can choose whether to consider only one of the low or high risk groups, or plot both groups together, or plot all lines.

    Use the '**Model Structure**' options on the left to choose whether to include deaths or not.

    Selecting '**Custom**' allows you to choose your own control measures. You can independently adjust the *infection rate* (how quickly the disease is transmitted between people) for each group. This corresponds to e.g. a lockdown for one group, and less strict measures for another.
    
    You may also directly compare the performance of **two** different custom strategies (how one performs relative to the other).

    All infection rates are given relative to a baseline level (at 100%) for COVID-19.

    ## Interpretation

    The plots will show a prediction for how coronavirus will affect the population. It is assumed that control measures are in place for a **maximum of one year**.
    
    We consider the effect of control in the **absence** of a vaccine. Of course, if a vaccine were introduced this would greatly help reduce the damage caused by COVID-19, and would further promote the use of the quarantine strategy before relying on the vaccine to generate [**herd immunity**](/intro).

    You can see how quickly the ICU capacity could be overwhelmed. You can also see how important it is to protect the high risk group (potentially by specifically reducing their transmission rate whilst allowing infection to spread more freely through lower risk groups).

    For further explanation, read the [**Introduction**](/intro).
    
    '''
    ,style={'fontSize': '100%'}
    ),
])


########################################################################################################################

layout_inter = html.Div([
    dbc.Row([
        # column_1,
        



# column_1 = 
                dbc.Col([

                        ##################################

                                        # form group 1
                                        dbc.FormGroup([

                                                        # start of col 1a
                                                        dbc.Col([
                                                                    dbc.Container([
                                                                    html.Div([
                                                                    ],style={'height': '2vh'}
                                                                    ),
                                                                    ]),

                                                                    dbc.Jumbotron([
                                                                            

                                                                            
                                                                            html.H4(
                                                                                html.Span(
                                                                                        'Control Choice',
                                                                                        id="tooltip-cont-choice",
                                                                                        style={"textDecoration": "underline", "cursor": "pointer"},
                                                                                )
                                                                                ,style={'fontSize': '150%'}),

                                                                            dbc.Tooltip(
                                                                                "Choose one of the preset strategies below, or define your own custom strategy. You may also choose how long your strategy is in place for using the months of control slider.",
                                                                                target="tooltip-cont-choice",
                                                                            ),

                                                                            html.H6('Strategy',style={'fontSize': '120%'}),
                                                                            

                                                                            html.Div([
                                                                                dbc.RadioItems(
                                                                                    id = 'preset',
                                                                                    options=[{'label': presets_dict[key],
                                                                                    'value': key} for key in presets_dict],
                                                                                    value= 'MSD'
                                                                                ),
                                                                            ],
                                                                            style={'fontSize': '80%'},
                                                                            ),

                                                                            
                                                                            html.H6('Months of Control',style={'fontSize': '120%'}),
                                                                            html.Div([
                                                                            dcc.RangeSlider(
                                                                                        id='month-slider',
                                                                                        min=0,
                                                                                        max=floor(params.months_run_for),
                                                                                        step=1,
                                                                                        # pushable=0,
                                                                                        marks={i: str(i) for i in range(0,floor(params.months_run_for)+1,3)},
                                                                                        value=[1,initial_month],
                                                                            ),
                                                                            ],
                                                                            style={'fontSize': '180%'},
                                                                            ),
                                                                                    
        ###################################



                                                                            dbc.Button([
                                                                                html.Span(
                                                                                        'Custom Options',
                                                                                        id="tooltip-custom",
                                                                                        style={"textDecoration": "underline", "cursor": "pointer"},
                                                                                ),
                                                                            
                                                                            
                                                                            ],
                                                                            color='success',
                                                                            className='mb-3',
                                                                            id="collapse-button-custom",
                                                                            style={'fontSize': '110%'} # 'margin-top': '1vh',
                                                                            ),

                                                                            dbc.Tooltip(
                                                                                "Choose your own custom strategy. You can compare two strategies directly or consider one only. The choice consists of selecting whether to accelerate or decelerate spread of COVID-19 (using the 'infection rate' sliders). You can choose a different infection rate for the low risk group compared to the high risk group.",
                                                                                target="tooltip-custom",
                                                                            ),

                                                                            
                                                                            dbc.Collapse(
                                                                                [


                                                                                            html.H6('Number Of Strategies',style={'fontSize': '100%'}),

                                                                                            html.Div([
                                                                                                    dbc.RadioItems(
                                                                                                    id = 'number-strats-slider',
                                                                                                    options=[
                                                                                                        {'label': 'One', 'value': 'one'},
                                                                                                        {'label': 'Two', 'value': 'two'}, #, 'disabled': True},
                                                                                                    ],
                                                                                                    value= 'one',
                                                                                                    inline=True,
                                                                                                    # labelStyle={'display': 'inline-block'}
                                                                                                    # style={'fontSize': '120%'}
                                                                                                    ),

                                                                                                    
                                                                                                    
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
                                                                                            ],id='things-not-grey'),


                                                                                            html.Div([
                                                                                                
                                                                                                    dbc.RadioItems(
                                                                                                    options=[
                                                                                                        {'label': 'One', 'value': 'one','disabled': True},
                                                                                                        {'label': 'Two', 'value': 'two','disabled': True}, #, 'disabled': True},
                                                                                                    ],
                                                                                                    value= 'one',
                                                                                                    inline=True
                                                                                                    ),
                                                                                                        

                                                                                                    html.H6('Low Risk Infection Rate (%)',style={'fontSize': '100%'}),
                                                                                                    dcc.Slider(
                                                                                                        id='grey-lr-slider',
                                                                                                        min=0,
                                                                                                        max=len(params.fact_v)-1,
                                                                                                        step = 1,
                                                                                                        marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                        value=initial_lr,
                                                                                                        disabled=True
                                                                                                    ),

                                                                                                    html.H6('High Risk Infection Rate (%)',style={'fontSize': '100%'}),
                                                                                                    dcc.Slider(
                                                                                                            id='grey-hr-slider',
                                                                                                            min=0,
                                                                                                            max=len(params.fact_v)-1,
                                                                                                            step = 1,
                                                                                                            marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                            value=initial_hr,
                                                                                                            disabled=True
                                                                                                            ),
                                                                                                            
                                                                                            ],id='things-grey'),


                                                                                            html.Div([
                                                                                                    html.H6('Strategy Two: Low Risk Infection Rate (%)',style={'fontSize': '120%'}),

                                                                                                    dcc.Slider(
                                                                                                        id='low-risk-slider-2',
                                                                                                        min=0,
                                                                                                        max=len(params.fact_v)-1,
                                                                                                        step = 1,
                                                                                                        marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                        value=6,
                                                                                                    ),

                                                                                                    html.H6('Strategy Two: High Risk Infection Rate (%)',style={'fontSize': '120%'}),
                                                                                                
                                                                                                    dcc.Slider(
                                                                                                        id='high-risk-slider-2',
                                                                                                        min=0,
                                                                                                        max=len(params.fact_v)-1,
                                                                                                        step = 1,
                                                                                                        marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                        value=6,
                                                                                                        ),
                                                                                            ],id='strat-2-id'),
                                                                
                                                                                ],
                                                                                # dbc.Card(dbc.CardBody("This content is hidden in the collapse-custom")),
                                                                                id="collapse-custom",
                                                                                is_open=True,
                                                                            ),


                                                                            dbc.Button([
                                                                                html.Span(
                                                                                        'Plot Settings',
                                                                                        id="tooltip-plot",
                                                                                        style={"textDecoration": "underline", "cursor": "pointer"},
                                                                                )
                                                                                    
                                                                            ],
                                                                            color='success',
                                                                            className='mb-3',
                                                                            id="collapse-button-plots",
                                                                            style={'display': 'none', 'fontSize': '110%'} # 'margin-top': '1vh',
                                                                            ),

                                                                            dbc.Tooltip(
                                                                                "Press this button to allow you to change the plot settings. You may change the timescale ('Years To Plot'), plot different risk groups ('Groups To Plot'), and different disease progress categories ('Categories To Plot'). 'How Many Plots' gives you the option of two different y-axis scales so that it is easier to compare hospital categories to the susceptible/infected/recovered categories.",
                                                                                target="tooltip-plot",
                                                                            ),
                                                                            
                                                                            dbc.Collapse(
                                                                                [
                        
                                                                                                        html.Div([

                                                                                                                    # start of row 1b
                                                                                                                    # dbc.Row([

                                                                                                                            # start of col 1b
                                                                                                                            # dbc.Col([
                                                                                                                                                    html.H6('Years To Plot',style={'fontSize': '120%'}),
                                                                                                                                                    dcc.Slider(
                                                                                                                                                        id='years-slider',
                                                                                                                                                        min=1,
                                                                                                                                                        max=3,
                                                                                                                                                        marks={i: str(i) for i in range(1,4)},
                                                                                                                                                        value=2,
                                                                                                                                                    ),

                                                                                                                                                    html.H6('How Many Plots',style={'fontSize': '120%'}),
                                                                                                                                                    dbc.RadioItems(
                                                                                                                                                        id='how-many-plots-slider',
                                                                                                                                                        options=[
                                                                                                                                                            {'label': 'One Plot', 'value': 'all'},
                                                                                                                                                            # {'label': 'One Plot: Hospital Categories', 'value': 'hosp'},
                                                                                                                                                            {'label': 'Two Plots (Different Axis Scales)', 'value': 'two'},
                                                                                                                                                        ],
                                                                                                                                                        value= 'all',
                                                                                                                                                        labelStyle = {'display': 'inline-block'}
                                                                                                                                                    ),


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

                                                                                                                                            # ],width = 6),
                                                                                                                                            # dbc.Col([

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
                                                                                                                                                                        value= ['S','I','R'],
                                                                                                                                                                        labelStyle = {'display': 'inline-block'}
                                                                                                                                                                    ),
                                                                                                                                                                    
                                                                                                                                            # ],width = 6),

                                                                                                                                    # end of row 1b
                                                                                                                                    # ]),

                                                                                                                    ],id='outputs-div',
                                                                                                                    
                                                                                                                    ),
                                                                                    ],
                                                                                    id="collapse-plots",
                                                                                    is_open=False,
                                                                                    ),

                                                                                    dbc.Button([
                                                                                        html.Span(
                                                                                                'Hospital Categories',
                                                                                                id="tooltip-hosp",
                                                                                                style={"textDecoration": "underline", "cursor": "pointer"},
                                                                                        )
                                                                                        
                                                                                        ],
                                                                                        color='danger',
                                                                                        className='mb-3',
                                                                                        id="collapse-button-hospital",
                                                                                        style={'fontSize': '110%'}
                                                                                        # style={'margin-top': '1vh'}
                                                                                    ),

                                                                                    dbc.Tooltip(
                                                                                        "Press this button to change which hospital categories are modelled.",
                                                                                        target="tooltip-hosp",
                                                                                    ),
                                                                                    
                                                                                    dbc.Collapse(
                                                                                            [
                                                                                            # html.H6('Optional Hospital Categories'),

                                                                                            dbc.RadioItems(
                                                                                                id = 'hosp-cats',
                                                                                                options=[
                                                                                                    {'label': 'Critical Care', 'value': 'True_crit'},
                                                                                                    {'label': 'Critical Care and Death', 'value': 'True_deaths'},
                                                                                                ],
                                                                                                value='True_deaths'
                                                                                            ),
                                                                                        ],
                                                                                        id="collapse-hospital",
                                                                                        is_open=False,
                                                                                    ),





#########################################################################################################################################################


                                                                        

                                                        

                                                                    ]),        
                                                        # end of col 1a
                                                        ],width = {"size":12, "offset": 0}),

                                        # end of form group 1
                                        ],
                                        row=True,
                                        # style={'fontSize':'100%'}
                                        ),



                        ],
                        width = 2,
                        ),
                        # end of first col



########################################################################################################################
## start row 1
# main_page = dbc.Row([
#                     column_1,

########################################################################################################################
                        # col 2
# column_2 =              
                        dbc.Col([

                                    # store results
                                    dcc.Store(id='sol-calculated'),# storage_type='session'),
                                    dcc.Store(id='sol-calculated-do-nothing'),# storage_type='session'),
            
                                    dbc.Col([

                                    dbc.Jumbotron([
                                    # tabs
                                    dbc.Tabs(id="interactive-tabs", active_tab='tab_0', 
                                        children=[

                                        # tab 0
                                        dbc.Tab(label='Overview',
                                         label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='tab_0', children = [html.Div(id = 'text-tab-0'),]),
#########################################################################################################################################################
                                        # tab 1
                                        dbc.Tab(label='Bar Graphs', label_style={"color": "#00AEF9" , 'fontSize':'120%' }, tab_id='tab_1', children = [
                                                                                                # dbc.Jumbotron([










                                                                                                            html.H4('Strategy Outcome',id='bar_page_title',className="display-4",style={'fontSize': '300%', 'margin-bottom': '2vh'}),

                                                                                                            # dbc.Col([
                                                                                                            # ],width=6),


                    
                                                                                        dbc.Col([
                                                                                                # dbc.Row([
                                                                                                        dbc.Row([



                                                                                                                dbc.Col([
                                                                                                                # html.Div([
                                                                                                                        html.Div(id='saved-clicks',style={'display': 'none'}),
                                                                                                                        dbc.Card([
                                                                                                                            dbc.CardHeader(html.H4(style={'fontSize': '180%'}, children = 'Explanation (click on plot titles)')),
                                                                                                                            dbc.CardBody([
                                                                                                                                    html.Div(id='explanation'),
                                                                                                                            ]
                                                                                                                            )
                                                                                                                        ],
                                                                                                                        color='light',
                                                                                                                        style = {'height': '90%', 'margin-bottom': '3vh'}
                                                                                                                        ),
                                                                                                                
                                                                                                                ],
                                                                                                                width = 12,
                                                                                                                md = 6,
                                                                                                                # style={'display': 'inline-block'}
                                                                                                                ),


                                                                                                                dbc.Col([
                                                                                                                # html.Div([
                                                                                                                    html.Div(
                                                                                                                        [
                                                                                                                        dbc.Row([
                                                                                                                            html.H4(style={'fontSize': '180%'}, children = 'Total Deaths (Percentage)',id='bar-plot-1-out'),
                                                                                                                            dbc.Spinner(html.Div(id="loading-bar-output-1")),
                                                                                                                        ]),
                                                                                                                        ],
                                                                                                                        id='bar-plot-1-title',style={ 'display':'block', 'textAlign': 'left'}),

                                                                                                                        dcc.Graph(id='bar-plot-1',style=bar_non_crit_style),
                                                                                                                
                                                                                                                ],
                                                                                                                align='center',
                                                                                                                width = 12,
                                                                                                                md = 6,
                                                                                                                # width=6
                                                                                                                ),
                                                                                                                

                                                                                                        ],
                                                                                                        # style={'height': 2*bar_height}
                                                                                                        ),
                                                                                                                
                                                                                                        html.Hr(),



                                                                                                        html.Div([
                                                                                                                    dbc.Row([
                                                                                                                                dbc.Col([
                                                                                                                                # html.Div([

                                                                                                                                            html.Div(
                                                                                                                                                [dbc.Row([##
                                                                                                                                                    html.H4(style={'fontSize': '180%'}, children = 'Peak ICU Bed Capacity Requirement'),
                                                                                                                                                    dbc.Spinner(html.Div(id="loading-bar-output-3")),
                                                                                                                                                ]),##
                                                                                                                                                ],
                                                                                                                                                id='bar-plot-3-title', style={'display':'block'}),

                                                                                                                                            dcc.Graph(id='bar-plot-3',style=bar_non_crit_style),
                                                                                                                                            
                                                                                                                                            # ]),
                                                                                                                                
                                                                                                                                ],
                                                                                                                                align='center',
                                                                                                                                
                                                                                                                                width = 12,
                                                                                                                                md = 6,
                                                                                                                                ),

                                                                                                                                dbc.Col([
                                                                                                                                            html.Div(
                                                                                                                                                    [dbc.Row([##
                                                                                                                                                        html.H4(style={'fontSize': '180%'}, children = 'Time ICU Bed Capacity Exceeded'),
                                                                                                                                                        dbc.Spinner(html.Div(id="loading-bar-output-4")),
                                                                                                                                                    ]),##
                                                                                                                                                    ],
                                                                                                                                            id='bar-plot-4-title',style={'display':'block'}),

                                                                                                                                            dcc.Graph(id='bar-plot-4',style=bar_non_crit_style),
                                                                                                                                ],
                                                                                                                                align='center',
                                                                                                                                width = 12,
                                                                                                                                md = 6,
                                                                                                                                ),
                                                                                                                    ],
                                                                                                                    # style={'height': 2*bar_height}
                                                                                                                    ),
                                                                                                                            
                                                                                                                    html.Hr(),

                                                                                                            ],
                                                                                                            id = 'bar-plots-crit'
                                                                                                            ),


                                                                                                                
                                                                                                    dbc.Row([


                                                                                                            dbc.Col([
                                                                                                                        html.Div(
                                                                                                                                [dbc.Row([##
                                                                                                                                    html.H4(style={'fontSize': '180%'}, children = 'Herd Immunity Threshold'),
                                                                                                                                    dbc.Spinner(html.Div(id="loading-bar-output-2")),
                                                                                                                                ]),##
                                                                                                                                ],
                                                                                                                        id='bar-plot-2-title',style={ 'display':'block'}),

                                                                                                                        dcc.Graph(id='bar-plot-2',style=bar_non_crit_style),
                                                                                                                        

                                                                                                            
                                                                                                            ],
                                                                                                            align='center',
                                                                                                            width = 12,
                                                                                                            md = 6,
                                                                                                            ),


                                                                                                            dbc.Col([
                                                                                                                    
                                                                                                                    html.Div(
                                                                                                                            [dbc.Row([##
                                                                                                                                html.H4(style={'fontSize': '180%'}, children = 'Time Until Herd Immunity Threshold Reached'),
                                                                                                                                dbc.Spinner(html.Div(id="loading-bar-output-5")),
                                                                                                                            ]),##
                                                                                                                            ],
                                                                                                                    id='bar-plot-5-title',style={ 'display':'block'}),

                                                                                                                    dcc.Graph(id='bar-plot-5',style=bar_non_crit_style),
                                                                                                                    

                                                                                                            
                                                                                                            ],
                                                                                                            align='center',
                                                                                                            width=12,
                                                                                                            md = 6,
                                                                                                            ),

                                                                                                    ],
                                                                                                    # style={'height': 2*bar_height}
                                                                                                    ),


                                                                                        ],width=True),
                                                                        ]),
#########################################################################################################################################################
                                                                
                                                                
                                                                

                                                                
                                                                # tab 2
                                                                dbc.Tab(label='Results and Explanation', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='DPC',children=[
                                                                                    
                                                                                                html.H4('Strategy Outcome',id='line_page_title',className="display-4",style={'fontSize': '300%'}),
                                                                                                dbc.Row([
                                                                                                        html.H4("Disease Progress Curves",
                                                                                                        style={'margin-left': '2vw', 'margin-bottom': '3vh','margin-top': '1vh','fontSize': '200%'}
                                                                                                        ),# style={'color':'blue'}),

                                                                                                        dbc.Spinner(html.Div(id="loading-line-output-1")),
                                                                                                        ],
                                                                                                        # justify='center',
                                                                                                        # style={'height':'6vh'}
                                                                                                ),

                                                                                                
                                                                                                dcc.Graph(id='line-plot-1',style={'display': 'none'}),

                                                                                                dbc.Container([html.Div([],style={'height': '3vh'})]),

                                                                                                dcc.Graph(id='line-plot-2',style={'display': 'none'}),

                                                                                                dbc.Container([html.Div([],style={'height': '1vh'})]),
                                                                                                html.Hr(className='my-2'),
                                                                                                dbc.Container([html.Div([],style={'height': '2vh'})]),

                                                                                                
                                                                                                dbc.Row([

                                                                                                    # dbc.Col([
                                                                                                    # dbc.Jumbotron([
                                                                                                        Results_explanation,
                                                                                                    # ]),
                                                                                                    # ],width=8),

                                                            ####################################################################################
                                                            ####################################################################################

                                                                ####################################################################################
                                                                ####################################################################################
                                                                                                        ]), # row below graphs
                                                                                            
                                                                                                    ]),
                                                                                                                dbc.Tab(label='Model Structure', label_style={"color": "#00AEF9", 'fontSize':'120%'}, tab_id='model_s',children=[
                                                                                                        
                                                                                                                                                html.Div([
                                                                                                                                                                # dbc.Col([
                                                                                                                                                                    dbc.Jumbotron([

                                                                                                                                                                    html.H4('Model Structure',
                                                                                                                                                                    style = {'margin-top': '1vh', 'fontSize': '300%'}),

                                                                                                                                                                    html.Hr(),
                                                                                                                                                                    dcc.Markdown(
                                                                                                                                                                    '''
                                                                                                                                                                    We present a compartmental model for COVID-19, split by risk categories. The model is very simplistic but still captures the basic spread mechanism. It is far simpler than the [Imperial College model](/https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf), but it uses similar parameter values and can capture much of the relevant information in terms of how effective control will be.

                                                                                                                                                                    It is intended solely as an illustrative, rather than predictive tool. We plan to increase the sophistication of the model and to update parameters as more (and better) data become available to us. In particular we will shortly be adding the real time global data feed as an input into the model, so that the simulation initial conditions will be based on current data.

                                                                                                                                                                    '''

                                                                                                                                                                    ),



                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585498493/Capture_yl2c6f.png',
                                                                                                                                                                    style={'width':'60%','display': 'block','margin-top': '1vh','margin-bottom': '1vh'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585498493/model_ddd3um.png',
                                                                                                                                                                    style={'width':'60%','display': 'block','margin-top': '1vh','margin-bottom': '1vh'}
                                                                                                                                                                    ),

                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    dcc.Markdown('''

                                                                                                                                                                    The selection of risk categories is done in the crudest way possible - an age split at 60 years (based on the age structure data below). A more nuanced split would give a more effective control result, since there are older people who are at low risk and younger people who are at high risk. In many cases, these people will have a good idea of which risk category they belong to.

                                                                                                                                                                    The model uses a weighted average across the age classes below and above 60 to calculate the probability of a member of each class getting hospitalised or needing critical care.
                                                                                                                                                                    '''),



                                                                                                                                                                    html.Hr(),
                                                                                                                                                                    


                                                                                                                                                                    html.H4('Parameter Values',style={'fontSize': '200%'}),

                                                                                                                                                                    html.H4('Age Structure',style={'fontSize': '180%'}),
                                                                                                                                                                    generate_table(df),
                                                                                                                                                                    
                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    The age data is taken from [**GOV.UK**](/https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest) and the hospitalisation and critical care data is from the [**Imperial College Paper**](/https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf) (Ferguson et al.).

                                                                                                                                                                    To find the probability of a low risk case getting hospitalised (or subsequently put in critical care), we take a weighted average by proportion of population.
                                                                                                                                                                    
                                                                                                                                                                    Setting the age split to be at 60 years gives us the following figures:
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'margin-top': '2vh','margin-bottom': '2vh'}
                                                                                                                                                                    ),


                                                                                                                                                                    dbc.Col([
                                                                                                                                                                        dbc.Table(
                                                                                                                                                                            [
                                                                                                                                                                                html.Thead(
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                    html.Th("Parameter description"),
                                                                                                                                                                                    html.Th("Symbol"),
                                                                                                                                                                                    html.Th("Value"),
                                                                                                                                                                                    html.Th("Source")
                                                                                                                                                                                    ])
                                                                                                                                                                                    ),
                                                                                                                                                                            ]
                                                                                                                                                                            + 
                                                                                                                                                                            
                                                                                                                                                                            [
                                                                                                                                                                            html.Tbody([
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Probability high risk infection needs hospital care"),
                                                                                                                                                                                        html.Td("h_H"),
                                                                                                                                                                                        html.Td('{0:.3f}'.format(params.frac_hosp_H)),
                                                                                                                                                                                        html.Td(html.A('Age table above (which uses Imperial and GOV.UK data)'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Probability low risk infection needs hospital care"),
                                                                                                                                                                                        html.Td("h_L"),
                                                                                                                                                                                        html.Td('{0:.3f}'.format(params.frac_hosp_L)),
                                                                                                                                                                                        html.Td(html.A('Age table above'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Probability high risk hospital case needs critical care"),
                                                                                                                                                                                        html.Td("c_H"),
                                                                                                                                                                                        html.Td('{0:.3f}'.format(params.frac_crit_H)),
                                                                                                                                                                                        html.Td(html.A('Age table above'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Probability low risk hospital case needs critical care"),
                                                                                                                                                                                        html.Td("c_L"),
                                                                                                                                                                                        html.Td('{0:.3f}'.format(params.frac_crit_L)),
                                                                                                                                                                                        html.Td(html.A('Age table above'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Probability survive critical care"),
                                                                                                                                                                                        html.Td("d"),
                                                                                                                                                                                        html.Td(str(0.5)),
                                                                                                                                                                                        html.Td(html.A('Ferguson et al.',href='https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Basic reproduction number"),
                                                                                                                                                                                        html.Td("R0"),
                                                                                                                                                                                        html.Td('{0:.1f}'.format(params.R_0)),
                                                                                                                                                                                        html.Td(html.A('Ferguson et al.',href='https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Infection rate"),
                                                                                                                                                                                        html.Td("beta"),
                                                                                                                                                                                        html.Td('{0:.3f}'.format(params.beta) + ' per day'),
                                                                                                                                                                                        html.Td(html.A('=mu*R0'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Average time infectious"),
                                                                                                                                                                                        html.Td("1/mu"),
                                                                                                                                                                                        html.Td('{0:.0f}'.format(1/params.recovery_rate) + ' Days'),
                                                                                                                                                                                        html.Td(html.A('Anderson et al',href='https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30567-5/fulltext'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Average time in hospital"),
                                                                                                                                                                                        html.Td("1/alpha"),
                                                                                                                                                                                        html.Td('{0:.0f}'.format(1/params.hosp_rate)  + ' Days'),
                                                                                                                                                                                        html.Td(html.A('Ferguson et al.**',href='https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf'))
                                                                                                                                                                                        ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Average time in critical care"),
                                                                                                                                                                                        html.Td("1/nu"),
                                                                                                                                                                                        html.Td('{0:.0f}'.format(1/params.death_rate)  + ' Days'),
                                                                                                                                                                                        html.Td(html.A('Ferguson et al.**',href='https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf'))

                                                                                                                                                                                    ]),        
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("NHS ICU beds"),
                                                                                                                                                                                        html.Td(''),
                                                                                                                                                                                        html.Td('8/100,000'),
                                                                                                                                                                                        html.Td(html.A('Ferguson et al.',href='https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf'))
                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("Initial number infected (split proportionally by risk)"),
                                                                                                                                                                                        html.Td("I_0"),
                                                                                                                                                                                        html.Td('625'),
                                                                                                                                                                                        html.Td(html.A('Anderson et al',href='https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30567-5/fulltext'))

                                                                                                                                                                                    ]),
                                                                                                                                                                                    html.Tr([ 
                                                                                                                                                                                        html.Td("UK population size (approx)"),
                                                                                                                                                                                        html.Td("N"),
                                                                                                                                                                                        html.Td('60M'),
                                                                                                                                                                                        html.Td(html.A('Anderson et al',href='https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30567-5/fulltext'))

                                                                                                                                                                                    ]),
                                                                                                                                                                                ]),
                                                                                                                                                                            ],
                                                                                                                                                                            bordered=True,
                                                                                                                                                                            # dark=True,
                                                                                                                                                                            hover=True,
                                                                                                                                                                            responsive=True,
                                                                                                                                                                            striped=True,
                                                                                                                                                                        ),
                                                                                                                                                                    ],
                                                                                                                                                                    ),

                                                                                                                                                                    html.P('** the Imperial paper uses 8 days in hospital if critical care is not required (as do we). It uses 16 days (with 10 in ICU) if critical care is required. Instead, if critical care is required we use 8 days in hospital (non-ICU) and then either recovery or a further 8 in intensive care (leading to either recovery or death).',
                                                                                                                                                                    style={'fontSize':'80%'}),









                                                                                                                                                                    ]),
                                                                                                                                                                    

                                                                                                                                                            # ],width={'size':8,'offset':2}
                                                                                                                                                            # ), 
                                                                                                                                            ],style={'fontSize': '100%'})
                                                                                                                        ]),

                                                                                                ]),
                                                                                    ]),


                                        
                                    ],width=11),

                        ],
                        width=10)
                        # end of col 2


########################################################################################################################
        # end of row 1
########################################################################################################################


    ]
    )],
    style={'fontSize' : '2vh'}
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
                        label='Introduction',value='intro',
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
            ], id='main-tabs', value='intro'),
        ], style={'width': '100vw'},
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
                    style={'margin-top': '2vh','fontSize': '7vh'}
                    ),

                    html.P('Disclaimer: this work is intended for educational purposes only and not decision making. There are many uncertainties in the COVID debate and there are limitations to the model.',
                    style={'margin-top': '1vh','margin-bottom': '1vh','fontSize': '1.5vh'}
                    ),
                    # html.Hr(),

                    # html.Div([
                    #     dcc.Markdown(
                    #     '''
                    #     An implementation of a model parameterised for COVID-19.
                        
                    #     ## Read our [**introduction**](/intro), experiment with the [**interactive model**](/inter), or explore [**real time data**](/data).
                        
                    #     Authors: Nick Taylor and Daniel Muthukrishna.
                    #     '''
                    #     ,style={'fontSize': 18}),
                    #     ]),

                    # ],fluid=True)
                    # html.P(,
                    # className="lead"),
                ],width=True,
                style={'margin-left': '10vh'}
                ),
            ],
            align="start",
            style={'backgroundColor': '#EEEEEE'}
            ),

        # ]),
        ##
        # navbar
        html.Div([navbar]),
        ##
        # # page content
        dcc.Location(id='url', refresh=False),

        html.Footer(["Authors: ",
                     html.A('Nick P. Taylor and Daniel Muthukrishna.') #, href='https://twitter.com/DanMuthukrishna'), ". ",
                    ],
                    style={'textAlign': 'center', 'fontSize': '1.6vh'}),

                    #  html.A('Source code', href='https://github.com/daniel-muthukrishna/covid19'), ". ",
                    #  "Data is taken from ",
                    #  html.A("Worldometer", href='https://www.worldometers.info/coronavirus/'), " if available or otherwise ",
                    #  html.A("John Hopkins University (JHU) CSSE", href="https://github.com/ExpDev07/coronavirus-tracker-api"), "."
        

        ],
        # style={'fontSize': '2vh'}
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
    else:
        return 'intro'




@app.callback(
            [Output('grey-lr-slider', 'value'),
            Output('grey-hr-slider', 'value')],
            [
            Input('preset', 'value'),
            ])
def preset_sliders(preset):
    if preset in preset_dict_low:
        return preset_dict_low[preset], preset_dict_high[preset]
    else: # 'N'
        return preset_dict_low['N'], preset_dict_high['N']



@app.callback(
    [Output("collapse-custom", "is_open"),
    Output("collapse-button-custom", "color")],
    [Input("collapse-button-custom", "n_clicks")],
    [State("collapse-custom", "is_open")],
)
def toggle_collapse(n, is_open):
    color = 'success'
    if n is not None and is_open:
        color = 'warning'
    else:
        color = 'success'
    if n:
        return [not is_open, color]
    return [is_open, color]



@app.callback(
    [Output("collapse-hospital", "is_open"),
    Output("collapse-button-hospital", "color")],
    [Input("collapse-button-hospital", "n_clicks")],
    [State("collapse-hospital", "is_open")],
)
def toggle_collapse(n, is_open):
    color = 'warning'
    if n is not None and not is_open:
        color = 'success'
    else:
        color = 'warning'
    if n:
        return [not is_open, color]
    return [is_open, color]

@app.callback(
    [Output("collapse-plots", "is_open"),
    Output("collapse-button-plots", "color")],
    [Input("collapse-button-plots", "n_clicks")],
    [State("collapse-plots", "is_open")],
)
def toggle_collapse(n, is_open):
    color = 'warning'
    if n is not None and not is_open:
        color = 'success'
    else:
        color = 'warning'
    if n:
        return [not is_open, color]
    return [is_open, color]

##############################################################################################################################
@app.callback([Output('infections-linear', 'figure'),
               Output('infections-log', 'figure'),
               Output('deaths-linear', 'figure'),
               Output('deaths-log', 'figure'),
               Output('active-linear', 'figure'),
               Output('active-log', 'figure'),
               Output('hidden-stored-data', 'children'),
               Output("loading-icon", "children")],
              [Input('button-plot', 'n_clicks'),
                Input('main-tabs', 'value'),
               Input('start-date', 'date'),
               Input('end-date', 'date'),
               Input('show-exponential-check', 'value'),
               Input('normalise-check', 'value')],
              [State('hidden-stored-data', 'children')] +
              [State(c_name, 'value') for c_name in COUNTRY_LIST])
def update_plots(n_clicks, tab, start_date, end_date, show_exponential, normalise_by_pop, saved_json_data, *args):
    if True: # tab ==
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
        for title in ['Cases', 'Deaths', 'Currently Infected']:
            if normalise_by_pop:
                axis_title = f"{title} (% of population)"
            else:
                axis_title = title
            fig_linear = []
            fig_log = []

            layout_linear = {
                'yaxis': {'title': axis_title, 'type': 'linear', 'showgrid': True},
                'showlegend': True,
            }
            layout_log = {
                'yaxis': {'title': axis_title, 'type': 'log', 'showgrid': True},
                'showlegend': True,
            }

            for fig in [fig_linear, fig_log]:
                if show_exponential:
                    fig.append(go.Scatter(x=[datetime.date(2020, 2, 20)],
                                        y=[0],
                                        mode='lines',
                                        line={'color': 'black', 'dash': 'dash'},
                                        showlegend=True,
                                        name=fr'Best exponential fits',
                                        yaxis='y1',
                                        legendgroup='group2', ))
                    label = fr'COUNTRY : best fit (doubling time)'
                else:
                    label = fr'COUNTRY'
                fig.append(go.Scatter(x=[datetime.date(2020, 2, 20)],
                                    y=[0],
                                    mode='lines+markers',
                                    line={'color': 'black'},
                                    showlegend=True,
                                    name=label,
                                    yaxis='y1',
                                    legendgroup='group2', ))

            for i, c in enumerate(country_names):
                if title not in country_data[c]:
                    continue
                if country_data[c] is None:
                    print("Cannot retrieve data from country:", c)
                    continue

                dates = country_data[c][title]['dates']
                xdata = np.arange(len(dates))
                ydata = country_data[c][title]['data']
                ydata = np.array(ydata).astype('float')

                if normalise_by_pop:
                    ydata = ydata/POPULATIONS[c] * 100

                date_objects = []
                for date in dates:
                    date_objects.append(datetime.datetime.strptime(date, '%Y-%m-%d').date())
                date_objects = np.asarray(date_objects)

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
                # log_yfit = b * xdata[model_date_mask] + logA
                lin_yfit = np.exp(logA) * np.exp(b * model_xdata)

                if show_exponential:
                    if np.log(2) / b >= 1000:
                        double_time = 'no growth'
                    else:
                        double_time = fr'{np.log(2) / b:.1f} days to double'
                    label = fr'{c.upper():<10s}: {np.exp(b):.2f}^t ({double_time})'
                else:
                    label = fr'{c.upper():<10s}'
                for fig in [fig_linear, fig_log]:
                    fig.append(go.Scatter(x=date_objects,
                                        y=ydata,
                                        mode='lines+markers',
                                        marker={'color': colours[i]},
                                        line={'color': colours[i]},
                                        showlegend=True,
                                        name=label,
                                        yaxis='y1',
                                        legendgroup='group1', ))
                    if show_exponential:
                        fig.append(go.Scatter(x=model_dates,
                                            y=lin_yfit,
                                            mode='lines',
                                            line={'color': colours[i], 'dash': 'dash'},
                                            showlegend=False,
                                            name=fr'Model {c.upper():<10s}',
                                            yaxis='y1',
                                            legendgroup='group1', ))

            out.append({'data': fig_linear, 'layout': layout_linear})
            out.append({'data': fig_log, 'layout': layout_log})

        out.append(json.dumps(country_data))
        out.append(None)

    return out


##################################################################################################################################





@app.callback(
    [
    Output('things-grey','style'),
    Output('things-not-grey','style'),
    
    Output('strat-2-id', 'style'),

    Output('strat-hr-infection','children'),
    Output('strat-lr-infection','children'),

    Output('groups-to-plot-radio','style'),
    Output('groups-checklist-to-plot','style'),

    Output('categories-to-plot-checklist','options'),                                   
    ],
    [
    Input('number-strats-slider', 'value'),
    Input('preset', 'value'),
    Input('hosp-cats', 'value'),
    ])
def invisible_or_not(num,preset,hosp_cats):
    
    style = None
    style_2 = {'display': 'none'}


    if preset!='C':
        style   = {'display': 'none'}
        style_2 = None

    if hosp_cats=='True_deaths':
        options=[
                    {'label': 'Susceptible', 'value': 'S'},
                    {'label': 'Infected', 'value': 'I'},
                    {'label': 'Recovered', 'value': 'R'},
                    {'label': 'Hospitalised', 'value': 'H'},
                    {'label': 'Critical Care', 'value': 'C'},
                    {'label': 'Deaths', 'value': 'D'}
                ]
    else:
        options=[
                    {'label': 'Susceptible', 'value': 'S'},
                    {'label': 'Infected', 'value': 'I'},
                    {'label': 'Recovered', 'value': 'R'},
                    {'label': 'Hospitalised', 'value': 'H'},
                    {'label': 'Critical Care', 'value': 'C'},
                    {'label': 'Deaths', 'value': 'D','disabled': True}
                ]
    
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

    

    return [style_2,style,says_strat_2,strat_H, strat_L ,groups_radio,groups_checklist,options]

########################################################################################################################

@app.callback(
    Output('sol-calculated', 'data'),
    [
    Input('preset', 'value'),
    # Input('years-slider', 'value'),
    Input('month-slider', 'value'),
    Input('low-risk-slider', 'value'),
    Input('high-risk-slider', 'value'),
    Input('low-risk-slider-2', 'value'),
    Input('high-risk-slider-2', 'value'),
    Input('number-strats-slider', 'value'),
    Input('hosp-cats', 'value'),
    ])
def find_sol(preset,month,lr,hr,lr2,hr2,num_strat,hosp): # years
    
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
    
    crit = True
    deaths = False
    if 'True_deaths' in hosp:
        deaths = True
    
    # t_stop = 365*years
    t_stop = 365*3


    months_controlled = [month_len*i for i in month]
    if month[0]==month[1]:
        months_controlled= None

    sols = []
    sols.append(simulator().run_model(beta_L_factor=lr,beta_H_factor=hr,t_control=months_controlled,critical=crit,death=deaths,T_stop=t_stop)) 
    if num_strat=='two':
        sols.append(simulator().run_model(beta_L_factor=lr2,beta_H_factor=hr2,t_control=months_controlled,critical=crit,death=deaths,T_stop=t_stop)) 
    
    return sols # {'sols': sols}


@app.callback(
    Output('sol-calculated-do-nothing', 'data'),
    [
    # Input('years-slider', 'value'),
    Input('hosp-cats', 'value'),
    ])
def find_sol_do_noth(hosp): # years

    crit = True
    deaths = False
    if 'True_deaths' in hosp:
        deaths = True
    
    # t_stop = 365*years
    t_stop = 365*3

    
    sol_do_nothing = simulator().run_model(beta_L_factor=1,beta_H_factor=1,t_control=None,critical=crit,death=deaths,T_stop=t_stop)
    
    return sol_do_nothing # {'do_nothing': sol_do_nothing}

########################################################################################################################


def cards_fn(death_stat_1st,dat3_1st,herd_stat_1st,color_1st_death,color_1st_herd,color_1st_ICU,width,number_of_crit_or_dead_text,crit_text_on_or_off):
    return dbc.Row([
        # dbc.Col([
            dbc.Col([
                dbc.Card(
                [
                    dbc.CardHeader(
                        html.Span(
                                number_of_crit_or_dead_text,
                                id="tooltip-dead",
                                style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        ),
                    dbc.CardBody([html.H1(str(round(death_stat_1st,1))+'%',className='card-title',style={'fontSize': '150%'})]),
                    dbc.CardFooter('compared to doing nothing'),

                ],color=color_1st_death,inverse=True
            )
            ],width=width,style={'textAlign': 'center'}),

            dbc.Tooltip(
                "This box shows the reduction in deaths (or critical care cases, depending on settings) due to the control strategy choice.",
                target="tooltip-dead",
            ),

            dbc.Col([
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.Span(
                                'ICU requirement:',
                                id="tooltip-ICU",
                                style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        ),
                    dbc.CardBody([html.H1(str(round(dat3_1st,1)) + 'x',className='card-title',style={'fontSize': '150%'})],),
                    dbc.CardFooter('multiple of current capacity'),

                ],color=color_1st_ICU,inverse=True
            )
            ],width=width,style=crit_text_on_or_off),

            dbc.Tooltip(
                "COVID-19 can cause a large number of serious illnesses very quickly. This box shows the extent to which the NHS capacity would be overwhelmed by the strategy choice (if nothing was done to increase capacity).",
                target="tooltip-ICU",
            ),
            dbc.Col([
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.Span(
                                'Herd immunity:',
                                id="tooltip-herd",
                                style={"textDecoration": "underline", "cursor": "pointer"},
                        ),
                        ),
                    dbc.CardBody([html.H1(str(round(herd_stat_1st,1))+'%',className='card-title',style={'fontSize': '150%'})]),
                    dbc.CardFooter('of safe threshold'),

                ],color=color_1st_herd,inverse=True
            )
            ],width=width,style={'textAlign': 'center'}),
            dbc.Tooltip(
                "This box shows how close to the safety threshold for herd immunity we got. If we reached (or exceeded) the threshold it will say 100%. However, this is the least important goal since an uncontrolled pandemic will reach safe levels of immunity very quickly, but cause lots of serious illness in doing so.",
                target="tooltip-herd",
            ),
        # ],width=True)
    ],
    no_gutters=True,
    style={'margin-top': '2vh', 'margin-bottom': '2vh','fontSize':'100%'})




def outcome_fn(month,beta_L,beta_H,death_stat_1st,herd_stat_1st,dat3_1st,death_stat_2nd,herd_stat_2nd,dat3_2nd,number_of_crit_or_dead_metric,hosp,preset,number_strategies,which_strat):
    
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
        Outcome_title = strat_name + ' Strategy ' + num_st + 'Outcome'
    else:
        Outcome_title = strat_name + ' Strategy Two Outcome'
    
    if 'True_crit' in hosp:
        crit_text_on_or_off   = {'display': 'none'}
    if 'True_deaths' in hosp:
        crit_text_on_or_off  = {'display': 'block','textAlign': 'center'}

    number_of_crit_or_dead_text = 'Reduction in ' + number_of_crit_or_dead_metric + ':' 

    if crit_text_on_or_off['display'] != 'none':
        width = 4
    else:
        width = 6
    
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
        return dbc.Jumbotron(
            html.Div([
            # dbc.Col([


                
            # dbc.Container([
            
                    dbc.Row([
                        html.H4(Outcome_title,className='display-4',style={'fontSize':'300%'}),
                    ],
                    justify='center'
                    ),

                    # html.Div([
                    # ],style={'height': '3vh'}
                    # ),
                    # ]),

                    dbc.Row([
                        html.P('In the absence of a vaccine, when compared to doing nothing:', style={'fontSize': '100%'}),
                    ],
                    justify='center', style={'margin-top': '1vh', 'margin-bottom': '1vh'}
                    ),

            
            # dbc
            dbc.Row([

                dbc.Col([
                    dbc.Table(
                        
                        [
                        html.Tbody([
                                html.Tr([ 
                                    html.Td(html.H5(["High Risk ",
                                        html.Span(
                                            "Infection Rate",
                                            id="tooltip-hr",
                                            style={"textDecoration": "underline", "cursor": "pointer"},
                                        ),
                                        ],style={'color': 'white', 'fontSize': '100%'})),
                                    html.Td(html.H5('{0:,.0f}'.format(100*beta_H) + '%',style={'color': 'white', 'fontSize': '100%'}))
                                    ]),
                                html.Tr([ 
                                    html.Td(html.H5("Low Risk Infection Rate",style={'color': 'white', 'fontSize': '100%'})),
                                    html.Td(html.H5('{0:,.0f}'.format(100*beta_L) + '%',style={'color': 'white', 'fontSize': '100%'}))
                                ]),
                                html.Tr([ 
                                    html.Td(html.H5("Control Starts",style={'color': 'white', 'fontSize': '100%'})),
                                    html.Td(html.H5([
                                        html.Span(
                                            'Month ' + str(month[0]),
                                            id="tooltip-month",
                                            style={"textDecoration": "underline", "cursor": "pointer"},
                                        ),
                                        ],style={'color': 'white', 'fontSize': '100%'}))
                                ]),
                                html.Tr([ 
                                    html.Td(html.H5("Control Ends",style={'color': 'white', 'fontSize': '100%'})),
                                    html.Td(html.H5('Month ' + str(month[1]),style={'color': 'white', 'fontSize': '100%'}))
                                ]),
                            ]),
                        ],
                        bordered=True,
                        dark=True,
                        hover=True,
                        responsive=True,
                        striped=True,
                    style={'margin-bottom': '2vh'} # 'margin-left': '1vh', 'margin-right': '1vh', 
                    ),
                ],
                width=12,
                md = 4,
                ),
            
                dbc.Tooltip(
                    "The Infection Rate relates to how quickly the disease is transmitted. Control measures can affect transmission rates (typically lowering them). Use the 'Control Choice' bar on the left to adjust.",
                    target="tooltip-hr",
                ),
                
                dbc.Tooltip(
                    "Use the 'Control Choice' bar on the left to adjust when we start controlling the epidemic. When control is not in place the infection rates remain at a baseline level (100%). When control is in place the infection rates are modified (by an amount depending on the choice of control)",
                    target="tooltip-month",
                ),
            
                dbc.Col([

                                html.H3('After 1 year:',style={'fontSize': '180%'}),
                    
                                cards_fn(death_stat_1st,dat3_1st,herd_stat_1st,color_1st_death,color_1st_herd,color_1st_ICU,width,number_of_crit_or_dead_text,crit_text_on_or_off),

                                html.H3('After 2 years:',style={'fontSize': '180%'}),

                                cards_fn(death_stat_2nd,dat3_2nd,herd_stat_2nd,color_2nd_death,color_2nd_herd,color_2nd_ICU,width,number_of_crit_or_dead_text,crit_text_on_or_off),


                ],
                width=12,
                md=8
                ),


            ],
            align='center',
            ),

            ],style=on_or_off)
        )



########################################################################################################################







########################################################################################################################


@app.callback(
            [Output('line-plot-intro', 'figure'),
            Output('line-plot-intro-2', 'figure')],
            [
            Input('intro-tabs', 'active_tab'),
            ],
            [
            State('hosp-cats', 'value'),
            State('sol-calculated-do-nothing', 'data'),
            ])
def intro_content(tab,hosp,sol_do_n): 
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
            
            deaths=False
            if 'True_deaths' in hosp:
                deaths=True
            sols.append(simulator().run_model(beta_L_factor=lr,beta_H_factor=hr,t_control=months_controlled,critical=True,death=deaths,T_stop=365*year_to_run)) 
            sols.append(sol_do_n)
            fig1 = figure_generator(sols,month,output_use,['BR'],'True_deaths','two',['BR'],'all')
            fig2 = figure_generator(sols,month,output_use_2,['BR'],'True_deaths','two',['BR'],'all')

        
        return fig1, fig2












@app.callback([
                Output('text-tab-0', 'children'),
                Output('bar_page_title', 'children'),
                Output('line_page_title', 'children'),


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

                Output('collapse-button-plots', 'style'),

                Output('line-plot-1', 'figure'),
                Output('line-plot-2', 'figure'),
                
                Output('loading-line-output-1','children'),
                
                
                Output('bar-plots-crit', 'style'),
                


                Output('bar-plot-1-out', 'children'),
                

                Output('line-plot-1', 'style'),
                Output('line-plot-2', 'style'),

                ],
                [
                Input('interactive-tabs', 'active_tab'),
                Input('main-tabs', 'value'),

                
                Input('sol-calculated', 'data'),

                # or any of the plot categories
                Input('groups-checklist-to-plot', 'value'),
                Input('groups-to-plot-radio','value'),                                      
                Input('how-many-plots-slider','value'),
                Input('categories-to-plot-checklist', 'value'),
                Input('years-slider', 'value'),


                ],
               [
                # State('url', 'pathname'),
                # State('intro-tabs', 'active_tab'),
                State('sol-calculated-do-nothing', 'data'),
                State('preset', 'value'),
                State('month-slider', 'value'),
                State('hosp-cats', 'value'),
                State('number-strats-slider', 'value'),
                ])
def render_interactive_content(tab,tab2,sols,groups,groups2,which_plots,output,years,sol_do_nothing,preset,month,hosp,num_strat): # pathname, tab_intro pathname


########################################################################################################################
    text_object_0 = ['']

    bar1 = dummy_figure
    bar2 = dummy_figure
    bar3 = dummy_figure
    bar4 = dummy_figure
    bar5 = dummy_figure

    
    Strat_outcome_title = presets_dict[preset] + ' Strategy Outcome'


    outputs_style = {'display': 'none'}

    fig1 = dummy_figure
    fig2 = dummy_figure


    bar_on_or_off = None
    bar1_title = 'Total Deaths (Percentage)'
    line_plot_style_1 = {'display': 'none'}
    line_plot_style_2 = {'display': 'none'}


    if tab2=='interactive':
   
   
        if preset!='C':
            num_strat = 'one'
            

        # crit = True    
        deaths = False
        if 'True_deaths' in hosp:
            deaths = True


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
        ########################################################################################################################
            #loop start
            if sols is not None and tab!='DPC':

                if deaths:
                    metric = 'deaths'
                else:
                    metric = 'critical care cases'


                for ii in range(len(sols)):
                    if sols[ii] is not None:
                        sol = sols[ii]
                        
                        yy = np.asarray(sol['y'])
                        tt = np.asarray(sol['t'])

                        
                        num_t_points = yy.shape[1]
                        # extract_info(yy,num_t_points,deaths)
                        metric_val_L_3yr, metric_val_H_3yr, ICU_val_3yr, herd_fraction_out, time_exc, time_reached = extract_info(yy,tt,num_t_points,deaths)
                        
                        crit_cap_data_L_3yr.append(metric_val_L_3yr) #
                        crit_cap_data_H_3yr.append(metric_val_H_3yr) #
                        ICU_data_3yr.append(ICU_val_3yr)
                        herd_list_3yr.append(herd_fraction_out) ##
                        time_exceeded_data.append(time_exc) ##
                        time_reached_data.append(time_reached) ##


                        num_t_2yr = ceil(2*num_t_points/3)
                        metric_val_L_2yr, metric_val_H_2yr, ICU_val_2yr, herd_fraction_out = extract_info(yy,tt,num_t_2yr,deaths)[:4]

                        crit_cap_data_L_2yr.append(metric_val_L_2yr) #
                        crit_cap_data_H_2yr.append(metric_val_H_2yr) #
                        ICU_data_2yr.append(ICU_val_2yr)
                        herd_list_2yr.append(herd_fraction_out) ##


                        num_t_1yr = ceil(num_t_points/3)
                        metric_val_L_1yr, metric_val_H_1yr, ICU_val_1yr, herd_fraction_out = extract_info(yy,tt,num_t_1yr,deaths)[:4]

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
            # tab 0
            if sols is not None and tab=='tab_0':

                text_object_0 = html.Div([
                    outcome_fn(month,sols[0]['beta_L'],sols[0]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[0],herd_list_2yr[0],ICU_data_2yr[0],metric,hosp,preset,number_strategies = num_strat,which_strat=1),
                    html.Hr(),
                    outcome_fn(month,sols[1]['beta_L'],sols[1]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[1],herd_list_2yr[1],ICU_data_2yr[1],metric,hosp,preset,number_strategies = num_strat,which_strat=2),
                    ],
                    style = {'fontSize': '2vh'}
                    )

                





        ########################################################################################################################
            bar1_title = 'Total Deaths (Percentage)'

            if tab!='tab_1':
                bar1 = dummy_figure
                bar2 = dummy_figure
                bar3 = dummy_figure
                bar4 = dummy_figure
                bar5 = dummy_figure

            if sols is not None and tab=='tab_1':

                if not deaths:
                    bar1_title = 'Maximum Percentage Of Population In Critical Care'



                crit_cap_bar_1yr = [crit_cap_data_L_1yr[i] + crit_cap_data_H_1yr[i] for i in range(len(crit_cap_data_H_1yr))]
                crit_cap_bar_3yr = [crit_cap_data_L_3yr[i] + crit_cap_data_H_3yr[i] for i in range(len(crit_cap_data_H_3yr))]


                bar1 = Bar_chart_generator(crit_cap_bar_1yr      ,text_addition='%'         , y_title='Population'                    , hover_form = '%{x}, %{y:.3%}'                                                   ,data_group=crit_cap_bar_3yr, yax_tick_form='.1%') # name1='Low Risk',name2='High Risk'
                bar2 = Bar_chart_generator(herd_list_1yr         ,text_addition='%'         , y_title='Percentage of Safe Threshold'  , hover_form = '%{x}, %{y:.1%}<extra></extra>'          ,color = 'mediumseagreen' ,data_group=herd_list_3yr,yax_tick_form='.1%',maxi=False,yax_font_size_multiplier=0.8) # preset = preset,
                bar3 = Bar_chart_generator(ICU_data_1yr          ,text_addition='x current' , y_title='Multiple of Current Capacity'  , hover_form = '%{x}, %{y:.1f}x Current<extra></extra>' ,color = 'powderblue'     ,data_group=ICU_data_3yr  ) # preset = preset,
                bar4 = Bar_chart_generator(time_exceeded_data    ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'peachpuff'   ) # preset = preset,
                bar5 = Bar_chart_generator(time_reached_data     ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'lemonchiffon') # preset = preset,

            if deaths:
                bar_on_or_off = None
            else:
                bar_on_or_off = {'display': 'none'}


        ########################################################################################################################

            if tab!='DPC':
                fig1 = dummy_figure
                fig2 = dummy_figure


            if sols is not None and tab=='DPC':
                output_2 = [i for i in output if i in ['C','H','D']]
                outputs_style = {'display': 'block', 'fontSize': '110%'}

                if len(output)>0:
                    fig1 = figure_generator(sols[:-1],month,output,groups,hosp,num_strat,groups2,which_plots,years)
                else:
                    fig1 = dummy_figure

                if len(output_2)>0:
                    fig2 = figure_generator(sols[:-1],month,output_2,groups,hosp,num_strat,groups2,which_plots,years)
                else:
                    fig2 = dummy_figure
            
        ##############

            fig_height = '55vh'
            fig_height_2 = '50vh'
            fig_width = '95%'

            if which_plots=='all':
                line_plot_style_1 = {'height': fig_height, 'width': fig_width, 'display': 'block'}
                line_plot_style_2 = {'display': 'none'}

            if which_plots=='two':
                line_plot_style_1 = {'height': fig_height_2, 'width': fig_width, 'display': 'block'}
                line_plot_style_2 = {'height': fig_height_2, 'width': fig_width, 'display': 'block'}
                

########################################################################################################################

    return [
    text_object_0,
    Strat_outcome_title,
    Strat_outcome_title,
    bar1,
    bar2,
    bar3,
    bar4,
    bar5,
    html.Div(), # 
    html.Div(), # 
    html.Div(), # 
    html.Div(), # 
    html.Div(), # 
    outputs_style,
    fig1,
    fig2,
    html.Div(),

    bar_on_or_off,
    bar1_title,
    line_plot_style_1,
    line_plot_style_2
    ]

########################################################################################################################











########################################################################################################################
@app.callback(
    Output('explanation', 'children'),
    [
    Input('bar-plot-1-title', 'n_clicks'),
    Input('bar-plot-2-title', 'n_clicks'),
    Input('bar-plot-3-title', 'n_clicks'),
    Input('bar-plot-4-title', 'n_clicks'),
    Input('bar-plot-5-title', 'n_clicks'),
    ],
    [State('saved-clicks','children')])
def use_clicks(nclicks_1,nclicks_2,nclicks_3,nclicks_4,nclicks_5,state): 

    explan = dcc.Markdown('''Click on the titles of each graph for an explanation of each plot''',style={'fontSize': '2vh'})
    txt1 = dcc.Markdown('''

                    This plot shows a prediction for the number of deaths caused by the epidemic in the absence of a vaccine. It also shows the split between the deaths in the high and low risk groups.
                    
                    Most outcomes result in a much higher proportion of high risk deaths, so it is critical that any strategy should protect the high risk.

                    ''',style={'fontSize': '2vh' })
    txt2 = dcc.Markdown('''

                    This plot shows how close to the 60% population immunity the strategy gets.
                    
                    Strategies with a lower *infection rate* can delay the course of the epidemic but once the strategies are lifted there is no protection through herd immunity. Strategies with a high infection rate can risk overwhelming healthcare capacity.

                    The optimal outcome is obtained by making sure the 60% that do get the infection are from the low risk group.

                    ''',style={'fontSize': '1.7vh' })
    txt3 = dcc.Markdown('''

                    This plot shows the maximum ICU capacity needed.
                    
                    Better strategies reduce the load on the healthcare system by reducing the numbers requiring Intensive Care at any one time.

                    ''',style={'fontSize': '2vh' })
    txt4 = dcc.Markdown('''

                    This plot shows the length of time for which ICU capacity is exceeded, over the calculated number of years.

                    Better strategies will exceed the ICU capacity for shorter lengths of time.

                    ''',style={'fontSize': '2vh' })
    txt5 = dcc.Markdown('''

                    This plot shows the length of time until the safe threshold for population immunity is 95% reached.
                    
                    We allow within 5% of the safe threshold, since some strategies get very close to full safety very quickly and then asymptotically approach it (but in practical terms this means the ppulation is safe).

                    The longer it takes to reach this safety threshold, the longer the population must continue control measures because it is at risk of a further epidemic.

                    ''',style={'fontSize': '1.7vh' })

    text_dict = {
        '1': txt1,
        '2': txt2,
        '3': txt3,
        '4': txt4,
        '5': txt5
    } 
    click_dict = {
        '1': nclicks_1,
        '2': nclicks_2,
        '3': nclicks_3,
        '4': nclicks_4,
        '5': nclicks_5
    } 

    if state is not None:
        state_dict = dict([i.split(':') for i in state.split(' ')])
            
        for key in state_dict.keys():
            current_state = click_dict[key]
            if current_state is None:
                current_state = 0

            if state_dict[key] == 'None':
                old_state = 0
            else:
                old_state = float(state_dict[key])

            if current_state > old_state:
                explan = text_dict[key]

    return [html.H3(explan),]

##################
@app.callback(
    Output('saved-clicks', 'children'),
    [
    Input('bar-plot-1-title', 'n_clicks'),
    Input('bar-plot-2-title', 'n_clicks'),
    Input('bar-plot-3-title', 'n_clicks'),
    Input('bar-plot-4-title', 'n_clicks'),
    Input('bar-plot-5-title', 'n_clicks'),
    ])
def save_clicks(nclicks_1,nclicks_2,nclicks_3,nclicks_4,nclicks_5):
    clicks_to_save = '1:{} 2:{} 3:{} 4:{} 5:{}'.format(nclicks_1,nclicks_2,nclicks_3,nclicks_4,nclicks_5)
    return clicks_to_save
########################################################################################################################











########################################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
    # app2.run_server(debug=True)










