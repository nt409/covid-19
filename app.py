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
from dan_get_data import get_data, COUNTRY_LIST_WORLDOMETER # , USE_API
from dan_constants import POPULATIONS
import datetime
import json


min_date = get_data('uk')['Currently Infected']['dates'][0]
max_date = get_data('uk')['Currently Infected']['dates'][-1]

min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )
max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d' )


COUNTRY_LIST_NICK = COUNTRY_LIST_WORLDOMETER

# if USE_API:
#     COUNTRY_LIST_NICK = COUNTRY_LIST
# else:
#     COUNTRY_LIST_NICK = COUNTRY_LIST_WORLDOMETER

# print(USE_API)
COUNTRY_LIST_NICK = sorted(COUNTRY_LIST_NICK)
COUNTRY_LIST_NICK.remove('world')
COUNTRY_LIST_NICK.remove('china-hong-kong-sar')

initial_country = COUNTRY_LIST_NICK.index('uk')

def begin_date(date,country='uk'):
    # if country is None:
    #     return 0.1, 0, 0, 0, 0

    
    date = datetime.datetime.strptime(date.split('T')[0], '%Y-%m-%d').date()
    
    try:
        country_data = get_data(country)
    except:
        country_data = get_data('uk')

    dates = np.asarray(country_data['Currently Infected']['dates'])
    
    currently_inf_data = np.asarray(country_data['Currently Infected']['data'])
    deaths_data        = np.asarray(country_data['Deaths']['data'])
    population_country = POPULATIONS[country]

    date_objects = []
    for dt in dates:
        date_objects.append(datetime.datetime.strptime(dt, '%Y-%m-%d').date())

    try:
        index = date_objects.index(date)
    except Exception as e:
        index = -1
        print('Date error; ',e)
    
    if index>=10:
        I0    = np.float(currently_inf_data[index])
        I_ten = np.float(currently_inf_data[index-10])
        D0    = np.float(deaths_data[index])
    else:
        index=10
        I0    = np.float(currently_inf_data[index])
        I_ten = np.float(currently_inf_data[index-10])
        D0    = np.float(deaths_data[index])
        print("dates didn't go far enough back")


    

    # of resolved cases, fatality rate is 0.9%
    p = 0.009
    R0 = D0*(1-p)/p

    R0 = R0/population_country
    D0 = D0/population_country

    I0 = 2*I0/population_country

    #  H rate is 4.4% so 
    hosp_proportion = 0.044
    #  30% of H cases critical
    crit_proportion = 0.3
    # and it takes 5 days from symptoms to get hospitalised... symptoms 5 days ago, infected 5 days before
    I_ten = 2*I_ten/population_country # only half reported

    Hospitalised_all = I_ten*hosp_proportion
    I0 = I0 - Hospitalised_all

    H0 = Hospitalised_all*(1-crit_proportion)
    C0 = Hospitalised_all*crit_proportion

    return I0, R0, H0, C0, D0



########################################################################################################################

# external_stylesheets = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
tab_label_color = 'black' # "#00AEF9"
external_stylesheets = dbc.themes.SPACELAB
# Cerulean
# COSMO
# JOURNAL
# Litera
# MINTY
# SIMPLEX - not red danger
# spacelab good too
# UNITED

app = dash.Dash(__name__, external_stylesheets=[external_stylesheets])

server = app.server

app.config.suppress_callback_exceptions = True
########################################################################################################################
# setup

initial_lr = 8
initial_hr = 5
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
                'H': 'Lockdown High Risk, No Social Distancing For Low Risk',
                'HL': 'Lockdown High Risk, Social Distancing For Low Risk',
                'Q': 'Lockdown All',
                'C': 'Custom'}

presets_dict_dropdown = {'N': 'Do Nothing',
                'MSD': 'Social Distancing',
                'H': 'High Risk: Lockdown, Low Risk: No Social Dist.',
                'HL': 'High Risk: Lockdown, Low Risk: Social Dist.',
                'Q': 'Lockdown All',
                'C': 'Custom'}

preset_dict_high = {'Q': 2, 'MSD': 7, 'HL': 2,  'H': 2, 'N':10}
preset_dict_low  = {'Q': 2, 'MSD': 7, 'HL': 7, 'H': 10, 'N':10}

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
group_strings = {'BR': ' All',
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

    This plot shows a prediction for the number of deaths (*or critical care depending on hospital category settings*) caused by the epidemic.
    
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
def time_exceeded_function(yy,tt,ICU_grow):
    ICU_capac = [params.ICU_capacity*(1 + ICU_grow*time/365 ) for time in tt]
    True_vec = [ (yy[params.C_H_ind,i]+yy[params.C_L_ind,i]) > ICU_capac[i] for i in range(len(tt))]
    Crit_vals = [ (yy[params.C_H_ind,i]+yy[params.C_L_ind,i])  for i in range(len(tt))]

    c_low = [-2]
    c_high = [-1]
    ICU = False
    if max(Crit_vals)>params.ICU_capacity:

        for i in range(len(True_vec)-1):
            if not True_vec[i] and True_vec[i+1]: # entering
                ICU = True
                y1 = 100*(yy[params.C_H_ind,i]+yy[params.C_L_ind,i])
                y2 = 100*(yy[params.C_H_ind,i+1]+yy[params.C_L_ind,i+1])
                t1 = tt[i]
                t2 = tt[i+1]
                t_int = t1 + (t2- t1)* abs((100*0.5*(ICU_capac[i]+ICU_capac[i+1]) - y1)/(y2-y1)) 
                c_low.append(t_int) # 0.5 * ( tt[i] + tt[i+1]))
            if True_vec[i] and not True_vec[i+1]: # leaving
                y1 = 100*(yy[params.C_H_ind,i]+yy[params.C_L_ind,i])
                y2 = 100*(yy[params.C_H_ind,i+1]+yy[params.C_L_ind,i+1])
                t1 = tt[i]
                t2 = tt[i+1]
                t_int = t1 + (t2- t1)* abs((100*0.5*(ICU_capac[i]+ICU_capac[i+1]) - y1)/(y2-y1)) 
                c_high.append(t_int) # 0.5 * ( tt[i] + tt[i+1]))
        


    if len(c_low)>len(c_high):
        c_high.append(tt[-1]+1)
    return c_low, c_high, ICU







########################################################################################################################
def extract_info(yy,tt,t_index,ICU_grow):
###################################################################
    # find percentage deaths/critical care
    metric_val_L_3yr = yy[params.D_L_ind,t_index-1]
    metric_val_H_3yr = yy[params.D_H_ind,t_index-1]

###################################################################
    ICU_val_3yr = [yy[params.C_H_ind,i] + yy[params.C_L_ind,i] for i in range(t_index)]
    ICU_capac = [params.ICU_capacity*(1 + ICU_grow*time/365 ) for time in tt]

    ICU_val_3yr = max([ICU_val_3yr[i]/ICU_capac[i] for i in range(t_index)])

###################################################################
    # find what fraction of herd immunity safe threshold reached
    herd_val_3yr = [yy[params.S_H_ind,i] + yy[params.S_L_ind,i] for i in range(t_index)]
    
    herd_lim = 1/(params.R_0)

    herd_fraction_out = min((1-herd_val_3yr[-1])/(1-herd_lim),1)

###################################################################
    # find time ICU capacity exceeded

    time_exc = 0

    # if True:
    c_low, c_high, ICU = time_exceeded_function(yy,tt,ICU_grow)

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
def human_format(num,dp=0):
    if num<1 and num>=0.1:
        return '%.2f' % num
    elif num<0.1:
        return '%.3f' % num
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    if dp==0 and not num/10<1:
        return '%.0f%s' % (num, ['', 'K', 'M', 'B'][magnitude])
    else:
        return '%.1f%s' % (num, ['', 'K', 'M', 'B'][magnitude])


########################################################################################################################
def figure_generator(sols,month,output,groups,num_strat,groups2,ICU_to_plot=False,vaccine_time=None,ICU_grow=None,comp_dn=False,starting_date=None,country = 'uk'): # hosp

    
    population_plot = POPULATIONS[country]

    if country in ['us','uk']:
        country_name = country.upper()
    else:
        country_name = country.title()
    
    font_size = 13
    
    lines_to_plot = []
    ymax = 0

    names = ['S','I','R','H','C','D']
    
    
    if num_strat=='one':
        group_use = groups
    if num_strat=='two' or comp_dn:
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
                            if comp_dn:
                                if ii == 0:
                                    line_style_use = 'solid'
                                else:
                                    line_style_use = 'dot'
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

                        yyy_p = (100*factor_L[group]*sol['y'][index[name],:len_to_plot] + 100*factor_H[group]*sol['y'][index[name] + params.number_compartments,:len_to_plot])

                        line =  {'x': xx[:len_to_plot], 'y': yyy_p,
                                'hovertemplate': group_hover_str +
                                                 longname[name] + ': %{y:.2f}%, ' + '%{text} <br>' +
                                                 'Time: %{x:.1f} Months<extra></extra>',
                                'text': [human_format(i*population_plot/100,dp=1) for i in yyy_p],
                                'line': {'color': str(colors[name]), 'dash': line_style_use }, 'legendgroup': name ,'name': longname[name] + name_string}
                        lines_to_plot.append(line)


        # setting up pink boxes
        ICU = False
        # print(ii,num_strat,group_use,output)
        if ii==0 and num_strat=='one' and len(group_use)>0 and len(output)>0: # 'True_deaths' in hosp 
            yyy = sol['y']
            ttt = sol['t']
            c_low, c_high, ICU = time_exceeded_function(yyy,ttt,ICU_grow)

    for line in lines_to_plot:
        ymax = max(ymax,max(line['y']))

    


    yax = dict(range= [0,min(1.1*ymax,100)])
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
        control_font_size = font_size*(18/24) # '10em'
        ICU_font_size = font_size*(18/24) # '10em'

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
        ICU_line = [100*params.ICU_capacity*(1 + ICU_grow*i/365) for i in sol['t']]
        lines_to_plot.append(
        dict(
        type='scatter',
            x=xx[:len_to_plot], y=ICU_line[:len_to_plot],
            mode='lines',
            opacity=0.8,
            legendgroup='thresholds',
            line=dict(
            color= 'maroon',
            dash = 'dash'
            ),
            hovertemplate= 'ICU Capacity<extra></extra>',
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
            hovertemplate= 'Vaccination starts<extra></extra>',
            name= 'Vaccination starts'))

    

    lines_to_plot.append(
    dict(
        type='scatter',
        x = [0,sol['t'][-1]],
        y = [ 0, population_plot],
        yaxis="y2",
        opacity=0,
        hoverinfo = 'skip',
        showlegend=False
    ))
    

    
    # yy2 = [0, 10**(-6), 2*10**(-6), 5*10**(-6), 10**(-5), 2*10**(-5), 5*10**(-5), 10**(-4), 2*10**(-4), 5*10**(-4), 10**(-3), 2*10**(-3), 5*10**(-3), 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100, 200]
    yy2 = [0]
    for i in range(8):
        yy2.append(10**(i-5))
        yy2.append(2*10**(i-5))
        yy2.append(5*10**(i-5))

    yy = [i for i in yy2] # 0.95*i


    for i in range(len(yy)-1):
        if yax['range'][1]>yy[i] and yax['range'][1] <= yy[i+1]:
            pop_vec_lin = np.linspace(0,yy2[i+1],11)

    vec = [i*(population_plot) for i in pop_vec_lin]

    log_bottom = -8
    log_range = [log_bottom,np.log10(yax['range'][1])]

    pop_vec_log_intermediate = np.linspace(log_range[0],ceil(np.log10(pop_vec_lin[-1])), 1+ ceil(np.log10(pop_vec_lin[-1])-log_range[0]) )

    pop_log_vec = [10**(i) for i in pop_vec_log_intermediate]
    vec2 = [i*(population_plot) for i in pop_log_vec]

    # date = datetime.datetime.strftime(date, '%Y-%#m-%d' )
    split = starting_date.split('-')

    day_start   = int(split[2]) #  int(starting_date.strftime('%d'))
    month_start = int(split[1]) #  int(starting_date.strftime('%m'))
    year_start  = int(split[0]) #  int(starting_date.strftime('%Y'))
    month_labelz = [[],[],[]]

    for j in range(4):
        for i in range(1,13):
            if i == month_start and j==0:
                month_labelz[0].append(datetime.date(year_start+j, i, day_start).strftime('%d %b-%y'))
                month_labelz[1].append(datetime.date(year_start+j, i, day_start).strftime('%d %b-%y'))
                month_labelz[2].append(datetime.date(year_start+j, i, day_start).strftime('%d %b-%y'))
            elif i==1:
                month_labelz[0].append(datetime.date(year_start+j, i, 1).strftime('%b %Y'))
                month_labelz[1].append(datetime.date(year_start+j, i, 1).strftime('%b %Y'))
                month_labelz[2].append(datetime.date(year_start+j, i, 1).strftime('%b %Y'))
            elif i==2:
                month_labelz[1].append(datetime.date(year_start+j, i, 1).strftime('%b %Y'))
                month_labelz[2].append(datetime.date(year_start+j, i, 1).strftime('%b %Y'))
                month_labelz[0].append(datetime.date(year_start+j, i, 1).strftime('%b'))
                
            elif i==3: # March
                month_labelz[2].append(datetime.date(year_start+j, i, 1).strftime('%b %Y'))
                month_labelz[0].append(datetime.date(year_start+j, i, 1).strftime('%b'))
                month_labelz[1].append(datetime.date(year_start+j, i, 1).strftime('%b'))
            else:
                month_labelz[0].append(datetime.date(year_start+j, i, 1).strftime('%b'))
                month_labelz[1].append(datetime.date(year_start+j, i, 1).strftime('%b'))
                month_labelz[2].append(datetime.date(year_start+j, i, 1).strftime('%b'))
    
    month_labels = []
    for list_m in month_labelz:
        month_labels.append(list_m[(month_start-1):(month_start-1+38)])


    days_till_end_of_month = max(month_len-day_start,1)
    time_vec = [0]
    for i in range(36):
        time_vec.append(days_till_end_of_month/month_len+i)

    time_axis_text = []
    time_axis_vals = []
    for N in range(1,4,1):
        time_axis_text.append([str(month_labels[N-1][N*i]) for i in range(ceil(len(time_vec)/N))])
        time_axis_vals.append([time_vec[N*i] for i in range(ceil(len(time_vec)/N))])



    # print(lines_to_plot[-1])

    layout = go.Layout(
                    annotations=annotz,
                    shapes=shapez,
                    template="simple_white",
                    font = dict(size= font_size), #'12em'),
                    legend = dict(
                       font=dict(size=font_size*(20/24)),#'8em'),
                       x = 0.5,
                       y = 1.03,
                       xanchor= 'center',
                       yanchor= 'bottom'
                   ),
                   legend_orientation = 'h',
                   legend_title='<b> Key <b>',
                   margin=dict(t=5, b=5, l=10, r=10,pad=15),
                   yaxis= dict(mirror= True,
                        title='Proportion of Population',
                        range= yax['range'],
                        showline=False,
                        automargin=True
                        # title_standoff = 10
                        # tickformat=yax_form
                   ),
                #    yaxis_tickformat = yax_form,

                   xaxis= dict(
                        # title='Time (Months)',
                        range= [0, (2/3)*max(sol['t'])/month_len],
                        showline=False,
                        # rangeslider_visible=True,
                        # linewidth=0,
                        ticktext = time_axis_text[1],
                        tickvals = time_axis_vals[1],
                        # automargin=False
                       ),
                    
                    yaxis2 = dict(
                        title = 'Population (' + country_name + ')',
                        overlaying='y1',
                        showline=False,
                        # showgrid=True,
                        range = yax['range'],
                        side='right',
                        ticktext = [human_format(0.01*vec[i]) for i in range(len(pop_vec_lin))],
                        tickvals = [i for i in  pop_vec_lin],
                        automargin=True
                   ),
                    updatemenus= [
                    dict(
                        buttons=list([
                            dict(
                                args=[{"yaxis": {'title': 'Percentage of Population', 'type': 'linear', 'range': yax['range'], 'automargin': True, 'showline':False},
                                "yaxis2": {'title': 'Population (' + country_name + ')','type': 'linear', 'overlaying': 'y1', 'range': yax['range'], 'ticktext': [human_format(0.01*vec[i]) for i in range(len(pop_vec_lin))], 'tickvals': [i for i in  pop_vec_lin],'automargin': True, 'showline':False,'side':'right'}
                                }], # tickformat
                                label="Linear",
                                method="relayout"
                            ),
                            dict(
                                args=[{"yaxis": {'title': 'Percentage of Population', 'type': 'log', 'range': log_range,'automargin': True, 'showline':False},
                                "yaxis2": {'title': 'Population (' + country_name + ')','type': 'log', 'overlaying': 'y1', 'range': log_range, 'ticktext': [human_format(0.01*vec2[i]) for i in range(len(pop_log_vec))], 'tickvals': [i for i in  pop_log_vec],'automargin': True, 'showline':False,'side':'right'}
                                }], # 'tickformat': yax_form_log,
                                label="Logarithmic",
                                method="relayout"
                            )
                    ]),
                    x= 0.5,
                    xanchor="right",
                    pad={"r": 5, "t": 30, "b": 10, "l": 5},
                    active=0,
                    y=-0.13,
                    showactive=True,
                    direction='up',
                    yanchor="top"
                    ),
                    dict(
                        buttons=list([
                            dict(
                                args = ["xaxis", {'range': [0, (1/3)*max(sol['t'])/month_len] , 'ticktext': time_axis_text[0], 'tickvals': time_axis_vals[0], 'showline':False}], # 'title': 'Time (Months)', 
                                label="Years: 1",
                                method="relayout"
                            ),
                            dict(
                                args = ["xaxis", {'range': [0, (2/3)*max(sol['t'])/month_len] , 'ticktext': time_axis_text[1], 'tickvals': time_axis_vals[1], 'showline':False}], #   'title': 'Time (Months)', 
                                label="Years: 2",
                                method="relayout"
                            ),
                            dict(
                                args = ["xaxis", {'range': [0, (3/3)*max(sol['t'])/month_len] , 'ticktext': time_axis_text[2], 'tickvals': time_axis_vals[2], 'showline':False}], # 'title': 'Time (Months)', 
                                label="Years: 3",
                                method="relayout"
                            )
                    ]),
                    x= 0.5,
                    xanchor="left",
                    pad={"r": 5, "t": 30, "b": 10, "l": 5},
                    showactive=True,
                    active=1,
                    direction='up',
                    y=-0.13,
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
                            dbc.CardBody([html.H1(str(round(death_stat_1st,1))+'%',  className='card-title',style={'fontSize': '150%'})]),
                            dbc.CardFooter('compared to doing nothing'),

                        ],color=color_1st_death, inverse=True
                    )
                    ],width=4,style={'textAlign': 'center'}),
    

                    dbc.Col([
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                        ['ICU requirement:']
                                ),
                            dbc.CardBody([html.H1(str(round(dat3_1st,1)) + 'x',className='card-title',style={'fontSize': '150%'})],),
                            dbc.CardFooter('multiple of capacity'),

                        ],color=color_1st_ICU, inverse=True
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

                        ],color=color_1st_herd, inverse=True
                    )
                    ],width=4,style={'textAlign': 'center'}),
                    
        ],
        no_gutters=True),
    
    # ],
    # width=True)

    ],style={'marginTop': '2vh', 'marginBottom': '2vh','fontSize':'75%'})




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


    red_col    = 'danger' # 'red' #  '#FF4136'
    orange_col = 'warning' # 'red' #  '#FF851B'
    green_col  = 'success' # 'red' #  '#2ECC40'
    color_1st_death = green_col
    if death_stat_1st<death_thresh1:
        color_1st_death = orange_col
    if death_stat_1st<death_thresh2:
        color_1st_death = red_col

    color_1st_herd = green_col
    if herd_stat_1st<herd_thresh1:
        color_1st_herd = orange_col
    if herd_stat_1st<herd_thresh2:
        color_1st_herd = red_col

    color_1st_ICU = green_col
    if dat3_1st>ICU_thresh1:
        color_1st_ICU = orange_col
    if dat3_1st>ICU_thresh2:
        color_1st_ICU = red_col

    
    color_2nd_death = green_col
    if death_stat_2nd<death_thresh1:
        color_2nd_death = orange_col
    if death_stat_2nd<death_thresh2:
        color_2nd_death = red_col

    color_2nd_herd = green_col
    if herd_stat_2nd<herd_thresh1:
        color_2nd_herd = orange_col
    if herd_stat_2nd<herd_thresh2:
        color_2nd_herd = red_col

    color_2nd_ICU = green_col
    if dat3_2nd>ICU_thresh1:
        color_2nd_ICU = orange_col
    if dat3_2nd>ICU_thresh2:
        color_2nd_ICU = red_col




    if on_or_off['display']=='none':
        return None
    else:
        return html.Div([


                
                    dbc.Row([
                        html.H3(Outcome_title,style={'fontSize':'250%'},className='display-4'),
                    ],
                    justify='center'
                    ),
                    html.Hr(),


                    dbc.Row([
                        html.I('Compared to doing nothing. Traffic light colours indicate relative success or failure.'),
                    ],
                    justify='center', style={'marginTop': '2vh'}
                    ),

            
            # dbc
            dbc.Row([

            
                dbc.Col([


                                html.H3('After 1 year:',style={'fontSize': '180%', 'marginTop': '3vh', 'marginBottom': '3vh'}),

                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button('Reduction in deaths 🛈',
                                                    color='primary',
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
                                                    color='primary',
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
                                                    color='primary',
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

                                html.H3('After 2 years:',style={'fontSize': '180%', 'marginTop': '3vh', 'marginBottom': '3vh'}),

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
                            # dbc.Jumbotron([##
                            dbc.Row([
    
    dbc.Col([
                            html.Div([
                html.Div([


    dbc.Tabs(id='intro-tabs',
             active_tab='tab_start',
             children = [
                
        
        dbc.Tab(label='Start Here', tab_style = {'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_start', children=[
        html.H3('Introduction',
        className = 'display-4',
        style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

        html.Hr(),

        dcc.Markdown('''
        The aim of this website is to demystify modelling of infectious diseases through short videos and interactive models which let you explore how different control strategies will affect the rate that covid-19 spreads. The website has been developed by experts in epidemiology and modelling from the University of Cambridge.
        ''',
        style={'textAlign': 'justify','marginTop': '5vh', 'marginBottom': '7vh'}),
        #  Nick and Cerian are based in the Department of Plant Sciences and Daniel is based at the Institute of Astronomy.
        
        html.Hr(),
        
        html.H3('Who is this website for?',
        className = 'display-4',
        style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

        html.Hr(),

        dcc.Markdown('''
        The content is targeted at people with little or no experience of modelling and might be used as a resource for anyone wishing to understand more about the standstill. We welcome any feedback you have as we develop this resource – just email us at: [covid.at.plants@gmail.com](/intro).
        ''',
        style={'textAlign': 'justify','marginTop': '5vh', 'marginBottom': '7vh'}),

        html.Hr(),
        
        html.H3('How to use the website',
        className = 'display-4',
        style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

        html.Hr(),

        dcc.Markdown('''
        It’s up to you how you explore our website. If you already know something about modelling you may want to jump straight to the interactive model. But if you’d like to know a little more about modelling and a detailed explanation of the output then click on the next tabs in this section.
        ''',
        style={'textAlign': 'justify','marginTop': '5vh', 'marginBottom': '7vh'}),
        


        ]),
        
        dbc.Tab(label='Introduction to modelling',  tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_0', children=[

            html.H3('Introduction to mathematical modelling',className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

            html.Hr(),

            dcc.Markdown('''
            Watch this video from Dr Cerian Webb, an expert in epidemiology and modelling from the University of Cambridge.
            ''',
            style={'textAlign': 'justify'}),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/v1585755877/WhatIsModellingv2_hhqe2h.mp4',
                    controls=True,
                    style={'maxWidth':'80%','height': 'auto','marginTop': '1vh','marginBottom': '3vh'}),
                    
                    ],
                    justify='center'
                    ),
            
                    

            
            html.Hr(),


            html.H3('Introducing SIR models',className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

            html.Hr(),

            dcc.Markdown('''
            Watch this explanation from Dr Cerian Webb, to find out more about basic epidemiological models.
            ''',
            style={'textAlign': 'justify'}),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/v1585814499/StandardSIRModel_hu5ztn.mp4',
                    controls=True,
                    style={'maxWidth':'80%','height': 'auto','marginTop': '1vh','marginBottom': '3vh'}),
                    
                    ],
                    justify='center'
                    ),
            
                    
            
            html.Hr(),


            html.H3('Introducing the basic reproductive number',className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

            html.Hr(),

            dcc.Markdown('''
            Watch Dr Cerian Webb introduce the basic reproductive number.
            ''',
            style={'textAlign': 'justify'}),

            dbc.Row([
                    
                    html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/v1585987204/AllAboutR_wmf2y1.mp4',
                    controls=True,
                    style={'maxWidth':'80%','height': 'auto','marginTop': '1vh','marginBottom': '3vh'}),
                    
                    ],
                    justify='center'
                    ),
            
            html.Hr(),
            
            html.H3('Definitions',className='display-4',
            style = { 'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '3vh'}),

            dcc.Markdown('''            

            There are **two vital concepts** that you need to understand before we can fully explore how the control measures work.
            '''),
            
            html.H4('1. Basic Reproduction Number',
            style = {'marginTop': '1vh', 'fontSize': '180%'}),

            dcc.Markdown('''            
            Any infectious disease requires both infectious individuals and susceptible individuals to be present in a population to spread. The higher the number of susceptible individuals, the faster it can spread since an infectious person can spread the disease to more susceptible people before recovering.

            The average number of infections caused by a single infected person is known as the '**basic reproduction number**' (*R*). If this number is less than 1 (each infected person infects less than one other on average) then the disease won't spread. If it is greater than 1 then the disease will spread. For COVID-19 most estimates for *R* are between 2 and 3. We use the value *R*=2.4.
            '''),

            html.H4('2. Herd Immunity',
            style = {'marginTop': '1vh', 'fontSize': '180%'}),
            
            dcc.Markdown('''            


            Once the number of susceptible people drops below a certain threshold (which is different for every disease, and in simpler models depends on the basic reproduction number), the population is no longer at risk of an epidemic (so any new infection introduced won't cause infection to spread through an entire population).

            Once the number of susceptible people has dropped below this threshold, the population is termed to have '**herd immunity**'. Herd immunity is either obtained through sufficiently many individuals catching the disease and developing personal immunity to it, or by vaccination.

            For COVID-19, there is a safe herd immunity threshold of around 60% (=1-1/*R*), meaning that if 60% of the population develop immunity then the population is **safe** (no longer at risk of an epidemic).

            Coronavirus is particularly dangerous because most countries have almost 0% immunity since the virus is so novel. Experts are still uncertain whether you can build immunity to the virus, but the drop in cases in China would suggest that you can. Without immunity it would be expected that people in populated areas get reinfected, which doesn't seem to have happened.
            
            A further concern arises over whether the virus is likely to mutate. However it is still useful to consider the best way to managing each strain.
            ''',
            style={'textAlign': 'justify'}),

            
        ]),

        dbc.Tab(label='COVID-19 Control Strategies',tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_control', children=[
            html.H3('Keys to a successful control strategy',
            className = 'display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

            html.Hr(),


            dcc.Markdown('''            
            There are three main goals a control strategy sets out to achieve:

            1. Reduce the number of deaths caused by the pandemic,

            2. Reduce the load on the healthcare system,

            3. Ensure the safety of the population in future.

            An ideal strategy achieves all of the above whilst also minimally disrupting the daily lives of the population.

            However, controlling COVID-19 is a difficult task, so there is no perfect strategy. We will explore the advantages and disadvantages of each strategy.
            ''',
            style={'textAlign': 'justify'}),
            
            html.Hr(),

            html.H3('Strategies',
            className='display-4',
            style = {'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

            html.Hr(),

            
            html.H4('Reducing the infection rate',
            style = {'fontSize': '180%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),


            dcc.Markdown('''            

            Social distancing, self isolation and quarantine strategies slow the rate of spread of the infection (termed the 'infection rate'). In doing so, we can reduce the load on the healthcare system (goal 2) and (in the short term) reduce the number of deaths.

            This has been widely referred to as 'flattening the curve'; buying nations enough time to bulk out their healthcare capacity. The stricter quarantines are the best way to minimise the death rate whilst they're in place. A vaccine can then be used to generate sufficient immunity.

            However, in the absence of a vaccine these strategies do not ensure the safety of the population in future (goal 3), meaning that the population is still highly susceptible and greatly at risk of a future epidemic. This is because these strategies do not lead to any significant level of immunity within the population, so as soon as the measures are lifted the epidemic restarts. Further, strict quarantines carry a serious economic penalty.

            COVID-19 spreads so rapidly that it is capable of quickly generating enough seriously ill patients to overwhelm the intensive care unit (ICU) capacity of most healthcase systems in the world. This is why most countries have opted for strategies that slow the infection rate. It is essential that the ICU capacity is vastly increased to ensure it can cope with the number of people that may require it.
            ''',
            style={'textAlign': 'justify'}),


            html.H4('Protecting the high risk',
            style = {'fontSize': '180%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

            


            dcc.Markdown('''            
            One notable feature of COVID-19 is that it puts particular demographics within society at greater risk. The elderly and the immunosuppressed are particularly at risk of serious illness caused by coronavirus.

            The **interactive model** presented here is designed to show the value is protecting the high risk members of society. It is critically important that the high risk don't catch the disease.

            If 60% of the population catch the disease, but all are classified as low risk, then very few people will get seriously ill through the course of the epidemic. However, if a mixture of high and low risk individuals catch the disease, then many of the high risk individuals will develop serious illness as a result.


            ''',
            style={'textAlign': 'justify'}
            ),

        ]),


        dbc.Tab(label='How to use', tab_style = { 'textAlign': 'center', 'cursor': 'pointer'}, label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='tab_1', children=[
                    



                    # dbc.Container(html.Div(style={'height':'5px'})),
                    html.H3('How to use the interactive model',className='display-4',
                    style = { 'fontSize': '250%', 'textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),

                    # html.H4('How to use the interactive model',className='display-4'),
                    
                    html.Hr(),
                    
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
                    style={'textAlign': 'justify'}
                    # style={'fontSize':20}
                    ),
                    
                    dbc.Row([
                    dbc.Button('Start Calculating', href='/inter', size='lg', color='primary',
                    style={'marginTop': '1vh','marginLeft': '2vw', 'fontSize': '100%'}
                    ),
                    ],
                    justify='center'),
                    # ],width={'size':3,'offset':1},
                    # ),
                    ],
                    style={'marginTop': '1vh'},
                    width = True),


                        #end of tab 1
                    ]),

            
    #end of tabs
    ])
],
style={'fontSize': '1.9vh'}
)


],
style= {'marginLeft': '5vw', 'marginRight': '5vw', 'marginBottom': '5vh', 'marginTop': '5vw',}
),

],
width=12,
xl=10
),

],
justify = 'center',
style= {'width': '90%','backgroundColor': '#f4f6f7'}
),
    
],
justify='center')

]),



Results_interpretation =  html.Div([
    
    dcc.Markdown('''

    The plots show a prediction for how coronavirus will affect the population. It is assumed that control measures are in place for a **maximum of 15 months**. Explore the effect of the different control measures and the amount of time they are implemented for.
    
    In the third line plot ('intensive care'), you can see how quickly the ICU capacity (relating to the number of intensive care beds available) could be overwhelmed (within a month if no control measures are implemented). Control measures allow us to 'flatten the curve'. In addition it's essential that the healthcare capacity is rapidly increased.

    You can also see how important it is to **protect the high risk group** (potentially by specifically reducing their transmission rate), since this is the group that contributes the most to the death rate (to see this, use the '**Groups to Plot**' checkboxes on the dropdown '**Plot Settings**').

    The figures also illustrate how a vaccine, coupled with an effective quarantine, can drastically decrease the death toll.

    For further explanation, read the [**Background**](/intro).
    
    '''
    ,style={'fontSize': '100%','textAlign': 'justify'},
    ),
])






#########################################################################################################################################################





layout_inter = html.Div([
    dbc.Row([
        # column_1,
        



                        html.Div([
                        dbc.Row([
                        dbc.Col([
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
                                         label_style={"color": tab_label_color, 'fontSize':'120%'},
                                         tab_id='tab_0',
                                         tab_style = {'minWidth':'50%','textAlign': 'center', 'cursor': 'pointer'},
                                         children = [
                                                    # html.Div([



                                                    # Instructions_layout,

                                                    # html.Hr(),

                                                    html.H3('Strategy Outcome',id='line_page_title',className="display-4",style={'fontSize': '250%','textAlign': 'center', 'marginTop': '3vh', 'marginBottom': '1vh'}),

                                                    # html.Hr(),

                                                    dcc.Markdown('''
                                                    *In this Section we explore possible outcomes of different choices of **COVID-19 control**.*
                                                    
                                                    *Pick a **strategy** below.*

                                                    '''
                                                    ,style = {'marginTop': '3vh', 'marginBottom': '3vh', 'textAlign': 'center'}
                                                    ),
                                                    # 1. **Pick your strategy** (bar below)
                                                    
                                                    # 2. Choose which **results** to display (button below).
                                             
                                             
                                                    html.Hr(),

                                                    # html.Hr(),

                                                # dbc.Row([
            
                                                        # dbc.Col([
                                                            # dbc.Jumbotron([
                                                                



############################################################################################################################################################################################################################
                                                                                            # html.Div([

                                                                                                        ##################################

                                                                                                                        # form group 1
                                                                                                                        dbc.FormGroup([

                                                                                ########################################################################################################################

                                                                                                                                                    dbc.Col([
                                                                                                                                                            


                                                                                                                                                            


                                                                                ########################################################################################################################


                                                                                                                                                            
                                                                                                                                                dbc.Row([
                                                                                                                                                        dbc.Col([

                                                                                                                                                            # dbc.Modal(
                                                                                                                                                            #     [
                                                                                                                                                            #         dbc.ModalHeader("Interactive Model",style={'display': 'block','textAlign': 'center'}),
                                                                                                                                                            #         dbc.ModalBody(
                                                                                                                                                            #             dcc.Markdown(
                                                                                                                                                            #             '''
                                                                                                                                                            #             This page illustrates the possible outcomes of different **COVID-19 control strategies**.

                                                                                                                                                            #             **Pick a strategy** and explore the **results**.
                                                                                                                                                            #             ''',style={'textAlign': 'center'}),
                                                                                                                                                            #             ),
                                                                                                                                                            #         dbc.ModalFooter(
                                                                                                                                                            #             dbc.Button("Close", color='success', id='close-modal', className="ml-auto")
                                                                                                                                                            #             ,style={'display': 'block','textAlign': 'center'}
                                                                                                                                                            #         ),
                                                                                                                                                            #     ],
                                                                                                                                                            #     id="modal",
                                                                                                                                                            #     size="md",
                                                                                                                                                            #     # backdrop='static',
                                                                                                                                                            #     is_open=True,
                                                                                                                                                            #     centered=True
                                                                                                                                                            # ),

                                                                                                                                                            html.H3(['When and where'],
                                                                                                                                                            style={'fontSize': '230%', 'marginTop': "3vh", 'marginBottom': "3vh", 'textAlign': 'center'}),

                                                                                                                                                            

                                                                                                                                                        dbc.Row([

                                                                                                                                                            
                                                                                                                                                            dbc.Col([
                                                                                                                                                            html.H6('Model Start Date',style={'fontSize': '100%', 'textAlign': 'center'}),

                                                                                                                                                            dbc.Row([
                                                                                                                                                            dcc.DatePickerSingle(
                                                                                                                                                                id='model-start-date',
                                                                                                                                                                min_date_allowed = min_date + datetime.timedelta(days=10), # datetime.date(2020, 2, 25),
                                                                                                                                                                max_date_allowed = max_date, #datetime.date.today() - datetime.timedelta(days=1),
                                                                                                                                                                initial_visible_month =  max_date, # datetime.date.today() - datetime.timedelta(days=1),
                                                                                                                                                                date = max_date, # datetime.date.today() - datetime.timedelta(days=1),
                                                                                                                                                                display_format='D-MMM-YYYY',
                                                                                                                                                                style={'textAlign': 'center', 'marginTop': '0.5vh', 'marginBottom': '2vh'}
                                                                                                                                                            ),
                                                                                                                                                            ],justify='center'),
                                                                                                                                                            ],width=4),



                                                                                                                                                                # dbc.Row([
                                                                                                                                                                dbc.Col([
                                                                                                                                                                
                                                                                                                                                                html.H6('Country',style={'fontSize': '100%', 'textAlign': 'center'}),

                                                                                                                                                                
                                                                                                                                                                    html.Div([
                                                                                                                                                                    dcc.Dropdown(
                                                                                                                                                                        id = 'model-country-choice',
                                                                                                                                                                        options=[{'label': c_name.title() if c_name not in ['us', 'uk'] else c_name.upper(), 'value': num} for num, c_name in enumerate(COUNTRY_LIST_NICK)],
                                                                                                                                                                        value= initial_country,
                                                                                                                                                                        clearable = False,
                                                                                                                                                                    ),],
                                                                                                                                                                    style={'cursor': 'pointer'}),

                                                                                                                                                                ],width=4),
                                                                                                                                                                # ],justify='center'),
                                                                                                                                                            
                                                                                                                                                        ],justify='center'),



                                                                                                                                                            html.Hr(),
                                                                                                                                                            
                                                                                                                                                            html.H3(['Pick Your Strategy ',
                                                                                                                                                            dbc.Button('🛈',
                                                                                                                                                            color='primary',
                                                                                                                                                            # className='mb-3',
                                                                                                                                                            id="popover-pick-strat-target",
                                                                                                                                                            size='md',
                                                                                                                                                            style = {'cursor': 'pointer', 'marginBottom': '0.5vh'}),
                                                                                                                                                            ],
                                                                                                                                                            className = 'display-4',
                                                                                                                                                            style={'fontSize': '230%', 'marginTop': "3vh", 'marginBottom': "3vh", 'textAlign': 'center'}),

                                                                                                                                                            


                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Pick Your Strategy'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                1a. Pick the **control timings** (how long control is applied for and when it starts).

                                                                                                                                                                1b. Pick the **type of control**.

                                                                                                                                                                1c. Introduce a **vaccine** if you would like.

                                                                                                                                                                2. Pick the results type.

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

                                                                                                                                                  


                                                                                                                                                        dcc.Markdown('''*Choose the type of control and when to implement it.*''', style = {'textAlign': 'center', 'marginBottom': '3vh'}), # 'textAlign': 'left', fs 80%

                                                                                                                                                        
                                                                                                                                                    dbc.Row([


                                                                                                                                                           



                                                                                                                                                            dbc.Col([
                                                                                                                                                                


                                                                                                                                                                html.H6([
                                                                                                                                                                '1a. Months of Control ',
                                                                                                                                                                dbc.Button('🛈',
                                                                                                                                                                color='primary',
                                                                                                                                                                size='sm',
                                                                                                                                                                id='popover-months-control-target',
                                                                                                                                                                style= {'cursor': 'pointer','marginBottom': '0.5vh'}),
                                                                                                                                                                ],
                                                                                                                                                                style={'fontSize': '100%','marginTop': '1vh', 'marginBottom': '1vh','textAlign': 'center'}),


                                                                                                                                                            
                                                                                                                                                                html.Div([
                                                                                                                                                                dcc.RangeSlider(
                                                                                                                                                                            id='month-slider',
                                                                                                                                                                            min=0,
                                                                                                                                                                            max=floor(params.max_months_controlling),
                                                                                                                                                                            step=1,
                                                                                                                                                                            # pushable=0,
                                                                                                                                                                            marks={i: str(i) for i in range(0,floor(params.max_months_controlling)+1,3)},
                                                                                                                                                                            value=[0,initial_month],
                                                                                                                                                                ),
                                                                                                                                                                ],
                                                                                                                                                                # style={'fontSize': '180%'},
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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                    html.H6([
                                                                                                                                                                '1b. Control Type ',
                                                                                                                                                                dbc.Button('🛈',
                                                                                                                                                                    color='primary',
                                                                                                                                                                    # className='mb-3',
                                                                                                                                                                    size='sm',
                                                                                                                                                                    id='popover-control-target',
                                                                                                                                                                    style={'cursor': 'pointer','marginBottom': '0.5vh'}
                                                                                                                                                                    ),
                                                                                                                                                                ],
                                                                                                                                                                style={'fontSize': '100%', 'marginTop': '1vh', 'marginBottom': '1vh','textAlign': 'center'}),


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

                                                                                                                                                                                                                                                                                                                                                                                                                                                                    



                                                                                                                                                            
                                                                                                                                                            ],
                                                                                                                                                            width=6,
                                                                                                                                                            ),



                                                                                                                                                        dbc.Col([



                                                                                                                                                                
                                                                                                                                                            html.H6([
                                                                                                                                                                '1c. Vaccination starts ',
                                                                                                                                                                dbc.Button('🛈',
                                                                                                                                                                color='primary',
                                                                                                                                                                size='sm',
                                                                                                                                                                id='popover-vaccination-target',
                                                                                                                                                                style= {'cursor': 'pointer','marginBottom': '0.5vh'}),
                                                                                                                                                                ],
                                                                                                                                                                style={'fontSize': '100%','marginTop': '1vh', 'marginBottom': '1vh', 'textAlign': 'center'}),


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
                                                                                                                                                            # style={'fontSize': '180%'},
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
                                                                                                                                                                placement='left',
                                                                                                                                                            ),


                                                                                                                                                                                                                                                                                                                                                                                                                                                                    html.H6([

                                                                                                                                                                '2. Results Type ',
                                                                                                                                                                dbc.Button('🛈',
                                                                                                                                                                    color='primary',
                                                                                                                                                                    # className='mb-3',
                                                                                                                                                                    size='sm',
                                                                                                                                                                    id='popover-res-type-target',
                                                                                                                                                                    style={'cursor': 'pointer','marginBottom': '0.5vh'}
                                                                                                                                                                    ),
                                                                                                                                                                ],
                                                                                                                                                                style={'fontSize': '100%', 'marginTop': '1vh', 'marginBottom': '1vh','textAlign': 'center'}),

                                                                                                                                                            dbc.Popover(
                                                                                                                                                                    [
                                                                                                                                                                    dbc.PopoverHeader('Results'),
                                                                                                                                                                    dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                    '''

                                                                                                                                                                    Choose between disease progress curves, bar charts and strategy overviews to explore the outcome of your strategy choice.

                                                                                                                                                                    '''
                                                                                                                                                                    ),),
                                                                                                                                                                    ],
                                                                                                                                                                    id = "popover-res-type",
                                                                                                                                                                    is_open=False,
                                                                                                                                                                    target="popover-res-type-target",
                                                                                                                                                                    placement='left',
                                                                                                                                                                ),

                                                                                                                                                            






                                                                                                                                                        ],width=6),

                                                                                                                                                    ]),



                                                                                                                                                dbc.Row([


                                                                                                                                                    dbc.Col([




                                                                                                                                                            html.Div([
                                                                                                                                                            dcc.Dropdown(
                                                                                                                                                                id = 'preset',
                                                                                                                                                                options=[{'label': presets_dict_dropdown[key],
                                                                                                                                                                'value': key} for key in presets_dict_dropdown],
                                                                                                                                                                value= 'MSD',
                                                                                                                                                                clearable = False,
                                                                                                                                                            ),],
                                                                                                                                                            style={'cursor': 'pointer'}),
                                                                                                                                                            


                                                                                                                                                            


                                                                                                                                                        ],
                                                                                                                                                        width=7,
                                                                                                                                                        ),



                                                                                                                                                        dbc.Col([



                                                                                                                                                            html.Div([
                                                                                                                                                            dcc.Dropdown(
                                                                                                                                                                id = 'dropdown',
                                                                                                                                                                options=[{'label': 'Disease Progress Curves','value': 'DPC_dd'},
                                                                                                                                                                {'label': 'Bar Charts','value': 'BC_dd'},
                                                                                                                                                                {'label': 'Strategy Overview','value': 'SO_dd'},
                                                                                                                                                                ],
                                                                                                                                                                value= 'DPC_dd',
                                                                                                                                                                clearable = False,
                                                                                                                                                            ),],
                                                                                                                                                            style={'cursor': 'pointer'}),


                                                                                                                                                            


                                                                                                                                                        ],width=5),
                                                                                                                                                    ]),







                                                                                                                                                        ],width=True),
                                                                                                                                                        #end of PYS row
                                                                                                                                                    ]),

                                                                                                                                                                    
                                                                                                                                                        html.Hr(),
                                                                                        ###################################



                                                                                                                                                    dbc.Row([
                                                                                                                                                        dbc.Col([
                                                                                                                                                            dbc.Row([

                                                                                                                                                            # dbc.Row([
                                                                                                                                                            # dbc.ButtonGroup(
                                                                                                                                                            #     children=[
                                                                                                                                                            #         dbc.Button("Disease Progress Curves",color='primary',outline=True,style={'minWidth': '17vw'},id='DPC_dd',active=True),
                                                                                                                                                            #         dbc.Button("Bar Charts",             color='primary',outline=True,style={'minWidth': '17vw'},id='BC_dd'),
                                                                                                                                                            #         dbc.Button("Strategy Overview",      color='primary',outline=True,style={'minWidth': '17vw'},id='SO_dd'),
                                                                                                                                                            #     ],
                                                                                                                                                            #     className="mb-3",
                                                                                                                                                            #     size='md',
                                                                                                                                                            #     # outline=True,
                                                                                                                                                            #     style = {'marginTop': '1vh', 'marginBottom': '2vh', 'textAlign': 'center'}
                                                                                                                                                            # ),
                                                                                                                                                            # ],
                                                                                                                                                            # justify='center'),




                                                                                                                                                            dbc.ButtonGroup([
                                                                                                                                                            dbc.Button('Custom Options',
                                                                                                                                                            color='primary',
                                                                                                                                                            outline=True,
                                                                                                                                                            className='mb-3',
                                                                                                                                                            id="collapse-button-custom",
                                                                                                                                                            style={'fontSize': '100%', 'cursor': 'pointer'}
                                                                                                                                                            ),

                                                                                                                                                            dbc.Button('🛈',
                                                                                                                                                            color='primary',
                                                                                                                                                            className='mb-3',
                                                                                                                                                            id="popover-custom-options-target",
                                                                                                                                                            style={'cursor': 'pointer'}
                                                                                                                                                            )
                                                                                                                                                            ],
                                                                                                                                                            ),
                                                                                                                                                            ],
                                                                                                                                                            style={'marginTop': '0vh', 'marginBottom': '0vh'},
                                                                                                                                                            justify='center'),


                                                                                                                                                            dbc.Popover(
                                                                                                                                                                [
                                                                                                                                                                dbc.PopoverHeader('Custom Options'),
                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                '''

                                                                                                                                                                Use this to choose your own custom strategy (you must first select 'Custom' in the '**1b. Control Type**' selector above). You can compare two strategies directly or consider one only.
                                                                                                                                                                
                                                                                                                                                                The choice consists of selecting by how much to decelerate spread of COVID-19 (using the 'infection rate' sliders).
                                                                                                                                                                
                                                                                                                                                                You can also choose different infection rates for the different risk groups.
                                                                                                                                                                '''
                                                                                                                                                                ),),
                                                                                                                                                                ],
                                                                                                                                                                id = "popover-custom-options",
                                                                                                                                                                is_open=False,
                                                                                                                                                                target="popover-custom-options-target",
                                                                                                                                                                placement='bottom',
                                                                                                                                                            ),
                                                                                                                                                            

                                                                                                                                                            
                                                                                                                                                            dbc.Collapse(
                                                                                                                                                                [
                                                                                                                                                                    dbc.Row([
                                                                                                                                                                        dbc.Col([
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                dbc.Row([
                                                                                                                                                                            dbc.Col([
                                                                                                                                                                                html.H6(['Critical Care Capacity Increase ',
                                                                                                                                                                                dbc.Button('🛈',
                                                                                                                                                                                    color='primary',
                                                                                                                                                                                    size='sm',
                                                                                                                                                                                    id='popover-cc-care-target',
                                                                                                                                                                                    style= {'cursor': 'pointer','marginBottom': '0.5vh'}),
                                                                                                                                                                                    ],
                                                                                                                                                                                    style={'fontSize': '100%','marginTop': '2vh', 'marginBottom': '1vh', 'textAlign': 'center'}),

                                                                                                                                                                                dcc.Slider(
                                                                                                                                                                                        id='ICU-slider',
                                                                                                                                                                                        min = 0,
                                                                                                                                                                                        max = 5,
                                                                                                                                                                                        step = 1,
                                                                                                                                                                                        marks={i: str(i)+'x' for i in range(6)},
                                                                                                                                                                                        value=1,
                                                                                                                                                                                        # style={'fontSize': '80%'}
                                                                                                                                                                                    ),
                                                                                                                                                                                
                                                                                                                                                                                dbc.Popover(
                                                                                                                                                                                    [
                                                                                                                                                                                    dbc.PopoverHeader('ICU Capacity'),
                                                                                                                                                                                    dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                                    '''

                                                                                                                                                                                    Select '0x' to assume that ICU capacity stays constant, or pick an amount by which capacity is bulked up each year (relative to the amount initially).

                                                                                                                                                                                    '''
                                                                                                                                                                                    ),),
                                                                                                                                                                                    ],
                                                                                                                                                                                    id = "popover-cc-care",
                                                                                                                                                                                    is_open=False,
                                                                                                                                                                                    target="popover-cc-care-target",
                                                                                                                                                                                    placement='top',
                                                                                                                                                                                ),
                                                                                                                                                                        ],width=6),],
                                                                                                                                                                        justify='center'),

                                                                                                                                                                        html.Hr(),

                                                                                                                                                                            
                                                                                                                                                                            
                                                                                                                                                                            dbc.Row([




                                                                                                                                                                            dbc.Col([
                                                                                                                                                                            html.H6('Number Of Strategies',style={'fontSize': '100%','textAlign': 'center'}),

                                                                                                                                                                            dcc.Markdown('''*Set '**1b. Control Type**' to 'Custom'.*''', style = {'fontSize': '70%', 'marginTop': '1vh', 'textAlign': 'center'}),
                                                                                                                                                                            dbc.Row([
                                                                                                                                                                            dbc.RadioItems(
                                                                                                                                                                            id = 'number-strats-radio',
                                                                                                                                                                            options=[
                                                                                                                                                                                {'label': 'One', 'value': 'one'},
                                                                                                                                                                                {'label': 'Two', 'value': 'two'},
                                                                                                                                                                            ],
                                                                                                                                                                            value= 'one',
                                                                                                                                                                            inline=True,
                                                                                                                                                                            labelStyle = {'fontSize': '80%'}
                                                                                                                                                                            ),
                                                                                                                                                                            ],justify='center'),
                                                                                                                                                                            ],width=6),

                                                                                                                                                                            ],justify='center',
                                                                                                                                                                            style={'marginTop': '2vh'}),



                                                                                                                                                                            html.Hr(),
                                                                                                                                                                            




                                                                                                                                                                            # html.Hr(),
                                                                                                                                                                            
                                                                                                                                                                            dcc.Markdown('''*To adjust the following, make sure '**1b. Control Type**' is set to 'Custom'.*''', style = {'fontSize': '80%', 'marginTop': '2vh', 'textAlign': 'center'}), # 'textAlign': 'left', 

                                                                                                                                                                            dbc.Row([

                                                                                                                                                                            dbc.Button('Infection rate 🛈',
                                                                                                                                                                                    size='sm',
                                                                                                                                                                                    color='primary',
                                                                                                                                                                                    className='mb-3',
                                                                                                                                                                                    # id="popover-custom-options-target",
                                                                                                                                                                                    id = 'popover-inf-rate-target',
                                                                                                                                                                                    style={'cursor': 'pointer'}
                                                                                                                                                                                    ),
                                                                                                                                                                            ],justify='center'),

                                                                                                                                                                            dbc.Popover(
                                                                                                                                                                                [
                                                                                                                                                                                dbc.PopoverHeader('Infection Rate'),
                                                                                                                                                                                dbc.PopoverBody(dcc.Markdown(
                                                                                                                                                                                '''

                                                                                                                                                                                The *infection rate* relates to how quickly the disease is transmitted. **Control** measures affect transmission/infection rates (typically lowering them).
                                                                                                                                                                            
                                                                                                                                                                                Adjust by choosing a preset strategy  or making your own custom choice ('**1b. Control Type**').


                                                                                                                                                                                '''
                                                                                                                                                                                ),),
                                                                                                                                                                                ],
                                                                                                                                                                                id = "popover-inf-rate",
                                                                                                                                                                                is_open=False,
                                                                                                                                                                                target="popover-inf-rate-target",
                                                                                                                                                                                placement='top',
                                                                                                                                                                            ),


                                                                                                                                                                            dbc.Row([
                                                                                                                                                                                dbc.Col([
                                                                                                                                                                                html.Div(id='strat-lr-infection',style = {'textAlign': 'center','fontSize': '100%'}),
                                                                                                                                                                                
                                                                                                                                                                                
                                                                                                                                                                                dcc.Slider(
                                                                                                                                                                                    id='low-risk-slider',
                                                                                                                                                                                    min=0,
                                                                                                                                                                                    max=len(params.fact_v)-1,
                                                                                                                                                                                    step = 1,
                                                                                                                                                                                    marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                    value=initial_lr,
                                                                                                                                                                                    # style={'fontSize': '80%'}
                                                                                                                                                                                ),


                                                                                                                                                                                html.Div(id='strat-hr-infection',style = {'textAlign': 'center','fontSize': '100%'}),
                                                                                                                                                                                dcc.Slider(
                                                                                                                                                                                        id='high-risk-slider',
                                                                                                                                                                                        min=0,
                                                                                                                                                                                        max=len(params.fact_v)-1,
                                                                                                                                                                                        step = 1,
                                                                                                                                                                                        marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                        value=initial_hr,
                                                                                                                                                                                        # style={'fontSize': '80%'}
                                                                                                                                                                                        ),
                                                                                                                                                                                ],width=6),

                                                                                                                                                                                        
                                                                                                                                                                                # style={'fontSize': '80%'}),

                                                                                                                                                                                # html.Hr(),



                                                                                                                                                                                dbc.Col([
                                                                                                                                                                                        html.H6('Strategy Two: Low Risk Infection Rate (%)',style={'fontSize': '100%','textAlign': 'center'}),

                                                                                                                                                                                        dcc.Slider(
                                                                                                                                                                                            id='low-risk-slider-2',
                                                                                                                                                                                            min=0,
                                                                                                                                                                                            max=len(params.fact_v)-1,
                                                                                                                                                                                            step = 1,
                                                                                                                                                                                            marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                            value=10,
                                                                                                                                                                                        ),

                                                                                                                                                                                        html.H6('Strategy Two: High Risk Infection Rate (%)', style = {'fontSize': '100%','textAlign': 'center'}),
                                                                                                                                                                                    
                                                                                                                                                                                        dcc.Slider(
                                                                                                                                                                                            id='high-risk-slider-2',
                                                                                                                                                                                            min=0,
                                                                                                                                                                                            max=len(params.fact_v)-1,
                                                                                                                                                                                            step = 1,
                                                                                                                                                                                            marks={i: '{0:,.0f}'.format(100*params.fact_v[i]) for i in range(0,len(params.fact_v),2)}, #str(5) if i == 0 else 
                                                                                                                                                                                            value=10,
                                                                                                                                                                                            ),
                                                                                                                                                                                ],width=6,
                                                                                                                                                                                id='strat-2-id'),
                                                                                                                                                                                
                                                                                                                                                                            ],justify='center'),

                                                                                                                                                                            # html.Hr(),


                                                                                                                                                                        ]),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'),


                                                                                                                                                
                                                                                                                                                                ],
                                                                                                                                                                id="collapse-custom",
                                                                                                                                                                is_open=False,
                                                                                                                                                            ),

                                                                                                                                                        ],
                                                                                                                                                        width=12,
                                                                                                                                                        ),
                                                                                                                                                        #### end of custom row
                                                                                                                                                        ]),

                                                                                                                                                        # dbc.Row([



                                                                                                
                                                                                                
                                                                                                
                                                                                                
                                                                                                
                                                                                                


                                                                                #########################################################################################################################################################


                                                                                                                                                    ],
                                                                                                                                                    width = True
                                                                                                                                                    ),

                                                                                                                                                    # ],
                                                                                                                                                    # style={'backgroundColor': "#FFFFFF", 'marginLeft': '1vw','marginRight': '1vw','marginBottom': '2vh','marginTop': '2vh'}
                                                                                                                                                    # ),        
                                                                                                                                        

                                                                                                                                                    # ],
                                                                                                                                                    # style={'backgroundColor': "#FFFFFF",'height': '100%','width': '100%'}),

                                                                                                                                                    # ],
                                                                                                                                                    # ),        
                                                                                                                                                    
                                                                                ########################################################################################################################

                                                                                                                        # end of form group 1
                                                                                                                        ],
                                                                                                                        row=True
                                                                                                                        ),
                                                                                ########################################################################################################################

                                                                                                        # ],
                                                                                                        # style={'marginLeft':'2vh', 'marginRight':'2vh'}
                                                                                                        # ),










############################################################################################################################################################################################################################    

                                                            # ]),



                                                        # ],
                                                        # width = 12,
                                                        # style={'height': '100%'}
                                                        # ),

                                                        # dbc.Col([

                                                        # dbc.Jumbotron([
                                                    html.Hr(), # style={'marginTop': '2vh'}
                                    ##############################################################################################################################################################################################################################
                                            # start of results col

                                                    html.Div([
                                             

                                                        
                                                        dbc.Row([
                                                                html.H3('Results',
                                                                className='display-4',
                                                                style={'fontSize': '250%', 'textAlign': 'center' ,'marginTop': "1vh",'marginBottom': "1vh"}),

                                                                dbc.Spinner(html.Div(id="loading-line-output-1")),
                                                                dbc.Spinner(html.Div(id="loading-sol-1")),
                                                                ],
                                                                justify='center',
                                                        ),
                                                        
                                             
                                                      

                                             
                                                        html.Div([





                                                                        
                                                                        dcc.Markdown('''
                                                                        *Click on the info buttons for explanations*.
                                                                        ''',style = {'textAlign': 'center','marginTop': '3vh', 'marginBottom': '3vh'}),



                                                                        dbc.Col([

                                                                                                dbc.Col([
                                                                                                    html.Div(
                                                                                                        [
                                                                                                        dbc.Row([
                                                                                                            html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [
                                                                                                            
                                                                                                            html.Div(['Plot: Total Deaths (Percentage) ',
                                                                                                            
                                                                                                            dbc.Button('🛈',
                                                                                                            color='primary',
                                                                                                            className='mb-3',
                                                                                                            size='sm',
                                                                                                            id="popover-bp1-target",
                                                                                                            style={'cursor': 'pointer'}
                                                                                                            )],
                                                                                                            style= {'textAlign': 'center'}),

                                                                                                            ]),
                                                                                                        ]
                                                                                                        ,justify='center'),
                                                                                                        ],
                                                                                                        id='bar-plot-1-title',style={ 'display':'block', 'textAlign': 'left'}),

                                                                                                        dcc.Graph(id='bar-plot-1',style=bar_non_crit_style),
                                                                                                
                                                                                                

                                                                                                
                                                                                        html.Hr(),




                                                                                                                    html.Div(
                                                                                                                        [dbc.Row([##
                                                                                                                            html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                                html.Div(['Plot: Peak ICU Bed Capacity Requirement ',
                                                                                                                                dbc.Button('🛈',
                                                                                                                                color='primary',
                                                                                                                                className='mb-3',
                                                                                                                                size='sm',
                                                                                                                                id="popover-bp3-target",
                                                                                                                                style={'cursor': 'pointer'}
                                                                                                                                )
                                                                                                                                ],style= {'textAlign': 'center'}),


                                                                                                                            ]),
                                                                                                                        ],
                                                                                                                        justify='center'),##
                                                                                                                        ],
                                                                                                                        id='bar-plot-3-title', style={'display':'block'}),

                                                                                                                        dcc.Graph(id='bar-plot-3',style=bar_non_crit_style),
                                                                                                                    
                                                                                                        

                                                                                                    html.Hr(),


                                                                                                                    html.Div(
                                                                                                                            [dbc.Row([##
                                                                                                                                html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                                    html.Div(['Plot: Time ICU (Current) Bed Capacity Exceeded ',
                                                                                                                                    dbc.Button('🛈',
                                                                                                                                    color='primary',
                                                                                                                                    className='mb-3',
                                                                                                                                    size='sm',
                                                                                                                                    id='popover-bp4-target',
                                                                                                                                    style={'cursor': 'pointer'}
                                                                                                                                    )

                                                                                                                                    ],style= {'textAlign': 'center'}),

                                                                                                                                ]),
                                                                                                                                
                                                                                                                            ],
                                                                                                                            justify='center'),##
                                                                                                                            ],
                                                                                                                    id='bar-plot-4-title',style={'display':'block'}),

                                                                                                                    dcc.Graph(id='bar-plot-4',style=bar_non_crit_style),
                                                                                                    
                                                                                            html.Hr(),



                                                                                                


                                                                                                        html.Div(
                                                                                                                [dbc.Row([##
                                                                                                                    html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                        html.Div(['Plot: Herd Immunity Threshold ',
                                                                                                                        dbc.Button('🛈',
                                                                                                                        color='primary',
                                                                                                                        className='mb-3',
                                                                                                                        size='sm',
                                                                                                                        id='popover-bp2-target',
                                                                                                                        style={'cursor': 'pointer'}
                                                                                                                        )
                                                                                                                        ],
                                                                                                                        style= {'textAlign': 'center'}), # id='bar-plot-2-out'),
                                                                                                                    

                                                                                                                    ]),

                                                                                                                ],
                                                                                                            justify='center'),##
                                                                                                                ],
                                                                                                        id='bar-plot-2-title',style={ 'display':'block'}),

                                                                                                        dcc.Graph(id='bar-plot-2',style=bar_non_crit_style),
                                                                                                        

                                                                                            

                                                                                            html.Hr(),


                                                                                                    
                                                                                                    html.Div(
                                                                                                            [dbc.Row([##

                                                                                                                    html.H4(style={'fontSize': '180%', 'textAlign': 'center'}, children = [

                                                                                                                    html.Div(['Plot: Time Until Herd Immunity Threshold Reached ',
                                                                                                                        dbc.Button('🛈',
                                                                                                                        color='primary',
                                                                                                                        className='mb-3',
                                                                                                                        size='sm',
                                                                                                                        id='popover-bp5-target',
                                                                                                                        style={'cursor': 'pointer'}
                                                                                                                        )
                                                                                                                    ],style= {'textAlign': 'center'}), # id='bar-plot-5-out'),

                                                                                                                    ]),
                                                                                                                    # dbc.Spinner(html.Div(id="loading-bar-output-5")),
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

                                                                
                                                                
                                                                
                                                                # dbc.Row([
                                                                #         html.H4("Disease Progress Curves",
                                                                #         style={'marginBottom': '2vh', 'textAlign': 'center' ,'marginTop': '1vh','fontSize': '180%'} # 'marginLeft': '2vw', 
                                                                #         ),

                                                                #         dbc.Spinner(html.Div(id="loading-line-output-1")),
                                                                #         ],
                                                                #         justify='center',
                                                                # ),

                                                                
                                                                dcc.Graph(id='line-plot-1',style={'height': '70vh', 'width': '97%'}), # figure=dummy_figure,


                                                                # dbc.Col([
                                                                        dbc.Row([
                                                                            dbc.Col([


                                                                                dbc.Row(
                                                                                    # dbc.ButtonGroup([
                                                                                    # dbc.Button('Plot Settings',
                                                                                    # color='primary',
                                                                                    # outline=True,
                                                                                    # className='mb-3',
                                                                                    # id="collapse-button-plots",
                                                                                    # style={'fontSize': '120%', 'cursor': 'pointer'}
                                                                                    # ),

                                                                                    # dbc.Button('🛈',
                                                                                    # color='primary',
                                                                                    # # className='mb-3',
                                                                                    # size = 'md',
                                                                                    # id="popover-plot-settings-target",
                                                                                    # style = {'cursor': 'pointer'}
                                                                                    # )
                                                                                    html.H4("Plot Settings ",
                                                                                    style={'marginBottom': '1vh', 'textAlign': 'center' ,'marginTop': '4vh','fontSize': '180%'}),
                                                                                    
                                                                                    # ],
                                                                                    # id='plot-settings-collapse',
                                                                                    # ),
                                                                                justify =  'center'
                                                                                ),

                                                                                dbc.Col([
                                                                                dcc.Markdown('''*Plot different disease progress categories, different risk groups, compare the outcome of your strategy with the outcome of 'Do Nothing', or plot the ICU capacity.*''', 
                                                                                style = {'textAlign': 'center', 'marginBottom': '1vh' , 'marginTop': '1vh'}),
                                                                                ],width={'size':8, 'offset': 2}),

                                                                                # dbc.Popover(
                                                                                #     [
                                                                                #     dbc.PopoverHeader('Plot Settings '),
                                                                                #     dbc.PopoverBody(dcc.Markdown(
                                                                                #     '''

                                                                                #     Press this button to allow you to change the plot settings.
                                                                                    
                                                                                #     You may plot different risk groups ('Groups To Plot'), and different disease progress categories ('Categories To Plot').

                                                                                #     '''
                                                                                #     ),),
                                                                                #     ],
                                                                                #     id = "popover-plot-settings",
                                                                                #     is_open=False,
                                                                                #     target="popover-plot-settings-target",
                                                                                #     placement='bottom',
                                                                                # ),
                                                                                
                                                                                # dbc.Collapse(
                                                                                #     [
                            
                                                                                                            dbc.Row([
                                                                                                                                        dbc.Col([


                                                                                                                                                html.H6('Categories To Plot',style={'fontSize': '100%','textAlign': 'center'}),
                                                                                                                                                dbc.Col([


                                                                                                                                                    dbc.Checklist(id='categories-to-plot-checklist',
                                                                                                                                                                    options=[
                                                                                                                                                                        {'label': 'Susceptible', 'value': 'S'},
                                                                                                                                                                        {'label': 'Infected', 'value': 'I'},
                                                                                                                                                                        {'label': 'Recovered', 'value': 'R'},
                                                                                                                                                                        {'label': 'Hospitalised', 'value': 'H'},
                                                                                                                                                                        {'label': 'Critical Care', 'value': 'C'},
                                                                                                                                                                        {'label': 'Deaths (Cumulative)', 'value': 'D'},
                                                                                                                                                                    ],
                                                                                                                                                                    value= ['S','I','R','H','C','D'],
                                                                                                                                                                    # inline=True,
                                                                                                                                                                    labelStyle = {'display': 'inline-block','fontSize': '80%'},
                                                                                                                                                                ),

                                                                                                                                                ],width={'size':6 , 'offset': 3}),

                                                                                                                                        ],width=4),





                                                                                                                                        dbc.Col([

                                                                                                                                                        html.H6('Groups To Plot',style={'fontSize': '100%','textAlign': 'center'}),


                                                                                                                                                        dbc.Col([
                                                                                                                                                            dbc.Checklist(
                                                                                                                                                                id = 'groups-checklist-to-plot',
                                                                                                                                                                options=[
                                                                                                                                                                    {'label': 'Low Risk Group', 'value': 'LR'},
                                                                                                                                                                    {'label': 'High Risk Group', 'value': 'HR'},
                                                                                                                                                                    {'label': 'All Risk Groups (Sum Of Both)', 'value': 'BR'},
                                                                                                                                                                ],
                                                                                                                                                                value= ['BR'],
                                                                                                                                                                # inline=True,
                                                                                                                                                                labelStyle = {'display': 'inline-block','fontSize': '80%'},
                                                                                                                                                            ),


                                                                                                                                                            dbc.RadioItems(
                                                                                                                                                                id = 'groups-to-plot-radio',
                                                                                                                                                                options=[
                                                                                                                                                                    {'label': 'Low Risk Group', 'value': 'LR'},
                                                                                                                                                                    {'label': 'High Risk Group', 'value': 'HR'},
                                                                                                                                                                    {'label': 'All Risk Groups (Sum Of Both)', 'value': 'BR'},
                                                                                                                                                                ],
                                                                                                                                                                value= 'BR',
                                                                                                                                                                # inline=True,
                                                                                                                                                                labelStyle = {'display': 'inline-block','fontSize': '80%'},
                                                                                                                                                            ),
                                                                                                                                                        ],width={'size':6 , 'offset': 3}),
                                                                                                                                        ],width=4),


                                                                                                                                        dbc.Col([


                                                                                                                                                        html.H6("Compare with 'Do Nothing'",style={'fontSize': '100%','textAlign': 'center'}),

                                                                                                                                                        dbc.Col([


                                                                                                                                                            dbc.Checklist(
                                                                                                                                                                id = 'plot-with-do-nothing',
                                                                                                                                                                options=[
                                                                                                                                                                    {'label': 'Compare', 'value': 1},
                                                                                                                                                                ],
                                                                                                                                                                value= 0,
                                                                                                                                                                labelStyle = {'display': 'inline-block','fontSize': '80%'},
                                                                                                                                                            ),

                                                                                                                                                        ],width={'size':6 , 'offset': 3}),

                                                                                                                                                        html.H6("Plot Intensive Care Capacity",style={'fontSize': '100%','textAlign': 'center', 'marginTop': '2vh'}),


                                                                                                                                                        dbc.Col([
                                                                                                                                                            dbc.Checklist(
                                                                                                                                                                id = 'plot-ICU-cap',
                                                                                                                                                                options=[
                                                                                                                                                                    {'label': 'Plot', 'value': 1},
                                                                                                                                                                ],
                                                                                                                                                                value= 0,
                                                                                                                                                                labelStyle = {'display': 'inline-block','fontSize': '80%'},
                                                                                                                                                            ),

                                                                                                                                                        ],width={'size':6 , 'offset': 3}
                                                                                                                                                        ,style={'marginBottom': '0vh'}
                                                                                                                                                        ),
                                                                                                                                                        dcc.Markdown('''
                                                                                                                                                        *ICU capacity will only be clear on small y-scales (hospital categories only).*
                                                                                                                                                        ''',style={'fontSize': '70%', 'marginTop': '0vh'}),


                                                                                                                                        ],width=4),


                                                                                                                                                                        

                                                                                                                        ],
                                                                                                                        id='outputs-div',
                                                                                                                        no_gutters=True,
                                                                                                                        # justify='center'
                                                                                                                        ),
                                                                                                                        
                                                                                                                        # html.Hr(),
                                                                                        # ],
                                                                                        # id="collapse-plots",
                                                                                        # is_open=False,
                                                                                        # ),

                                                                                ],
                                                                                width=12,
                                                                                ),


                                                                                # end of plot settings row
                                                                                ],
                                                                                justify='center',
                                                                                no_gutters=True
                                                                                # style={'textAlign': 'center'}
                                                                                ),















                                                                # html.Hr(),

                                                                # html.H4("Hospital Categories",
                                                                # style={'marginBottom': '2vh', 'textAlign': 'center' ,'marginTop': '5vh','fontSize': '180%'} # 'marginLeft': '2vw', 
                                                                # ),

                                                                # dcc.Graph(id='line-plot-2',style={'height': '70vh', 'width': '95%'}), # figure=dummy_figure,

                                                                # html.H4("Intensive Care - 'flatten the curve'",
                                                                # style={'marginBottom': '2vh', 'textAlign': 'center' ,'marginTop': '5vh','fontSize': '180%'} # 'marginLeft': '2vw', 
                                                                # ),

                                                                # dcc.Graph(id='line-plot-3',style={'height': '70vh', 'width': '95%'}), # figure=dummy_figure,


                                                    ]
                                                    ),
                                             
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











                                                        #end of jumbo
                                                        # ]), 


                                                        # end of col
                                            #             ],
                                            #             width = 12
                                            #             ),


                                            #         # end of row
                                            # ],
                                            # # no_gutters=True
                                            # ),
                                                
                                                    
                                             ##################################################################################################
                                                    
                                                    html.Hr(style={'marginTop': '5vh'}),
    
                                                    html.H3("Interpretation", className="display-4",style={'fontSize': '250%','textAlign': 'center', 'marginTop': '1vh', 'marginBottom': '1vh'}),
                                                    html.Hr(),

                                                    # dbc.Jumbotron([
                                                    Results_interpretation,
                                                    # ]),

                                             
                                             
                                             


                                         ],
                                         
                                         ),
#########################################################################################################################################################
                                                                                                                dbc.Tab(label='Model Explanation', label_style={"color": tab_label_color, 'fontSize':'120%'}, tab_id='model_s',
                                                                                                                tab_style = {'minWidth':'50%','textAlign': 'center', 'cursor': 'pointer'},
                                                                                                                children=[
                                                                                                        
                                                                                                                                                html.Div([
                                                                                                                                                                # dbc.Col([

                                                                                                                                                                    html.H3('Model Explanation',
                                                                                                                                                                    className = 'display-4',
                                                                                                                                                                    style = {'marginTop': '1vh', 'marginBottom': '1vh', 'textAlign': 'center', 'fontSize': '250%'}),

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
                                                                                                                                                                    style = {'textAlign': 'justify'}

                                                                                                                                                                    ),



                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/Capture_lomery.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto', 'display': 'block','marginTop': '1vh','marginBottom': '1vh'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    dcc.Markdown('''

                                                                                                                                                                    The selection of risk categories is done in the crudest way possible - an age split at 60 years (based on the age structure data below). A more nuanced split would give a more effective control result, since there are older people who are at low risk and younger people who are at high risk. In many cases, these people will have a good idea of which risk category they belong to.

                                                                                                                                                                    *For the more mathematically inclined reader, a translation of the above into a mathematical system is described below.*

                                                                                                                                                                    ''',style={'textAlign': 'justify','marginTop' : '2vh','marginBottom' : '2vh'}),
                                                                                                                                                                    
                                                                                                                                                                    

                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/eqs_f3esyu.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '1vh','marginBottom': '1vh'})
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),


                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/text_toshav.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '1vh','marginBottom': '1vh'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),

                                                                                                                                                                    
                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    Of those requiring critical care, we assume that if they get treatment, a fraction *1-d* recover. However, if they don't receive it all die. The number able to get treatment must be lower than the number of ICU beds available.
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify'}),



                                                                                                                                                                    html.Hr(),

                                                                                                                                                                    html.H4('Parameter Values',style={'fontSize': '180%', 'textAlign': 'center'}),

                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    The model uses a weighted average across the age classes below and above 60 to calculate the probability of a member of each class getting hospitalised or needing critical care. Our initial conditions are updated to match new data every day (meaning the model output is updated every day, although in '**Custom Options**' there is the choice to start from any given day).

                                                                                                                                                                    We assume a 10 day delay on hospitalisations, so we use the number infected 10 days ago to inform the number hospitalised (0.044 of infected) and in critical care (0.3 of hospitalised). We calculate the approximate number recovered based on the number dead, assuming that 0.009 infections cause death. All these estimates are as per the Imperial College paper ([**Ferguson et al**](https://spiral.imperial.ac.uk/handle/10044/1/77482)).

                                                                                                                                                                    The number of people infected, hospitalised and in critical care are calculated from the recorded data. We assume that only half the infections are reported ([**Fraser et al.**](https://science.sciencemag.org/content/early/2020/03/30/science.abb6936)), so we double the recorded number of current infections.

                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify'}),



                                                                                                                                                                    

                                                                                                                                                                    dbc.Row([
                                                                                                                                                                    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1586179930/table_jo7uex.png',
                                                                                                                                                                    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '1vh','marginBottom': '1vh'}
                                                                                                                                                                    ),
                                                                                                                                                                    ],
                                                                                                                                                                    justify='center'
                                                                                                                                                                    ),



                                                                                                                                                                    html.P('** the Imperial paper uses 8 days in hospital if critical care is not required (as do we). It uses 16 days (with 10 in ICU) if critical care is required. Instead, if critical care is required we use 8 days in hospital (non-ICU) and then either recovery or a further 8 in intensive care (leading to either recovery or death).',
                                                                                                                                                                    style={'fontSize':'80%'}),

                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    Please use the following links: [**Ferguson et al**](https://spiral.imperial.ac.uk/handle/10044/1/77482), [**Anderson et al**](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30567-5/fulltext) and [**Zhao et al**](https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0006158&type=printable)
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify'}),


                                                                                                                                                                    html.H4('Age Structure',style={'fontSize': '180%', 'textAlign': 'center'}),
                                                                                                                                                                    
                                                                                                                                                                    dcc.Markdown('''
                                                                                                                                                                    The age data is taken from [**GOV.UK**](https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest) and the hospitalisation and critical care data is from the [**Imperial College Paper**](https://spiral.imperial.ac.uk/handle/10044/1/77482) (Ferguson et al.). This means that the age structure won't be accurate when modelling other countries.

                                                                                                                                                                    To find the probability of a low risk case getting hospitalised (or subsequently put in critical care), we take a weighted average by proportion of population.

                                                                                                                                                                    *The table below shows the age structure data that was used to calculate these weighted averages across the low risk category (under 60) and high risk (over 60) category.*
                                                                                                                                                                    
                                                                                                                                                                    ''',
                                                                                                                                                                    style={'textAlign': 'justify','marginTop': '2vh','marginBottom': '2vh'}
                                                                                                                                                                    
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
                        style= {'width': '90%', 'marginLeft': '5vw', 'marginRight': '5vw', 'marginTop': '10vh', 'marginBottom': '5vh'}
                        ),

                    
                        ],
                        width=12,
                        xl=10),
                        
                        ],
                        justify='center'
                        )





                        ],
                        style= {'width': '90%', 'backgroundColor': '#f4f6f7', 'marginLeft': '5vw', 'marginRight': '5vw', 'marginBottom': '5vh'}
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
    style={'fontSize' : '1.9vh'},
    id='main-page-id'
    )




















navbar = html.Nav([
        html.Div([
            dcc.Tabs([
                dcc.Tab(children=
                        layout_intro,
                        label='Background',value='intro',
                        style={'fontSize':'1.9vh'}
                        ), #
                dcc.Tab(children=
                        layout_inter,
                        label='Interactive Model',value='interactive',
                        style={'fontSize':'1.9vh'}
                        ),
                dcc.Tab(children=
                        layout_dan,
                        label='Real-Time Global Data Feed',
                        style={'fontSize':'1.9vh'},
                        value='data',
                        ), #disabled=True),
            ], id='main-tabs', value='inter'),
        ], style={'width': '100vw'}, # , 'display': 'flex', 'justifyContent': 'center'},
        ),
    ],)










# app.layout
        
page_layout = html.Div([
    
            
            dbc.Row([
                dbc.Col([
                    html.H3(children='Modelling control of COVID-19',
                    className="display-4",
                    style={'marginTop': '1vh', 'textAlign': 'center','fontSize': '6vh'}
                    ),



                    # html.P(
                    # 'Disclaimer:',
                    # style = {'marginTop': '0.5vh', 'fontSize': '100%','marginBottom': '0vh', 'color': '#446E9B', 'fontWeight': 'bold'}
                    # ),
                    html.P(
                    'Disclaimer: This work is for educational purposes only and not for accurate prediction of the pandemic.',
                    style = {'marginTop': '0vh','marginBottom': '0vh', 'color': '#446E9B', 'fontWeight': 'bold'}
                    ),
                    html.P(
                    'There are many uncertainties in the COVID debate. The model is intended solely as an illustrative rather than predictive tool.',
                    style = {'marginTop': '0vh','marginBottom': '3vh', 'color': '#446E9B', 'fontWeight': 'bold'}
                    ), # 

                ],width=True,
                style={'textAlign': 'center'}
                ),
            ],
            align="center",
            style={'backgroundColor': '#e9ecef'}
            ),

        # navbar
        html.Div([navbar]),
        ##

        # # page content
        dcc.Location(id='url', refresh=False),

        html.Footer('This page is intended for illustrative/educational purposes only, and not for accurate prediction of the pandemic.',
                    style={'textAlign': 'center', 'fontSize': '100%', 'marginBottom': '1.5vh' , 'color': '#446E9B', 'fontWeight': 'bold'}),
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
        # 
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





########################################################################################################################
# collapse
def toggle(n, is_open):
    if n:
        return not is_open
    return is_open


for p in ["custom"]: # , "hospital"]:
    app.callback(
        Output(f"collapse-{p}", "is_open"),
        # Output(f"collapse-button-{p}", "color")],
        [Input(f"collapse-button-{p}", "n_clicks")
        ],
        [State(f"collapse-{p}", "is_open")],
    )(toggle)


########################################################################################################################
# popovers


for p in [ "rt-data" ,"pick-strat","control", "months-control", "vaccination", "res-type" , "cc-care" ,"custom-options", "inf-rate", "inf-tab", "cont-tab", "example","red-deaths","ICU","herd","bp1","bp2","bp3","bp4","bp5"]: #  "plot-settings"
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

    Output('groups-to-plot-radio','style'),
    Output('groups-checklist-to-plot','style'),
    Output('plot-with-do-nothing','options'),

    ],
    [
    Input('number-strats-radio', 'value'),
    Input('preset', 'value'),
    Input('plot-with-do-nothing', 'value')
    ])
def invisible_or_not(num,preset,do_nothing):
    # print('invis or not')

    do_nothing_dis = False
    do_n_val = 1
    if num=='two':
        strat_H = [html.H6('Strategy One: High Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        strat_L = [html.H6('Strategy One: Low Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        groups_checklist = {'display': 'none'}
        groups_radio = None
        says_strat_2 = None
        do_nothing_dis = True
        do_n_val = 0

    else:
        strat_H = [html.H6('High Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        strat_L = [html.H6('Low Risk Infection Rate (%)',style={'fontSize': '100%'}),]
        if preset=='N':
            do_nothing_dis = True
            do_n_val = 0

        if do_nothing==[1]:
            groups_checklist = {'display': 'none'}
            groups_radio = None
        else:
            groups_checklist = None
            groups_radio = {'display': 'none'}
        says_strat_2 = {'display': 'none'}

    if preset!='C':
        says_strat_2 = {'display': 'none'}

    options=[{'label': 'Compare', 'value': do_n_val, 'disabled': do_nothing_dis}]
    

    return [says_strat_2,strat_H, strat_L ,groups_radio,groups_checklist,options]

########################################################################################################################




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
    # print('preset sliders')
    
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
    return preset_dict_low[preset], preset_dict_high[preset], dis, dis, options
    



@app.callback(
    [Output('sol-calculated', 'data'),
    Output('loading-sol-1','children')],
    [
    Input('preset', 'value'),
    Input('month-slider', 'value'),
    Input('low-risk-slider', 'value'),
    Input('high-risk-slider', 'value'),
    Input('low-risk-slider-2', 'value'),
    Input('high-risk-slider-2', 'value'),
    Input('number-strats-radio', 'value'),
    Input('vaccine-slider', 'value'),
    Input('ICU-slider', 'value'),
    Input('model-start-date', 'date'),
    Input('model-country-choice', 'value'),
    ])
def find_sol(preset,month,lr_in,hr_in,lr2_in,hr2_in,num_strat,vaccine,ICU_grow,date,country_num):

    
    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    if vaccine==9:
        vaccine = None

    I0, R0, H0, C0, D0 = begin_date(date,country)

    # if preset is None:
    #     preset = 'N'

    if preset=='C':
        lr = params.fact_v[int(lr_in)]
        hr = params.fact_v[int(hr_in)]
    else:
        lr = params.fact_v[preset_dict_low[preset]]
        hr = params.fact_v[preset_dict_high[preset]]

    if preset=='N':
        month=[0,0]
    
    lr2 = params.fact_v[int(lr2_in)]
    hr2 = params.fact_v[int(hr2_in)]
    
    t_stop = 365*3


    months_controlled = [month_len*i for i in month]
    if month[0]==month[1]:
        months_controlled= None

    sols = []
    sols.append(simulator().run_model(beta_L_factor=lr,beta_H_factor=hr,t_control=months_controlled,T_stop=t_stop,vaccine_time=vaccine,I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,ICU_grow=ICU_grow))
    if num_strat=='two':
        sols.append(simulator().run_model(beta_L_factor=lr2,beta_H_factor=hr2,t_control=months_controlled,T_stop=t_stop,vaccine_time=vaccine,I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,ICU_grow=ICU_grow))
    
    return sols, None


@app.callback(
    Output('sol-calculated-do-nothing', 'data'),
    [
    Input('ICU-slider', 'value'),
    Input('model-start-date', 'date'),
    Input('model-country-choice', 'value'),
    ])
def find_sol_do_noth(ICU_grow,date,country_num):

    # ctx = dash.callback_context
    # print('find sol do nothing')

    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    I0, R0, H0, C0, D0 = begin_date(date,country)

    t_stop = 365*3
    
    sol_do_nothing = simulator().run_model(beta_L_factor=1,beta_H_factor=1,t_control=None,T_stop=t_stop,I0=I0,R0=R0,H0=H0,C0=C0,D0=D0,ICU_grow=ICU_grow)
    
    return sol_do_nothing











###################################################################################################################################################################################


@app.callback([ 

                Output('DPC-content', 'style'),
                Output('bc-content', 'style'),
                Output('strategy-outcome-content', 'style'),
                

                Output('line_page_title', 'children'),
                

                Output('strategy-outcome-content', 'children'),


                Output('bar-plot-1', 'figure'),
                Output('bar-plot-2', 'figure'),
                Output('bar-plot-3', 'figure'),
                Output('bar-plot-4', 'figure'),
                Output('bar-plot-5', 'figure'),


                Output('line-plot-1', 'figure'),


                
                Output('loading-line-output-1','children'),
                
                

                ],
                [
                Input('interactive-tabs', 'active_tab'),



                Input('main-tabs', 'value'),

                
                Input('sol-calculated', 'data'),

                # or any of the plot categories
                Input('groups-checklist-to-plot', 'value'),
                Input('groups-to-plot-radio','value'),                                      
                Input('categories-to-plot-checklist', 'value'),
                Input('plot-with-do-nothing','value'),
                Input('plot-ICU-cap','value'),

                
                Input('dropdown', 'value'),
                Input('model-country-choice', 'value'),



                ],
               [
                State('sol-calculated-do-nothing', 'data'),
                State('preset', 'value'),
                State('month-slider', 'value'),

                State('number-strats-radio', 'value'),
                State('vaccine-slider', 'value'),
                State('ICU-slider','value'),
                State('model-start-date','date'),
                ])
def render_interactive_content(tab,tab2,sols,groups,groups2,output,plot_with_do_nothing,plot_ICU_cap,results_type,country_num,sol_do_nothing,preset,month,num_strat,vaccine_time,ICU_grow,date):

    # ctx = dash.callback_context
    # print(ctx.triggered[0]['prop_id'],'render')



    try:
        country = COUNTRY_LIST_NICK[country_num]
    except:
        country = 'uk'

    

    
    DPC_style = {'display' : 'none'}
    BC_style  = {'display': 'none'}
    SO_style  = {'display' : 'none'}

    if results_type == 'BC_dd':
        BC_style = {'display': 'block'}

    elif results_type == 'SO_dd':
        SO_style = {'display': 'block'}

    else: # 'DPC
        DPC_style = {'display': 'block'}
        results_type = 'DPC_dd' # in case wasn't


########################################################################################################################

    # if preset is None:
    #     preset = 'N'

    Strat_outcome_title = presets_dict[preset] + ' Strategy Outcome'
    strategy_outcome_text = ['']


    if sols is None or tab2!='interactive' or results_type!='BC_dd':
        bar1 = dummy_figure
        bar2 = dummy_figure
        bar3 = dummy_figure
        bar4 = dummy_figure
        bar5 = dummy_figure

    if sols is None or tab2!='interactive' or results_type!='DPC_dd':
        fig1 = dummy_figure



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

                        dcc.Markdown('''
                            *Click on the info buttons for explanations*.
                            ''',style = {'textAlign': 'center', 'marginTop': '3vh', 'marginBottom': '3vh'}),
                        
                        outcome_fn(month,sols[0]['beta_L'],sols[0]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[0],herd_list_2yr[0],ICU_data_2yr[0],preset,number_strategies = num_strat,which_strat=1), # hosp,
                        html.Hr(),
                        outcome_fn(month,sols[1]['beta_L'],sols[1]['beta_H'],crit_cap_quoted_1yr[0],herd_list_1yr[0],ICU_data_1yr[0],crit_cap_quoted_2yr[1],herd_list_2yr[1],ICU_data_2yr[1],preset,number_strategies = num_strat,which_strat=2), # hosp,
                        ],
                        style = {'fontSize': '2vh'}
                        )

                
                ########################################################################################################################
                # BC results

                if results_type=='BC_dd':

                    crit_cap_bar_1yr = [crit_cap_data_L_1yr[i] + crit_cap_data_H_1yr[i] for i in range(len(crit_cap_data_H_1yr))]
                    crit_cap_bar_3yr = [crit_cap_data_L_3yr[i] + crit_cap_data_H_3yr[i] for i in range(len(crit_cap_data_H_3yr))]


                    bar1 = Bar_chart_generator(crit_cap_bar_1yr      ,text_addition='%'         , y_title='Population'                    , hover_form = '%{x}, %{y:.3%}'                         ,color = 'powderblue' ,data_group=crit_cap_bar_3yr, yax_tick_form='.1%')
                    bar2 = Bar_chart_generator(herd_list_1yr         ,text_addition='%'         , y_title='Percentage of Safe Threshold'  , hover_form = '%{x}, %{y:.1%}<extra></extra>'          ,color = 'powderblue' ,data_group=herd_list_3yr,yax_tick_form='.1%',maxi=False,yax_font_size_multiplier=0.8)
                    bar3 = Bar_chart_generator(ICU_data_1yr          ,text_addition='x current' , y_title='Multiple of Capacity'  , hover_form = '%{x}, %{y:.1f}x Current<extra></extra>' ,color = 'powderblue'     ,data_group=ICU_data_3yr  )
                    bar4 = Bar_chart_generator(time_exceeded_data    ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'powderblue'   )
                    bar5 = Bar_chart_generator(time_reached_data     ,text_addition=' Months'   , y_title='Time (Months)'                 , hover_form = '%{x}: %{y:.1f} Months<extra></extra>'   ,color = 'powderblue')



        ########################################################################################################################
            # DPC results

            if results_type=='DPC_dd':
                
                date = datetime.datetime.strptime(date.split('T')[0], '%Y-%m-%d')
                date = datetime.datetime.strftime(date, '%Y-%#m-%d' )


                if vaccine_time==9:
                    vaccine_time = None

                if plot_with_do_nothing==[1] and num_strat=='one' and preset!='N':
                    sols_to_plot = sols
                    comp_dn = True
                else:
                    sols_to_plot = sols[:-1]
                    comp_dn = False

                if plot_ICU_cap==[1]:
                    ICU_plot = True
                else:
                    ICU_plot = False



                if len(output)>0:
                    fig1 = figure_generator(sols_to_plot,month,output,groups,num_strat,groups2,vaccine_time=vaccine_time,ICU_grow=ICU_grow, ICU_to_plot=ICU_plot ,comp_dn=comp_dn,starting_date=date, country = country)
                else:
                    fig1 = dummy_figure

            
        ##############

                
    # print('render end')

########################################################################################################################

    return [
    DPC_style,
    BC_style,
    SO_style,

    Strat_outcome_title,

    strategy_outcome_text,

    bar1,
    bar2,
    bar3,
    bar4,
    bar5,

    fig1,

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
            #    Input('main-tabs', 'value'),
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
              [State('hidden-stored-data', 'children')] +
              [State(c_name, 'value') for c_name in COUNTRY_LIST])
def update_plots(n_clicks, start_date, end_date, show_exponential, normalise_by_pop,
                 align_cases_check, align_cases_input, align_deaths_check, align_deaths_input, align_active_cases_check,
                 align_active_cases_input, align_daily_cases_check, align_daily_cases_input,
                 align_daily_deaths_check, align_daily_deaths_input, saved_json_data, *args):

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
                    x=0,
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










