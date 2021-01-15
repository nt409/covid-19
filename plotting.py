import plotly.graph_objects as go
from plotly.subplots import make_subplots
from parameters_cov import params
import pandas as pd
import numpy as np
from math import ceil
import datetime

from data_constants import POPULATIONS

month_len = 365/12


longname = {'S': 'Susceptible',
        'I': 'Infected',
        'R': 'Recovered (total)',
        'H': 'Hospitalised',
        'C': 'Critical',
        'D': 'Deaths (total)',
}


index = {'S': params.S_L_ind,
        'I': params.I_L_ind,
        'R': params.R_L_ind,
        'H': params.H_L_ind,
        'C': params.C_L_ind,
        'D': params.D_L_ind,
        }



colors = {'S': 'blue',
        'I': 'orange',
        'R': 'green',
        'H': 'red',
        'C': 'black',
        'D': 'purple',
        }











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
def time_exceeded_function(yy,tt,ICU_grow):
    ICU_capac = [params.ICU_capacity*(1 + ICU_grow*time/365 ) for time in tt]
    Exceeded_vec = [ (yy[params.C_H_ind,i]+yy[params.C_L_ind,i]) > ICU_capac[i] for i in range(len(tt))]
    Crit_vals = [ (yy[params.C_H_ind,i]+yy[params.C_L_ind,i])  for i in range(len(tt))]

    c_low = [-2]
    c_high = [-1]
    ICU = False
    if max(Crit_vals)>params.ICU_capacity:
        if Exceeded_vec[0]: # if exceeded at t=0
            c_low.append(0)
        for i in range(len(Exceeded_vec)-1):
            if not Exceeded_vec[i] and Exceeded_vec[i+1]: # entering
                ICU = True
                y1 = 100*(yy[params.C_H_ind,i]+yy[params.C_L_ind,i])
                y2 = 100*(yy[params.C_H_ind,i+1]+yy[params.C_L_ind,i+1])
                t1 = tt[i]
                t2 = tt[i+1]
                t_int = t1 + (t2- t1)* abs((100*0.5*(ICU_capac[i]+ICU_capac[i+1]) - y1)/(y2-y1)) 
                c_low.append(t_int) # 0.5 * ( tt[i] + tt[i+1]))
            if Exceeded_vec[i] and not Exceeded_vec[i+1]: # leaving
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
    c_low, c_high, _ = time_exceeded_function(yy,tt,ICU_grow)

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
                    # xaxis=dict(showline=False),
                    yaxis = dict(
                        automargin = True,
                        # showline=False,
                        title = y_title,
                        title_font = dict(size=yax_font_size),
                    ),
                    showlegend = show_ledge,

                    transition = {'duration': 500},
                   )



    return {'data': traces, 'layout': layout}























########################################################################################################################
def solnIntoDataframe(sol,startdate):
    time = pd.Series([startdate + datetime.timedelta(days=i) for i in sol['t']])
    df = pd.DataFrame(time)
    df.columns = ['t']

    sol['y'] = np.asarray(sol['y'])
    for name in index.keys():
        
        y_Low   = 100*pd.Series(sol['y'][index[name],:]).values
        y_High  = 100*pd.Series(sol['y'][index[name]+params.number_compartments,:]).values
        y_Total = y_Low + y_High


        df[longname[name]+': LR'] = y_Low
        df[longname[name]+': HR'] = y_High
        df[longname[name]+': BR'] = y_Total

    return df



















def string_function(len_sols,num_strat,ss,comp_dn):
    if len_sols>1:
        strat_list = [': Strategy',': Do Nothing']
    else:
        strat_list = ['']

    linestyle_numst = ['solid','dash','dot','dashdot','longdash','longdashdot']

    if num_strat=='one':
        name_string = strat_list[ss]
        line_style_use = 'solid' # linestyle['BR']
        if comp_dn:
            if ss == 0:
                line_style_use = 'solid'
            else:
                line_style_use = 'dot'
    else:
        name_string = ': Strategy ' + str(ss+1)
        line_style_use = linestyle_numst[ss]
    
    return line_style_use, name_string





def yaxis_function(Yrange,population_plot,country_name):

    yy2 = [0]
    for i in range(8):
        yy2.append(10**(i-5))
        yy2.append(2*10**(i-5))
        yy2.append(5*10**(i-5))

    yy = [i for i in yy2]

    pop_vec_lin = np.linspace(0,1,11) # temp
    for i in range(len(yy)-1):
        if Yrange[1]>yy[i] and Yrange[1] <= yy[i+1]:
            pop_vec_lin = np.linspace(0,yy2[i+1],11)

    linTicks = [i*(population_plot) for i in pop_vec_lin]
    LinText = [human_format(0.01*ll) for ll in linTicks]

    log_bottom = -8
    log_range = [log_bottom,np.log10(100)]



    pop_log_vec = [10**(i) for i in range(log_range[0], int(log_range[1]+1),2)] # will always give 6 values
    logTicks = [i*(population_plot) for i in pop_log_vec]

    LogText = [human_format(0.01*ll) for ll in logTicks]


    yAxisPopLinear = {
    # 'title': f'Population ({country_name})', 
    'fixedrange': True,
    'type': 'linear', 
    'range': Yrange,
    'ticktext': LinText,
    'tickvals': pop_vec_lin,
    'automargin': True}

    yAxisPopLog    = {
    # 'title': f'Population ({country_name})',
    'fixedrange': True,
    'type': 'log',
    'range': log_range,
    'ticktext': LogText,
    'tickvals': pop_log_vec,
    'automargin': True}
    
    return yAxisPopLinear, yAxisPopLog






def annotations_shapes_function(month_cycle,month,preset,startdate,ICU,font_size,c_low,c_high,Yrange):
    annotz = []
    shapez = []


    blue_opacity = 0.25
    if month_cycle is not None:
        blue_opacity = 0.1

    if month[0]!=month[1] and preset != 'N':
        shapez.append(dict(
                # filled Blue Control Rectangle
                type="rect",
                x0= startdate+datetime.timedelta(days=month_len*month[0]), #month_len*
                y0=0,
                x1= startdate+datetime.timedelta(days=month_len*month[1]), #month_len*
                y1=100,
                line=dict(
                    color="LightSkyBlue",
                    width=0,
                ),
                fillcolor="LightSkyBlue",
                opacity= blue_opacity
            ))
            
    if ICU:

        yval_pink = 0.4
        yval_blue = 0.6


        for c_min, c_max in zip(c_low, c_high):
            if c_min>=0 and c_max>=0:
                shapez.append(dict(
                        # filled Pink ICU Rectangle
                        type="rect",
                        x0= startdate+datetime.timedelta(days=c_min), #month_len*  ##c_min/month_len,
                        y0=0,
                        x1= startdate+datetime.timedelta(days=c_max), #c_max/month_len,
                        y1=100,
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
                        x  = startdate+datetime.timedelta(days=0.5*(c_min+c_max)), # /month_len
                        y  = yval_pink,
                        text="​🚑",
                        showarrow=False,
                        textangle= 0,
                        font=dict(
                            size= 20,
                            color="purple"
                        ),
                        opacity=0.5,
                        xref = 'x',
                        yref = 'paper',
                ))

    else:
        yval_blue = 0.6




    if month[0]!=month[1] and preset!='N':
        annotz.append(dict(
                x  = startdate+datetime.timedelta(days=month_len*max(0.5*(month[0]+month[1]), 0.5)),
                y  = yval_blue,
                text="​😷​",
                font=dict(
                    size= 20,
                    color="blue"
                ),
                showarrow=False,
                opacity=0.5,
                xref = 'x',
                yref = 'paper',
        ))

    
    if month_cycle is not None:

        for i in range(0,len(month_cycle),2):
            shapez.append(dict(
                    # filled Blue Control Rectangle
                    type="rect",
                    x0= startdate+datetime.timedelta(days=month_len*month_cycle[i]),
                    y0=0,
                    x1= startdate+datetime.timedelta(days=month_len*month_cycle[i+1]),
                    y1=100,
                    line=dict(
                        color="LightSkyBlue",
                        width=0,
                    ),
                    fillcolor="LightSkyBlue",
                    opacity=0.3
                ))


        
    return annotz, shapez











def lineplot(sols,population_plot,startdate,num_strat,comp_dn):
    cats_to_plot = ['S','I','R','H','C','D']
    lines_to_plot = []
    group_use = ['BR']

    ss = -1
    for sol in sols:
        dataframe = solnIntoDataframe(sol,startdate)
        ss += 1
        if num_strat == 'one' and not comp_dn and ss>0:
            pass
        else:
            for name in cats_to_plot:
                for group in group_use:
                    if name in ['H','D']:
                        vis = True
                    else:
                        vis = False

                    line_style_use, name_string = string_function(len(sols),num_strat,ss,comp_dn)
                    xx = [startdate + datetime.timedelta(days=i) for i in sol['t']]
                    yyy_p = np.asarray(dataframe[f'{longname[name]}: {group}'])
                    
                    line =  {'x': xx, 'y': yyy_p,
                            'hovertemplate': '%{y:.2f}%, %{text}',
                            'visible': vis,
                            'text': [human_format(i*population_plot/100,dp=1) for i in yyy_p],
                            'line': {'color': str(colors[name]), 'dash': line_style_use }, 'legendgroup': name,
                            'name': longname[name] + name_string}
                    lines_to_plot.append(line)

    return lines_to_plot, xx



def stackPlot(sols,population_plot,startdate):
    lines_to_plot = []

    sol = sols[0]
    dataframe = solnIntoDataframe(sol,startdate)

    group_strings = {'BR': ' All',
        'HR': ' High Risk',
        'LR': ' Low Risk'}



    group_use = ['HR','LR']

    cats_to_plot = ['D']

    for name in cats_to_plot:
        for group in group_use:
            
            name_string = ':' + group_strings[group]

            xx = [startdate + datetime.timedelta(days=i) for i in sol['t']]

            yyy_p = np.asarray(dataframe[f'{longname[name]}: {group}'])
            
            # points_per_bar = 2
            # xx = [xx[i] for i in range(1,len(xx),points_per_bar)]
            # yyy_p = [yyy_p[i] for i in range(1,len(yyy_p),points_per_bar)]

            line =  dict(
                    x=xx,
                    y=yyy_p,
                    hovertemplate='%{y:.2f}%, %{text}',
                    text=[human_format(i*population_plot/100,dp=1) for i in yyy_p],
                    # type='bar',
                    legendgroup=name,
                    name=longname[name] + name_string,
                    # 'width': [1]*len(yyy_p),
                    )
                    
            if group=='LR':
                line['marker'] = dict(color='LightSkyBlue')
            elif group=='HR':
                line['marker'] = dict(color='orange')
            
            line['visible'] = False
            
            lines_to_plot.append(line)

    return lines_to_plot







def uncertPlot(upper_lower_sol,population_plot,startdate):
    lines_to_plot = []
    ii = -1
    name = 'D'


    for sol in upper_lower_sol:
        ii += 1
        sol['y'] = np.asarray(sol['y'])
                
        if ii == 0:
            fill = None
            label_add = '; lower estimate'
        else:
            fill = 'tonexty'
            label_add = '; upper estimate'

        xx = [startdate + datetime.timedelta(days=i) for i in sol['t']]

        yyy_p = (100*sol['y'][index[name],:] + 100*sol['y'][index[name] + params.number_compartments,:])
        
        line =  {'x': xx, 'y': yyy_p,
                'hovertemplate': '%{y:.2f}%, %{text}',
                'text': [human_format(i*population_plot/100,dp=1) for i in yyy_p],
                'line': {'width': 0, 'color': str(colors[name])},
                'fillcolor': 'rgba(128,0,128,0.4)',
                'visible': False,
                'showlegend': False,
                'fill': fill,
                'name': longname[name] + label_add}
        lines_to_plot.append(line)

    return lines_to_plot


def prevDeaths(previous_deaths,startdate,population_plot):
    lines_to_plot = []
    x0 = 0
    if previous_deaths is not None:
        x_deaths = [startdate - datetime.timedelta(days=len(previous_deaths) - i ) for i in range(len(previous_deaths))]
        y_deaths = [100*float(i)/population_plot for i in previous_deaths]

        lines_to_plot.append(
        dict(
        type='scatter',
            x = x_deaths,
            y = y_deaths,
            mode='lines',
            opacity=0.85,
            showlegend=False,
            legendgroup='deaths',
            line=dict(
            color= 'purple',
            visible= False,
            dash = 'dash'
            ),
            hovertemplate = '%{y:.2f}%, %{text}',
            text = [human_format(i*population_plot/100,dp=1) for i in y_deaths],
            name= 'Recorded deaths'))
        x0 = x_deaths[0]
    return lines_to_plot, x0


def CategoryFunction(CategoryList,Indices,Name,lines_to_plot_line,population_plot,country_name,BooleanString,xAxis):

    FalseList = [False]*len(lines_to_plot_line)
    LineMaxTemp = 0
    for index, line in enumerate(lines_to_plot_line):
        if index in Indices: # corresponds to hosp/crit
            FalseList[index] = True
            LineMaxTemp = max(max(1.1*line['y']),0.01,LineMaxTemp) # makes sure above 0
            LineRangeTemp = [0,min(LineMaxTemp,100)]
    
    yAxisPopLinTemp, _ = yaxis_function(LineRangeTemp,population_plot,country_name)
    
    CategoryDict = dict(
        args=[{
        "visible": 
        FalseList + BooleanString
        },
        {
        "xaxis": xAxis,
        "yaxis": yAxisPopLinTemp,
        }],
        label  = Name,
        method = "update"
    )
    CategoryList.append(CategoryDict)
    return CategoryList




def MultiFigureGenerator(upper_lower_sol,sols,month,num_strat,ICU_to_plot=False,
        vaccine_time=None,ICU_grow=None,comp_dn=False,country = 'uk',month_cycle=None,
        preset=None,startdate=None, previous_deaths=None):
    
    # cats_to_plot = ['S','I','R','C','H','D']
    try:
        population_plot = POPULATIONS[country]
    except:
        population_plot = 100

    if country in ['us','uk']:
        country_name = country.upper()
    else:
        country_name = country.title()
    
    font_size = 12
    
   
   
    lines_to_plot_line, xx = lineplot(sols,population_plot,startdate,num_strat,comp_dn)
    lines_to_plot_stack = stackPlot(sols,population_plot,startdate)
    lines_to_plot_uncert = uncertPlot(upper_lower_sol,population_plot,startdate)
    lines_PrevDeaths, x0 = prevDeaths(previous_deaths,startdate,population_plot)
    
    # setting up pink boxes
    ICU = False
    if num_strat=='one': #  and len(cats_to_plot)>0:
        yyy = np.asarray(sols[0]['y'])
        ttt = sols[0]['t']
        c_low, c_high, ICU = time_exceeded_function(yyy,ttt,ICU_grow)
    
    ymax = 0.001
    for line in lines_to_plot_line:
        if line['visible']:
            ymax = max(ymax,max(line['y']))

    yRange = [0,min(1.1*ymax,100)]
    
    ##
    yMaxDeath = 0.001
    for line in lines_to_plot_uncert:
        yMaxDeath = max(yMaxDeath,max(line['y']))

    yMaxDeathRange = [0,min(1.1*yMaxDeath,100)]


    moreLines = []
    if ICU_to_plot: # and 'C' in cats_to_plot:
        ICU_line = [100*params.ICU_capacity*(1 + ICU_grow*i/365) for i in sols[0]['t']]
        moreLines.append(
        dict(
        type='scatter',
            x=xx, y=ICU_line,
            mode='lines',
            opacity=0.5,
            legendgroup='thresholds',
            line=dict(
            color= 'black',
            dash = 'dot'
            ),
            hovertemplate= 'ICU Capacity<extra></extra>',
            name= 'ICU Capacity'))

    if vaccine_time is not None and vaccine_time>0:
        moreLines.append(
        dict(
        type='scatter',
            x=[startdate+datetime.timedelta(days=month_len*vaccine_time),
            startdate+datetime.timedelta(days=month_len*vaccine_time)],
            y=[yRange[0],100],
            mode='lines',
            opacity=0.2,
            legendgroup='thresholds',
            line=dict(
            color= 'green',
            ),
            hovertemplate= 'Vaccination starts<extra></extra>',
            name= 'Vaccination starts'))

    
    
    # moreLines.append(
    # dict(
    #     type='scatter',
    #     x = [xx[0],xx[-1]],
    #     y = [0, population_plot],
    #     # yaxis="y2",
    #     opacity=0,
    #     showlegend=False,
    #     hoverinfo='skip',
    # ))

    controlLines = []
    if month[0]!=month[1] and preset != 'N':
        controlLines.append(
        dict(
        type='scatter',
            x=[startdate+datetime.timedelta(days=month_len*month[0]),
            startdate+datetime.timedelta(days= month_len*month[0])], # +1 to make it visible when at 0
             y=[0,100],
            mode='lines',
            opacity=0.9,
            legendgroup='control',
            visible = False,
            line=dict(
            color= 'blue',
            dash = 'dash'
            ),
            hovertemplate= 'Control starts<extra></extra>',
            name= 'Control starts'))
        controlLines.append(
        dict(
        type='scatter',
            x=[startdate+datetime.timedelta(days=month_len*month[1]),
            startdate+datetime.timedelta(days=month_len*month[1])],
            y=[0,100],
            mode='lines',
            opacity=0.9,
            legendgroup='control',
            visible = False,
            line=dict(
            color= 'blue',
            dash = 'dot'
            ),
            hovertemplate= 'Control ends<extra></extra>',
            name= 'Control ends'))




    yAxisPopLinear, yAxisPopLog = yaxis_function(yRange,population_plot,country_name)
    yAxisPopLinearDeaths, _ = yaxis_function(yMaxDeathRange,population_plot,country_name)

    annotz, shapez = annotations_shapes_function(month_cycle,month,preset,startdate,ICU,font_size,c_low,c_high,yRange)

    xAx2Year = {'range': [xx[0], xx[-1]],'hoverformat':'%d %b','fixedrange': True}

    xAx2FromFeb = {'range': [x0, xx[-1]],'hoverformat':'%d %b','fixedrange': True}








    CategoryList = []

    BoolString = [False]*len(lines_to_plot_stack) + [False]*len(lines_to_plot_uncert)  + [True]*len(moreLines)  + [False]*len(controlLines)  + [False]*len(lines_PrevDeaths)
    config = lines_to_plot_line, population_plot, country_name, BoolString, xAx2Year
    
    CategoryList = CategoryFunction(CategoryList,[3,4],'Hosp. Categories',*config)
    CategoryList = CategoryFunction(CategoryList,[0,1,2,3,4,5],'All',*config)
    CategoryList = CategoryFunction(CategoryList,[1,3,4,5],'Pathology',*config)



    for index, line in enumerate(lines_to_plot_line):
        
        CategoryList = CategoryFunction(CategoryList,[index],line['name'],*config)
        
        if line['name']=='Deaths (total)':
            LineMax = max(max(1.1*line['y']),0.01) # makes sure above 0
            LineRange = [0,min(LineMax,100)]
            yAxisPopLinStack, _ = yaxis_function(LineRange,population_plot,country_name)







    if vaccine_time is not None and vaccine_time>0:
        annotz.append(go.layout.Annotation(x=startdate+datetime.timedelta(days=month_len*vaccine_time),
                        y=1,
                        text='💉',
                        showarrow=False,
                        xref='x',
                        yref='paper',
                        xanchor="right",
                        yanchor="top",
                        opacity=0.5,
                        font={"size": 20}
                        )
                        )


    layout = go.Layout(
                    annotations=annotz,
                    shapes=shapez,
                    autosize= False,
                    barmode = 'stack',
                    template="simple_white",
                    font = dict(size=font_size),
                    margin=dict(t=0, b=0, l=0, r=0,
                                pad=0),
                    yaxis  = yAxisPopLinear,
                    hovermode='x',
                    xaxis= xAx2Year,
                        updatemenus = [

                                            dict(
                                                buttons= CategoryList,
                                            x= 0.07,
                                            xanchor="center",
                                            active=0,
                                            y=-0.350,
                                            showactive=True,
                                            direction='up',
                                            yanchor="top"
                                            ),



                                            dict(
                                                buttons=list([
                                                    dict(
                                                    args=[{
                                                    "yaxis": yAxisPopLinear,
                                                    }],
                                                    label="Axis: Linear",
                                                    method="relayout"
                                                ),
                                                dict(
                                                    args=[{"yaxis": yAxisPopLog}],
                                                    label="Axis: Logarithmic",
                                                    method="relayout"
                                                )
                                        ]),
                                        x= 0.93,
                                        xanchor="center",
                                        active=0,
                                        y=-0.20,
                                        showactive=True,
                                        direction='up',
                                        yanchor="top"
                                        ),

                                        dict(
                                                buttons=list([
                                                    dict(
                                                    args=[
                                                    {'visible': [True]*len(lines_to_plot_line) + [False]*len(lines_to_plot_stack) + [False]*len(lines_to_plot_uncert)  + [True]*len(moreLines)  + [False]*len(controlLines)  + [False]*len(lines_PrevDeaths)
                                                    },
                                                    {
                                                    # "shapes": shapez,
                                                    "xaxis": xAx2Year
                                                    },
                                                    ],
                                                    label="Plot: Line",
                                                    method="update"
                                                ),
                                                dict(
                                                    args=[
                                                    {"visible": [False]*len(lines_to_plot_line) + [True]*len(lines_to_plot_stack) + [False]*len(lines_to_plot_uncert)  + [True]*len(moreLines)  + [True]*len(controlLines) + [False]*len(lines_PrevDeaths)},
                                                    {
                                                    # "shapes":[],
                                                    "yaxis": yAxisPopLinStack,
                                                    "xaxis": xAx2Year,
                                                    # "barmode":'stack',
                                                    },
                                                    ],
                                                    label="Plot: Stacked",
                                                    method="update"
                                                ),
                                                dict(
                                                    args=[
                                                    {"visible": [False]*(len(lines_to_plot_line)-1) + [True] + [False]*len(lines_to_plot_stack) + [True]*len(lines_to_plot_uncert)  + [True]*len(moreLines)  + [False]*len(controlLines) + [True]*len(lines_PrevDeaths)},
                                                    {
                                                    # "shapes": shapez,
                                                    "yaxis": yAxisPopLinearDeaths,
                                                    "xaxis": xAx2FromFeb
                                                    },
                                                    ],
                                                    label="Plot: Fatalities",
                                                    method="update"
                                                )
                                        ]),
                                        x= 0.93,
                                        xanchor="center",
                                        active=0,
                                        y=-0.35,
                                        showactive=True,
                                        direction='up',
                                        yanchor="top"
                                        )
                                        
                                        ],
                                        legend = dict(
                                                        font=dict(size=font_size*(20/24)),
                                                        x = 0.5,
                                                        y = 1.03,
                                                        xanchor= 'center',
                                                        yanchor= 'bottom'
                                                    ),
                                        legend_orientation  = 'h',
                                        legend_title        = '<b> Key </b>',
                            )



    linesUse = lines_to_plot_line + lines_to_plot_stack + lines_to_plot_uncert + moreLines + controlLines + lines_PrevDeaths

    return {'data': linesUse, 'layout': layout}










def test_bar_plot(outputs, title):

    traces = []

    names = [
        'True positive',
        'False positive',
        'True negative',
        'False negative',
            ]
    colors = ['red']*2 + ['blue']*2

    line = go.Bar(
            x=names,
            y=outputs,
            marker_color = colors,
            showlegend=False
        )
    traces.append(line)

    layout = go.Layout(
                template="simple_white",
                title=title,
                xaxis=dict(
                        title='Test outcome',
                        fixedrange= True, 
                        ),
                yaxis={'title': 'Proportion of tests returned', 'fixedrange': True},
        )

    
    return {'data': traces, 'layout': layout}



def test_bar_plot2(outputs, title):

    positive = [ outputs[i]/(sum(outputs[:2])) for i in [0,1]]
    negative = [ outputs[i]/(sum(outputs[2:])) for i in [2,3]]
    
    fig = make_subplots(rows=1, cols=2)

    names = [
        'True positive',
        'False positive',
        'True negative',
        'False negative',
            ]
    
    colors = ['red']*2 + ['blue']*2

    line = go.Bar(
            x=names[:2],
            y=positive,
            marker_color = colors[:2],
            showlegend=False
        )
    
    fig.add_trace(line, row=1, col=1)

    line = go.Bar(
            x=names[2:],
            y=negative,
            marker_color = colors[2:],
            showlegend=False
        )
    
    fig.add_trace(line, row=1, col=2)

    layout = go.Layout(
                template="simple_white",
                title=title,
        )

    fig.update_layout(layout)
    
    fig.update_xaxes(title_text='Test outcome', fixedrange= True, row=1, col=1)
    fig.update_xaxes(title_text='Test outcome', fixedrange= True, row=1, col=2)

    fig.update_yaxes(title_text='Proportion of positive tests', fixedrange= True, row=1, col=1)
    fig.update_yaxes(title_text='Proportion of negative tests', fixedrange= True, row=1, col=2)
    
    return fig





def vaccine_plot(df, c_names, normalise_by_pop):
    
    yaxis_title = "People"

    if normalise_by_pop:
        yaxis_title += " (% of population)"

    traces = []

    for country in c_names:
        this_country = df[df['location']==country]
        df_plot = this_country[this_country['people_vaccinated']!=""]

        xx = df_plot['date']
        yy = df_plot['people_vaccinated'].astype(int)
        if normalise_by_pop:
            yy = yy/POPULATIONS[country.lower()] * 100

        line = go.Scatter(
                x=xx,
                y=yy,
                name=country,
                mode='lines+markers'
            )

        traces.append(line)

    layout = go.Layout(
                template="plotly_white",
                # title=", ".join(c_names),
                height=450,
                xaxis=dict(
                        fixedrange= True,
                        ),
                yaxis=dict(title=yaxis_title,
                    fixedrange= True),
                legend= dict(
                    x = 0.5,
                    font=dict(size=11),
                    y = 1.03,
                    xanchor= 'center',
                    yanchor= 'bottom'
                ),
                margin= {'l': 0, 'b': 10, 't': 10, 'r': 10, 'pad': 0},
            )

    
    return {'data': traces, 'layout': layout}