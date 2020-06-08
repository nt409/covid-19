import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from parameters_cov import params, df2
import pandas as pd
import numpy as np
from math import floor, ceil
import datetime
from dan_constants import POPULATIONS

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

    # print(c_low,c_high)
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
        # print(index[name])
        # print(sol['y'][index[name],:])
        #str()
        y_Low   = 100*pd.Series(sol['y'][index[name],:]).values
        y_High  = 100*pd.Series(sol['y'][index[name]+params.number_compartments,:]).values
        y_Total = y_Low + y_High


        # df = df.assign(e = pd.Series(sol['y'][index[name],:]).values)
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





def yaxis_function(Yrange,population_plot):

    yy2 = [0]
    for i in range(8):
        yy2.append(10**(i-5))
        yy2.append(2*10**(i-5))
        yy2.append(5*10**(i-5))

    yy = [i for i in yy2]


    for i in range(len(yy)-1):
        if Yrange[1]>yy[i] and Yrange[1] <= yy[i+1]:
            pop_vec_lin = np.linspace(0,yy2[i+1],11)

    linTicks = [i*(population_plot) for i in pop_vec_lin]

    log_bottom = -8
    log_range = [log_bottom,np.log10(Yrange[1])]

    pop_vec_log_intermediate = np.linspace(log_range[0],ceil(np.log10(pop_vec_lin[-1])), 1+ ceil(np.log10(pop_vec_lin[-1])-log_range[0]) )

    pop_log_vec = [10**(i) for i in pop_vec_log_intermediate]
    logTicks = [i*(population_plot) for i in pop_log_vec]
    
    return linTicks, pop_vec_lin, log_range, logTicks, pop_log_vec


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
                y1=Yrange[1],
                line=dict(
                    color="LightSkyBlue",
                    width=0,
                ),
                fillcolor="LightSkyBlue",
                opacity= blue_opacity
            ))
            
    if ICU: #  and 'C' in cats_to_plot:
        # if which_plots=='two':
        control_font_size = font_size*(22/24) # '10em'
        ICU_font_size = font_size*(22/24) # '10em'

        yval_pink = 0.3
        yval_blue = 0.82


        for c_min, c_max in zip(c_low, c_high):
            if c_min>=0 and c_max>=0:
                shapez.append(dict(
                        # filled Pink ICU Rectangle
                        type="rect",
                        x0= startdate+datetime.timedelta(days=c_min), #month_len*  ##c_min/month_len,
                        y0=0,
                        x1= startdate+datetime.timedelta(days=c_max), #c_max/month_len,
                        y1=Yrange[1],
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
                        text="<b>ICU<br>" + "<b> Capacity<br>" + "<b> Exceeded",
                        # hoverinfo='ICU Capacity Exceeded',
                        showarrow=False,
                        textangle= 0,
                        font=dict(
                            size= ICU_font_size,
                            color="purple"
                        ),
                        opacity=0.6,
                        xref = 'x',
                        yref = 'paper',
                ))

    else:
        control_font_size = font_size*(30/24) #'11em'
        yval_blue = 0.4




    if month[0]!=month[1] and preset!='N':
        annotz.append(dict(
                x  = startdate+datetime.timedelta(days=month_len*max(0.5*(month[0]+month[1]), 0.5)),
                y  = yval_blue,
                text="<b>Control<br>" + "<b> In <br>" + "<b> Place",
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
    
    if month_cycle is not None:
        for i in range(0,len(month_cycle),2):
            shapez.append(dict(
                    # filled Blue Control Rectangle
                    type="rect",
                    x0= startdate+datetime.timedelta(days=month_len*month_cycle[i]),
                    y0=0,
                    x1= startdate+datetime.timedelta(days=month_len*month_cycle[i+1]),
                    y1=Yrange[1],
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
                    if name in ['S','R']:
                        vis = False
                    else:
                        vis = True

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

    # name= cats_to_plot[0]
    # cats_to_plot = ['S','I','R','H','C','D']
    cats_to_plot = ['D']

    for name in cats_to_plot:
        for group in group_use:

            
            name_string = ':' + group_strings[group]

            xx = [startdate + datetime.timedelta(days=i) for i in sol['t']]
            # print(xx[0])
            yyy_p = np.asarray(dataframe[f'{longname[name]}: {group}'])
            
            xx = [xx[i] for i in range(1,len(xx),2)]
            yyy_p = [yyy_p[i] for i in range(1,len(yyy_p),2)]


            line =  {'x': xx, 'y': yyy_p,
                    'hovertemplate': '%{y:.2f}%, %{text}',
                    'text': [human_format(i*population_plot/100,dp=1) for i in yyy_p],
                    'type': 'bar',
                    'legendgroup': name ,
                    'name': longname[name] + name_string}
                    
            if group=='LR':
                line['marker'] = dict(color='LightSkyBlue')
            elif group=='HR':
                line['marker'] = dict(color='orange')
            
            line['visible'] = False
            
            lines_to_plot.append(line)

    return lines_to_plot, xx







def uncertPlot(upper_lower_sol,population_plot,startdate):
    lines_to_plot = []
    ii = -1
    name = 'D'
    # group_use = ['BR']

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
                # 'legendgroup': name,
                'visible': False,
                'showlegend': False,
                'fill': fill,
                'name': longname[name] + label_add}
        lines_to_plot.append(line)

    return lines_to_plot


def prevDeaths(previous_deaths,startdate,population_plot):
    lines_to_plot = []
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
            legendgroup='deaths',
            line=dict(
            color= 'purple',
            dash = 'dash'
            ),
            hovertemplate = '%{y:.2f}%, %{text}',
            text = [human_format(i*population_plot/100,dp=1) for i in y_deaths],
            name= 'Recorded deaths'))
        x0 = x_deaths[0]
    return lines_to_plot, x0


def MultiFigureGenerator(upper_lower_sol,sols,month,num_strat,ICU_to_plot=False,vaccine_time=None,ICU_grow=None,comp_dn=False,country = 'uk',month_cycle=None,preset=None,startdate=None, previous_deaths=None):
    
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
    
    
    # if num_strat=='one':
    #     group_use = groups
    # if num_strat=='two' or comp_dn:
    #     group_use = groups2
   
   
    lines_to_plot_line, xx = lineplot(sols,population_plot,startdate,num_strat,comp_dn)
    lines_to_plot_stack, xxx = stackPlot(sols,population_plot,startdate)
    lines_to_plot_uncert = uncertPlot(upper_lower_sol,population_plot,startdate)
    lines_PrevDeaths, x0 = prevDeaths(previous_deaths,startdate,population_plot)
    # lines_to_plot, xx = lineplot(sols,cats_to_plot,group_use,population_plot,startdate,num_strat,comp_dn)
    # for line in lines_to_plot_stack:
    #     line['visible'] = False


    # print(xx[0])
    # setting up pink boxes
    ICU = False
    if num_strat=='one': #  and len(cats_to_plot)>0:
        yyy = np.asarray(sols[0]['y'])
        ttt = sols[0]['t']
        c_low, c_high, ICU = time_exceeded_function(yyy,ttt,ICU_grow)
    
    ymax = 0.01
    for line in lines_to_plot_line:
        if line['visible']:
            ymax = max(ymax,max(line['y']))
            # print(ymax,line['name'])

    yax = dict(range= [0,min(1.1*ymax,100)])
    
    # print(ymax)
    ##



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

    if vaccine_time is not None:
        moreLines.append(
        dict(
        type='scatter',
            x=[startdate+datetime.timedelta(days=month_len*vaccine_time),
            startdate+datetime.timedelta(days=month_len*vaccine_time)],
            y=[yax['range'][0],yax['range'][1]],
            mode='lines',
            opacity=0.9,
            legendgroup='thresholds',
            line=dict(
            color= 'green',
            dash = 'dash'
            ),
            hovertemplate= 'Vaccination starts<extra></extra>',
            name= 'Vaccination starts'))

    
    
    moreLines.append(
    dict(
        type='scatter',
        x = [xx[0],xx[-1]],
        y = [0, population_plot],
        yaxis="y2",
        opacity=0,
        hoverinfo = 'skip',
        showlegend=False
    ))

    controlLines = []
    if month[0]!=month[1] and preset != 'N':
        controlLines.append(
        dict(
        type='scatter',
            x=[startdate+datetime.timedelta(days=month_len*month[0]),
            startdate+datetime.timedelta(days= month_len*month[0])], # +1 to make it visible when at 0
             y=[0,ymax],
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
            y=[0,ymax],
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




    linTicks, pop_vec_lin, log_range, logTicks, pop_log_vec = yaxis_function(yax['range'],population_plot)
    annotz, shapez = annotations_shapes_function(month_cycle,month,preset,startdate,ICU,font_size,c_low,c_high,yax['range'])


    

    layout = go.Layout(
                    annotations=annotz,
                    shapes=shapez,
                    barmode = 'stack',
                    template="simple_white",
                    font = dict(size= font_size),
                    margin=dict(t=5, b=5, l=10, r=10,pad=15),
                    # visible= [True]*len(lines_to_plot_line) + [False]*len(lines_to_plot_stack)  + [True]*len(moreLines),
                    yaxis= dict(mirror= True,
                            title='Percentage of Total Population',
                            range= yax['range'],
                            fixedrange= True,
                            automargin=True,
                            type = 'linear'
                    ),
                    hovermode='x',
                    xaxis= dict(
                            range= [xx[0], xx[floor((2/3)*len(xx))]],
                            hoverformat='%d %b',
                            fixedrange= True,
                        ),
                        updatemenus = [dict(
                                                buttons=list([
                                                    dict(
                                                        args = ["xaxis", {'range': [xx[0], xx[floor((1/3)*len(xx))]],
                                                        'hoverformat':'%d %b',
                                                        'fixedrange': True,
                                                        }],
                                                        label="Years: 1",
                                                        method="relayout"
                                                    ),
                                                    dict(
                                                        args = ["xaxis", {'range': [xx[0], xx[floor((2/3)*len(xx))]],
                                                        'hoverformat':'%d %b',
                                                        'fixedrange': True,
                                                        }],
                                                        label="Years: 2",
                                                        method="relayout"
                                                    ),
                                                    dict(
                                                        args = ["xaxis", {'range': [xx[0], xx[-1]],
                                                        'hoverformat':'%d %b',
                                                        'fixedrange': True,
                                                        }],
                                                        label="Years: 3",
                                                        method="relayout"
                                                    )
                                            ]),
                                            x= 0.3,
                                            xanchor = 'left',
                                            pad={"r": 5, "t": 30, "b": 10, "l": 5},
                                            showactive=True,
                                            active=1,
                                            direction='up',
                                            y=-0.13,
                                            yanchor="top"
                                            ),
                                            dict(
                                                buttons=list([
                                                    dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population','fixedrange': True, 'type': 'linear', 'range': yax['range'], 'automargin': True}, # , 'showline':False
                                                    "yaxis2": {'title': 'Population (' + country_name + ')','fixedrange': True, 'type': 'linear', 'overlaying': 'y1', 'range': yax['range'], 'ticktext': [human_format(0.01*linTicks[i]) for i in range(len(pop_vec_lin))], 'tickvals': [i for i in  pop_vec_lin],'automargin': True,'side':'right'} # , 'showline':False,
                                                    }], # tickformat
                                                    label="Linear",
                                                    method="relayout"
                                                ),
                                                dict(
                                                    args=[{"yaxis": {'title': 'Percentage of Total Population', 'fixedrange': True, 'type': 'log', 'range': log_range,'automargin': True}, # , 'showline':False
                                                    "yaxis2": {'title': 'Population (' + country_name + ')', 'fixedrange': True, 'type': 'log', 'overlaying': 'y1', 'range': log_range, 'ticktext': [human_format(0.01*logTicks[i]) for i in range(len(pop_log_vec))], 'tickvals': [i for i in  pop_log_vec],'automargin': True,'side':'right'} #  'showline':False,
                                                    }], # 'tickformat': yax_form_log,
                                                    label="Logarithmic",
                                                    method="relayout"
                                                )
                                        ]),
                                        x= 0.3,
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
                                                    args=[
                                                    {'visible':
                                                    [True]*len(lines_to_plot_line) + [False]*len(lines_to_plot_stack) + [False]*len(lines_to_plot_uncert)  + [True]*len(moreLines)  + [False]*len(controlLines)
                                                    },
                                                    {
                                                    # "annotations": annotz,
                                                    "shapes": shapez
                                                    },
                                                    ],
                                                    label="Line",
                                                    method="update"
                                                ),
                                                dict(
                                                    args=[
                                                    {"visible":[False]*len(lines_to_plot_line) + [True]*len(lines_to_plot_stack) + [False]*len(lines_to_plot_uncert)  + [True]*len(moreLines)  + [True]*len(controlLines)},
                                                    {
                                                    # "annotations":[],
                                                    "shapes":[],
                                                    "barmode":'stack'
                                                    },
                                                    ],
                                                    label="Stacked Bar",
                                                    method="update"
                                                ),
                                                dict(
                                                    args=[
                                                    {"visible":[False]*(len(lines_to_plot_line)-1) + [True] + [False]*len(lines_to_plot_stack) + [True]*len(lines_to_plot_uncert)  + [True]*len(moreLines)  + [False]*len(controlLines)},
                                                    {
                                                    # "annotations":[],
                                                    # "shapes":[],
                                                    "shapes": shapez,
                                                    # "barmode":'stack'
                                                    },
                                                    ],
                                                    label="Fatalities",
                                                    method="update"
                                                )
                                        ]),
                                        x= 0.7,
                                        xanchor="left",
                                        pad={"r": 5, "t": 30, "b": 10, "l": 5},
                                        active=0,
                                        y=-0.13,
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
                                        yaxis2 = dict(
                                                        title = 'Population (' + country_name + ')',
                                                        overlaying='y1',
                                                        fixedrange= True,
                                                        # showline=False,
                                                        range = yax['range'],
                                                        side='right',
                                                        ticktext = [human_format(0.01*linTicks[i]) for i in range(len(pop_vec_lin))],
                                                        tickvals = [i for i in  pop_vec_lin],
                                                        automargin=True
                                                    )
                            )

    linesUse = lines_to_plot_line + lines_to_plot_stack + lines_to_plot_uncert + moreLines + controlLines

    return {'data': linesUse, 'layout': layout}















