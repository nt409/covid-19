import plotly.graph_objects as go






# appearance
backgd_color = None
disclaimerColor = '#e9ecef'

# figure appearance
# bar_height = '100'
# bar_width  =  '100%'
bar_non_crit_style = {'display': 'block' }



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

scatter = dict(
                    x = [0,1],
                    y = [0,1],
                    mode = 'lines',
                    showlegend=False,
                    line = {'width': 0},
                    )
                    
dummy_figure = dict(data=[scatter], layout= {'template': 'simple_white', 'annotations': annotz, 'xaxis': {'fixedrange': True}, 'yaxis': {'fixedrange': True}})







#

presets_dict = {'N': 'Do nothing',
                'MSD': 'Tiered approach',
                'H': 'Lockdown high risk, no social distancing for low risk',
                'HL': 'Lockdown high risk, social distancing for low risk',
                'Q': 'Lockdown',
                'LC': 'Lockdown cycles',
                'C': 'Custom'}

presets_dict_dropdown = {'N': 'Do nothing',
                'MSD': 'Tiered approach',
                'H': 'High risk: lockdown, low risk: no social distancing',
                'HL': 'High risk: lockdown, low risk: social distancing',
                'Q': 'Lockdown',
                'LC': 'Lockdown cycles (switching lockdown on and off)',
                'C': 'Custom'}

ld = 3
sd = 4
noth = 10

preset_dict_high = {'Q': ld, 'MSD': sd, 'LC': ld, 'HL': ld,  'H': ld,  'N':noth}
preset_dict_low  = {'Q': ld, 'MSD': sd, 'LC': ld, 'HL': sd, 'H': noth, 'N':noth}

initial_strat = 'MSD'

initial_hr = preset_dict_high[initial_strat]
initial_lr = preset_dict_low[initial_strat]
