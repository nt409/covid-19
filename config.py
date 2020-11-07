import plotly.graph_objects as go






# appearance
backgd_color = None
disclaimerColor = '#e9ecef'

# figure appearance
bar_height = '100'
bar_width  =  '100'
bar_non_crit_style = {'height': bar_height, 'width': bar_width, 'display': 'block' }



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

presets_dict = {'N': 'Do Nothing',
                'MSD': 'Social Distancing',
                'H': 'Lockdown High Risk, No Social Distancing For Low Risk',
                'HL': 'Lockdown High Risk, Social Distancing For Low Risk',
                'Q': 'Lockdown',
                'LC': 'Lockdown Cycles',
                'C': 'Custom'}

presets_dict_dropdown = {'N': 'Do Nothing',
                'MSD': 'Social Distancing',
                'H': 'High Risk: Lockdown, Low Risk: No Social Distancing',
                'HL': 'High Risk: Lockdown, Low Risk: Social Distancing',
                'Q': 'Lockdown',
                'LC': 'Lockdown Cycles (switching lockdown on and off)',
                'C': 'Custom'}

ld = 5
sd = 8
noth = 10

preset_dict_high = {'Q': ld, 'MSD': sd, 'LC': ld, 'HL': ld,  'H': ld,  'N':noth}
preset_dict_low  = {'Q': ld, 'MSD': sd, 'LC': ld, 'HL': sd, 'H': noth, 'N':noth}

initial_strat = 'Q'

initial_hr = preset_dict_high[initial_strat]
initial_lr = preset_dict_low[initial_strat]
