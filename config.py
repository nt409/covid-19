presets_dict = {'N': 'Do Nothing',
                'MSD': 'Social Distancing',
                'H': 'Lockdown High Risk, No Social Distancing For Low Risk',
                'HL': 'Lockdown High Risk, Social Distancing For Low Risk',
                'Q': 'Lockdown All',
                'LC': 'Lockdown Cycles',
                'C': 'Custom'}

presets_dict_dropdown = {'N': 'Do Nothing',
                'MSD': 'Social Distancing',
                'H': 'High Risk: Lockdown, Low Risk: No Social Distancing',
                'HL': 'High Risk: Lockdown, Low Risk: Social Distancing',
                'Q': 'Lockdown All',
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
