import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from math import ceil, cos, pi
import numpy as np
from scipy.integrate import ode
import datetime


from config import presets_dict
from parameters_cov import params
from data_scraper import get_data
from data_constants import POPULATIONS

##
# -----------------------------------------------------------------------------------
##
def ode_system(t, y,
                    beta_L_factor, beta_H_factor,
                    vaccinate_rate, ICU_grow,
                    days_since_peak):

    # print(t)

    S_L = y[params.S_L_ind]
    I_L = y[params.I_L_ind]
    H_L = y[params.H_L_ind]
    C_L = y[params.C_L_ind]
    
    S_H = y[params.S_H_ind]
    I_H = y[params.I_H_ind]
    H_H = y[params.H_H_ind]
    C_H = y[params.C_H_ind]

    vaccine_effect_H = 0
    vaccine_effect_L = 0
    if y[params.S_H_ind] > (1-params.vaccinate_percent)*params.hr_frac: # vaccinate 90%
        vaccine_effect_H = vaccinate_rate
    elif y[params.S_L_ind] > (1-params.vaccinate_percent)*(1-params.hr_frac):  # vaccinate 90%
        vaccine_effect_L = vaccinate_rate

    ICU_capac = params.ICU_capacity*(1 + ICU_grow*t/365 )

    
    if C_H+C_L > 0:
        ICU_capacYoung = C_L/(C_H+C_L)*ICU_capac
        ICU_capacOld   = C_H/(C_H+C_L)*ICU_capac
    else:
        ICU_capacYoung = (1-params.hr_frac)*ICU_capac
        ICU_capacOld   = params.hr_frac*ICU_capac


    C_L_to_R_L =  min(ICU_capacYoung,C_L)*params.crit_recovery
    C_L_to_D_L = (min(ICU_capacYoung,C_L)*params.crit_death
                        + params.noICU*(params.crit_death + params.crit_recovery)*max(C_L-ICU_capacYoung,0) # all without crit care die
                    )
    
    C_H_to_R_H =  min(ICU_capacOld,C_H)*params.crit_recovery
    C_H_to_D_H = (min(ICU_capacOld,C_H)*params.crit_death
                        + params.noICU*(params.crit_death + params.crit_recovery)*max(C_H-ICU_capacOld,0) # all without crit care die
                    )
    
    beta = params.beta*(1 + 0.2*cos((t+days_since_peak)*2* pi /365) )

    dydt = [-S_L*beta*( (beta_L_factor**2)*I_L  +  (beta_H_factor*beta_L_factor)*I_H ) - vaccine_effect_L, # dS
            +S_L*beta*( (beta_L_factor**2)*I_L  +  (beta_H_factor*beta_L_factor)*I_H ) - params.mu_L*I_L - params.gamma_L*I_L + (1-params.hr_frac)*params.import_rate, # dI
            I_L*params.mu_L    + H_L*params.recover_L + C_L_to_R_L + vaccine_effect_L  - (1-params.hr_frac)*params.import_rate,  # dR
            I_L*params.gamma_L - H_L*params.recover_L - H_L*params.crit_L,         # dH
            H_L*params.crit_L  - (C_L_to_R_L + C_L_to_D_L),                          # dC
            + C_L_to_D_L,                                                      # dD

            -S_H*beta*( (beta_H_factor*beta_L_factor)*I_L + (beta_H_factor**2)*I_H) - vaccine_effect_H , # dS
            +S_H*beta*( (beta_H_factor*beta_L_factor)*I_L + (beta_H_factor**2)*I_H) - params.mu_H*I_H - params.gamma_H*I_H + params.hr_frac*params.import_rate, # dI
            I_H*params.mu_H    + H_H*params.recover_H  + C_H_to_R_H + vaccine_effect_H -  params.hr_frac*params.import_rate,           # dR
            I_H*params.gamma_H - H_H*params.recover_H - H_H*params.crit_H,         # dH
            H_H*params.crit_H  - (C_H_to_R_H + C_H_to_D_H),                          # dC
            + C_H_to_D_H                                                       # dD
            ]
    
    return dydt



##
#--------------------------------------------------------------------
##

def set_initial_condition(I0,R0,H0,C0,D0):

    prop_hosp_H = (params.hr_frac*params.frac_hosp_H)   /( (1-params.hr_frac)*params.frac_hosp_L   +    params.hr_frac*params.frac_hosp_H  )

    prop_crit_H = (params.hr_frac*params.frac_hosp_H*params.frac_crit_H)/( (1-params.hr_frac)*params.frac_hosp_L*params.frac_crit_L     +    params.hr_frac*params.frac_hosp_H*params.frac_crit_H)
    
    prop_rec_H = (params.hr_frac*(1-params.frac_hosp_H))/( (1-params.hr_frac)*(1 - params.frac_hosp_L)   +  params.hr_frac*(1 - params.frac_hosp_H) )

    I0_L = (1-params.hr_frac)*params.N*I0
    R0_L = (1-prop_rec_H)    *params.N*R0
    H0_L = (1-prop_hosp_H)   *params.N*H0
    C0_L = (1-prop_crit_H)   *params.N*C0
    D0_L = (1-prop_crit_H)   *params.N*D0
    S0_L = (1-params.hr_frac)*params.N - I0_L - R0_L - C0_L - H0_L - D0_L
    
    I0_H = params.hr_frac*params.N*I0
    R0_H = prop_rec_H    *params.N*R0
    H0_H = prop_hosp_H   *params.N*H0
    C0_H = prop_crit_H   *params.N*C0
    D0_H = prop_crit_H   *params.N*D0
    S0_H = params.hr_frac*params.N - I0_H - R0_H - C0_H - H0_H - D0_H

    y0 = [S0_L,
        I0_L,
        R0_L,
        H0_L,
        C0_L,
        D0_L,
        S0_H,
        I0_H,
        R0_H,
        H0_H,
        C0_H,
        D0_H,
        ]
    
    return y0



def determine_betas(t, t_control, beta_L_factor, beta_H_factor, let_HR_out):
    
    # to avoid the boundary
    # t = t+0.001 
    
    if t_control is None:
        beta_L_factor = 1
        beta_H_factor = 1
    else:
        
        if len(t_control)==2:
            if t<t_control[0] or t>=t_control[1]: # outside of window
                beta_L_factor = 1
                beta_H_factor = 1
        
        elif len(t_control)>2: # lockdown cycle
            control = False
            for i in range(0,len(t_control)-1,2):
                if not control and (t>=t_control[i] and t<t_control[i+1]): # in a control period
                    control = True
            if not control:
                beta_L_factor = 1
                if let_HR_out:
                    beta_H_factor = 1

    return np.float(beta_L_factor), np.float(beta_H_factor)
##
# ----------------------------------------------------------
##
def solve_it(y0,
                beta_L_factor, beta_H_factor,
                t_control, vaccine_time,
                ICU_grow, let_HR_out, date):
    
    # remove vaccine, beta stuff, let HR out

    y_list = []
    tVecFinal = []
    
    T_stop = 365*0.75 + 0.02
    if vaccine_time is None:
        vaccine_timing = [0]
    else:
        # - 0.01 so different to control month
        vaccine_timing = [365*(vaccine_time/12)+0.01]
    
    if t_control is not None:
        KeyTimesList = [0,T_stop] + t_control + vaccine_timing
        KeyTimesList.sort()
    else:
        KeyTimesList = [0,T_stop]

    # could have several '0' entries - filter so that just one
    MoreThan0 = filter(lambda number: number > 0, KeyTimesList)
    MoreThan0 = list(MoreThan0)
    KeyTimesList = [0] + MoreThan0

    tVecList = []
    for i in range(len(KeyTimesList)-1):
        tVecList.append(np.linspace(KeyTimesList[i],
                                    KeyTimesList[i+1],
                                    max(1+ceil((KeyTimesList[i+1]-KeyTimesList[i])/2),4)
                                    )
                        )

    
    # remove final value from list (since the Key times needed below are the starting points for the solver)
    StartTimesList = KeyTimesList[:-1]
    
    odeSolver = ode(ode_system,jac=None)
    # odeSolver.set_integrator('dopri5')

    y0_new = None
    for t, tim in zip(StartTimesList,tVecList):
        if y0_new is None:
            y0_new = np.asarray(y0)
        else:
            y0_new = odeSolver.y
        
        yy2  = np.zeros((len(y0),len(tim)))

        odeSolver.set_initial_value(y0_new,t)

        vaccinate_rate = 0
        if vaccine_time is not None:
            if t>=365*(vaccine_time/12) - 1: # -1 for rounding error
                vaccinate_rate = params.vaccinate_rate
        
        betaLfact, betaHfact = determine_betas(t, t_control, beta_L_factor, beta_H_factor, let_HR_out)
        
        ICU_grow = np.float(ICU_grow)
        
        
        split_date = date.split('-')
        
        year = int(split_date[0])
        month = int(split_date[1])
        day = int(split_date[2])

        start_date = datetime.date(year, month, day)
        peak_date = datetime.date(2021, 1, 1)

        days_since_peak = start_date-peak_date
        
        days_since_peak = float(days_since_peak.days)

        odeSolver.set_f_params(betaLfact,
                                    betaHfact,
                                    vaccinate_rate,
                                    ICU_grow,
                                    days_since_peak)
        
        ##
        for ind, tt in enumerate(tim[1:]):
            if odeSolver.successful():
                yy2[:,ind] = odeSolver.y
                odeSolver.integrate(tt)
            else:
                raise RuntimeError('ode solver unsuccessful')
        
        if t != StartTimesList[-1]:
            y_list.append(yy2[:,:-1])
            tVecFinal.append(tim[:-1])
##----------------------------------------------------------------------------------
    yy2[:,-1] = odeSolver.y
    y_list.append(yy2)
    tVecFinal.append(tim)
     
    t_out  = np.concatenate((tVecFinal))
    y_out  = np.concatenate((y_list),axis=1)

    return y_out, t_out

##
#--------------------------------------------------------------------
##
def run_model(I0,R0,H0,C0,D0,
                beta_L_factor=1,
                beta_H_factor=1,
                t_control=None,
                vaccine_time=None,
                date=None,
                ICU_grow=0,
                let_HR_out=True):

    print('running model')

    y0 = set_initial_condition(I0,R0,H0,C0,D0)

    otherArgs = beta_L_factor, beta_H_factor, t_control, vaccine_time, ICU_grow, let_HR_out, date

    y_out, tVec = solve_it(y0, *otherArgs)

    print('finished running')


    dicto = {'y': y_out,'t': tVec,'beta_L': beta_L_factor,'beta_H': beta_H_factor}
    return dicto

#--------------------------------------------------------------------





def test_probs(have_it,sens,spec):
    
    true_pos = have_it*sens
    false_neg = have_it*(1-sens)

    false_pos = (1-have_it)*(1-spec)
    true_neg = (1-have_it)*spec
    
    return true_pos, false_pos, true_neg, false_neg










def begin_date(date,country='uk'):

    date = datetime.datetime.strptime(date.split('T')[0], '%Y-%m-%d').date()
    pre_defined = False



    try:
        country_data = get_data(country)
        min_date = country_data['Cases']['dates'][0]
        max_date = country_data['Cases']['dates'][-1]
        min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )
        max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d' )
    except:
        print("Cannnot get country data from:",country)
        pre_defined = True
        min_date = '2020-2-15' # first day of data
        min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d' )

        max_date = datetime.datetime.today() - datetime.timedelta(days=2)
        max_date = str(max_date).split(' ')[0]

    if country_data is None:
        print("Country data none")
        pre_defined = True

    try:
        population_country = POPULATIONS[country]
    except:
        population_country = 100*10**6
        print("Cannot get country population")
        pre_defined = True

    
    if not pre_defined:
        worked = True

        dates = np.asarray(country_data['Cases']['dates'])
        deaths_data = np.asarray(country_data['Deaths']['data'])
        cases       = np.asarray(country_data['Cases']['data'])

        

        date_objects = []
        for dt in dates:
            date_objects.append(datetime.datetime.strptime(dt, '%Y-%m-%d').date())

        try:
            index = date_objects.index(date)
        except Exception as e: # defaults to today if error
            index = -1
            worked = False
            print('Date error; ',e)
        
        if index>=26:
            pass # all good
        else:
            index=-1
            worked = False
            print("dates didn't go far enough back")      
        
        try:
            I0    = np.float(cases[index]) - np.float(cases[index - 10]) # all cases in the last 10 days

            # 5 days from symptoms to get hospitalised... symptoms 5 days ago, infected 5 days before.
            # Anyone from previous 8 days could be in hosp 10-18 days
            # Anyone from previous 8 days could be in crit 18-26 days
            I_hosp_delay = np.float(cases[index - 10]) - np.float(cases[index - 18])  #sum( [np.float(currently_inf_data[index - i]) for i in range(10,19) ]  ) - 7*np.float(currently_inf_data[index - 18]) # counted too many times
            I_crit_delay = np.float(cases[index - 18]) - np.float(cases[index - 26])  #sum( [np.float(currently_inf_data[index - i]) for i in range(18,27) ]  ) - 7*np.float(currently_inf_data[index - 26]) # counted too many times
            # print(I0,I_hosp_delay,I_crit_delay)
        except:
            worked = False
            I0           = 0.01 # np.float(currently_inf_data[index])
            I_hosp_delay = 0.01 # np.float(currently_inf_data[index-10])
            I_crit_delay = 0.01 # np.float(currently_inf_data[index-18])
            print("dates didn't go far enough back, I_hosp_delay")      
        
        D0    = np.float(deaths_data[index])

        prev_deaths = deaths_data[:index]
        # of resolved cases, fatality rate is 0.9%
        p = 0.009
        R0 = D0*(1-p)/p

        R0 = R0/population_country
        D0 = D0/population_country

        factor_infections_underreported = 1.5*2
        # only small fraction of cases reported (and usually only symptomatic) symptomatic is 50%

        I0           = factor_infections_underreported*I0/population_country
        I_hosp_delay = factor_infections_underreported*I_hosp_delay/population_country
        I_crit_delay = factor_infections_underreported*I_crit_delay/population_country



        #  H rate for symptomatic is 4.4% so 
        hosp_proportion = 0.044
        #  30% of H cases critical
        crit_proportion = 0.3 # 0.3

        H0 = I_hosp_delay*hosp_proportion
        C0 = I_crit_delay*hosp_proportion*crit_proportion
        # print(H0,C0)

        I0 = I0 - H0 - C0 # since those in hosp/crit will be counted in current numbers
        return I0, R0, H0, C0, D0, worked, min_date, max_date, prev_deaths
    else:
        return 0.0015526616816533823, 0.011511334132676547, 1.6477539091227494e-05, 7.061802467668927e-06, 0.00010454289323318761, False, min_date, max_date, prev_deaths # if data collection fails, use UK on 8th April as default






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

    ],style={'marginTop': '20px', 'marginBottom': '20px','fontSize':'75%'})





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
                    justify='center', style={'marginTop': '20px'}
                    ),

            
            # dbc
            dbc.Row([

            
                dbc.Col([


                                html.H3('After 1 year:',style={'fontSize': '150%', 'marginTop': '30px', 'marginBottom': '30px'}),

                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button('Reduction in deaths 🛈',
                                                    color='primary',
                                                    className='mb-3',
                                                    id="popover-red-deaths-target",
                                                    size='sm',
                                                    style = {'cursor': 'pointer'}),
                                                    dbc.Popover(
                                                        [
                                                        dbc.PopoverHeader('Reduction in deaths'),
                                                        dbc.PopoverBody(html.Div(
                                                        'This box shows the reduction in deaths due to the control strategy choice.'
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
                                                    style={'cursor': 'pointer'}
                                                    ),

                                                    
                                                    dbc.Popover(
                                                        [
                                                        dbc.PopoverHeader('ICU requirement'),
                                                        dbc.PopoverBody(html.Div(
                                                        'COVID-19 can cause a large number of serious illnesses very quickly. This box shows the extent to which the NHS capacity would be overwhelmed by the strategy choice (if nothing was done to increase capacity).'
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
                                                    style={'cursor': 'pointer'}
                                                    ),               
                                                                        
                                                    dbc.Popover(
                                                        [
                                                        dbc.PopoverHeader('Herd immunity'),
                                                        dbc.PopoverBody(dcc.Markdown(
                                                        '''

                                                        This box shows how close to the safety threshold for herd immunity we got. If we reached (or exceeded) the threshold it will say 100%.
                                                        
                                                        However, this is the least important goal since an uncontrolled pandemic will reach safe levels of immunity very quickly, but cause lots of serious illness in doing so.
                                                        ''',
                                                        style={'font-family': 'sans-serif'}
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

                                html.H3('After 2 years:',style={'fontSize': '150%', 'marginTop': '30px', 'marginBottom': '30px'}),

                                cards_fn(death_stat_2nd,dat3_2nd,herd_stat_2nd,color_2nd_death,color_2nd_herd,color_2nd_ICU),


                ],
                width=12,
                ),


            ],
            align='center',
            ),

            ],style=on_or_off)




