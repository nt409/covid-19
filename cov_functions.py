from parameters_cov import params
from math import exp, ceil, log, floor, sqrt
import numpy as np
from scipy.integrate import ode
from scipy.stats import norm, gamma
import pandas as pd
##
# -----------------------------------------------------------------------------------
##
def ode_system(t, y,
                    beta_L_factor, beta_H_factor,
                    vaccinate_rate, ICU_grow):

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
    
    dydt = [-S_L*( params.beta*(beta_L_factor**2)*I_L  +  (params.beta*((beta_H_factor*beta_L_factor)**1))*I_H ) - vaccine_effect_L, # dS
            +S_L*( params.beta*(beta_L_factor**2)*I_L  +  (params.beta*((beta_H_factor*beta_L_factor)**1))*I_H ) - params.mu_L*I_L - params.gamma_L*I_L + (1-params.hr_frac)*params.import_rate, # dI
            I_L*params.mu_L    + H_L*params.recover_L + C_L_to_R_L + vaccine_effect_L  - (1-params.hr_frac)*params.import_rate,  # dR
            I_L*params.gamma_L - H_L*params.recover_L - H_L*params.crit_L,         # dH
            H_L*params.crit_L  - (C_L_to_R_L + C_L_to_D_L),                          # dC
            + C_L_to_D_L,                                                      # dD

            -S_H*( (params.beta*((beta_H_factor*beta_L_factor)**1))*I_L + params.beta*(beta_H_factor**2)*I_H) - vaccine_effect_H , # dS
            +S_H*( (params.beta*((beta_H_factor*beta_L_factor)**1))*I_L + params.beta*(beta_H_factor**2)*I_H) - params.mu_H*I_H - params.gamma_H*I_H + params.hr_frac*params.import_rate, # dI
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
                ICU_grow, let_HR_out):
    
    # remove vaccine, beta stuff, let HR out

    y_list = []
    tVecFinal = []
    
    T_stop = 365*2
    if vaccine_time is None:
        vaccine_timing = [0]
    else:
        # - 0.01 so different to control month
        vaccine_timing = [365*(vaccine_time/12)-0.01]
    
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

        odeSolver.set_f_params(betaLfact,
                                    betaHfact,
                                    vaccinate_rate,
                                    ICU_grow)
        
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
                ICU_grow=0,
                let_HR_out=True):

    print('running model')

    y0 = set_initial_condition(I0,R0,H0,C0,D0)

    otherArgs = beta_L_factor, beta_H_factor, t_control, vaccine_time, ICU_grow, let_HR_out

    y_out, tVec = solve_it(y0, *otherArgs)

    print('finished running')


    dicto = {'y': y_out,'t': tVec,'beta_L': beta_L_factor,'beta_H': beta_H_factor}
    return dicto

#--------------------------------------------------------------------

# y = run_model(0.01,0,0,0,0)
# print(y)




def test_probs(have_it,sens,spec):
    
    true_pos = have_it*sens
    false_neg = have_it*(1-sens)

    false_pos = (1-have_it)*(1-spec)
    true_neg = (1-have_it)*spec
    
    return true_pos, false_pos, true_neg, false_neg
