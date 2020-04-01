from parameters_cov import params
from math import exp, ceil, log, floor, sqrt
import numpy as np
from scipy.integrate import ode
from scipy.stats import norm, gamma
import pandas as pd
##
# -----------------------------------------------------------------------------------
##
class simulator:
    def __init__(self):
        pass
    ##
#-----------------------------------------------------------------
        
    ##
    def poly_solv_ode(self,t,y,beta_L_factor=1,beta_H_factor=1,t_control=None): # critical=True,death=True
        ##
        S_L = y[params.S_L_ind]
        I_L = y[params.I_L_ind]
        H_L = y[params.H_L_ind]
        C_L = y[params.C_L_ind]


        S_H = y[params.S_H_ind]
        I_H = y[params.I_H_ind]
        H_H = y[params.H_H_ind]
        C_H = y[params.C_H_ind]


        # if not critical:
        #     params.crit_L = 0
        #     params.crit_H = 0
        #     params.recover_L = 0
        #     params.recover_H = 0
        
        death_on = 1
        # if not death:
        #     death_on = 0
        
        if t_control is None:
            beta_L_factor = 1
            beta_H_factor = 1

        if t_control is not None and len(t_control)>1:
            if t<t_control[0] or t>t_control[1]:
                beta_L_factor = 1
                beta_H_factor = 1

        if C_L>params.ICU_capacity:
            C_L_to_R_L = params.ICU_capacity*params.crit_recovery
            C_L_to_D_L = params.ICU_capacity*params.crit_death + (params.crit_death + params.crit_recovery)*(C_L-params.ICU_capacity) # all without crit care die
        else:
            C_L_to_R_L = C_L*params.crit_recovery
            C_L_to_D_L = C_L*params.crit_death

        if C_H>params.ICU_capacity:
            C_H_to_R_H = params.ICU_capacity*params.crit_recovery
            C_H_to_D_H = params.ICU_capacity*params.crit_death + (params.crit_death + params.crit_recovery)*(C_H-params.ICU_capacity) # all without crit care die
        else:
            C_H_to_R_H = C_H*params.crit_recovery
            C_H_to_D_H = C_H*params.crit_death
        
        dydt = [-S_L*( params.beta*(beta_L_factor**2)*I_L  +  (params.beta*((beta_H_factor*beta_L_factor)**1))*I_H ), # dS
                +S_L*( params.beta*(beta_L_factor**2)*I_L  +  (params.beta*((beta_H_factor*beta_L_factor)**1))*I_H ) - params.mu_L*I_L - params.gamma_L*I_L, # dI
                I_L*params.mu_L    + H_L*params.recover_L + death_on*C_L_to_R_L,            # dR
                I_L*params.gamma_L - H_L*params.recover_L - H_L*params.crit_L,         # dH
                H_L*params.crit_L  - death_on*(C_L_to_R_L + C_L_to_D_L),                          # dC
                + death_on*C_L_to_D_L,                                                      # dD
                -S_H*( (params.beta*((beta_H_factor*beta_L_factor)**1))*I_L + params.beta*(beta_H_factor**2)*I_H), # dS
                +S_H*( (params.beta*((beta_H_factor*beta_L_factor)**1))*I_L + params.beta*(beta_H_factor**2)*I_H) - params.mu_H*I_H - params.gamma_H*I_H, # dI
                I_H*params.mu_H    + H_H*params.recover_H  + death_on*C_H_to_R_H,           # dR
                I_H*params.gamma_H - H_H*params.recover_H - H_H*params.crit_H,         # dH
                H_H*params.crit_H  - death_on*(C_H_to_R_H + C_H_to_D_H),                          # dC
                + death_on*C_H_to_D_H                                                       # dD
                ]
        
        return dydt
    ##
    #--------------------------------------------------------------------
    ##
    def poly_calc_ode(self,I0_L,I0_H,beta_L_factor,beta_H_factor,t_control,T_stop): # critical,death

        I_L_0 = I0_L
        S_L_0 = (1-params.hr_frac)*params.N - I_L_0
        
        I_H_0 = I0_H
        S_H_0 = params.hr_frac*params.N - I_H_0
        
        
        y0 = [S_L_0,
            I_L_0,
            0,
            0,
            0,
            0,
            S_H_0,
            I_H_0,
            0,
            0,
            0,
            0
            ]

        sol = ode(self.poly_solv_ode,jac=None).set_integrator('dopri5').set_f_params(beta_L_factor,beta_H_factor,t_control) # ,critical,death
        
        tim = np.linspace(0,T_stop, 301) # use 141 time values

        # tim = np.linspace(0,10, 141)
        
        sol.set_initial_value(y0,tim[0])

        y_out = np.zeros((len(y0),len(tim)))
        
        i2 = 0
        y_out[:,0] = sol.y
        for t in tim[1:]:
                if sol.successful():
                    sol.integrate(t)
                    i2=i2+1
                    y_out[:,i2] = sol.y
                else:
                    raise RuntimeError('ode solver unsuccessful')
        
        return y_out, tim

    ##
    #--------------------------------------------------------------------
    ##
    def run_model(self,I0_L=(1-params.hr_frac)*(params.initial_infections/params.UK_population)*params.N,I0_H=params.hr_frac*(params.initial_infections/params.UK_population)*params.N,beta_L_factor=1,beta_H_factor=1,t_control=None,T_stop=params.T_stop): # critical=True,death=True,
        # print(I0_H*60*10**6,I0_L*60*10**6,params.hr_frac)
        y_out, tim = self.poly_calc_ode(I0_L=I0_L,I0_H=I0_H,beta_L_factor=beta_L_factor,beta_H_factor=beta_H_factor,t_control=t_control,T_stop=T_stop) # critical=critical,death=death,
        dicto = {'y': y_out,'t': tim,'beta_L': beta_L_factor,'beta_H': beta_H_factor}
        return dicto
#--------------------------------------------------------------------


