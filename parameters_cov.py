from math import log, exp
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from math import ceil
#------------------------------------------------------------

# https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest






df2 = pd.DataFrame({'Age': ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80+'],
   'Hosp': [0.1,0.3,1.2,3.2,4.9,10.2,16.6,24.3,27.3],
   'Crit': [5,5,5,5,6.3,12.2,27.4,43.2,70.9],
   'Pop':  [11.8,9.5,16.2,13.3,14.6,12.1,10.8,7.1,4.6]
   })

divider = -2

# print()
# exit()
fact_v = np.concatenate([[0.01,0.05,0.1],np.linspace(0.25,2,8)]) # np.linspace(0.05,1.2,21) # [0.1,1,4] # np.linspace(0.05,8,11) #[0.05,1,10] #np.linspace(0.05,8,3) # [0.1,1,2,4] #np.linspace(0.5,4,5)
# fact_v = [0.05,1,1.5]
months_run_for = 12

# divider = int(divider)
# hr_frac = 0.15
hr_frac = sum(df2.Pop[divider:])/100

T_stop = 30*months_run_for

df2 = df2.assign(pop_low_prop=lambda x: x.Pop/(100*(1-hr_frac)),
    pop_high_prop=lambda x: x.Pop/(100*(hr_frac)))

# print(df2['pop_low_prop'])
# # print(df2.pop_low_prop[-4:])
# x1 = df2.ix[divider:,'pop_low_prop'] # 'pop_low_prop'])
# print(x1)
    


# df2.loc[:,'pop_low_prop'][divider:] = 0
# df2.loc[:,'pop_high_prop'][:divider] = 0
# print(df2.loc[(df2.shape[0]-1+divider):,'pop_low_prop'])# = 0
# print(df2.loc[:(df2.shape[0]-1+divider),'pop_high_prop'])# = 0

df2.loc[  (df2.shape[0]-1+divider): ,'pop_low_prop' ] = 0
df2.loc[ :(df2.shape[0]-1+divider)  ,'pop_high_prop'] = 0

# print(df2.loc[(df2.shape[0]-1+divider):,'pop_low_prop'])# = 0
# print(df2.loc[:(df2.shape[0]-1+divider),'pop_high_prop'])# = 0

df2 = df2.assign(   weighted_hosp_low=lambda x: (x.Hosp/100)*x.pop_low_prop,
                    weighted_hosp_high=lambda x:(x.Hosp/100)*x.pop_high_prop,
                    weighted_crit_low=lambda x: (x.Crit/100)*x.pop_low_prop,
                    weighted_crit_high=lambda x:(x.Crit/100)*x.pop_high_prop)


# print(df2)
frac_hosp_L = sum(df2.weighted_hosp_low)
frac_hosp_H = sum(df2.weighted_hosp_high)
frac_crit_L = sum(df2.weighted_crit_low)
frac_crit_H = sum(df2.weighted_crit_high)

hosp_rate = 1/8
death_rate = 1/8


crit_L      = hosp_rate*frac_crit_L
recover_L   = hosp_rate*(1-frac_crit_L)
crit_H      = hosp_rate*frac_crit_H
recover_H   = hosp_rate*(1-frac_crit_H) 

crit_death     =  death_rate*0.5
crit_recovery  =  death_rate*0.5

recov_rate = 1/7
R_0        = 2.4
# frac_hosp_H = 0.14 # for high risk group
# frac_hosp_L = 0.005 # 0.01 # = low risk hosp/high risk hosp
number_compartments = 6

N    = 1 # 66*(10**6)
beta = R_0*recov_rate/N # R_0 mu/N

mu_L    = recov_rate*(1-frac_hosp_L)
gamma_L = recov_rate*frac_hosp_L
mu_H    = recov_rate*(1-frac_hosp_H)
gamma_H = recov_rate*frac_hosp_H

hospital_production_rate = 8*10**(-3)
ICU_factor = 1
ICU_capacity = ICU_factor*8/100000 # 0.001 #  8/100000 # = 0.00008

UK_population = 60 * 10**(6)



class Parameters:
    def __init__(self): # ,mu_L=mu_L,mu_H=mu_H,gamma_L=gamma_L,gamma_H=gamma_H,beta=beta,N=N,hr_frac=hr_frac,crit_L=crit_L,crit_H=crit_H,recover_L=recover_L,recover_H=recover_H,death=death):
        self.mu_L  = mu_L
        self.mu_H  = mu_H
        self.gamma_L = gamma_L
        self.gamma_H = gamma_H
        self.beta  = beta
        self.N  = N
        self.hr_frac  = hr_frac
        self.crit_L = crit_L
        self.crit_H = crit_H
        self.recover_L = recover_L
        self.recover_H = recover_H
        self.crit_death = crit_death
        self.crit_recovery = crit_recovery
        self.hospital_production_rate = hospital_production_rate
        self.T_stop = T_stop
        self.ICU_capacity = ICU_capacity
        self.fact_v = fact_v
        self.months_run_for = months_run_for
        self.R_0 = R_0
        self.UK_population = UK_population


        self.number_compartments = number_compartments


        self.S_L_ind = 0
        self.I_L_ind = 1
        self.R_L_ind = 2
        self.H_L_ind = 3
        self.C_L_ind = 4
        self.D_L_ind = 5
        self.S_H_ind = number_compartments + 0
        self.I_H_ind = number_compartments + 1
        self.R_H_ind = number_compartments + 2
        self.H_H_ind = number_compartments + 3
        self.C_H_ind = number_compartments + 4
        self.D_H_ind = number_compartments + 5




params = Parameters()