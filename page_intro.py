import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

from parameters_cov import age_risk_df_out

########################################################################################################################
layout_model = html.Div([

    html.H3('Model Explanation',className='intro-heading'),

    html.Hr(),

    dcc.Markdown(className="markdown",children=
    '''
    *Underlying all of the predictions is a mathematical model. In this Section we explain how the mathematical model works.*

    We present a compartmental model for COVID-19, split by risk categories. That is to say that everyone in the population is **categorised** based on **disease status** (susceptible/ infected/ recovered/ hospitalised/ critical care/ dead) and based on **COVID risk**.
    
    The model is very simplistic but still captures the basic spread mechanism. It is far simpler than the [**Imperial College model**](https://spiral.imperial.ac.uk/handle/10044/1/77482), but it uses similar parameter values and can capture much of the relevant information in terms of how effective control will be.

    It is intended solely as an illustrative, rather than predictive, tool. We plan to increase the sophistication of the model and to update parameters as more (and better) data become available to us.
    
    We have **two risk categories**: high and low. **Susceptible** people get **infected** after contact with an infected person (from either risk category). A fraction of infected people (*h*) are **hospitalised** and the rest **recover**. Of these hospitalised cases, a fraction (*c*) require **critical care** and the rest recover. Of those in critical care, a fraction (*d*) **die** and the rest recover.

    The recovery fractions depend on which risk category the individual is in.

    '''),



    dbc.Row([
    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/Capture_lomery.png',
    style={'maxWidth':'90%','height': 'auto', 'display': 'block','marginTop': '10px','marginBottom': '10px'}
    ),
    ],
    justify='center'
    ),

    dcc.Markdown(className="markdown",children='''

    The selection of risk categories is done in the simplest way possible - an age split at 60 years (based on the age structure data below). A more nuanced split would give a more effective control result, since there are older people who are at low risk and younger people who are at high risk. In many cases, these people will have a good idea of which risk category they belong to.

    *For the more mathematically inclined reader, a translation of the above into a mathematical system is described below.*

    '''),
    
    

    dbc.Row([
    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/eqs_f3esyu.png',
    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '10px','marginBottom': '10px'})
    ],
    justify='center'
    ),


    dbc.Row([
    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1585831217/text_toshav.png',
    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '10px','marginBottom': '10px'}
    ),
    ],
    justify='center'
    ),

    
    dcc.Markdown(className="markdown",children='''
    Of those requiring critical care, we assume that if they get treatment, a fraction *1-d* recover. If they do not receive it they die, taking 2 days. The number able to get treatment must be lower than the number of ICU beds available.
    '''),



    html.Hr(),

    html.H4('Parameter Values',className="intro-sub-heading"),

    dcc.Markdown(className="markdown",children='''
    The model uses a weighted average across the age classes below and above 60 to calculate the probability of a member of each class getting hospitalised or needing critical care. Our initial conditions are updated to match new data every day (meaning the model output is updated every day, although in '**Custom Options**' there is the choice to start from any given day).

    We assume a 10 day delay on hospitalisations, so we use the number infected 10 days ago to inform the number hospitalised (0.044 of infected) and in critical care (0.3 of hospitalised). We calculate the approximate number recovered based on the number dead, assuming that 0.009 infections cause death. All these estimates are as per the Imperial College paper ([**Ferguson et al**](https://spiral.imperial.ac.uk/handle/10044/1/77482)).

    The number of people infected, hospitalised and in critical care are calculated from the recorded data. We assume that only half the infections are reported ([**Fraser et al.**](https://science.sciencemag.org/content/early/2020/03/30/science.abb6936)), so we double the recorded number of current infections. The estimates for the initial conditions are then distributed amongst the risk groups. These proportions are calculated using conditional probability, according to risk (so that the initial number of infections is split proportionally by size of the risk categories, whereas the initially proportion of high risk deaths is much higher than low risk deaths).

    '''),



    

    dbc.Row([
    html.Img(src='https://res.cloudinary.com/hefjzc2gb/image/upload/v1586345773/table_fhy8sf.png',
    style={'maxWidth':'90%','height': 'auto','display': 'block','marginTop': '10px','marginBottom': '10px'}
    ),
    ],
    justify='center'
    ),



    html.Div('** the Imperial paper uses 8 days in hospital if critical care is not required (as do we). It uses 16 days (with 10 in ICU) if critical care is required. Instead, if critical care is required we use 8 days in hospital (non-ICU) and then either recovery or a further 8 in intensive care (leading to either recovery or death).',
    style={'fontSize':'70%'}),

    dcc.Markdown(className="markdown",children='''
    Please use the following links: [**Ferguson et al**](https://spiral.imperial.ac.uk/handle/10044/1/77482), [**Anderson et al**](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30567-5/fulltext) and [**Zhao et al**](https://journals.plos.org/plosntds/article/file?rev=2&id=10.1371/journal.pntd.0006158&type=printable)
    '''),


    html.H4('Age Structure',className="intro-sub-heading"),
    
    dcc.Markdown(className="markdown",children='''
    The age data is taken from [**GOV.UK**](https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest) and the hospitalisation and critical care data is from the [**Imperial College Paper**](https://spiral.imperial.ac.uk/handle/10044/1/77482) (Ferguson et al.). This means that the age structure will not be accurate when modelling other countries.

    To find the probability of a low risk case getting hospitalised (or subsequently put in critical care), we take a weighted average by proportion of population. Note that the figures below are proportion of *symptomatic* cases that are hospitalised, which we estimate to be 55% of cases ([**Ferguson et al.**](https://spiral.imperial.ac.uk/handle/10044/1/77482)). The number requiring critical care is a proportion of this hospitalised number.

    *The table below shows the age structure data that was used to calculate these weighted averages across the low risk category (under 60) and high risk (over 60) category.*
    
    '''),

    dbc.Table.from_dataframe(age_risk_df_out, striped=True, bordered = True, hover=True)

])

















########################################################################################################################
layout_intro = html.Div([
    
    dbc.Tabs(id='intro-tabs',
             active_tab='tab_0',
             children = [
                
        
        
        
        dbc.Tab(labelClassName='tab', label='Introduction to modelling', tab_id='tab_0', children=[

            
            html.H3('Introduction to mathematical modelling',className='intro-heading'),

            html.Hr(),

            dcc.Markdown(className="markdown",children='''
            Watch this video from Dr Cerian Webb, an expert in epidemiology and modelling from the University of Cambridge.
            '''),

            html.Div(html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1586536825/WhatIsModellingv2_172141_jwpplb.mp4', #vc_h264
                    controls=True,className="video"),
            className="video-container"),
                    
                    

            
            html.Hr(),


            html.H3('Introducing SIR models',className='intro-heading'),

            html.Hr(),

            dcc.Markdown(className="markdown",children='''
            Watch this explanation from Dr Cerian Webb, to find out more about basic epidemiological models.
            '''),

                    
            html.Div(html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1585814499/StandardSIRModel_hu5ztn.mp4', # 
                    controls=True,className="video")
            ,className="video-container"),
            
                    
            
            html.Hr(),


            html.H3('Introducing the basic reproductive number',className='intro-heading'),

            html.Hr(),

            dcc.Markdown(className="markdown",children='''
            Watch Dr Cerian Webb introduce the basic reproductive number.
            '''),

                    
            html.Div(html.Video(src='https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1586536823/AllAboutR_173637_poxzmb.mp4',
                    controls=True,className="video")
            ,className="video-container"),
            


            html.Hr(),

            html.H3('Introducing Herd Immunity',className='intro-heading'),

            html.Hr(),


            dcc.Markdown(className="markdown",children='''
            Watch Dr Cerian Webb introduce the concept of herd immunity.
            '''),

            html.Div(html.Video(src="https://res.cloudinary.com/hefjzc2gb/video/upload/c_scale,q_auto,w_800/v1588167893/HerdImmunity_144205_dyhaiy.mp4",
                    controls=True,className="video")
            ,className="video-container"),
            
        ]),
            

        dbc.Tab(labelClassName='tab', label='Definitions', tab_id='tab_defs',
            children=[
                
            html.H3('Definitions',className='intro-heading'),

            html.Hr(),

            dcc.Markdown(className="markdown",children='''            

            There are two key concepts that you need to understand before we can fully explore how the control measures work.
            '''),
            
            html.H4('1. Basic Reproduction Number',className="intro-sub-heading"),

            dcc.Markdown(className="markdown",children='''
            Any infectious disease requires both infectious individuals and susceptible individuals to be present in a population to spread. The higher the number of susceptible individuals, the faster it can spread since an infectious person can spread the disease to more susceptible people before recovering.

            The average number of infections caused by a single infected person is known as the '**effective reproduction number**' (*R*). If this number is less than 1 (each infected person infects less than one other on average) then the disease will not continue to spread. If it is greater than 1 then the disease will spread. For COVID-19 most estimates for *R* are between 2 and 3. We use the value *R*=2.4.
            '''),

            html.H4('2. Herd Immunity',className="intro-sub-heading"),
            
            dcc.Markdown(className="markdown",children='''            


            Once the number of susceptible people drops below a certain threshold (which is different for every disease, and in simpler models depends on the basic reproduction number), the population is no longer at risk of an epidemic (so any new infection introduced will not cause infection to spread through an entire population).

            Once the number of susceptible people has dropped below this threshold, the population is termed to have '**herd immunity**'. Herd immunity is either obtained through sufficiently many individuals catching the disease and developing personal immunity to it, or by vaccination.

            For COVID-19, there is a safe herd immunity threshold of around 60% (=1-1/*R*), meaning that if 60% of the population develop immunity then the population is **safe** (no longer at risk of an epidemic).

            Coronavirus is particularly dangerous because most countries have almost 0% immunity since the virus is so novel. Experts are still uncertain whether you can build immunity to the virus, but the drop in cases in China would suggest that you can. Without immunity it would be expected that people in populated areas get reinfected, which doesn't seem to have happened.
            
            A further concern arises over whether the virus is likely to mutate. However it is still useful to consider the best way to managing each strain.
            '''),

        ]),


        dbc.Tab(labelClassName='tab', label='COVID-19 Control Strategies', tab_id='tab_control', children=[

            html.H3('Keys to a successful control strategy', className='intro-heading'),

            html.Hr(),


            dcc.Markdown(className="markdown",children='''            
            There are three main goals a control strategy sets out to achieve:

            1. Reduce the number of deaths caused by the pandemic,

            2. Reduce the load on the healthcare system,

            3. Ensure the safety of the population in future.

            An ideal strategy achieves all of the above whilst also minimally disrupting the daily lives of the population.

            However, controlling COVID-19 is a difficult task, so there is no perfect strategy. We will explore the advantages and disadvantages of each strategy.
            '''),
            
            html.Hr(),

            html.H3('Strategies',className='intro-heading'),

            html.Hr(),

            
            html.H4('Reducing the infection rate',className="intro-sub-heading"),


            dcc.Markdown(className="markdown",children='''            

            Social distancing, self isolation and quarantine strategies slow the rate of spread of the infection (termed the 'infection rate'). In doing so, we can reduce the load on the healthcare system (goal 2) and (in the short term) reduce the number of deaths.

            This has been widely referred to as 'flattening the curve'; buying nations enough time to bulk out their healthcare capacity. The stricter quarantines are the best way to minimise the death rate whilst they're in place. A vaccine can then be used to generate sufficient immunity.

            However, in the absence of a vaccine these strategies do not ensure the safety of the population in future (goal 3), meaning that the population is still highly susceptible and greatly at risk of a future epidemic. This is because these strategies do not lead to any significant level of immunity within the population, so as soon as the measures are lifted the epidemic restarts. Further, strict quarantines carry a serious economic penalty.

            COVID-19 spreads so rapidly that it is capable of quickly generating enough seriously ill patients to overwhelm the intensive care unit (ICU) capacity of most healthcase systems in the world. This is why most countries have opted for strategies that slow the infection rate. It is essential that the ICU capacity is vastly increased to ensure it can cope with the number of people that may require it.
            '''),


            html.H4('Protecting the high risk',className="intro-sub-heading"),

            


            dcc.Markdown(className="markdown",children='''            
            One notable feature of COVID-19 is that it puts particular demographics within society at greater risk. The elderly and the immunosuppressed are particularly at risk of serious illness caused by coronavirus.

            The **interactive model** presented here is designed to show the value is protecting the high risk members of society. It is critically important that the high risk do not catch the disease.

            If 60% of the population catch the disease, but all are classified as low risk, then very few people will get seriously ill through the course of the epidemic. However, if a mixture of high and low risk individuals catch the disease, then many of the high risk individuals will develop serious illness as a result.


            '''),

        ]),


        dbc.Tab(labelClassName='tab', label='How to use', tab_id='tab_1', children=[
                    



                    html.H3('How to use the interactive model',className='intro-heading'),

                    html.Hr(),
                    
                    dbc.Col([
                    dcc.Markdown(className="markdown",children='''


                    We present a model parameterised for COVID-19. The interactive element allows you to predict the effect of different **control measures**.

                    We use **control** to mean an action taken to try to reduce the severity of the epidemic. In this case, control measures (e.g. social distancing and quarantine/lockdown) will affect the '**infection rate**' (the rate at which the disease spreads through the population).

                    Stricter measures (e.g. lockdown) have a more dramatic effect on the infection rate than less stringent measures.
                    
                    To start predicting the outcome of different strategies, press the button below!

                    '''),
                    

                    dbc.Row([
                    dbc.Button('Start Calculating', href='/interactive-model', size='lg', color='primary',
                    style={'marginTop': '10px', 'textAlign': 'center', 'fontSize': '100%'}
                    ),
                    ],
                    justify='center'),
                    # ],width={'size':3,'offset':1},
                    # ),
                    ],
                    style={'marginTop': '10px'},
                    width = True),


                    #end of tab 1
                    ]),
                    
        dbc.Tab(labelClassName='tab',
                    label='Model Explanation', 
                    tab_id='tab_explan',
                    children=layout_model),

            
    #end of tabs
    ])

],className='intro-container')