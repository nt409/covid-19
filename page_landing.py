import dash_bootstrap_components as dbc
import dash_html_components as html


layout_enter = html.Div([
    
    html.Div([
        html.Div(html.Img(src='/assets/images/mainlogo.svg',className="main-logo"),className="main-logo-container"),
        html.H1('LowHighCovid',className="home-title"),
        html.H2('Covid modelling and real-time data HQ',className="home-page-desc"),
    ],className="title-container"),

    html.Div([
    html.Div([
        html.Div([
            html.Div(html.Img(src='/assets/images/main_backgd.svg',className="main-img"),className="landing-logo"),
            html.H2('Understand the pandemic',className="home-sub-title"),
            html.P('An introduction to mathematical modelling presented by experts in epidemiology from the University of Cambridge.',className="home-sub-text"),
            dbc.Button('Learn more',color='primary', href='/intro',className='page-button'),
        ],className="main-container",id='main-intro-cont'),


        html.Div([
            html.Div(html.Img(src='/assets/images/main_inter.svg',className="main-img"),className="landing-logo"),
            html.H2('Which restrictions, and when?',className="home-sub-title"),
            html.P('See how different control measures implemented today could impact on infections, hospitalisations and deaths.',className="home-sub-text"),
            dbc.Button('Start predicting',color='primary', href='/inter',className='page-button'),
                                
        ],className="main-container light",id='main-inter-cont'),

        html.Div([
            html.Div(html.Img(src='/assets/images/main_test.svg',className="main-img test"),className="landing-logo"),
            html.H2('How good can a test be?',className="home-sub-title"),
            html.P('We use conditional probability to show you how likely different tests really are to give correct results.',className="home-sub-text"),
            dbc.Button('Learn more',color='primary', href='/tests', className='page-button'),
        ],className="main-container",id='main-tests-cont'),


        html.Div([
            html.Div(html.Img(src='/assets/images/main_data.svg',className="main-img"),className="landing-logo data"),
            html.H2('Explore real-time data',className="home-sub-title"),
            html.P('Real-time data on coronavirus cases and deaths from hundreds of countries around the world.',className="home-sub-text"),
            dbc.Button('View global data', color='primary', href='/data', className='page-button'),
        ],className="main-container light",id='main-data-cont'),
    ],id="main-items-container"),
    ],id="gray-cont"),


],className='main-page')