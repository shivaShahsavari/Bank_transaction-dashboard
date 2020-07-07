#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

import numpy as np
import pandas as pd

from datetime import datetime
import pdb

import networkx as nx
from pyvis import network as net
from itertools import product

from datetime import datetime,date
import pdb
import json


### Data import and preprocessing

df=pd.read_csv('data/threeMonths.csv')

df['datee'] = pd.to_datetime(df['datee']).dt.date

print('Finding unique accounts')
unique_accounts=df['a_key'].append(df['b_key'],ignore_index=True).unique().tolist()
print('Done')

print('Sorting unique accounts')
unique_accounts.sort()
print('Done')

print('Finding min and max dates and amounts')
min_date=df['datee'].min()
max_date=df['datee'].max()
min_abs_amount=df['amount'].min()
max_abs_amount=df['amount'].max()
print('Done')


### Test-cases
test_cases=[
    [], # case 0 (Default test in which app is initialized)
    ['A36860','A26417',date(2019,6,18),date(2019,6,24),min_abs_amount,max_abs_amount], # case 1
    ['A1000','A44134',date(2019,6,25),date(2019,7,1),min_abs_amount,max_abs_amount], # case 2
     ]


### Utility functions

def filter_df(key,start_date=min_date,end_date=max_date,start_amount=min_abs_amount,end_amount=max_abs_amount):
    
    # key-based filtering
    condition=((df['originn']==key) | (df['dest']==key))
    
    # date-based filtering
    if (start_date is not None and end_date is not None):
        date_condition=(df['datee']>=start_date) & (df['datee']<=end_date)
        condition=condition & date_condition
    
    # amount-based filtering
    if (start_amount is not None and end_amount is not None):    
        amount_condition=(df['amount']>=start_amount) & (df['amount']<=end_amount)
        condition=condition & amount_condition
    
    return df[condition]


### Network similarity functions

def _is_close(d1, d2, atolerance=0, rtolerance=0):
    # Pre-condition: d1 and d2 have the same keys at each level if they are dictionaries
    if not isinstance(d1, dict) and not isinstance(d2, dict):
        return abs(d1 - d2) <= atolerance + rtolerance * abs(d2)
    return all(all(_is_close(d1[u][v], d2[u][v]) for v in d1[u]) for u in d1)

def unique(list1): 
    
    # intilize a null list 
    unique_list = [] 
        
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return  unique_list

def precssr(list1):
    res=[x[0] for x in unique(list1)]
    return res
        
def simrank_similarity_Incoming(G, source=None, target=None, importance_factor=0.9, max_iterations=100, tolerance=1e-4):
    prevsim = None
        # build up our similarity adjacency dictionary output
    newsim = {u: {v: 1 if u == v else 0 for v in G} for u in G}
        # These functions compute the update to the similarity value of the nodes
        # `u` and `v` with respect to the previous similarity values.
    avg_sim = lambda s: sum(newsim[w][x] for (w, x) in s) / len(s) if s else 0.0
    sim = lambda u, v: importance_factor * avg_sim(list(product(
    precssr(list(G.in_edges(u, data=False))),
    precssr(list(G.in_edges(v, data=False)))
    )))
    for _ in range(max_iterations):
        if prevsim and _is_close(prevsim, newsim, tolerance):
            break
        prevsim = newsim
        newsim = {u: {v: sim(u, v) if u is not v else 1
                    for v in newsim[u]} for u in newsim}
    if source is not None and target is not None:
        return newsim[source][target]
    if source is not None:
        return newsim[source]
    return newsim

def sim_outgoing_two_nodes(sim_dict,a,b):
    return sim_dict[a][b]

def sim_incoming_two_nodes(sim_dict,a,b):
    return sim_dict[a][b]

### App definition
    
app = dash.Dash(__name__)
app.title = 'Van Lanschot Dashboard'

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout=html.Div([
        html.Div(
            children=[
            html.Div(html.Img(src=app.get_asset_url('download.png'),style={ 'height': '100%', 'width':'100%'  }), style={'display': 'inline-block', 'height': '60px', 'width': '201px'}),
            html.Div(html.Img(src=app.get_asset_url('jads-logo.png'),style={ 'height': '100%', 'width':'100%','margin-right':'0px' }), style={'display': 'inline-block', 'height': '60px','textAlign':'right', 'width':'162px'}) #why are there two styles?
            ]
        ),
        html.Div(
            className="row",
            style={'height': '46px',
                   'margin-top': '10px',
                   'margin-bottom': '10px',
                   'margin-left': '4px'},
            children=[
                html.Div(
                    #className="col-1",
                    style={'width': '10vw'},
                    children=[
                        dcc.Input(
                            id='input_1',
                            type='text',
                            value=unique_accounts[7],
                            style={'height': '47px'}
                         )
                    ]                        
                ),
                html.Div(
                    #className="col-1",
                    style={'width': '10vw'},
                    children=[
                        dcc.Input(
                         id='input_2',
                         type='text',
                         style={'height': '47px'}
                         )
                    ]                        
                ),
                html.Div(
                    #className="col-4",
                    style={'width': '28vw'},
                    children=[
                        dcc.DatePickerRange(
                            id='datepicker',
                            display_format='DD MM YYYY',
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            calendar_orientation='vertical'
                        )  
                    ]
                ),
                html.Div(
                    #className="col-3",
                    style={'padding-top': '12px',
                        'width': '26vw'},
                    children=[
                        dcc.RangeSlider(
                            id='amount-slider',
                            step=1, # todo: set this in callback according to slider value
                        ),
                        html.Div(
                            className='row',
                            style={'margin-top': '-24px'},
                            children=[
                                html.Div(
                                    id='amount-slider-min-text',
                                    style={'width': '50%',
                                    'textAlign': 'left'}
                                ),
                                html.Div(
                                    id='amount-slider-max-text',
                                    style={'width':'50%','textAlign':'right'}
                                )
                            ]
                        )
                    ]
                ),
                #html.Div( # REJECT ME WHEN MERGING PLEASE ONLY IF OTHER CODE IS INPUT TEXT INSTEAD OF TEXTAREA
                #    #className="col-1",
                #    id='textArea',
                #    children=[  
                #        dcc.Textarea( # todo: change this into DIV
                #            placeholder='SimRank',
                #            #value='Similarity',
                #            style={'width': '100%',
                #            'height': '46px',
                #            'text-align': 'center'}
                #        )
                #    ],
                #    style={'height': '46px',
                #        'width': '12vw'}
                #),
                html.Div(
                    #className="col-1",
                    style={'width': '10vw',
                        'text-align': 'center'},
                    children=[
                        html.Button(
                            id='submit-button',
                            children='Submit',
                            n_clicks=0,
                            style={'height': '46px'}
                        )
                    ]
                ),
                html.Div(
                    className="col-1",
                    children=[
                        dcc.Dropdown(
                            id='test_case_dropdown',
                            options=[
                                {'label':i,'value':i} for i in range(len(test_cases))
                            ],
                            value=0
                        )
                    ]
                ),
                html.Div(
                    style={
                        'display':'none'
                    },
                    id='intermediate_div',
                    children=[]
                ),
                html.Div(
                    style={
                        'display':'none'
                    },
                    id='initialization_div',
                    children='A10' # initialize key_1 from here
                )
            ]
        ),
        html.Div(
            className="row",
            style={'height': '80vh'},
            children=[
                html.Div(
                    #className='col-6',
                    style={'width': '50vw'},
                    children=[
                        html.Div(
                            children=[
                                html.Iframe(
                                    id='mapx', 
                                    width='100%', 
                                    height='100%'
                                )
                            ],
                            style={'width': '100%','height':'100%'}
                        )
                    ],
                    # style={'width': '100%','height':'100%'}
                ),
                html.Div(
                    #className='col-6',
                    style={'width': '50vw',
                        'display': 'inline-block'},
                    children=[
                        dcc.Tabs([
                            dcc.Tab(
                                label='Time (1)',
                                #style={'height': '5vh'},
                                children=[
                                    dcc.Graph(
                                        id='timegraph',
                                        style={'width': '100%',
                                            'display': 'inline-block'} #'display': 'inline-block'
                                    )
                                ]
                            ),
                            dcc.Tab(
                                label='Time (2)',
                                #style={'height': '5vh'},
                                children=[
                                    dcc.Graph(
                                        id='cumgraph',
                                        style={'width': '100%'}
                                    )
                                ]
                            ),
                            dcc.Tab(
                                label='Frequency', 
                                #style={'height': '5vh'},
                                children=[
                                    dcc.Graph(
                                        id='freqgraph',
                                        style={'width': '100%',
                                            'display': 'inline-block'}
                                    )
                                ]
                            ),
                             dcc.Tab(
                                label='Similarity', 
                                #style={'height': '5vh'},
                                children=[
                                    html.Div(   
                                        id='sim_table_div',
                                        style={#'height': '70vh',
                                            'width': '50vw'
                                            #'display': 'inline-block',
                                            #'margin-left': '1px',
                                            },
                                        children=[
                
                                        ]
                                    )
                                
                                ]
                            ),
                            dcc.Tab(
                                label='Stats',
                                #style={'height': '5vh'},
                                children=[
                                    html.Div(
                                        id='stats_table_1_div',
                                        style={'width': '50vw'}
                                        # style={'width': '100%','height':'100%'}
                                    ),
                                    html.Div(
                                        id='stats_table_2_div',
                                        style={'width': '50vw'}
                                        # style={'width': '100%','height':'100%'}
                                    ),
                                    html.Div(
                                        id='stats_table_1_2_div',
                                        style={'width': '50vw'}
                                        # style={'width': '100%','height':'100%'}
                                    )
                                ]
                            )
                        ])
                    ]
                )
            ]
        )
])

    
### Callbacks

# Update key_1
@app.callback(
    Output('input_1','value'),
    [Input('initialization_div','children'),
     Input('test_case_dropdown','value')])
def update_key_1(key_1,test):
    
    if test!=0: return test_cases[test][0]
    
    if key_1 not in unique_accounts:
        idx=np.searchsorted(unique_accounts,key_1)
        if (idx==0) or (idx==len(unique_accounts)):
            return dash.no_update
        else:
            return unique_accounts[idx-1]
    else:
        return key_1        
        
# Update key_2 based on key_1
@app.callback(
    Output('input_2','value'),
    [Input('input_1','value'),
     Input('test_case_dropdown','value')])
def update_key_2(key_1,test):
    
    if test!=0: return test_cases[test][1]
    
    df_key_1=filter_df(key_1)
    
    if (df_key_1.empty) or (key_1 is None):
        print("Preventing update of key 2")
        raise PreventUpdate
    
    df_key_1['other_key']=np.where(df_key_1['a_key']==key_1,df_key_1['b_key'],df_key_1['a_key'])
    key_2=df_key_1['other_key'].groupby(df_key_1['other_key']).count().idxmax()
        
    return key_2


# Update minimum and maximum dates and amounts based on key_1 and key_2
@app.callback(
    [Output('datepicker','min_date_allowed'),
     Output('datepicker','max_date_allowed'),
     Output('datepicker','initial_visible_month'),
     Output('datepicker','start_date'),
     Output('datepicker','end_date'),
     Output('amount-slider','min'),
     Output('amount-slider','max'),
     Output('amount-slider','value')],
    [Input('input_1','value'),
     Input('input_2','value'),
     Input('test_case_dropdown','value')])
def update_dates_and_amounts(key_1,key_2,test):
    
    if test!=0: return min_date,max_date,test_cases[test][2],test_cases[test][2],test_cases[test][3],min_abs_amount,max_abs_amount,[test_cases[test][4],test_cases[test][5]]
    
    # get global df filtered according to key_1 and key_2
    
    df_key_1=filter_df(key_1)
    df_key_2=filter_df(key_2)
    
    if (key_1 is None) or (key_2 is None) or (df_key_1.empty) or (df_key_2.empty):
        print("Preventing update of dates and amounts")
        raise PreventUpdate
        
    # calculate min and max dates
    
    min_date_1=df_key_1['datee'].min()
    max_date_1=df_key_1['datee'].max()
    
    min_date_2=df_key_2['datee'].min()
    max_date_2=df_key_2['datee'].max()
    
    min_date_12=min_date_1 if (min_date_1<min_date_2) else min_date_2
    max_date_12=max_date_1 if (max_date_1>max_date_2) else max_date_2
    
    # calculate min and max absolute amounts
    
    min_amount_1=df_key_1['amount'].min()
    max_amount_1=df_key_1['amount'].max()
    
    min_amount_2=df_key_2['amount'].min()
    max_amount_2=df_key_2['amount'].max()
    
    min_amount_12=min_amount_1 if min_amount_1<=min_amount_2 else min_amount_2
    max_amount_12=max_amount_1 if max_amount_1>=max_amount_2 else max_amount_2
    
    return min_date_12,max_date_12,min_date_12,min_date_12,max_date_12,min_amount_12,max_amount_12,[min_amount_12,max_amount_12]


# Update text representing the min and max value of amount slider
@app.callback(
    [Output('amount-slider-min-text','children'),
     Output('amount-slider-max-text','children')],
    [Input('amount-slider','value')])
def update_amount_text(value):
    
    if (value is None):
        print("Preventing update of amount text")
        raise PreventUpdate

    return 'Min: {}'.format(value[0]),'Max: {}'.format(value[1])


# Update intermediate div
@app.callback(
    Output('intermediate_div','children'),
    [Input('input_1','value'),
     Input('input_2','value'),
     Input('datepicker','start_date'),
     Input('datepicker','end_date'),
     Input('amount-slider','value')])
def update_intermediate_div(key_1,key_2,start_date,end_date,amount_range):
  
    if ((key_1 is None) or (key_2 is None) or (start_date is None) or (end_date is None) or (amount_range is None)):
        print('Preventing update of intermediate div')
        raise PreventUpdate
        
    # Calculate necessary variables
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()
    
    df_key_1=filter_df(key_1,sd,ed,amount_range[0],amount_range[1]).sort_values('datee')   
    df_key_2=filter_df(key_2,sd,ed,amount_range[0],amount_range[1]).sort_values('datee')
    df_key_1_2=df_key_1[(df_key_1['originn']==key_2) | (df_key_1['dest']==key_2)].sort_values('datee')
    
    df_key_1['other_key']=np.where(df_key_1['a_key']==key_1,df_key_1['b_key'],df_key_1['a_key'])
    df_key_2['other_key']=np.where(df_key_2['a_key']==key_2,df_key_2['b_key'],df_key_2['a_key'])
    df_key_1_2['other_key']=np.where(df_key_1_2['a_key']==key_1,df_key_1_2['b_key'],df_key_1_2['a_key'])
  
    df_key_1['amount_signed']=np.where(df_key_1['other_key']==df_key_1['originn'],df_key_1['amount'],df_key_1['amount'].apply(lambda x:-x))
    df_key_2['amount_signed']=np.where(df_key_2['other_key']==df_key_2['originn'],df_key_2['amount'],df_key_2['amount'].apply(lambda x:-x))
    df_key_1_2['amount_signed']=np.where(df_key_1_2['other_key']==df_key_1_2['originn'],df_key_1_2['amount'],df_key_1_2['amount'].apply(lambda x:-x))

    
    datasets = {
         'df_key_1': df_key_1.to_json(orient='split', date_format='iso'),
         'df_key_2': df_key_2.to_json(orient='split', date_format='iso'),
         'df_key_1_2': df_key_1_2.to_json(orient='split', date_format='iso'),
         'key_1':key_1,
         'key_2':key_2,
     }
    
    return json.dumps(datasets)


 # Update Stats Tab
@app.callback(
    [Output('stats_table_1_div','children'),
     Output('stats_table_2_div','children'),
     Output('stats_table_1_2_div','children')],
    [Input('submit-button','n_clicks')],
    [State('intermediate_div','children')])
def update_stat_tables(n_clicks,jsonified_data):
    
    # error checks
    if ((n_clicks is None) or (jsonified_data == [])):
        print('Returning dash.no_update')
        raise PreventUpdate
    
    # if jsonified_data==[]: return dash.no_update
    
    # get variables from jsonidied_date
    datasets=json.loads(jsonified_data)
    df_key_1=pd.read_json(datasets['df_key_1'],orient='split')
    df_key_2=pd.read_json(datasets['df_key_2'],orient='split')
    df_key_1_2=pd.read_json(datasets['df_key_1_2'],orient='split')
    
    key_1=datasets['key_1']
    key_2=datasets['key_2']
    
    df_key_1_sent=df_key_1[df_key_1['originn']==key_1]
    df_key_2_sent=df_key_2[df_key_2['originn']==key_2]
    df_key_1_2_sent=df_key_1_2[df_key_1_2['originn']==key_1]
    
    df_key_1_received=df_key_1[df_key_1['dest']==key_1]
    df_key_2_received=df_key_2[df_key_2['dest']==key_2]
    df_key_1_2_received=df_key_1_2[df_key_1_2['dest']==key_1]
      
    stats_df_key_1=pd.concat([
        df_key_1_received['amount'].describe(percentiles=[.5]),
        df_key_1_sent['amount'].describe(percentiles=[.5]),
        df_key_1['amount_signed'].describe(percentiles=[.5])],
        axis=1
    ).reset_index()
    
    stats_df_key_2=pd.concat([
        df_key_2_received['amount'].describe(percentiles=[.5]),
        df_key_2_sent['amount'].describe(percentiles=[.5]),
        df_key_2['amount_signed'].describe(percentiles=[.5])],
        axis=1
    ).reset_index()
    
    stats_df_key_1_2=pd.concat([
        df_key_1_2_received['amount'].describe(percentiles=[.5]),
        df_key_1_2_sent['amount'].describe(percentiles=[.5]),
        df_key_1_2['amount_signed'].describe(percentiles=[.5])],
        axis=1
    ).reset_index()
    
    stats_df_key_1.columns=['index','Incoming','Outgoing','All']
    stats_df_key_2.columns=['index','Incoming','Outgoing','All']
    stats_df_key_1_2.columns=['index','Incoming','Outgoing','All']
    
    columns=[[
        {"name" : key,"id" : 'index'},
        {"name" : 'Incoming', "id" : 'Incoming'},
        {"name" : 'Outgoing', "id" :'Outgoing'},
        {"name" : 'All', "id" : 'All'}
    ] for key in [key_1,key_2,key_1+' -> '+key_2]]

    
    data=[stats_df_key_1.round(2).to_dict('records'),
          stats_df_key_2.round(2).to_dict('records'),
          stats_df_key_1_2.round(2).to_dict('records')]

    tables=[dash_table.DataTable(
        columns=columns[i],
        data=data[i],
        
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'textAlign': 'left'
        },
        style_table={
            'maxHeight': '300px',
            'overflowY': 'scroll'
        },
        style_cell={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white',
            'textAlign': 'left'
        },
        style_cell_conditional=[
        {'if': {'column_id': 'Incoming'},
         'width': '20%'},
        {'if': {'column_id': 'Outgoing'},
         'width': '20%'},
        {'if': {'column_id': 'All'},
         'width': '20%'}]

    ) for i in range(3)]


    return tables[0],tables[1],tables[2]


# Update Time (1) Tab
@app.callback(
    Output('timegraph','figure'),
    [Input('submit-button','n_clicks')],
    [State('intermediate_div','children')])
def update_time_graphs(n_clicks,jsonified_data):
    
    # error checks
    if ((n_clicks is None) or (jsonified_data == [])):
        print('Returning dash.no_update')
        return dash.no_update
        
    # get variables from jsonidied_data
    datasets=json.loads(jsonified_data)
    df_key_1=pd.read_json(datasets['df_key_1'],orient='split')
    df_key_2=pd.read_json(datasets['df_key_2'],orient='split')
    df_key_1_2=pd.read_json(datasets['df_key_1_2'],orient='split')
    
    key_1=datasets['key_1']
    key_2=datasets['key_2']
    
    # assign colors to different accounts
      
    accounts_key_1=df_key_1['a_key'].append(df_key_1['b_key'],ignore_index=True)
    accounts_key_2=df_key_2['a_key'].append(df_key_2['b_key'],ignore_index=True)
    accounts_combined=accounts_key_1.append(accounts_key_2,ignore_index=True)
    unique_accounts_combined=accounts_combined.unique()
    
    dcs=px.colors.qualitative.D3 # discrete color scale
    unique_account_colors={acc:dcs[i%len(dcs)] for (i,acc) in enumerate(unique_accounts_combined)}   
  
    df_key_1['col']=df_key_1['other_key'].apply(lambda acc:unique_account_colors[acc])
    df_key_2['col']=df_key_2['other_key'].apply(lambda acc:unique_account_colors[acc])
    df_key_1_2['col']=df_key_1_2['other_key'].apply(lambda acc:unique_account_colors[acc])
    
    # make subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.02,
        # subplot_titles=(key_1,key_2,key_1+' - '+key_2)
    )
    
    # key_1
    fig.add_trace(
        go.Scatter(
            x=df_key_1['datee'],
            y=df_key_1['amount_signed'],
            text=df_key_1['other_key'],
            mode='markers',
            marker={
                'color':df_key_1['col'],
                'size': 10,
                'opacity': 0.25,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),     
        row=1,
        col=1
    )
       
    # key_2
    fig.add_trace(
        go.Scatter(
            x=df_key_2['datee'],
            y=df_key_2['amount_signed'],
            text=df_key_2['other_key'],
            mode='markers',
            marker={
                'color':df_key_2['col'],
                'size': 10,
                'opacity': 0.25,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),
        row=2,
        col=1
    )
       
    # key_1_2
    fig.add_trace(
        go.Scatter(
            x=df_key_1_2['datee'],
            y=df_key_1_2['amount_signed'],
            text=df_key_1_2['other_key'],
            mode='markers',
            marker={
                'color':df_key_1_2['col'],
                'size': 10,
                'opacity': 0.25,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name='Transactions'
        ),
        row=3,
        col=1
    )
    
    # xaxis properties
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    # yaxis properties
    fig.update_yaxes(title_text="Amount", row=1, col=1)
    fig.update_yaxes(title_text="Amount", row=2, col=1)
    fig.update_yaxes(title_text="Amount", row=3, col=1)
    
    # layout
    fig.update_layout(
        autosize=False,
        # width=800,
        # height=800,      
        margin={'l' : 0, 'r' : 0, 'b' : 0, 't' : 0, 'autoexpand' : True},
        hovermode='closest',
        showlegend=False
    )
    
    return fig


# Update Time (2) Tab
@app.callback(
    Output('cumgraph','figure'),
    [Input('submit-button','n_clicks')],
    [State('intermediate_div','children')])
def update_cum_graphs(n_clicks,jsonified_data):
    
    # error checks
    if ((n_clicks is None) or (jsonified_data == [])):
        print('Returning dash.no_update')
        return dash.no_update
        
    # get variables from jsonidied_data
    datasets=json.loads(jsonified_data)
    df_key_1=pd.read_json(datasets['df_key_1'],orient='split')
    df_key_2=pd.read_json(datasets['df_key_2'],orient='split')
    df_key_1_2=pd.read_json(datasets['df_key_1_2'],orient='split')
    
    key_1=datasets['key_1']
    key_2=datasets['key_2']
    
    df_key_1['cum_amount']=df_key_1['amount_signed'].cumsum()
    df_key_2['cum_amount']=df_key_2['amount_signed'].cumsum()
    df_key_1_2['cum_amount']=df_key_1_2['amount_signed'].cumsum()
    
    # make subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.02,
        # subplot_titles=(key_1,key_2,key_1+' - '+key_2)
    )
    
    # key_1
    fig.add_trace(
        go.Scatter(
            x=df_key_1['datee'],
            y=df_key_1['cum_amount'],
            text=key_1,
            mode='lines',
            marker={
                'size': 10,
                'opacity': 0.25,
                'line': {'width': 0.5, 'color': 'white'}
            },
            line={
                # 'color':unique_account_colors[key_1] if key_1 in unique_account_colors else None
            },
            name='Cumulative Transactions'
        ),     
        row=1,
        col=1
    )
    
    # key_2
    fig.add_trace(
        go.Scatter(
            x=df_key_2['datee'],
            y=df_key_2['cum_amount'],
            text=key_2,
            mode='lines',
            marker={
                'size': 10,
                'opacity': 0.25,
                'line': {'width': 0.5, 'color': 'white'}
            },
            line={
                # 'color':unique_account_colors[key_2] if key_2 in unique_account_colors else None
            },
            name='Cumulative Transactions'
        ),
        row=2,
        col=1
    )
    
    # key_1_2
    fig.add_trace(
        go.Scatter(
            x=df_key_1_2['datee'],
            y=df_key_1_2['cum_amount'],
            text=key_1,
            mode='lines',
            marker={
                'size': 10,
                'opacity': 0.25,
                'line': {'width': 0.5, 'color': 'white'}
            },
            line={
                # 'color':unique_account_colors[key_1] if key_1 in unique_account_colors else None
            },
            name='Cumulative Transactions'
        ),
        row=3,
        col=1
    )
    
    # xaxis properties
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    # yaxis properties
    fig.update_yaxes(title_text="Amount", row=1, col=1)
    fig.update_yaxes(title_text="Amount", row=2, col=1)
    fig.update_yaxes(title_text="Amount", row=3, col=1)
    
    # layout
    fig.update_layout(
        autosize=False,
        # width=800,
        # height=800,      
        margin={'l' : 0, 'r' : 0, 'b' : 0, 't' : 0, 'autoexpand' : True},
        hovermode='closest',
        showlegend=False
    )

    return fig


# Update Frequency Tab
@app.callback(
    Output('freqgraph','figure'),
    [Input('submit-button','n_clicks')],
    [State('intermediate_div','children')])
def update_frequency_graphs(n_clicks,jsonified_data):
    
    # error checks
    if ((n_clicks is None) or (jsonified_data == [])):
        print('Returning dash.no_update')
        return dash.no_update
    
    # get variables from jsonidied_data
    datasets=json.loads(jsonified_data)
    df_key_1=pd.read_json(datasets['df_key_1'],orient='split')
    df_key_2=pd.read_json(datasets['df_key_2'],orient='split')
    df_key_1_2=pd.read_json(datasets['df_key_1_2'],orient='split')
    
    key_1=datasets['key_1']
    key_2=datasets['key_2']
    
    # convert strings to dates
    df_key_1['datee']=pd.to_datetime(df_key_1['datee']).dt.date
    df_key_2['datee']=pd.to_datetime(df_key_2['datee']).dt.date
    df_key_1_2['datee']=pd.to_datetime(df_key_1_2['datee']).dt.date
    
    # calculate consecutive transaction intervals
    intervals_1=(df_key_1['datee']-df_key_1['datee'].shift(1)).dropna().apply(lambda x: x.days)
    intervals_2=(df_key_2['datee']-df_key_2['datee'].shift(1)).dropna().apply(lambda x: x.days)
    intervals_1_2=(df_key_1_2['datee']-df_key_1_2['datee'].shift(1)).dropna().apply(lambda x: x.days)

    # make subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.02
    )
    
    # key_1
    fig.add_trace(go.Histogram(
            x=intervals_1,
            name='Intervals'
        ),
        row=1,
        col=1
    )
       
    # key_2
    fig.add_trace(go.Histogram(
            x=intervals_2,
            name='Intervals'
        ),
        row=2,
        col=1
    )
        
    # key_1_2
    fig.add_trace(go.Histogram(
        x=intervals_1_2,
        name='Intervals'
        ),
        row=3,
        col=1
    )
    
    # xaxis properties
    fig.update_xaxes(title_text="Interval (days)", row=3, col=1)
    
    # yaxis properties
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    
    # layout
    fig.update_layout(
        autosize=False,
        # width=800,
        # height=800,
        margin={'l': 0, 'b': 0, 't': 0, 'r': 0, 'autoexpand' : True},
        hovermode='closest',
        showlegend=False
    )
    
    return fig


# Update network based on all inputs
@app.callback(
    Output('mapx','srcDoc'), 
    [Input('submit-button','n_clicks')],
    [State('input_1','value'),
     State('input_2','value'),
     State('amount-slider','value'),
     State('datepicker','start_date'),
     State('datepicker','end_date')])
def update_network(n_clicks,Account_1,Account_2,amount_range,start_date,end_date):
    
    if ((Account_1 is None) or (Account_2 is None) or (start_date is None) or (end_date is None)):
        print('Preventing update of network')
        raise PreventUpdate
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()
    granular_bank_data=df[(df['datee'] >= sd) & (df['datee'] <= ed) & (df['amount']>=amount_range[0]) & (df['amount']<=amount_range[1])]
    tot_bank=filter_df(Account_1,sd,ed,amount_range[0],amount_range[1])

    """granular_bank_data=bank_data[(bank_data['datee'] >= sd) & (bank_data['datee'] <= ed)]
    sent_bank=granular_bank_data[granular_bank_data['originn']==Account_1]
    recieved_bank=granular_bank_data[granular_bank_data['dest']==Account_1]
    tot_bank=pd.concat([sent_bank,recieved_bank])
    
    edges = pd.DataFrame({'source': tot_bank['originn'],'target':tot_bank['dest']
                          ,'weight': tot_bank['amount']
                          ,'color': ['g' if x<=200 else 'r' for x in tot_bank['amount']]
                         })"""
    adj_data = tot_bank
    if Account_1 != Account_2:
        a_temp=filter_df(Account_2,sd,ed,amount_range[0],amount_range[1])
        a=a_temp[(a_temp['originn']==Account_2) & (a_temp['dest'] != Account_1)]
        b=a_temp[(a_temp['dest']==Account_2) & (a_temp['originn'] != Account_1)]
        adj_data=pd.concat([adj_data,a])
        adj_data=pd.concat([adj_data,b])
    for i in adj_data['originn'].unique():
        for j in adj_data['dest'].unique():
            if i != Account_1 and j != Account_1 and i != Account_2 and j != Account_2:
                origin_i=granular_bank_data[(granular_bank_data['originn']== i) & (granular_bank_data['dest']== j)]
                adj_data=pd.concat([adj_data,origin_i])    
    two_edges=pd.DataFrame({'source': adj_data['originn'],'target':adj_data['dest']
                          ,'weight': adj_data['amount']
                          ,'color': ['green' if x<=100 else 'red' for x in adj_data['amount']]
                         })
     
    G_two_edge = nx.from_pandas_edgelist(two_edges,'source','target', edge_attr=['weight','color'],create_using=nx.MultiDiGraph()) 
    output_filename='TwoEdge_net_updated.html'
    # make a pyvis network
    network_class_parameters = {"notebook": True, "height": "98vh", "width":"98vw", "bgcolor": None,"font_color": None, "border": 0, "margin": 0, "padding": 0} # 
    pyvis_graph = net.Network(**{parameter_name: parameter_value for parameter_name,
                                 parameter_value in network_class_parameters.items() if parameter_value}, directed=True) 
    sources = two_edges['source']
    targets = two_edges['target']
    weights = two_edges['weight']
    color = two_edges['color']
    edge_data = zip(sources, targets, weights, color)
    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]
        c = e[3]
        pyvis_graph.add_node(src,title=src)
        pyvis_graph.add_node(dst,title=dst)
        pyvis_graph.add_edge(src,dst,value=w,color=c)   
    #pyvis_graph.show_buttons(filter_=['nodes','edges','physics'])   
    pyvis_graph.set_options("""
var options = {
  "nodes": {
    "borderWidthSelected": 3,
    "color": {
      "border": "rgba(43,124,233,1)",
      "background": "rgba(109,203,252,1)",
      "highlight": {
        "border": "rgba(55,123,233,1)",
        "background": "rgba(255,248,168,1)"
      }
    },
    "font": {
      "size": 15,
      "face": "tahoma"
    },
    "size": 17
  },
  "edges": {
    "arrowStrikethrough": false,
    "color": {
      "inherit": true
    },
    "smooth": {
      "forceDirection": "none",
      "roundness": 0.35
    }
  },
  "physics": {
    "forceAtlas2Based": {
      "springLength": 100,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based",
    "timestep": 0.49
  }
}
""")
    pyvis_graph.save_graph(output_filename)
    return open(output_filename, 'r').read()


# Update Similarity Tab
@app.callback(Output('sim_table_div','children'),
              [Input('submit-button','n_clicks')],
              [State('input_1','value'),
               State('input_2','value'),
               State('amount-slider','value'),
               State('datepicker','start_date'),
               State('datepicker','end_date')])
def update_output_div_similaritytable(n_clicks,Account_1,Account_2,amount_range,start_date,end_date):
    
    if ((Account_1 is None) or (Account_2 is None) or (start_date is None) or (end_date is None)):
        print('Preventing update of time and frequency graphs')
        raise PreventUpdate
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()
    granular_bank_data=df[(df['datee'] >= sd) & (df['datee'] <= ed) & (df['amount']>=amount_range[0]) & (df['amount']<=amount_range[1])]
    tot_bank=filter_df(Account_1,sd,ed,amount_range[0],amount_range[1])
    adj_data = tot_bank
    if Account_1 != Account_2:
        a_temp=filter_df(Account_2,sd,ed,amount_range[0],amount_range[1])
        a=a_temp[(a_temp['originn']==Account_2) & (a_temp['dest'] != Account_1)]
        b=a_temp[(a_temp['dest']==Account_2) & (a_temp['originn'] != Account_1)]
        adj_data=pd.concat([adj_data,a])
        adj_data=pd.concat([adj_data,b])
    for i in adj_data['originn'].unique():
        for j in adj_data['dest'].unique():
            if i != Account_1 and j != Account_1 and i != Account_2 and j != Account_2:
                origin_i=granular_bank_data[(granular_bank_data['originn']== i) & (granular_bank_data['dest']== j)]
                adj_data=pd.concat([adj_data,origin_i])

    two_edges=pd.DataFrame({'source': adj_data['originn'],'target':adj_data['dest']
                          ,'weight': adj_data['amount']
                          ,'color': ['green' if x<=100 else 'red' for x in adj_data['amount']]
                         })
    G_two_edge = nx.from_pandas_edgelist(two_edges,'source','target', edge_attr=['weight','color'],create_using=nx.MultiDiGraph()) 
    sim_outgoing = nx.simrank_similarity(G_two_edge)
    sim_incoming = simrank_similarity_Incoming(G_two_edge)
    new_out={}
    for i in sim_outgoing.keys():
        for j in sim_outgoing.keys():
            new_out[i +'_'+j]=sim_outgoing_two_nodes(sim_outgoing,i,j)
    new_in={}
    for i in sim_outgoing.keys():
        for j in sim_outgoing.keys():
            new_in[i +'_'+j]=sim_incoming_two_nodes(sim_incoming,i,j)
    result_df=[]
    result_df=pd.DataFrame(new_out.keys())
    result_df['Accounts']=pd.DataFrame(new_out.keys())
    result_df['sim_out']=pd.DataFrame(new_out.values())
    result_df['sim_in']=pd.DataFrame(new_in.values())
    result_df1=result_df[['Accounts','sim_out','sim_in']]
    
    columns=[
        {"name" : 'Accounts', "id" : 'Accounts'},
        {"name" : 'sim_out', "id" :'sim_out'},
        {"name" : 'sim_in', "id" : 'sim_in'}
    ]
    
    data=result_df1.to_dict('records')


    table=dash_table.DataTable(
        columns=columns,
        data=data,
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        style_header={'backgroundColor': 'rgb(30, 30, 30)',
        'textAlign': 'left'},
        style_table={
        'width': '100%',
        #'height': '70vh',
        'maxHeight': '70vh',
        #'overflowY': 'hidden'
    },
    fixed_rows={ 'headers': True, 'data': 0 },
    style_cell_conditional=[
        {'if': {'column_id': 'Accounts'},
         'width': '50%'},
        {'if': {'column_id': 'sim_out'},
         'width': '25%'},
        {'if': {'column_id': 'sim_in'},
         'width': '25%'}],
    style_cell={
        'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white',
        'textAlign': 'left'
    },


    )
    return table
# Update similarity of key_1 and key_2 based on outgoing and incoming edges

"""@app.callback(Output('textArea','children'),
              [Input('submit-button','n_clicks')],
              [State('input_1','value'),
               State('input_2','value'),
               State('amount-slider','value'),
               State('datepicker','start_date'),
               State('datepicker','end_date')])
def update_output_div_similarity(n_clicks,Account_1,Account_2,amount_range,start_date,end_date):
    
    if ((Account_1 is None) or (Account_2 is None) or (start_date is None) or (end_date is None)):
        print('Preventing update of similarity')
        raise PreventUpdate
    sd=datetime.strptime(start_date.split(' ')[0],'%Y-%m-%d').date()
    ed=datetime.strptime(end_date.split(' ')[0],'%Y-%m-%d').date()
    granular_bank_data=df[(df['datee'] >= sd) & (df['datee'] <= ed) & (df['amount']>=amount_range[0]) & (df['amount']<=amount_range[1])]
    tot_bank=filter_df(Account_1,sd,ed,amount_range[0],amount_range[1])
    adj_data = tot_bank
    if Account_1 != Account_2:
        a_temp=filter_df(Account_2,sd,ed,amount_range[0],amount_range[1])
        a=a_temp[(a_temp['originn']==Account_2) & (a_temp['dest'] != Account_1)]
        b=a_temp[(a_temp['dest']==Account_2) & (a_temp['originn'] != Account_1)]
        adj_data=pd.concat([adj_data,a])
        adj_data=pd.concat([adj_data,b])
    for i in adj_data['originn'].unique():
        for j in adj_data['dest'].unique():
            if i != Account_1 and j != Account_1 and i != Account_2 and j != Account_2:
                origin_i=granular_bank_data[(granular_bank_data['originn']== i) & (granular_bank_data['dest']== j)]
                adj_data=pd.concat([adj_data,origin_i])

    two_edges=pd.DataFrame({'source': adj_data['originn'],'target':adj_data['dest']
                          ,'weight': adj_data['amount']
                          ,'color': ['green' if x<=100 else 'red' for x in adj_data['amount']]
                         })
    G_two_edge = nx.from_pandas_edgelist(two_edges,'source','target', edge_attr=['weight','color'],create_using=nx.MultiDiGraph()) 
    sim_outgoing = nx.simrank_similarity(G_two_edge)
    sim_incoming = simrank_similarity_Incoming(G_two_edge)
   
    sim1_2=str(sim_outgoing_two_nodes(sim_outgoing,Account_1,Account_2))+" "+str((sim_incoming_two_nodes(sim_incoming,Account_1,Account_2)))
    return sim1_2
"""

### Run App
if __name__ == '__main__':
    app.run_server(debug=True)