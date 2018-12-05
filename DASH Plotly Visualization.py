import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go
from plotly import tools
from dash.dependencies import Input, Output,State

#import numpy as np

#import mysql.connector
#from sqlalchemy import create_engine

#def setting_connect(MYSQL_USER,MYSQL_PASSWORD,MYSQL_HOST_IP,MYSQL_PORT,MYSQL_DATABASE):
#    engine=create_engine('mysql+mysqlconnector://'+MYSQL_USER+':'+MYSQL_PASSWORD+'/'+MYSQL_DATABASE+'?host='+MYSQL_HOST_IP+'?port='+MYSQL_PORT)
#    return engine.connect()
#
#conn = setting_connect('root','admin@localhost','localhost','3306','ujiantitanic')

app = dash.Dash() # make python obj with Dash() method
app.title = 'RETAIL ANALYSIS'; # set web title
global_mean=0

conn = sql.connect("datasets\Retail.sqlite", check_same_thread=False)
#sqlite.connect(":memory:", check_same_thread=False)
df_year = pd.read_sql_query("select DISTINCT year from retail;", conn)
df_store = pd.read_sql_query("select DISTINCT store from retail;", conn)

#data_all=pd.read_csv("datasets\Retail.sqlite")
global_data_output=pd.DataFrame()

app.layout = html.Div(children=[
    dcc.Tabs(id="tabs", value='tab-1', 
        style={
            'fontFamily': 'system-ui',
            'maxWidth': '1000px',
            'margin': '0 auto',
            'background-color': 'rgb(0,255,0)',
            'opacity':'0.6'
        },
        content_style={
            'fontFamily': 'Arial',
            'borderLeft': '1px solid #d6d6d6',
            'borderRight': '1px solid #d6d6d6',
            'borderBottom': '1px solid #d6d6d6',
            'padding': '44px',
#            'background-color': 'rgb(0,0,255)',
#            'opacity':'0.6'
        }, 
        children=[
                dcc.Tab(label='Table Sales Report', value='tab-1', children=[
                html.Div([
                    html.H1(id='judultabel',children='',style={'text-align':'center', 'color': '#008080'}),
                    html.Table([
                            html.Tr([
                                        html.Td([html.P('Financial Year :')],style={'width': '150px'}),
                                        html.Td([
                                                    dcc.RangeSlider(
                                                                        id='slide-table-tahun-anggaran',            
                                                                        count=1,
                                                                        min=df_year['year'].min(),
                                                                        max=df_year['year'].max(),
                                                                        marks={str(i): i for i in df_year['year'].unique()},
                                                                        value=[df_year['year'].min(),df_year['year'].max()]
                                                                    )
                                                ],style={'margin': '0 auto','width': '500px'}),
                                    ]),
                            html.Tr([html.Td([html.P('     ')],style={'width': '150px'})]),
                            ],style={'margin': '0 auto'}),
                    html.P('TOP 10 Whole Sales',style={'text-align':'center', 'color': '#008080'}),
                    dcc.Graph(
                                id='tablegrafiksalestop',
                                figure={
                                    'data': []
                                }
                        ),
                    html.P('BOTTOM 10 Whole Sales',style={'text-align':'center', 'color': '#008080'}),
                    dcc.Graph(
                                id='tablegrafiksalesbottom',
                                figure={
                                    'data': []
                                }
                        )                                                        
                        ], style={ 'width' : '1000px', 'margin': '0 auto'}),
                ]),
            dcc.Tab(label='Graphics Sales Report', value='tab-2', children=[
                html.Div([
                    html.H1(id='judulgrafik',children='',style={'text-align':'center', 'color': '#008080'}),
                    html.Table([
                            html.Tr([
                                    html.Td([html.P('Report Type : ')],style={'width': '150px'}),                                            
                                    html.Td([
                                                dcc.Dropdown(
                                                            id='ddlreporttype',
                                                            options=[{'label': 'Store Sales Report ', 'value': 'Store'},
                                                                    {'label': 'Sales Report by Dept', 'value': 'Dept'},
                                                                    {'label': 'Sales Report by Week', 'value': 'week'},
                                                                    {'label': 'Sales Report by Store Size', 'value':'Size'},
                                                                    {'label': 'Sales Report by Store Type', 'value':'Type'}, 
                                                                    ],
                                                            value='Store'
                                                        )
    
                                            ],style={'margin': '0 auto','width': '500px'})
                                    ]),
                            html.Tr([
                                        html.Td([html.P('Financial Year :')],style={'width': '150px'}),
                                        html.Td([
                                                    dcc.RangeSlider(
                                                                        id='slide-tahun-anggaran',            
                                                                        count=1,
                                                                        min=df_year['year'].min(),
                                                                        max=df_year['year'].max(),
                                                                        marks={str(i): i for i in df_year['year'].unique()},
                                                                        value=[df_year['year'].min(),df_year['year'].max()]
                                                                    )
                                                ],style={'margin': '0 auto','width': '500px'}),
                                    ]),
                            html.Tr([html.Td([html.P(children='',id='label_mean')],style={'width': '400px'})]),
                            html.Tr([html.Td([html.P('     ')],style={'width': '150px'})]),
                            ],style={'margin': '0 auto'}),
                    dcc.Graph(
                                id='grafiksales',
                                figure={
                                    'data': []
                                }
                        ),                                                                                                       
                        ], style={ 'width' : '1000px', 'margin': '0 auto'}),
                ]),
            dcc.Tab(label='Feature Importance Relation', value='tab-3', children=[
                html.Div([
#                    html.H1(id='judulgrafik',children='',style={'text-align':'center', 'color': '#008080'}),
                     html.Table([
                             html.Tr([
                                    html.Td([html.P('Store : ')],style={'width': '150px'}),                                            
                                    html.Td([
                                                dcc.Dropdown(
                                                            id='ddlStore',
                                                            options=[{'label':'All Store', 'value':'all'}]+[{'label':'Store '+str(i), 'value':i} for i in df_store['Store']],
                                                            value=['all'],
                                                            multi=True,
                                                        )
    
                                            ],style={'margin': '0 auto','width': '500px'}),
                                                
                                    html.Td([
                                                html.Button('SHOW', id='buttonshow')
                                            ],style={'margin': '0 auto','width': '200px'}),
#                                    html.Td([html.P(id='cektoko',children='')],style={'width': '150px'}),
                                    ]),
                             
                             html.Tr([
                                    html.Td([html.P('Features : ')],style={'width': '150px'}),                                            
                                    html.Td([
                                                dcc.Dropdown(
                                                            id='ddlfeatures',
                                                            options=[{'label': 'Number of Department', 'value': 'Dept'},
                                                                    {'label': 'Store Size', 'value': 'Size'},
                                                                    {'label': 'Markdown', 'value': 'Markdown'},
                                                                    {'label': 'Temperature', 'value': 'Temperature'},
                                                                    {'label': 'CPI', 'value': 'CPI'},
                                                                    {'label': 'Fuel Price', 'value': 'Fuel_Price'},
                                                                    {'label': 'Unemployment', 'value': 'Unemployment'},
                                                                    ],
                                                            value='Dept'
                                                        )
    
                                            ],style={'margin': '0 auto','width': '500px'})
                                    ]),
                                                
                            html.Tr([
                                    html.Td([html.P('Financial Year :')],style={'width': '150px'}),
                                    html.Td([
                                                dcc.RangeSlider(
                                                                    id='slide-table-tahun-anggaran-scatter',            
                                                                    count=1,
                                                                    min=df_year['year'].min(),
                                                                    max=df_year['year'].max(),
                                                                    marks={str(i): i for i in df_year['year'].unique()},
                                                                    value=[df_year['year'].min(),df_year['year'].max()]
                                                                )
                                            ],style={'margin': '0 auto','width': '500px'}),
                                    ]),
                            html.Tr([html.Td([html.P('     ')],style={'width': '150px'})]),
                            ],style={'margin': '0 auto'}),
                    dcc.Graph(
                                id='grafikfeatureimportance',
                                figure={
                                    'data': []
                                }
                        ),                                                                                                       
                        ], style={ 'width' : '1000px', 'margin': '0 auto'}),
                ])
            ])
])


@app.callback(
    Output('grafikfeatureimportance', 'figure'),
    [Input('ddlfeatures','value'),Input('buttonshow','n_clicks'),Input('slide-table-tahun-anggaran-scatter','value')],
    [State('ddlStore','value')]
)
def gambar_grafik_feature_importance(features,gapake,periode,daftar_toko) :
    temp_daftar_toko=daftar_toko.copy()
#    list_not_all=[]
    if daftar_toko.count('all')>0 : 
        daftar_toko=df_store['Store'].values
        
    daftar_toko=list(set(daftar_toko))
    str1 = ''.join(str(e)+',' for e in daftar_toko)
    daftar_toko='('+str1[:-1]+')'
    del str1
    
    df_scatter = pd.read_sql_query(
    'select Store,avg(Size)''Size'',avg(temperature)''Temperature'',avg(cpi)''CPI'',avg(fuel_price)''Fuel_Price'',avg(unemployment)''Unemployment'',count(dept)''Dept'',sum(weekly_sales)''Weekly_Sales'''
    +' ,(avg(Markdown1)+avg(Markdown2)+avg(Markdown3)+avg(Markdown4)+avg(Markdown5))''Markdown'''
    +' from retail'
    +' where store in '+daftar_toko
    +' and year between '+str(periode[0])+' and '+str(periode[1])
    +' group by Store,Date,week,year,IsHoliday'
    +';', conn)
    df_scatter['Store no']=df_scatter['Store']
    
    fig = tools.make_subplots(rows=1, cols=1)
    if temp_daftar_toko.count('all')>0 : 
        df_scatter['Store no'][~df_scatter['Store'].isin(temp_daftar_toko)]='All'
    

        toko='All'
        trace1 = go.Scatter(
                    x = df_scatter[df_scatter['Store no']==toko][features], 
                    y = df_scatter[df_scatter['Store no']==toko]['Weekly_Sales'], 
                    mode = 'markers',
                    text = df_scatter[df_scatter['Store no']==toko]['Store'],
                    name = 'All Store ',
                    showlegend=True
    #                    hovertext= ('Num of Dept :', ' Whole Sales :')
                    )
        
        fig.append_trace(trace1,1,1)
        fig['layout']['xaxis'+str(1)].update(title=features.capitalize())
        fig['layout']['yaxis'+str(1)].update(title='Whole Sales')     
    for toko in sorted(df_scatter[df_scatter['Store no']!='All']['Store'].unique()):
        trace1 = go.Scatter(
                    x = df_scatter[df_scatter['Store']==toko][features], 
                    y = df_scatter[df_scatter['Store']==toko]['Weekly_Sales'], 
                    mode = 'markers',
                    text = df_scatter[df_scatter['Store']==toko]['Store'],
                    name = 'Store ' + str(toko),
                    showlegend=True
#                    hovertext= ('Num of Dept :', ' Whole Sales :')
                    )
        
        fig.append_trace(trace1,1,1)
        fig['layout']['xaxis'+str(1)].update(title=features.capitalize())
        fig['layout']['yaxis'+str(1)].update(title='Whole Sales')
               
    fig['layout'].update(
                        height=400, 
                        width=1000,
                        hovermode='closest',
                        #title='HUE='+hue.upper()+'  Colomns='+kolom.upper()
                        ) 
    del df_scatter
    return fig     

@app.callback(
    Output('tablegrafiksalestop', 'figure'),
    [Input('slide-table-tahun-anggaran','value')]
)
def update_isitable(periode) :
    global global_data_output
    df = pd.read_sql_query(
    ' select t1.Store,t1.Type,t1.Temperature,t1.CPI,t1.Fuel_Price,t1.Unemployment,t2.Dept,t1.Size,t1.Weekly_sales'
    +' from ('
            +'select subt1.Store,subt1.Type,round(avg(subt1.temperature))''Temperature'',round(avg(subt1.cpi),2)''CPI'',round(avg(subt1.fuel_price),2)''Fuel_Price'',round(avg(subt1.unemployment),2)''Unemployment'''
            +',round(avg(subt1.Size))''Size'''
            +',sum(subt1.weekly_sales)''weekly_sales'''
            +' from retail subt1'
            +' where subt1.year between '+str(periode[0])+' and '+str(periode[1])
            +' group by subt1.Store'
    +') t1'
    +' left join (select sub2.store,max(sub2.dept) Dept from'
        +' ('
        +' select r2.store,r2.week,count(r2.Dept) Dept'
        +' from retail r2'
        +' where (r2.year between '+str(periode[0])+' and '+str(periode[1])+')'
        +' group by r2.store,r2.week,r2.Date,r2.year,r2.IsHoliday'
        +' )sub2'
        +' group by sub2.store'
    +' ) t2'
    +' on t2.store=t1.store'
    +' order by t1.weekly_sales desc'
    +';', conn)
    
    df['Total_Sales']=df['weekly_sales'].apply(lambda x: "{0:,.2f}".format(x))

    global_data_output=df.tail(10).copy()
    df=df.head(10)
    return {
        'data': [
            go.Table(
                    header=dict(values=['<b>'+col.upper()+'</b>' for col in df[['Store','Type','Temperature','CPI','Fuel_Price','Unemployment','Dept','Size','Total_Sales']].columns],
                                fill = dict(color='#C2D4FF'),
                                align = 'center'
                                
                                ),
                    cells=dict(values=[df[col] for col in df[['Store','Type','Temperature','CPI','Fuel_Price','Unemployment','Dept','Size','Total_Sales']].columns],
                               fill = dict(color='#F5F8FF'),
                               align = 'center',
                               font = dict(color = 'black', size = 11)
                               )
                               
                            )
        ],
        'layout': go.Layout(
            margin={'l': 40, 'b': 10, 't': 10, 'r': 10},
            height=300, 
            width=1000,
        )
    }
    
@app.callback(
    Output('tablegrafiksalesbottom', 'figure'),
    [Input('tablegrafiksalestop','figure')]
)
def update_isitable_bottom(periode) :    
    data_output=global_data_output
    
    return {
        'data': [
            go.Table(
                    header=dict(values=['<b>'+col.upper()+'</b>' for col in data_output[['Store','Type','Temperature','CPI','Fuel_Price','Unemployment','Dept','Size','Total_Sales']].columns],
                                fill = dict(color='#C2D4FF'),
                                align = 'center'
                                
                                ),
                    cells=dict(values=[data_output[col] for col in data_output[['Store','Type','Temperature','CPI','Fuel_Price','Unemployment','Dept','Size','Total_Sales']].columns],
                               fill = dict(color='#F5F8FF'),
                               align = 'center',
                               font = dict(color = 'black', size = 11)
                               )
                               
                            )
        ],
        'layout': go.Layout(
            margin={'l': 40, 'b': 10, 't': 10, 'r': 10},
            height=300, 
            width=1000,
        )
    }

@app.callback(
    Output('label_mean', 'children'),
    [Input('grafiksales','figure')] 
)

def isi_mean(gapake) :
#    data_output['Weekly_Sales'].apply(lambda x: "{0:,.2f}".format(x))
    return 'Rata-rata Sales : '+"{0:,.2f}".format(global_mean)
    
@app.callback(
    Output('grafiksales', 'figure'),
    [Input('ddlreporttype','value'),
     Input('slide-tahun-anggaran','value')] 
)

def gambar_histo_sales(jenis_report,periode) :
    global global_mean
    sql_select=''
    if jenis_report=='Store': sql_select=jenis_report
    else : sql_select='Store,'+jenis_report
    df = pd.read_sql_query(
    'select '+sql_select+',year,sum(Weekly_Sales)Weekly_Sales'
    +' from retail '
    +' where year between '+str(periode[0])+' and '+str(periode[1])
    +' group by '+jenis_report+',year'
    +' order by Weekly_Sales desc'
    +';', conn) 
    
    
    fig = tools.make_subplots(rows=4, cols=1,
                              subplot_titles=['Total Sales '+str(periode[0])+' to '+str(periode[1])] + list(df['year'].unique().astype(str)),
#                              vertical_spacing=1
                              )
    showlegenda=True    
#    if jenis_report=='week' : temp=data_output.groupby([jenis_report]).sum().reset_index().sort_values('week')
#    temp=df.copy()
    if jenis_report=='Size':
        temp=df.groupby([jenis_report]).mean().reset_index().sort_values('Size',ascending=True)
        temp['Weekly_Sales']=df.groupby([jenis_report]).sum().reset_index().sort_values('Size',ascending=True)['Weekly_Sales']
    else:
        temp=df.groupby([jenis_report]).mean().reset_index().sort_values('Weekly_Sales',ascending=True)
        temp['Weekly_Sales']=df.groupby([jenis_report]).sum().reset_index().sort_values('Weekly_Sales',ascending=True)['Weekly_Sales']

    
    tempy=temp['Weekly_Sales'].copy()
    
    global_mean=tempy.mean()
    
    batas_atas=tempy.mean()+tempy.std()
    batas_bawah=tempy.mean()-tempy.std()
    tempy[tempy<=batas_atas]=0
    if jenis_report!='Type' : tempx=round(temp[jenis_report])
    else : tempx=temp[jenis_report]
    
    trace1 = go.Bar(
        x=tempx,
        y=tempy,
        hoverinfo = 'x+y',
        text=temp['Store'],
        width=0.8, 
        name='ABOVE STD',
        showlegend=showlegenda,
        marker=dict(
            color='rgba(50, 171, 96, 1.0)'
        )
    )
    tempy=temp['Weekly_Sales'].copy()
    tempy[(tempy>batas_atas)|(tempy<batas_bawah)]=0
    trace2 = go.Bar(
        x=tempx,
        y=tempy,
        hoverinfo = 'x+y',
        text=temp['Store'],
        width=0.8,
        name='NORMAL',
        showlegend=showlegenda,
        marker=dict(
            color='rgba(55, 128, 191, 1.0)'
        )
    )
    
    tempy=temp['Weekly_Sales'].copy()
    tempy[tempy>=batas_bawah]=0
    trace3 = go.Bar(
        x=tempx,
        y=tempy,
        hoverinfo = 'x+y',
        text=temp['Store'],
        width=0.8,
        name='BELOW STD',
        showlegend=showlegenda,
        marker=dict(
            color='rgba(219, 64, 82, 1.0)'
        )
    )
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,1)
    fig.append_trace(trace3,1,1)
    fig['layout']['xaxis'+str(1)].update(title=jenis_report.capitalize())
    if (jenis_report!='week') : 
        fig['layout']['xaxis'+str(1)].update(type='category')
    fig['layout']['yaxis'+str(1)].update(title='Total Transaction')
#    fig['layout']['margin'+str(1)].update(b= 40)

    showlegenda=False     
    
    pos=2
    for tahun in list(reversed(df['year'].unique())):
        
#        if jenis_report=='week':temp=data_output[data_output['year']==tahun].sort_values('week')
        if jenis_report=='Size':
            temp=df[df['year']==tahun].sort_values('Size',ascending=True)
        else:
            temp=df[df['year']==tahun].sort_values('Weekly_Sales',ascending=True)
        tempy=temp['Weekly_Sales'].copy()
        batas_atas=tempy.mean()+tempy.std()
        batas_bawah=tempy.mean()-tempy.std()
        tempy[tempy<=batas_atas]=0
        trace1 = go.Bar(
            x=tempx,
            y=tempy,
            hoverinfo = 'x+y',
            text=temp['Store'],
            width=0.8, 
            name='ABOVE STD',
            showlegend=showlegenda,
            marker=dict(
                color='rgba(50, 171, 96, 1.0)'
            )
        )
        tempy=temp['Weekly_Sales'].copy()
        tempy[(tempy>batas_atas)|(tempy<batas_bawah)]=0
        trace2 = go.Bar(
            x=tempx,
            y=tempy,
            hoverinfo = 'x+y',
            text=temp['Store'],
            width=0.8,
            name='NORMAL',
            showlegend=False,
            marker=dict(
                color='rgba(55, 128, 191, 1.0)'
            )
        )
        
        tempy=temp['Weekly_Sales'].copy()
        tempy[tempy>=batas_bawah]=0
        trace3 = go.Bar(
            x=tempx,
            y=tempy,
            hoverinfo = 'x+y',
            text=temp['Store'],
            width=0.8,
            name='BELOW STD',
            showlegend=showlegenda,
            marker=dict(
                color='rgba(219, 64, 82, 1.0)'
            )
        )
        fig.append_trace(trace1,pos,1)
        fig.append_trace(trace2,pos,1)
        fig.append_trace(trace3,pos,1)
        fig['layout']['xaxis'+str(pos)].update(title=jenis_report.capitalize())
        if (jenis_report!='week'): fig['layout']['xaxis'+str(pos)].update(type='category')
        fig['layout']['yaxis'+str(pos)].update(title='Total Transaction')
#        fig['layout']['margin'+str(pos)].update(b= 40)
        pos+=1
        showlegenda=False 
        
    fig['layout'].update(
                        height=1200, 
                        width=1000,
                        #title='HUE='+hue.upper()+'  Colomns='+kolom.upper()
                        ) 
    del df,temp,tempy
    return fig 

if __name__ == '__main__':
    # run server on port 1997
    # debug=True for auto restart if code edited
    app.run_server(debug=True) 