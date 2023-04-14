import dash 
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from flask import Flask

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import os
import json

# Loading data 
df = pd.read_csv(r'Copy of TS2 Find At Location LOG - APR.csv')

# Initializing app
server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)

# Creating app layout 

app.layout = dbc.Container([

    # Creating Header for Application
    dbc.Row([
        # Creating a header using dbc.Label
        dbc.Label('Machine Learning Dashboard', className='h1', style={'text-align': 'center', 'margin-top': '12px'})
            ]),
    # End of Row 1

    # Creating a Row with 2 columns for graphs
    dbc.Row([
    
        # Column 1 for Graph 1
        dbc.Col([
                # Creating a header using dbc.Label
                dbc.Label('Graph 1', className='h2', style={'text-align': 'center', 'margin-top': '12px'}),
                # Creating a Graph using dcc.Graph
                dcc.Graph(id='graph1', figure={})
                ]),
                # End of Column 1

        # Column 2 for Graph 2
        dbc.Col([
                # Creating a header using dbc.Label
                dbc.Label('Graph 2', className='h2', style={'text-align': 'center', 'margin-top': '12px'}),
                # Creating a Graph using dcc.Graph
                dcc.Graph(id='graph2', figure={})
                ])
                # End of Column 2

            ]),
        # End of Row 2

])




# Creating callbacks for widgets in application

# Running app
if __name__ == '__main__':
    app.run_server(debug=True)