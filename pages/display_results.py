import dash
from dash import html, callback
import plotly.graph_objects as go
from dash import dcc
from dash.dependencies import Input, Output, State

import time
import numpy as np
import financial_calculator
import output_calculator

dash.register_page(__name__, "/results")
data = None


def layout():
    return html.Div(id='parent', children=[
        html.H1(id='H1', children='calculating...', style={'textAlign': 'center', 'marginTop': 40,
                                                           'marginBottom': 40}),
        html.Div(id='content', children=[
            html.Div(children=[
                dcc.Dropdown(id='dropdown1',
                             options=[
                                 {'label': 'irr', 'value': 'irr'},
                                 {'label': 'lcoe', 'value': 'lcoe'},
                             ],
                             value='irr',
                             style=dict(width='100%')),
                dcc.Dropdown(id='dropdown2',
                             options=[
                                 {'label': 'battery hours', 'value': 'battery hours'},
                             ],
                             value='battery hours',
                             style=dict(width='100%'))],
                style=dict(display='flex')),
            dcc.Graph(id='bar_plot')], style={'display': 'none'})
        ])


@callback(Output(component_id='bar_plot', component_property='figure'),
          [Input(component_id='dropdown1', component_property='value'),
           Input(component_id='dropdown2', component_property='value')])
def graph_update(dropdown1_value, dropdown2_value):
    values = data[dropdown1_value]
    fig = go.Figure(
        [go.Scatter(x=tuple(values.keys()), y=tuple(values.values()), line=dict(color='firebrick', width=4))])

    fig.update_layout(title='',
                      xaxis_title=dropdown2_value,
                      yaxis_title=dropdown1_value
                      )
    return fig


@callback([Output(component_id='content', component_property='style'),
           Output(component_id='H1', component_property='children'),
           Output(component_id='dropdown1', component_property='value')],
          Input(component_id='inputs', component_property='data'))
def create_data(inputs):
    num_of_years, grid_size, data_file, land_size, panel_num = inputs
    data_file = ("data/" + data_file[0]) if data_file else 'test.csv'
    if panel_num == '':
        panel_num = 13648
    global data
    data = {'irr': {}, 'lcoe': {}}
    output = output_calculator.OutputCalculator(num_of_years, grid_size, data_file)
    # output = output_calculator.OutputCalculator(25, 5000, location=(30.658611, 35.236667), panel_num=13648)
    finance = financial_calculator.FinancialClaculator(num_of_years, land_size, panel_num=panel_num)
    start_time = time.time()
    for x in np.arange(2, 10, 0.5):
        output.battery_hours = x
        finance.battery_block_num = output.battery_blocks
        output.run()
        data['irr'][x] = finance.get_irr(output.output, output.purchased_from_grid)
        data['lcoe'][x] = finance.get_lcoe(output.output, output.purchased_from_grid)
    print(f"calculation took: {(time.time() - start_time)} seconds")
    return {'display': 'block'}, 'results of financial simulation', 'irr'


if __name__ == '__main__':
    app = dash.Dash()
    app.layout = layout()
    port = 5000
    # webbrowser.open_new("http://localhost:{}".format(port))
    app.run(port=port)
