from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash
import dash_uploader as du

dash.register_page(__name__, path="/")


def layout():
    return html.Center(children=[
        html.Div(id='form', children=[
            html.Center(children=[
                html.Div(children=[
                    html.Label(children="number of years:", style={"margin-right": "15px"}),
                    dcc.Slider(1, 30, 1, id="num_of_years", value=25, marks=None,
                               tooltip={"placement": "bottom", "always_visible": True})
                ], style={"display": "grid", "grid-template-columns": "50% 50%"}),
                html.Div(children=[
                    html.Label(children="grid_size:", style={"margin-right": "15px"}),
                    dcc.Slider(2000, 10000, 100, id="grid_size", value=5000, marks=None,
                               tooltip={"placement": "bottom", "always_visible": True})
                ], style={"display": "grid", "grid-template-columns": "50% 50%"}),
                html.Div(children=[
                    html.Label(children="data file:", style={"margin-right": "15px"}),
                    du.Upload(id='data_file', filetypes=['csv'])
                ], style={"display": "grid", "grid-template-columns": "50% 50%"})
            ], style={'width': "100%", 'margin-right': "15px"}),
            html.Center(children=[
                html.Div(children=[
                    html.Label(children="land size:", style={"margin-right": "15px"}),
                    dcc.Slider(10, 120, 1, id="land_size", value=97, marks=None,
                               tooltip={"placement": "bottom", "always_visible": True})
                ], style={"display": "grid", "grid-template-columns": "50% 50%"}),
                html.Div(children=[
                    html.Label(children="number of panels:", style={"margin-right": "15px"}),
                    dcc.Input(min=5000, max=20000, step=1, id="panel_num", value=13648, required=True)
                ], style={"display": "grid", "grid-template-columns": "50% 50%"}),
            ], style={'width': "100%", 'margin-right': "15px"})
        ], style=dict(display='flex')),
        dcc.Link(html.Button('Submit', id='submit-val', n_clicks=None, form='form', className="submit__button"
                             , style={'margin-top': '30px'}),
                 href="/results")
    ])


@callback(
    Output(component_id='inputs', component_property='data'),
    Input(component_id='submit-val', component_property='n_clicks'),
    State(component_id='num_of_years', component_property='value'),
    State(component_id='grid_size', component_property='value'),
    State(component_id='data_file', component_property='fileNames'),
    State(component_id='land_size', component_property='value'),
    State(component_id='panel_num', component_property='value')
)
def update_output(n_clicks, num_of_years, grid_size, data_file, land_size, panel_num):
    if n_clicks is None:
        return
    data = [num_of_years, grid_size, data_file, land_size, panel_num]
    print(data)
    return data


@callback(
    Output("collapse1", "is_open"),
    [Input("collapse-button1", "n_clicks")],
    [State("collapse1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app = dash.Dash(__name__)
    app.layout = layout()
    port = 5000
    app.run(debug=True, port=port)
