from dash import Dash, html, dcc
import dash
import dash_uploader as du


def setup():
    app = Dash(__name__, use_pages=True)
    server = app.server

    du.configure_upload(app, "data/", use_upload_id=False)
    app.layout = html.Div([
        dcc.Store(id='inputs'),
        html.H1('PV storage simulation', style={'textAlign': 'center'}),
        dash.page_container
    ])
    return app


if __name__ == '__main__':
    app = setup()
    port = 5000
    app.run(debug=True, port=port)
