from dash import Dash, html, dcc
import dash
import dash_uploader as du


def setup():
    app = Dash(__name__, use_pages=True)

    du.configure_upload(app, "data/", use_upload_id=False)
    app.layout = html.Div([
        dcc.Store(id='inputs'),
        html.H1('PV storage simulation', style={'textAlign': 'center'}),
        dash.page_container
    ])
    return app


if __name__ == '__main__':
    app = setup()
    server = app.server
    app.run(debug=True)
