__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'

import dash
from dash import Dash, html, dcc, ctx, dash_table, Input, Output, State

import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir)))
from Tools.db_tools import DbManager
from Tools.exp_tools import Experiment


def spread_hist(frame, lower, upper, bitdepth=None):
    if bitdepth is not None:
        f = 2 ** bitdepth - 1
    else:
        f = 1.0
    return f * np.clip((frame.astype(np.float64) - f * lower) / (f * upper - f * lower), a_min=0, a_max=1)


def create_sliders(channel_df):
    slider_children = []
    for i, channel in channel_df.iterrows():
        label = html.Div(channel['channel_name'], className='slider_label')
        slider = dcc.RangeSlider(
            id={'type': 'dynamic-slider', 'index': i},
            min=0,
            max=1,
            value=[channel['min_val'], channel['max_val']],
            marks=None, allowCross=False, updatemode='drag', className='slider_widget'
        )
        slider_children.append(label)
        slider_children.append(slider)
    return slider_children


class AnnotationPackage:
    def __init__(self, experiment_id, ap_id):
        self.experiment = Experiment(experiment_id)
        self.df = self.get_ap()
        self.droplets = self.df['droplet_id'].unique().copy()
        self.frames = np.moveaxis(self.experiment.db.filter_dataset(self.droplets), 3, 1)
        self.label_types = self.df['label_type'].unique().tolist()

        
        # Create dataframe that stores information about the channel intensities for plotting
        self.LUTs = self.experiment.handler.get_LUTs()
        self.channel_df = self.experiment.handler.channel_df
        self.channel_df.loc[:, ['min_val', 'max_val']] = [0.0, 1.0]
        self.sliders = create_sliders(self.channel_df)
        self.bitdepth = 16

        self.i_max = len(self.droplets) - 1
        for i, droplet_id in enumerate(self.droplets):
            self.i = i
            if np.any(self.df.query(f'droplet_id == {droplet_id}')['status'].values == "pending"):
                break

    
    def get_ap(self):
        self.experiment.connect_db()
        return self.experiment.db.get_annotations(source='manual').query(f'ap_id == "{ap_id}"').set_index('annotation_id')


    def current_annotation(self):
        self.df = self.get_ap()
        annotations = self.df.query(f'droplet_id == {self.droplets[self.i]}').reset_index()
        annotations.drop(columns=['experiment_id', 'ap_id', 'source', 'droplet_id', 'timestamp'], inplace=True)
        annotations = annotations.rename(columns={'label_type': 'Label Type', 'value': 'Value', 'status': 'Status', 'annotation_id': 'ID'})
        return annotations.to_dict('records')
        

    def current_frame(self):
        frame = self.frames[self.i].copy()
        for chan, info in self.channel_df.iterrows():
            frame[chan, :, :] = spread_hist(frame[chan, :, :], info['min_val'], info['max_val'], bitdepth=self.bitdepth)
        frame = (frame // (2 ** (self.bitdepth - 8))).astype(int)  # convert to 8 bit image
        rgb_frame = np.array([LUT[channel] for channel, LUT in zip(frame, self.LUTs)]).astype(float)
        composite = np.sum(rgb_frame, axis=0)
        composite[composite > 255] = 255
        separate_panel = np.hstack(rgb_frame)
        panel = np.concatenate([composite, separate_panel], axis=1).astype(np.uint8)

        fig = px.imshow(panel)
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_xaxes(visible=False, showticklabels=False)
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        return fig

    def current_progress(self):
        progress = round(100 * (self.i +1)/ (self.i_max+1),1)
        progress_label = f'{self.i+1}/{self.i_max+1} frames annotated ({progress} %)'
        return progress_label

# CLI argument parser
parser = argparse.ArgumentParser(description="Launch Droplet Annotator")
parser.add_argument('--experiment_id', required=True, help='Experiment ID')
parser.add_argument('--ap_id', required=True, help='Annotation package ID')
args = parser.parse_args()

experiment_id = args.experiment_id
ap_id = args.ap_id
ap = AnnotationPackage(experiment_id=experiment_id, ap_id=ap_id)

app = Dash(__name__)
app.layout = html.Div(
    id='mainPanel',
    children=[
        html.H2("Droplet Annotator", style={"textAlign": "center", "marginTop": "20px"}),

        # Image + progress label
        html.Div([
            dcc.Graph(
                id="image-display",
                figure=ap.current_frame(),
                style={
                    "width": "100%",
                    "margin": "0 auto"
                }
            ),
            html.Div(id="progress-label", children=ap.current_progress(), style={"textAlign": "center", "fontSize": "1.1rem", "margin": "10px"})
        ], style={"margin": "20px auto", "maxWidth": "80%", "padding": "20px", "border": "1px solid #ccc", "borderRadius": "8px", "backgroundColor": "#fafafa"}),

        # Navigation buttons
        html.Div([
            html.Button("Previous", id="prev-button", n_clicks=0, className="nav-button"),
            html.Button("Next", id="next-button", n_clicks=0, className="nav-button"),
        ], style={"display": "flex", "justifyContent": "center", "gap": "20px", "marginBottom": "20px"}),

        # Channel sliders
        html.Div(
            ap.sliders,
            style={
                "margin": "10px auto",
                "maxWidth": "800px",
                "padding": "15px",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fcfcfc"
            }
        ),

        # Annotation table
        html.Div([
            dash_table.DataTable(
                id='annotation-table',
                columns=[
                    {'name': 'Label Type', 'id': 'Label Type', 'editable': False},
                    {'name': 'Value', 'id': 'Value', 'editable': True, 'type': 'numeric'},
                    {'name': 'Status', 'id': 'Status', 'editable': False},
                    {'name': 'ID', 'id': 'ID', 'editable': False}
                ],
                data=ap.current_annotation(),
                editable=True,
                style_table={
                    'overflowX': 'auto',
                    'marginTop': '10px',
                    'border': '1px solid #ccc',
                    'borderRadius': '8px',
                    'padding': '10px'
                },
                style_header={
                    'backgroundColor': '#f5f5f5',
                    'fontWeight': 'bold',
                    'fontSize': '1.1rem',
                    'textAlign': 'center',
                    'borderBottom': '2px solid #d3d3d3'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontSize': '1.05rem',
                    'minWidth': '100px',
                    'maxWidth': '200px',
                    'whiteSpace': 'normal',
                    'border': '1px solid #e0e0e0'
                },
                style_data={
                    'backgroundColor': '#ffffff',
                    'color': '#333333'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Value'},
                        'textAlign': 'right'
                    },
                    {
                        'if': {'state': 'active'},
                        'backgroundColor': '#e3f2fd',
                        'border': '2px solid #2196f3'
                    },
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': '#f0f8ff',
                        'border': '2px solid #2196f3'
                    }
                ],
            )
        ], style={"margin": "20px auto", "maxWidth": "900px"}),

        # Hidden keyboard trigger
        html.Button(id='next-trigger-button', style={'display': 'none'}),

        # Tip
        html.Div("Use Enter to save input and Shift+Enter to go to the next droplet.", style={"textAlign": "center", "color": "#555", "fontSize": "0.85rem", "marginTop": "20px"})
    ],
    style={"fontFamily": "Segoe UI, sans-serif", "color": "#333", "paddingBottom": "50px"}
)

def update_display_logic():
    ap.experiment.connect_db()
    fig = ap.current_frame()
    label = ap.current_progress()
    table_data = ap.current_annotation()
    return fig, label, table_data


@app.callback(
    Output("image-display", "figure", allow_duplicate=True),
    Output("progress-label", "children"),
    Output("annotation-table", "data", allow_duplicate=True),
    Input("next-button", "n_clicks"),
    Input("prev-button", "n_clicks"),
    Input("next-trigger-button", "n_clicks"),
    prevent_initial_call=True
)
def update_display(next_clicks, prev_clicks, proxy_clicks):
    trigger = ctx.triggered_id
    if trigger == "next-button" or trigger == "next-trigger-button":
        ap.i = min(ap.i + 1, ap.i_max)
    elif trigger == "prev-button":
        ap.i = max(ap.i - 1, 0)
    return update_display_logic()


@app.callback(
    Output("image-display", "figure", allow_duplicate=True),
    Input({"type": "dynamic-slider", "index": dash.ALL}, "value"),
    prevent_initial_call = True
)
def update_brightness(slider_values):
    for i, (min_val, max_val) in enumerate(slider_values):
        ap.channel_df.loc[i, 'min_val'] = min_val
        ap.channel_df.loc[i, 'max_val'] = max_val
    return ap.current_frame()

@app.callback(
    Output('annotation-table', 'data', allow_duplicate=True),
    Input('annotation-table', 'data'),
    State('annotation-table', 'data_previous'),
    prevent_initial_call=True
)
def save_table_annotation(current, previous):
    if previous is None:
        raise dash.exceptions.PreventUpdate

    ap.experiment.connect_db()
    diffs = [curr for curr, prev in zip(current, previous) if curr['Value'] != prev['Value']]

    for row in diffs:
        annotation_id = int(row['ID'])
        try:
            new_value = int(row['Value'])
        except (ValueError, TypeError):
            continue
        ap.experiment.db.update_annotation(annotation_id, value=new_value, status="completed")

    return ap.current_annotation()


if __name__ == "__main__":
    app.run_server(debug=True)
