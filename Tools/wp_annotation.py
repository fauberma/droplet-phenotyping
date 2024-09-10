__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'


from dash import Dash, html, dcc, ctx, dash_table, ALL
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir)))
from Tools.db_tools import DbManager
from Tools.leica_tools import RawLoader


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


class WP:
    def __init__(self, expID, WP_ID):
        self.base = os.path.join(os.getenv('EXP_DIR'), expID, WP_ID)
        self.rawloader = RawLoader(expID)
        self.LUTs = self.rawloader.get_LUTs()
        self.channel_df = self.rawloader.channel_df
        self.channel_df.loc[:, 'min_val'] = 0.0
        self.channel_df.loc[:, 'max_val'] = 1.0
        self.n_channels = len(self.channel_df.index)
        self.sliders = create_sliders(self.channel_df)

        self.bitdepth = 16
        self.csv = os.path.join(self.base, WP_ID + '.csv')

        self.df = pd.read_csv(self.csv, index_col='i')
        self.i_max = self.df.index.size
        self.annotations = self.df.drop(columns=['GlobalID']).columns

        # Check if predictions have been made in droplet_df
        self.droplet_df = pd.read_csv(os.path.join(self.base, os.pardir, 'droplets.csv'), index_col='GlobalID')
        self.predictions = self.droplet_df.loc[
            self.df['GlobalID'], self.droplet_df.columns.str.startswith('PREDICTED')]

        self.frames = np.load(os.path.join(self.base, WP_ID + '.npy'))
        _, y, x, c = self.frames.shape
        self.df.loc[self.i_max, :] = pd.NA  # add last row with NA to df
        self.frames = np.vstack([self.frames, np.ones((1, y, x, c)) * 10])  # add gray frame as last frame

        for mode in self.annotations:
            self.mode = mode
            self.i = self.get_first_unlabeled(mode)
            if self.i < self.i_max:
                break

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

        progress = 100 * self.i / self.i_max
        progress_label = f'{self.i}/{self.i_max} frames annotated'
        placeholder = self.df.loc[self.i, self.mode]
        disable_input = self.i == self.i_max

        existing_anns = self.df.loc[self.i, :]
        if self.i < self.i_max:
            existing_anns = pd.concat([existing_anns, self.predictions.loc[self.df.loc[self.i, 'GlobalID'], :]])
            existing_anns = existing_anns.to_frame().rename(columns={0: 'Value'})
        existing_anns = existing_anns.reset_index().to_dict('records')

        return progress, progress_label, disable_input, placeholder, fig, existing_anns

    def next_frame(self, skip_annotated=False):
        self.i = min(self.i + 1, self.i_max)

        while skip_annotated and self.i < self.i_max:
            if np.any(self.df.loc[self.i, :] == 10):
                self.i = min(self.i + 1, self.i_max)
                continue
            if self.df.isnull().loc[self.i, self.mode]:
                break
            self.i = min(self.i + 1, self.i_max)

        return self.current_frame()

    def previous_frame(self):
        self.i = max(self.i - 1, 0)
        return self.current_frame()

    def switch_mode(self, new_mode):
        self.i = self.get_first_unlabeled(new_mode)
        self.mode = new_mode
        return self.current_frame()

    def get_first_unlabeled(self, mode):
        return self.df.loc[:, mode].isnull().idxmax()

    def save_inputs(self, user_input):
        if user_input != 'nan' and user_input is not None and self.i < self.i_max:
            self.df.loc[self.i, self.mode] = user_input
        self.df.iloc[0:self.i_max, :].to_csv(self.csv)

dbm = DbManager()
wp = None

app = Dash(external_stylesheets=[dbc.themes.LUX, ])

app.layout = html.Div(children=[
    html.Div([
        html.H1(id='header', children=html.Center('WP Annotator'), className='header'),

        dcc.Dropdown(id='WP', options=dbm.existing_wps['expID'] + ' ' + dbm.existing_wps['WP_ID'], clearable=False,
                     placeholder='Select Workpackage', className='dropDownMenu'),

        dcc.RadioItems(id='annotation_mode', inline=False, className='modeSelector'),

        dash_table.DataTable(id='existing_annotations_table',
                             style_cell={'textAlign': 'left', 'fontSize': '12px', 'backgroundColor': '#333333',
                                         'border': '#333333'},),
        dash_table.DataTable(id='existing_predictions_table',
                             style_cell={'textAlign': 'left', 'fontSize': '12px', 'backgroundColor': '#333333',
                                         'border': '#333333'}),
    ], className='sidePanel'),


    html.Div([
        dbc.Progress(
            id='progress',
            label=f'{0}/{0} frames annotated',
            value=0,
            className='progressBar'
        ),

        # Image plot
        dcc.Graph(id='frame_image', className='graph-container'),

        dbc.Row([
            dbc.Button('Back', id='go_back', n_clicks=0, color='Primary'),
            dcc.Input(
                id="input_field",
                type='number',
                placeholder="Number of cells",
                min=0,
                max=10,
                debounce=True,
                autoComplete='sus'
            ),
            dbc.Button('Next', id='go_next', n_clicks=0, color='Primary'),
            dbc.Button('Skip annotated', id='skip_annotated', n_clicks=0, color='Primary'),
        ], className='inputRow'
        ),

        html.Div(id='slider_container'),

    ], className='mainPanel'),
    ],
)


@app.callback(
    Output(component_id='progress', component_property='label', allow_duplicate=True),
    Output(component_id='progress', component_property='value', allow_duplicate=True),
    Output(component_id='annotation_mode', component_property='options', allow_duplicate=True),
    Output(component_id='annotation_mode', component_property='value', allow_duplicate=True),
    Output(component_id='frame_image', component_property='figure', allow_duplicate=True),
    Output(component_id='slider_container', component_property='children'),
    Input('WP', 'value'),
    prevent_initial_call=True
)
def set_wp(select):
    expID, WP_ID = select.split(' ')
    global wp
    wp = WP(expID, WP_ID)

    progress, progress_label, disable_input, placeholder, fig, existing_annotations = wp.current_frame()
    return progress_label, progress, wp.annotations, wp.mode, fig, wp.sliders


@app.callback(
    Output(component_id='progress', component_property='value'),
    Output(component_id='progress', component_property='label'),
    Output(component_id='input_field', component_property='disabled'),
    Output(component_id='input_field', component_property='value'),
    Output(component_id='frame_image', component_property='figure'),
    Output(component_id='existing_annotations_table', component_property='data'),
    Input(component_id='input_field', component_property='value'),
    Input(component_id='go_back', component_property='n_clicks'),
    Input(component_id='go_next', component_property='n_clicks'),
    Input(component_id='skip_annotated', component_property='n_clicks'),
    Input(component_id='annotation_mode', component_property='value'),
    [Input({'type': 'dynamic-slider', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def update(user_input, goback, gonext, skip_annotated, new_mode, slider_values):
    global wp
    # What did trigger the callback?
    # User switched the annotation mode. Get the first un-annotated image and jump to this image
    if ctx.triggered_id == 'annotation_mode':
        return wp.switch_mode(new_mode)

    # User pressed back button
    elif ctx.triggered_id == 'go_back':
        return wp.previous_frame()

    elif ctx.triggered_id == 'input_field':
        wp.save_inputs(user_input)
        return wp.next_frame(skip_annotated=True)

    elif ctx.triggered_id == 'go_next':
        wp.save_inputs(user_input)
        return wp.next_frame()

    elif ctx.triggered_id == 'skip_annotated':
        wp.save_inputs(user_input)
        return wp.next_frame(skip_annotated=True)

    else:
        for i, (min_val, max_val) in enumerate(slider_values):
            wp.channel_df.loc[i, 'min_val'] = min_val
            wp.channel_df.loc[i, 'max_val'] = max_val
        return wp.current_frame()


if __name__ == '__main__':
    app.run_server(debug=True)
