import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
from skimage import filters
import logging
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dask import delayed, compute
from dask.distributed import Client
from PIL import Image, ImageDraw

from Tools.db_tools import DbManager
from Tools.leica_tools import LeicaHandler

# Configure the logger for the notebook
logger = logging.getLogger()  # Root logger
logger.setLevel(logging.INFO)

# StreamHandler for Jupyter Notebook
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Experiment:
    def __init__(self, expID, mode='leica'):
        self.expID = expID
        self.mode = mode
        self.exp_dir = os.getenv('EXP_DIR')
        if not self.exp_dir or not os.path.exists(self.exp_dir):
            raise ValueError("Invalid or missing EXP_DIR environment variable.")

        self.dir = os.path.join(self.exp_dir, expID)
        self.param_df = pd.read_excel(os.path.join(self.dir, 'setup.xlsx'), sheet_name='parameters', header=None, index_col=0)
        self.channel_df = pd.read_excel(os.path.join(self.dir, 'setup.xlsx'), sheet_name='channels', index_col=0)
        self.frame_df = pd.read_excel(os.path.join(self.dir, 'setup.xlsx'), sheet_name='raw_data', index_col=0)
        self.conditions = self.frame_df.columns.drop(['droplet_size', 'size_range', 'image_index', 't_index', 'path'], errors='ignore').to_list()
        self.annotations = self.param_df.loc['annotations', :].to_list()


        # So far only Leica Data Hamdler implemented
        handlers = {'leica': LeicaHandler(self.frame_df, self.channel_df)}
        if self.mode in handlers:
            self.handler = handlers[self.mode]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    def get_droplet_df(self, as_multiindex=False):
        csv_path = os.path.join(self.dir, 'droplets.csv')
        if os.path.isfile(csv_path):
            droplet_df = pd.read_csv(csv_path, index_col='GlobalID').convert_dtypes()
            if as_multiindex:
                return droplet_df.set_index(pd.MultiIndex.from_product([[self.expID], droplet_df.index], names=['expID', 'GlobalID']))
            else:
                return droplet_df
        else:
            print('No droplets were detected yet.')

    def update_droplet_df(self, droplet_df):
        csv_path = os.path.join(self.dir, 'droplets.csv')
        droplet_df.to_csv(csv_path)

    def preview_detect_droplets(self, frameID, mode):
        df = self._process_frame(frameID, mode)
        frame, meta = self.handler.get_frame(frameID)
        self.visualize_droplets(frame, df)

    def visualize_droplets(self, frame, df, factors=None, save='preview.png'):
        if factors is not None:
            frame = factors.reshape((-1, 1, 1)) * frame
        LUTs = self.handler.get_LUTs()
        array_8bit = np.clip(frame//256, a_min=0, a_max=255).astype(int)
        rgb = np.sum([LUT[channel] for LUT, channel in zip(LUTs, array_8bit)], axis=0)
        rgb = np.clip(rgb, a_min=0, a_max=255).astype(np.uint8)

        im = Image.fromarray(rgb)
        image_draw = ImageDraw.Draw(im)
        colors = {False: (255, 255, 255), True: (255, 0, 0)}
        for _, drop in df.iterrows():
            image_draw.rectangle(xy=(drop.x_min, drop.y_min, drop.x_max, drop.y_max),
                                 outline=colors[drop.get('outlier', False)], width=4)
        im.save(os.path.join(self.dir, save))

    def detect_droplets(self, mode: str = 'constant'):
        """
        Detects droplets in experiment frames using Hough transform.

        Parameters:
        - mode (str): Detection mode ('constant' or 'sweep').

        Returns:
        - pd.DataFrame: DataFrame containing detected droplet properties.
        """
        df_list = [self._process_frame(frameID, mode) for frameID in self.frame_df.index]

        droplets = pd.concat(df_list, ignore_index=True)
        droplets = droplets.merge(self.frame_df[self.conditions], on='frameID', how='left').rename_axis('GlobalID')
        self.update_droplet_df(droplets)
        DbManager().add_dataset(self)

    def track_droplets(self, track_by='time'):
        def visualize(data):
            fig, ax = plt.subplots(figsize=(15, 15),dpi=600)
            ax.scatter(data['x'], data['y'], c=data['time'], cmap='viridis',s=0.1,marker='x')
            plt.savefig('lol.png')

        df = self.get_droplet_df()
        assert track_by in self.conditions, f"Track by '{track_by}' is not a valid condition"
        groups = df.groupby([c for c in self.conditions if c != track_by])
        groups.apply(visualize)

    def detect_outliers(self, model_name):
        def prepare_inference(element):
            element['outlier_input'] = tf.cast(element['frame'][:, :, tf.constant(0)], tf.float32) / 65535
            return element

        dbm = DbManager()
        model = load_model(os.path.join(os.getenv('MODEL_DIR'), 'outlier_detection', model_name))
        ds = dbm.get_dataset(self.expID)
        globalIDs = np.array([element['GlobalID'] for element in ds.as_numpy_iterator()])
        dataset = ds.map(prepare_inference).batch(32)
        y_predict = np.argmax(model.predict(dataset), axis=-1).astype(bool)

        droplet_df = self.get_droplet_df()
        droplet_df.loc[globalIDs, 'outlier'] = y_predict
        self.update_droplet_df(droplet_df)

        with PdfPages(os.path.join(self.dir, 'outlier_summary.pdf')) as pdf:
            for outlier in [True, False]:
                fig, axs = plt.subplots(figsize=(8,8), ncols=8, nrows=8)
                subset = droplet_df.query(f'outlier == {outlier}').copy()
                size = subset.index.size
                IDs = subset.sample(min(size, 64)).index
                frames = dbm.filter_dataset(self.expID, IDs)[:, :, :, 0]
                for i, ax in enumerate(axs.flatten()):
                    ax.imshow(frames[i]/65535, cmap='gray', vmin=0, vmax=1)
                    ax.grid(False)
                    ax.set_yticks([])
                    ax.set_xticks([])
                if outlier:
                    fig.suptitle(f'Samples of detected outliers ({size} in total)', fontsize=15)
                else:
                    fig.suptitle(f'Samples of detected non-outliers ({size} in total)', fontsize=15)
                pdf.savefig(fig)
                plt.close()

    def cell_count(self, model_name):
        def prepare_data(element):
            image = tf.cast(element['frame'], tf.float32)
            image = tf.math.log(image+1)
            image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
            element['cell_count_input'] = image
            return element

        dbm = DbManager()
        model = load_model(os.path.join(os.getenv('MODEL_DIR'), 'cell_count', model_name))
        prefix = model_name.split('.')[0]
        tags = [layer.name[:-7] for layer in model.layers[-4:]]


        ds = dbm.get_dataset(self.expID)
        globalIDs = np.array([element['GlobalID'] for element in ds.as_numpy_iterator()])
        dataset = ds.map(prepare_data).batch(32)
        y_predict = np.argmax(np.array(model.predict(dataset)), axis=-1).transpose()

        droplet_df = self.get_droplet_df()
        droplet_df.loc[globalIDs, ['_'.join((prefix, tag)) for tag in tags]] = y_predict
        self.update_droplet_df(droplet_df)

    def generate_wp(self, sample_size=100, exclude_query='index != index'):
        if exclude_query == '':
            exclude_query = 'index != index'

        dbm = DbManager()
        droplet_df = self.get_droplet_df()
        existing_WPs = dbm.get_wps(self, filter_annotations='None')
        wpID = 'WP_' + str(dbm.existing_wps.query(f'expID == "{self.expID}"').index.size + 1)
        droplet_df.drop(index=existing_WPs['GlobalID'], inplace=True)
        droplet_df.drop(index=droplet_df.query(exclude_query).index, inplace=True)
        selection = droplet_df.groupby('frameID').sample(sample_size).index

        wp = pd.DataFrame(index=selection, columns=self.annotations).reset_index().rename_axis(index='i')
        frames = np.moveaxis(dbm.filter_dataset(self.expID, wp['GlobalID']), 3, 1)

        os.mkdir(os.path.join(self.dir, wpID))
        wp.to_csv(os.path.join(self.dir, wpID, f'{wpID}.csv'))
        np.save(os.path.join(self.dir, wpID, f'{wpID}.npy'), frames)

    def make_binary(self, frame: np.ndarray) -> np.ndarray:
        channel_index = 0
        bg_subtracted = frame[channel_index] - filters.gaussian(frame[channel_index], 3)
        t = filters.threshold_otsu(bg_subtracted)
        return bg_subtracted < t

    def _process_frame(self, frameID, mode):
        logging.info(f"Processing frame {frameID}")
        frame, meta = self.handler.get_frame(frameID)
        binary = self.make_binary(frame)
        del frame
        if mode == 'constant':
            radii = [(meta['droplet_size'] * meta['scale']) // 2]
        elif mode == 'sweep' and 'size_range' in meta.index:
            mean = (meta['droplet_size'] * meta['scale']) // 2
            span = (meta['size_range'] * meta['scale']) // 2
            radii = np.arange(mean - span, mean + span)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Use 'constant' or 'sweep'.")

        threshold = meta.get('threshold', 0.7)

        #Perform Hough circle detection
        hough_res = hough_circle(binary, radii)
        score, x, y, r = hough_circle_peaks(
            hough_res, radii,
            normalize=True,
            min_xdistance=int(radii[0] * 1.4),
            min_ydistance=int(radii[0] * 1.4),
            threshold=threshold
        )
        df = pd.DataFrame({'Score': score, 'x': x, 'y': y, 'r': r}).astype({'Score': float, 'x': int, 'y': int, 'r': int})
        df = df.query(f'y > r and {binary.shape[0]} - y > r and x > r and {binary.shape[1]} - x > r').reset_index()

        if not df.empty:
            df['x_min'], df['y_min'], df['x_max'], df['y_max'] \
                = df['x'] - df['r'], df['y'] - df['r'], df['x'] + df['r'], df['y'] + df['r']
            df['x_shape'], df['y_shape'], df['frameID'] \
                = df['x_max'] - df['x_min'], df['y_max'] - df['y_min'], meta['frameID']
        logging.info(f'{df.index.size} droplets in frame {meta["frameID"]} detected')
        return df

if __name__ == '__main__':
    pass