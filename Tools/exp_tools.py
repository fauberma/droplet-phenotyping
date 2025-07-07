import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import logging
from PIL import Image, ImageDraw
from Tools.db_tools import DbManager
from Tools.leica_tools import LeicaHandler

# Configure the logger for the notebook
logger = logging.getLogger()  # Root logger
logger.setLevel(logging.INFO)


class Experiment:
    def __init__(self, exp_id, mode='leica'):
        self.exp_id = exp_id
        self.mode = mode
        self.exp_dir = os.getenv('EXP_DIR')
        if not self.exp_dir or not os.path.exists(self.exp_dir):
            raise ValueError("Invalid or missing EXP_DIR environment variable.")

        self.dir = os.path.join(self.exp_dir, exp_id)

        if mode == 'leica':
            self.frame_df = pd.read_excel(os.path.join(self.dir, 'setup.xlsx'), sheet_name='raw_data', index_col=0)
            self.conditions = self.frame_df.columns.drop(['image_index', 't_index', 'path'], errors='ignore').to_list()
            try:
                channel_df = pd.read_excel(os.path.join(self.dir, 'setup.xlsx'), sheet_name='channels', index_col='channel_index')
                self.handler = LeicaHandler(self.frame_df, channel_df)
            except:
                self.handler = LeicaHandler(self.frame_df, channel_df=None)
                with pd.ExcelWriter(os.path.join(self.dir, 'setup.xlsx'), mode='a') as writer:
                    self.handler.channel_df.to_excel(writer, sheet_name='channels', index=True)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.db = DbManager(exp_id)

    def get_droplet_df(self):
        df = self.db.get_droplets()
        if df.empty:
            print('No droplets were detected yet.')
            return None
        return df.astype({'outlier': bool})

    def annotate_frame(self, frame_id):
        frame, meta = self.handler.get_frame(frame_id)
        df = self.get_droplet_df().query('frame_id == @frame_id')
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

    def detect_droplets(self, model_name, TILE_SIZE=512, OVERLAP=0.2):

        def suppress_bboxes(droplets):
            # Filter out droplets with zero width or height
            valid = (droplets["x_max"] > droplets["x_min"]) & (droplets["y_max"] > droplets["y_min"])
            droplets = droplets[valid].reset_index(drop=True)

            centers = np.stack([(droplets["x_min"] + droplets["x_max"]) / 2, (droplets["y_min"] + droplets["y_max"]) / 2], axis=0)
            sizes = np.stack([droplets["x_max"] - droplets["x_min"], droplets["y_max"] - droplets["y_min"]], axis=0)
            min_dist = np.sqrt(np.sum(np.square(sizes),axis=0))

            dists = np.sum(np.square(centers.reshape(2, -1, 1) - centers.reshape(2, 1, -1)), axis=0)
            np.fill_diagonal(dists, np.inf)
            conflicts = np.where(dists < min_dist.reshape(-1,1))

            droplets['keep'] = True
            for bbox1, bbox2 in zip(conflicts[0], conflicts[1]):
                if not droplets.at[bbox1, 'keep'] or not droplets.at[bbox2, 'keep']:
                    continue
                if droplets.at[bbox1, 'confidence'] > droplets.at[bbox2, 'confidence']:
                    droplets.at[bbox2, 'keep'] = False
                else:
                    droplets.at[bbox1, 'keep'] = False
            return droplets.query('keep == True').drop(columns=['keep'])

        def frame_generator():
            for frame_id in self.frame_df.index:
                yield self.handler.get_frame(frame_id)

        model = load_model(os.path.join(os.getenv('MODEL_DIR'), 'droplet_detection', model_name), compile=False)
        df_list = []
        for frame_id in self.frame_df.index:

            # divide full frame into sliding window coordinates
            frame, meta = self.handler.get_frame(frame_id)
            frame = (frame[0]/ 256).astype(np.uint8)
            frame = np.moveaxis(np.array([frame, frame, frame]),0,-1)
            coords = self.sliding_window(frame.shape[1], frame.shape[0], TILE_SIZE, OVERLAP)

            # model inference
            tiles = []
            for x_min, y_min, x_max, y_max in coords:
                tile = tf.constant(frame[y_min:y_max, x_min:x_max, :], dtype=tf.float32)
                tiles.append(tf.image.resize(tile, (512,512)))
            predictions = model.predict(tf.stack(tiles, axis=0))

            # parse predictions
            df = []
            for i, (n, (x_min, y_min, x_max, y_max)) in enumerate(zip(predictions['num_detections'],coords)):
                x_len = x_max-x_min
                y_len = y_max-y_min
                if n > 0:
                    bbox = predictions['boxes'][i][:n].copy()
                    bbox *= np.array([[x_len, y_len, x_len, y_len]])
                    bbox += np.array([[x_min, y_min, x_min, y_min]])
                    bbox = np.round(bbox).astype(int)
                    df.append(pd.DataFrame([bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3], predictions['confidence'][i][:n], predictions['classes'][i][:n]], index=['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'outlier']).transpose())
            df = pd.concat(df, ignore_index=True)
            df['frame_id'] = frame_id
            df[['x_min', 'x_max']] = df[['x_min', 'x_max']].clip(0, frame.shape[1])
            df[['y_min', 'y_max']] = df[['y_min', 'y_max']].clip(0, frame.shape[0])

            df = suppress_bboxes(df)
            df_list.append(df.drop(columns=['confidence']))

        droplet_df = pd.concat(df_list, ignore_index=True)
        droplet_df.index.name='droplet_id'
        droplet_df.reset_index(inplace=True)


        self.db.add_droplets(droplet_df)

        tfrecord_manifest = {
            'n_frames': droplet_df.index.size,
            'y_shape': 128,
            'x_shape': 128,
            'n_channels': self.handler.channel_df.index.size,
            'channel_info': self.handler.channel_df['channel_name'].to_dict(),
        }

        self.db.add_dataset(frame_generator(), tfrecord_manifest)

    def cell_count(self, model_name):
        def prepare_data(element):
            image = tf.cast(element['frame'], tf.float32)
            image = tf.math.log(image+1)
            image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
            element['cell_count_input'] = image
            return element

        model = load_model(os.path.join(os.getenv('MODEL_DIR'), 'cell_count', model_name), compile=False)
        prefix = model_name.split('.')[0]
        tags = [layer.name[:-7] for layer in model.layers[-4:]]


        ds = self.db.get_dataset()
        droplet_ids = np.array([element['droplet_id'] for element in ds.as_numpy_iterator()])
        dataset = ds.map(prepare_data).batch(32)
        y_predict = np.argmax(np.array(model.predict(dataset)), axis=-1).transpose()


        rows = []

        for droplet_id, droplet in zip(droplet_ids, y_predict):
            for tag, value in zip(tags, droplet):
                rows.append({
                    "droplet_id": droplet_id,
                    "label_type": tag,
                    "value": value,
                    "source": prefix,
                })

        annotation_df = pd.DataFrame(rows)
        self.db.add_annotations(annotation_df)

    def load_wps(self):
        # check for Work Packages with droplet annotations
        files = glob.glob(os.path.join(self.dir, 'WP_*', '*.csv'))
        data = []
        for file in files:
            WP_ID = re.search(r'WP_[0-9]+', file).group()
            df = pd.read_csv(file, index_col='i')
            df.drop(columns=['GlobalID'], errors='ignore', inplace=True)
            labels = list(df.columns)
            data.append([WP_ID, file, labels])

        return pd.DataFrame(data, columns=['WP_ID', 'csv_file', 'labels']).reset_index(drop=True)

    def generate_wp(self, sample_size=100, exclude_query='index != index'):
        if exclude_query == '':
            exclude_query = 'index != index'

        dbm = DbManager()
        droplet_df = self.get_droplet_df()
        existing_WPs = dbm.get_wps(self, filter_annotations='None')
        wpID = 'WP_' + str(dbm.existing_wps.query(f'exp_id == "{self.exp_id}"').index.size + 1)
        droplet_df.drop(index=existing_WPs['GlobalID'], inplace=True)
        droplet_df.drop(index=droplet_df.query(exclude_query).index, inplace=True)
        selection = droplet_df.groupby('frameID').sample(sample_size).index

        wp = pd.DataFrame(index=selection, columns=self.annotations).reset_index().rename_axis(index='i')
        frames = np.moveaxis(dbm.filter_dataset(self.exp_id, wp['global_id']), 3, 1)

        os.mkdir(os.path.join(self.dir, wpID))
        wp.to_csv(os.path.join(self.dir, wpID, f'{wpID}.csv'))
        np.save(os.path.join(self.dir, wpID, f'{wpID}.npy'), frames)

    @staticmethod
    def sliding_window(full_width, full_height, TILE=512, OVERLAP=0.2):

        coords = []
        for y_min in range(0, full_height, TILE - int(TILE * OVERLAP)):
            for x_min in range(0, full_width, TILE - int(TILE * OVERLAP)):
                x_max = min(full_width, x_min + TILE)
                y_max = min(full_height, y_min + TILE)
                coords.append((x_min, y_min, x_max, y_max))
        return coords

if __name__ == '__main__':
    pass