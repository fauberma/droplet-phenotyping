import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from skimage import filters
from skimage.transform import hough_circle, hough_circle_peaks
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
            self.conditions = self.frame_df.columns.drop(['image_index', 't_index', 'path', 'droplet_size', 'size_range'], errors='ignore').to_list()
            try:
                channel_df = pd.read_excel(os.path.join(self.dir, 'setup.xlsx'), sheet_name='channels', index_col='channel_index')
                self.handler = LeicaHandler(self.frame_df, channel_df)
            except:
                self.handler = LeicaHandler(self.frame_df, channel_df=None)
                with pd.ExcelWriter(os.path.join(self.dir, 'setup.xlsx'), mode='a') as writer:
                    self.handler.channel_df.to_excel(writer, sheet_name='channels', index=True)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.connect_db()
    
    def connect_db(self):
        self.db = DbManager(self.exp_id)

    def get_droplet_df(self):
        df = self.db.get_droplets().astype({'outlier': bool})
        if df.empty:
            print('No droplets were detected yet.')
            return None
        return df.join(self.frame_df[self.conditions], on='frame_id')
    
    def get_annotations(self, source, remove_duplicates=False):
        df = self.db.get_annotations(source=source)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if remove_duplicates:
            df = df.sort_values('timestamp', ascending=False).drop_duplicates(subset=['droplet_id', 'label_type', 'source'])
        
        return df.pivot(index='droplet_id', columns='label_type', values='value').join(self.get_droplet_df())

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

    def detect_droplets_legacy(self, mode='sweep'):
        df_list = []
        for frame_id in self.frame_df.index:
            frame, meta = self.handler.get_frame(frame_id)
            bg_subtracted = frame[0] - filters.gaussian(frame[0], 3)
            t = filters.threshold_otsu(bg_subtracted)
            binary = bg_subtracted < t
            del frame, bg_subtracted
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
                df['x_shape'], df['y_shape'], df['frame_id'] \
                    = df['x_max'] - df['x_min'], df['y_max'] - df['y_min'], meta['frame_id']
            df_list.append(df)
        
        droplet_df = pd.concat(df_list, ignore_index=True)
        droplet_df.index.name='droplet_id'
        droplet_df.reset_index(inplace=True)
        droplet_df['outlier'] = False

        self.db.add_droplets(droplet_df)

        tfrecord_manifest = {
            'n_frames': droplet_df.index.size,
            'y_shape': 128,
            'x_shape': 128,
            'n_channels': self.handler.channel_df.index.size,
            'channel_info': self.handler.channel_df['channel_name'].to_dict(),
        }
        
        def frame_generator():
            for frame_id in self.frame_df.index:
                yield self.handler.get_frame(frame_id)

        self.db.add_dataset(frame_generator(), tfrecord_manifest)

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

    def generate_wp(self, labels, size, subset_query=None, droplet_ids=None):
        if droplet_ids is None:
            droplet_df = self.get_droplet_df()
            existing_annotations = self.db.get_annotations(source='manual')
            droplet_df.drop(index=existing_annotations['droplet_id'].unique(), inplace=True)
            if subset_query is not None:
                droplet_df = droplet_df.query(subset_query)
            selection = droplet_df.sample(size).index
        else:
            selection = droplet_ids

        ap = pd.DataFrame(index=selection, columns=labels).reset_index().rename_axis(index='i')
        frames = np.moveaxis(self.db.filter_dataset(ap['droplet_id']), 3, 1)

        ap_id = 'AP_1'
        while True:
            if os.path.isdir(os.path.join(self.dir, ap_id)):
                ap_id = 'AP_' + str(int(ap_id.split('_')[1]) + 1)
            else:
                break
        os.mkdir(os.path.join(self.dir, ap_id))
        ap.to_csv(os.path.join(self.dir, ap_id, f'{ap_id}.csv'))
        np.save(os.path.join(self.dir, ap_id, f'{ap_id}.npy'), frames)

        annotations = ap.melt(id_vars='droplet_id', var_name='label_type',value_name='value')
        annotations['ap_id'] = ap_id
        annotations['status'] = 'pending'
        self.db.add_annotations(annotations)

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