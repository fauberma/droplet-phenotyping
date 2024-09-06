__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'


import numpy as np
import random
import pandas as pd
import tensorflow as tf
import keras
import re
import os
import glob
from functools import partial
import yaml
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.models import Model, load_model
from Tools.leica_tools import RawLoader
from Tools.sample_tools import Sample

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

class DbManager:
    def __init__(self):
        self.exp_dir = os.getenv('EXP_DIR')
        self.db_dir = os.getenv('DB_DIR')
        self.existing_dbs = self.load_dbs()
        self.existing_wps = self.load_wps()

    def generate_drop_register(self, expID, mode='constant'):
        rawloader = RawLoader(expID)
        df_list = []
        for frameID in rawloader.frame_df.index:
            df_list.append(Sample(expID, frameID).detect_droplets(mode=mode, return_df=True))
        drop_register = pd.concat(df_list).reset_index(drop=True).rename_axis('GlobalID')
        rawloader.update_dropregister(drop_register)

    def load_wps(self):
        wps = pd.DataFrame(columns=['expID', 'WP_ID', 'csv_file', 'annotated', 'labels'], dtype=object)
        files = glob.glob(os.path.join(self.exp_dir, '*', 'WP_*', '*.csv'))
        for i, file in enumerate(files):
            expID = re.search(os.getenv('expID_pattern'), file).group()
            WP_ID = re.search('WP_[0-9]', file).group()
            df = pd.read_csv(file, index_col='i')
            df.drop(columns=['GlobalID', ], inplace=True)
            labels = list(df.columns)
            annotated = 100 - 100 * df.isna().values.sum() // df.size
            wps.loc[i, ['expID', 'WP_ID', 'csv_file', 'annotated', 'labels']] = expID, WP_ID, file, annotated, labels
            wps.sort_values(by=['expID', 'WP_ID'], inplace=True)
            wps = wps.reset_index(drop=True)
        return wps

    def get_wps(self, expID, allow_partial=False, as_multiindex=False):
        rawloader = RawLoader(expID)
        wps = self.existing_wps.query(f'expID == "{expID}"')
        if wps.shape[0] > 0:
            df = pd.concat([pd.read_csv(csv) for csv in wps['csv_file']], keys=wps['WP_ID'].tolist(), names=['WP_ID', ])
            df.reset_index(level=0, inplace=True)
            if allow_partial:
                df.dropna(axis='index', how='all', subset=rawloader.annotations, inplace=True, ignore_index=True)
            else:
                df.dropna(axis='index', how='any', subset=rawloader.annotations, inplace=True, ignore_index=True)
            df = df.convert_dtypes()
            if as_multiindex:
                return df.set_index(pd.MultiIndex.from_product([[expID], df['GlobalID']], names=['expID', 'GlobalID']))
            else:
                return df
        else:
            print('No existing Work Packages found')
            return pd.DataFrame(columns=['WP_ID', 'i', 'GlobalID'])

    def generate_wp(self, expID, sample_size=100, exclude_query='index != index'):
        rawloader = RawLoader(expID)
        drop_register = rawloader.get_dropregister()
        existing_WPs = self.get_wps(expID)
        wpID = f'WP_{len(existing_WPs["WP_ID"].unique()) +1}'
        drop_register.drop(index=existing_WPs['GlobalID'], inplace=True)
        drop_register.drop(index=drop_register.query(exclude_query).index, inplace=True)
        selection = drop_register.groupby('frameID').sample(sample_size).index

        wp = pd.DataFrame(index=selection, columns=rawloader.annotations).reset_index().rename_axis(index='i')
        frames = np.moveaxis(self.filter_db(expID, wp['GlobalID']), 3, 1)

        os.mkdir(os.path.join(rawloader.exp_dir, wpID))
        wp.to_csv(os.path.join(rawloader.exp_dir, wpID, f'{wpID}.csv'))
        np.save(os.path.join(rawloader.exp_dir, wpID, f'{wpID}.npy'), frames)
        self.existing_wps = self.load_wps()

    def load_dbs(self):
        dbs = pd.DataFrame(columns=['expID', 'n_frames', 'y_shape', 'x_shape', 'n_channels', 'channel_info', 'tfrecord_annotations']).set_index('expID')
        spec_files = glob.glob(os.path.join(self.db_dir, '*', 'db_spec.yml'))
        for file in spec_files:
            with open(file, 'r') as f:
                data = yaml.safe_load(f)
            expID = re.search('[A-Z]{2,4}_[A-Z0-9]{2,4}_[0-9]{3}', file).group()
            dbs.loc[expID, :] = pd.Series(data)
        return dbs.sort_index()

    @staticmethod
    def serialize_example(image, dropletID, expID):
        serial_im = tf.io.serialize_tensor(image)
        feature = {
            'frame': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_im.numpy()])),
            'GlobalID': tf.train.Feature(int64_list=tf.train.Int64List(value=[dropletID])),
            'expID': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(expID)]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    @staticmethod
    def _parse_function(proto, tensor_shape):
        # Define your feature description
        feature_description = {
            'frame': tf.io.FixedLenFeature([], tf.string),
            'GlobalID': tf.io.FixedLenFeature([], tf.int64),
            'expID': tf.io.FixedLenFeature([], tf.string),
        }

        parsed_features = tf.io.parse_single_example(proto, feature_description)
        frame = tf.reshape(tf.io.parse_tensor(parsed_features['frame'], out_type=tf.uint16), tensor_shape)
        parsed_features['frame'] = frame
        return parsed_features

    def generate_tfrecord(self, expID, shape=(128, 128)):
        rawloader = RawLoader(expID)
        drop_register = rawloader.get_dropregister()
        drop_register['batch_id'] = drop_register.index // int(os.getenv('batch_size'))
        drop_register['batch_pos'] = drop_register.index % int(os.getenv('batch_size'))

        if not os.path.isdir(os.path.join(self.db_dir, expID)):
            os.mkdir(os.path.join(self.db_dir, expID))
        else:
            print('DB exists already')
            return

        data = {
            'n_frames': drop_register.index.size,
            'y_shape': shape[0],
            'x_shape': shape[1],
            'n_channels': rawloader.channel_df.index.size,
            'channel_info': rawloader.channel_df['channel_name'].to_dict(),
        }
        with open(os.path.join(os.getenv('DB_DIR'), expID, 'db_spec.yml'), 'w') as outfile:
            yaml.dump(data, outfile)

        frame, meta = rawloader.get_frame(0)
        for batch_id, batch_subset in drop_register.groupby('batch_id'):
            fname = os.path.join(self.db_dir, expID, f'{batch_id}.tfrecord')
            with tf.io.TFRecordWriter(fname) as writer:

                for frameID, frame_subset in batch_subset.groupby('frameID'):

                    if frameID != meta['frameID']:
                        frame, meta = rawloader.get_frame(frameID)

                    for dropID, drop in frame_subset.iterrows():
                        droplet_frame = np.moveaxis(frame[:, drop['y_min']:drop['y_max'], drop['x_min']:drop['x_max']], 0, 2)
                        preprocessed_image = tf.convert_to_tensor(droplet_frame, dtype=tf.uint16)
                        preprocessed_image = tf.cast(tf.image.resize(preprocessed_image, size=[shape[0], shape[1]]), tf.uint16)
                        writer.write(self.serialize_example(preprocessed_image, dropID, expID))
        self.existing_dbs = self.load_dbs()

    def get_dataset(self, expID, return_spec=False, shuffle=False):
        fnames = glob.glob(os.path.join(self.db_dir, expID, '*.tfrecord'))
        with open(os.path.join(self.db_dir, expID, 'db_spec.yml'), 'r') as file:
            db_spec = yaml.safe_load(file)
            shape = (db_spec['y_shape'], db_spec['x_shape'], db_spec['n_channels'])
        if shuffle:
            random.shuffle(fnames)
        dataset = tf.data.TFRecordDataset(fnames, num_parallel_reads=5)
        parsed_dataset = dataset.map(partial(self._parse_function, tensor_shape=shape))
        if return_spec:
            return parsed_dataset, db_spec
        else:
            return parsed_dataset

    def get_datasets(self, expIDs, shuffle=False):
        fnames = []
        for expID in expIDs:
            fnames.extend(glob.glob(os.path.join(self.db_dir, expID, '*.tfrecord')))
            with open(os.path.join(self.db_dir, expID, 'db_spec.yml'), 'r') as file:
                db_spec = yaml.safe_load(file)
                shape = (db_spec['y_shape'], db_spec['x_shape'], db_spec['n_channels'])
        if shuffle:
            random.shuffle(fnames)
        dataset = tf.data.TFRecordDataset(fnames)
        return dataset.map(partial(self._parse_function, tensor_shape=shape))

    def filter_db(self, expID, GlobalIDs):
        y, x, c = self.existing_dbs.loc[expID, ['y_shape', 'x_shape', 'n_channels']]
        df = pd.DataFrame(GlobalIDs, columns=['GlobalID']).reset_index()
        df['tfrecord'] = (df['GlobalID'] // int(os.getenv('batch_size'))).astype(int)
        frame_array = np.zeros((len(GlobalIDs), y, x, c), dtype=np.uint16)
        with open(os.path.join(self.db_dir, expID, 'db_spec.yml'), 'r') as file:
            db_spec = yaml.safe_load(file)
            shape = (db_spec['y_shape'], db_spec['x_shape'], db_spec['n_channels'])
        for tfrecord, subset in df.groupby('tfrecord'):
            dataset = tf.data.TFRecordDataset(os.path.join(os.getenv('DB_DIR'), expID, f'{tfrecord}.tfrecord'))
            parsed_dataset = dataset.map(partial(self._parse_function, tensor_shape=shape))
            for element in parsed_dataset.as_numpy_iterator():
                if element['GlobalID'] in subset['GlobalID'].values:
                    position = subset.query(f'GlobalID == {element["GlobalID"]}').index
                    frame_array[position, :, :, :] = element['frame']
        return frame_array

    def detect_outliers(self, expID, model_name):
        def prepare_inference(element):
            element['outlier_input'] = tf.cast(element['frame'][:, :, tf.constant(0)], tf.float32) / 65535
            return element

        rawloader = RawLoader(expID)
        drop_register = rawloader.get_dropregister()
        dataset = self.get_dataset(expID).map(prepare_inference).batch(32)
        model = keras.models.load_model(os.path.join(os.getenv('MODEL_DIR'), model_name))
        y_predict_raw = model.predict(dataset)
        globalIDs = [e['GlobalID'] for e in dataset.unbatch().as_numpy_iterator()]
        y_predict = np.argmax(y_predict_raw, axis=-1).astype(bool)
        drop_register.loc[globalIDs, 'outlier'] = y_predict
        rawloader.update_dropregister(drop_register)

        with PdfPages(os.path.join(rawloader.exp_dir, 'outlier_summary.pdf')) as pdf:
            for outlier in [True, False]:
                fig, axs = plt.subplots(figsize=(8,8), ncols=8, nrows=8)
                subset = drop_register.query(f'outlier == {outlier}').copy()
                size = subset.index.size
                IDs = subset.sample(min(size, 64)).index
                frames = self.filter_db(expID, IDs)[:, :, :, 0]
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

    def cell_count(self, expID, model_name):
        def prepare_data(element):
            image = tf.cast(element['frame'], tf.float32)
            image = tf.math.log(image+1)
            image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
            element['cell_count_input'] = image
            return element

        model = load_model(os.path.join(os.getenv('MODEL_DIR'), model_name))
        prefix = model_name.split('.')[0]
        tags = [layer.name[:-7] for layer in model.layers[-4:]]

        rawloader = RawLoader(expID)
        drop_register = rawloader.get_dropregister()
        ds = self.get_dataset(expID)
        globalIDs = np.array([element['GlobalID'] for element in ds.as_numpy_iterator()])

        dataset = ds.map(prepare_data).batch(32)
        y_predict = np.argmax(np.array(model.predict(dataset)), axis=-1).transpose()
        drop_register.loc[globalIDs, ['_'.join((prefix, tag)) for tag in tags]] = y_predict
        rawloader.update_dropregister(drop_register)

    def get_models(self, type):
        pass

    def get_expIDs(self):
        return ['NKIP_FA_066', 'NKIP_FA_067']

    def new_experiment(self, expID):
        pass


if __name__ == "__main__":
    pass
