__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'


import numpy as np
import random
import pandas as pd
import tensorflow as tf
import re
import os
import glob
from functools import partial
import yaml
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

class DbManager:
    def __init__(self):
        """Initialize DbManager with necessary directories."""
        self.exp_dir = os.getenv('EXP_DIR')
        self.db_dir = os.getenv('DB_DIR')
        self.model_dir = os.getenv('MODEL_DIR')
        if not self.exp_dir or not os.path.exists(self.exp_dir):
            raise ValueError("Invalid or missing EXP_DIR environment variable.")
        if not self.db_dir or not os.path.exists(self.db_dir):
            raise ValueError("Invalid or missing DB_DIR environment variable.")
        if not self.model_dir or not os.path.exists(self.model_dir):
            raise ValueError("Invalid or missing MODEL_DIR environment variable.")
        self.existing_dbs = self.load_datasets()
        self.existing_wps = self.load_wps()

    def load_wps(self) -> pd.DataFrame:
        files = glob.glob(os.path.join(self.exp_dir, '*', 'WP_*', '*.csv'))
        data = []
        for file in files:
            try:
                expID = re.search(os.getenv('expID_pattern'), file).group()
                WP_ID = re.search(r'WP_[0-9]+', file).group()
                df = pd.read_csv(file, index_col='i')
                df.drop(columns=['GlobalID'], errors='ignore', inplace=True)
                labels = list(df.columns)
                annotated = 100 - 100 * df.isna().values.sum() // df.size
                data.append([expID, WP_ID, file, annotated, labels])
            except (AttributeError, FileNotFoundError, pd.errors.EmptyDataError) as e:
                print(f"Error processing {file}: {e}")
        return (pd.DataFrame(data, columns=['expID', 'WP_ID', 'csv_file', 'annotated', 'labels'])
                .sort_values(by=['expID', 'WP_ID'])
                .reset_index(drop=True))

    def get_wps(self, experiment, filter_annotations='full', as_multiindex=False):
        wps = self.existing_wps.query(f'expID == "{experiment.expID}"')
        if wps.shape[0] > 0:
            df = pd.concat([pd.read_csv(csv) for csv in wps['csv_file']], keys=wps['WP_ID'].tolist(), names=['WP_ID', ])
            df.reset_index(level=0, inplace=True)
            if filter_annotations == 'full':
                df.dropna(axis='index', how='any', subset=experiment.annotations, inplace=True, ignore_index=True)
            elif filter_annotations == 'partial':
                df.dropna(axis='index', how='all', subset=experiment.annotations, inplace=True, ignore_index=True)
            df = df.convert_dtypes()
            if as_multiindex:
                return df.set_index(pd.MultiIndex.from_product([[experiment.expID], df['GlobalID']], names=['expID', 'GlobalID']))
            else:
                return df
        else:
            print('No existing Work Packages found')
            return pd.DataFrame(columns=['WP_ID', 'i', 'GlobalID'])

    def load_datasets(self):
        dbs = {}
        for file in glob.glob(os.path.join(self.db_dir, '*', 'db_spec.yml')):
            expID = re.search(os.getenv('expID_pattern'), file).group()
            with open(file, 'r') as f:
                dbs[expID] = yaml.safe_load(f)
        return pd.DataFrame(dbs).transpose().sort_index()

    def add_dataset(self, experiment, shape=(128, 128)): #Experiment object needs to be passed directly and cannot be created due to circular import issue
        droplet_df = experiment.get_droplet_df()
        droplet_df['batch_id'] = droplet_df.index // 1024
        droplet_df['batch_pos'] = droplet_df.index % 1024

        if not os.path.isdir(os.path.join(self.db_dir, experiment.expID)):
            os.mkdir(os.path.join(self.db_dir, experiment.expID))
        else:
            print('DB exists already')
            return

        data = {
            'n_frames': droplet_df.index.size,
            'y_shape': shape[0],
            'x_shape': shape[1],
            'n_channels': experiment.channel_df.index.size,
            'channel_info': experiment.channel_df['channel_name'].to_dict(),
        }
        with open(os.path.join(self.db_dir, experiment.expID, 'db_spec.yml'), 'w') as outfile:
            yaml.dump(data, outfile)

        frame, meta = experiment.handler.get_frame(0)
        for batch_id, batch_subset in droplet_df.groupby('batch_id'):
            fname = os.path.join(self.db_dir, experiment.expID, f'{batch_id}.tfrecord')
            with tf.io.TFRecordWriter(fname) as writer:

                for frameID, frame_subset in batch_subset.groupby('frameID'):

                    if frameID != meta['frameID']:
                        frame, meta = experiment.handler.get_frame(frameID)

                    for dropID, drop in frame_subset.iterrows():
                        droplet_frame = np.moveaxis(frame[:, drop['y_min']:drop['y_max'], drop['x_min']:drop['x_max']], 0, 2)
                        preprocessed_image = tf.convert_to_tensor(droplet_frame, dtype=tf.uint16)
                        preprocessed_image = tf.cast(tf.image.resize(preprocessed_image, size=[shape[0], shape[1]]), tf.uint16)
                        writer.write(self.serialize_example(preprocessed_image, dropID, experiment.expID))
        self.existing_dbs = self.load_datasets()

    def get_dataset(self, expIDs, return_spec=False, shuffle=False, num_parallel_reads=5):
        """
        Loads a TFRecord dataset for one or more experiment IDs, ensuring compatibility.

        Parameters:
        - expIDs (str or list of str): Single experiment ID or a list of experiment IDs.
        - return_spec (bool): Whether to return the dataset specifications.
        - shuffle (bool): Whether to shuffle the dataset files.
        - num_parallel_reads (int): Number of threads for parallel file reading.

        Returns:
        - tf.data.Dataset: Parsed dataset object.
        - list of dict (optional): Specifications for each dataset if `return_spec` is True.

        Raises:
        - ValueError: If `n_channels`, `x_shape`, or `y_shape` differ between datasets.
        """
        # Ensure expIDs is a list
        if isinstance(expIDs, str):
            expIDs = [expIDs]

        fnames = []
        specs = []
        shapes = None  # Placeholder to check shape consistency

        # Collect filenames and validate specs
        for expID in expIDs:
            exp_fnames = glob.glob(os.path.join(self.db_dir, expID, '*.tfrecord'))
            if not exp_fnames:
                raise FileNotFoundError(f"No TFRecord files found for experiment ID: {expID}")

            fnames.extend(exp_fnames)

            # Load the specification
            with open(os.path.join(self.db_dir, expID, 'db_spec.yml'), 'r') as file:
                db_spec = yaml.safe_load(file)
                specs.append(db_spec)

                # Validate shape consistency
                current_shape = (db_spec['y_shape'], db_spec['x_shape'], db_spec['n_channels'])
                if shapes is None:
                    shapes = current_shape
                elif shapes != current_shape:
                    raise ValueError(
                        f"Shape mismatch detected! Dataset {expID} has shape {current_shape}, "
                        f"but expected {shapes}."
                    )

        # Shuffle file names if requested
        if shuffle:
            random.shuffle(fnames)

        # Prepare dataset
        dataset = tf.data.TFRecordDataset(fnames, num_parallel_reads=num_parallel_reads)
        parsed_dataset = dataset.map(partial(self._parse_function, tensor_shape=shapes))

        if return_spec:
            return parsed_dataset, specs
        else:
            return parsed_dataset

    def filter_dataset(self, expID, GlobalIDs):
        y, x, c = self.existing_dbs.loc[expID, ['y_shape', 'x_shape', 'n_channels']]
        df = pd.DataFrame(GlobalIDs, columns=['GlobalID']).reset_index()
        df['tfrecord'] = (df['GlobalID'] // 1024).astype(int)
        frame_array = np.zeros((len(GlobalIDs), y, x, c), dtype=np.uint16)
        with open(os.path.join(self.db_dir, expID, 'db_spec.yml'), 'r') as file:
            db_spec = yaml.safe_load(file)
            shape = (db_spec['y_shape'], db_spec['x_shape'], db_spec['n_channels'])
        for tfrecord, subset in df.groupby('tfrecord'):
            dataset = tf.data.TFRecordDataset(os.path.join(self.db_dir, expID, f'{tfrecord}.tfrecord'))
            parsed_dataset = dataset.map(partial(self._parse_function, tensor_shape=shape))
            for element in parsed_dataset.as_numpy_iterator():
                if element['GlobalID'] in subset['GlobalID'].values:
                    position = subset.query(f'GlobalID == {element["GlobalID"]}').index
                    frame_array[position, :, :, :] = element['frame']
        return frame_array

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

    def get_models(self, model_type: str) -> list[str]:
        """Retrieve available models of the specified type."""
        models = glob.glob(os.path.join(self.model_dir, model_type, '*.h5'))
        return [os.path.basename(model).split('.')[0] for model in models]

    def get_experiments(self) -> list[str]:
        """Retrieve all experiment IDs based on the setup files."""
        pattern = os.getenv('expID_pattern')
        return sorted(
            re.search(pattern, file).group()
            for file in glob.glob(os.path.join(self.exp_dir, '*', 'setup.xlsx'))
            if re.search(pattern, file)
        )


if __name__ == "__main__":
    pass
