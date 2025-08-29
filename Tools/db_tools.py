__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'


import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
import json
from dotenv import load_dotenv
import sqlite3

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

class DbManager:
    def __init__(self, exp_id):
        """Initialize DbManager with necessary directories."""
        self.exp_id = exp_id
        self.db_dir = os.getenv('DB_DIR')
        if not self.db_dir or not os.path.exists(self.db_dir):
            raise ValueError("Invalid or missing DB_DIR environment variable.")


        self.dir = os.path.join(self.db_dir, exp_id)
        os.makedirs(self.dir, exist_ok=True)
        self.conn = sqlite3.connect(os.path.join(self.dir, 'droplets.db'))
        self._initialize_schema()

    def _initialize_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS droplets (
            droplet_id INTEGER PRIMARY KEY,
            experiment_id TEXT,
            frame_id INTEGER,
            x_min INTEGER,
            y_min INTEGER,
            x_max INTEGER,
            y_max INTEGER,
            outlier BOOLEAN            
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            droplet_id INTEGER,
            experiment_id TEXT,
            label_type TEXT,
            value TEXT,
            ap_id TEXT,
            source TEXT,
            timestamp TEXT,
            status TEXT,
            FOREIGN KEY (droplet_id) REFERENCES droplets(droplet_id)
        )""")
        self.conn.commit()

    def add_droplets(self, df):
        """
        Insert one or more droplets into the database using executemany.

        Parameters
        ----------
        droplets : pd.Series or pd.DataFrame
            - pd.Series: one droplet
            - pd.DataFrame: multiple droplets
        """
        # Normalize to DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        elif not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pd.Series or pd.DataFrame")

        # Ensure experiment ID is present
        df["experiment_id"] = self.exp_id

        # Fill in required fields if missing (optional safety check)
        required =["droplet_id", "experiment_id", "frame_id", "x_min", "y_min", "x_max", "y_max", "outlier"]

        for field in required:
            if field not in df.columns:
                raise ValueError(f"Missing required droplet field: '{field}'")

        #records = df[required.keys()].to_records(index=False)
        records = list(df[required].itertuples(index=False, name=None))

        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO droplets (
                    droplet_id, experiment_id, frame_id, x_min, y_min, x_max, y_max, outlier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

    def get_droplets(self):
        query = "SELECT * FROM droplets WHERE experiment_id = ?"
        df = pd.read_sql(query, self.conn, params=(self.exp_id,), index_col='droplet_id')
        return df

    def add_annotations(self, annotations):
        """
        Add annotations to the database.

        Parameters
        ----------
        annotations : pd.Series or pd.DataFrame
            - pd.Series: represents one annotation (with all required fields)
            - pd.DataFrame: multiple annotations with the same schema
        """

        # Handle single-row case
        if isinstance(annotations, pd.Series):
            df = annotations.to_frame().T  # Convert to single-row DataFrame
        elif isinstance(annotations, pd.DataFrame):
            df = annotations
        else:
            raise TypeError("Expected a pd.Series (single row) or pd.DataFrame (multiple rows)")

        # Fill missing fields with defaults
        if "experiment_id" not in df.columns:
            df["experiment_id"] = self.exp_id
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp.now().isoformat()
        if "ap_id" not in df.columns:
            df["ap_id"] = None
        if "source" not in df.columns:
            df["source"] = "manual"
        if "status" not in df.columns:
            df["status"] = "completed"

        # Check required columns
        required = ["experiment_id", "droplet_id", "label_type", "value", "ap_id", "source", "timestamp", "status"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required annotation column: '{col}'")

        # Prepare and insert
        records = list(df[required].itertuples(index=False, name=None))

        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO annotations (
                    experiment_id, droplet_id, label_type, value,
                    ap_id, source, timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

    def update_annotation(self, annotation_id, value, status='completed', timestamp=None):
        """
        Update the value, status, and timestamp of an annotation using its annotation_id.
        """
        timestamp = timestamp or pd.Timestamp.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE annotations
            SET value = ?, timestamp = ?, status = ?
            WHERE annotation_id = ?
        """, (value, timestamp, status, annotation_id))
        
        self.conn.commit()

        if cursor.rowcount == 0:
            print(f"Warning: No annotation found with annotation_id = {annotation_id}")

    def get_annotations(self, droplet_ids=None, label_type=None, source=None):
        query = "SELECT * FROM annotations"
        clauses = []
        params = []

        if droplet_ids:
            placeholders = ','.join(['?'] * len(droplet_ids))
            clauses.append(f"droplet_id IN ({placeholders})")
            params.extend(droplet_ids)

        if label_type:
            if isinstance(label_type, (list, tuple)):
                placeholders = ','.join(['?'] * len(label_type))
                clauses.append(f"label_type IN ({placeholders})")
                params.extend(label_type)
            else:
                clauses.append("label_type = ?")
                params.append(label_type)

        if source:
            if isinstance(source, (list, tuple)):
                placeholders = ','.join(['?'] * len(source))
                clauses.append(f"source IN ({placeholders})")
                params.extend(source)
            else:
                clauses.append("source = ?")
                params.append(source)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        return pd.read_sql(query, self.conn, params=params)

    def remove_annotations(self, annotation_ids):
        if isinstance(annotation_ids, (int, str)):
                annotation_ids = [annotation_ids]

        placeholders = ','.join(['?'] * len(annotation_ids))
        query = f"DELETE FROM annotations WHERE annotation_id IN ({placeholders})"
        self.conn.execute(query, annotation_ids)
        self.conn.commit()

    def add_dataset(self, frame_generator, tfrecord_manifest):
        droplet_df = self.get_droplets()
        droplet_df['batch_id'] = droplet_df.index // 1024
        droplet_df['batch_pos'] = droplet_df.index % 1024

        self.update_manifest({'tfrecord': tfrecord_manifest})
        shape = (tfrecord_manifest['y_shape'], tfrecord_manifest['x_shape'])

        frame, meta = next(frame_generator)
        for batch_id, batch_subset in droplet_df.groupby('batch_id'):
            fname = os.path.join(self.dir, f'{batch_id}.tfrecord')
            with tf.io.TFRecordWriter(fname) as writer:

                for frame_id, frame_subset in batch_subset.groupby('frame_id'):

                    if frame_id != meta['frame_id']:
                        frame, meta = next(frame_generator)

                    assert frame_id == meta['frame_id'], 'Problems with frame_id ordering'

                    for droplet_id, drop in frame_subset.iterrows():
                        droplet_frame = np.moveaxis(frame[:, drop['y_min']:drop['y_max'], drop['x_min']:drop['x_max']], 0, 2)
                        preprocessed_image = tf.convert_to_tensor(droplet_frame, dtype=tf.uint16)
                        preprocessed_image = tf.cast(tf.image.resize(preprocessed_image, size=[shape[0], shape[1]]), tf.uint16)
                        writer.write(self.serialize_example(preprocessed_image, droplet_id=droplet_id, exp_id=self.exp_id))

    def get_dataset(self, num_parallel_reads=5):

        fnames = glob.glob(os.path.join(self.dir, '*.tfrecord'))
        if not fnames:
            raise FileNotFoundError(f"No TFRecord files found for experiment ID: {self.exp_id}")

        # Prepare dataset
        dataset = tf.data.TFRecordDataset(fnames, num_parallel_reads=num_parallel_reads)
        parsed_dataset = dataset.map(self.parse_function)
        return parsed_dataset

    def filter_dataset(self, droplet_ids):
        with open(os.path.join(self.dir, 'manifest.json'), 'r') as file:
            manifest = json.load(file)
        tfrecord_spec = manifest['tfrecord']
        y, x, c = (tfrecord_spec['y_shape'], tfrecord_spec['x_shape'], tfrecord_spec['n_channels'])

        df = pd.DataFrame(droplet_ids, columns=['droplet_id']).reset_index()
        df['tfrecord'] = (df['droplet_id'] // 1024).astype(int)
        frame_array = np.zeros((len(droplet_ids), y, x, c), dtype=np.uint16)

        for tfrecord, subset in df.groupby('tfrecord'):
            dataset = tf.data.TFRecordDataset(os.path.join(self.dir, f'{tfrecord}.tfrecord'))
            parsed_dataset = dataset.map(self.parse_function)
            for element in parsed_dataset.as_numpy_iterator():
                if element['droplet_id'] in subset['droplet_id'].values:
                    position = subset.query(f'droplet_id == {element["droplet_id"]}').index
                    frame_array[position, :, :, :] = element['frame']
        return frame_array

    def update_manifest(self, updates: dict):
        path = os.path.join(self.dir, "manifest.json")
        manifest = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                manifest = json.load(f)

        manifest.update(updates)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def get_manifest(self):
        path = os.path.join(self.dir, "manifest.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

    @staticmethod
    def serialize_example(image, droplet_id, exp_id):
        serial_im = tf.io.serialize_tensor(image)
        y_shape, x_shape, n_channels = image.shape
        feature = {
            'frame': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial_im.numpy()])),
            'droplet_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[droplet_id])),
            'experiment_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(exp_id)])),
            'y_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_shape])),
            'x_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[x_shape])),
            'n_channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[n_channels])),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    @staticmethod
    def parse_function(proto):
        # Define your feature description
        feature_description = {
            'frame': tf.io.FixedLenFeature([], tf.string),
            'droplet_id': tf.io.FixedLenFeature([], tf.int64),
            'experiment_id': tf.io.FixedLenFeature([], tf.string),
            'y_shape': tf.io.FixedLenFeature([], tf.int64),
            'x_shape': tf.io.FixedLenFeature([], tf.int64),
            'n_channels': tf.io.FixedLenFeature([], tf.int64),
        }

        parsed_features = tf.io.parse_single_example(proto, feature_description)
        tensor_shape = (parsed_features['y_shape'], parsed_features['x_shape'], parsed_features['n_channels'])
        frame = tf.reshape(tf.io.parse_tensor(parsed_features['frame'], out_type=tf.uint16), tensor_shape)
        parsed_features['frame'] = frame
        return parsed_features

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    pass
