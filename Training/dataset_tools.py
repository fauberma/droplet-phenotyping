import sys
import os
import random

sys.path.append(os.path.join(os.getcwd(), os.pardir))

import tensorflow as tf
from Tools.db_tools import DbManager
from functools import partial

def get_annotation_tables(annotations):
    annotations = annotations.copy()
    labels = annotations.columns.tolist()
    annotations['key'] = annotations.index.map(lambda idx: f"{idx[0]}|{idx[1]}")
    keys = tf.constant(annotations['key'].values)

    lookup_tables = {}

    for label in labels:
        values = tf.constant(annotations[label].values, dtype=tf.int64)

        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1
        )
        lookup_tables[label] = table
    return lookup_tables

def prepare_data(element, lookup_tables):

    # Build composite key
    key = tf.strings.join([tf.as_string(element['droplet_id']), element['experiment_id']], separator='|')

    # Prepare image
    image = tf.cast(element['frame'], tf.float32)
    image = tf.gather(image, indices=[0, 2, 3], axis=2)
    image.set_shape([128, 128, 3])
    image = tf.math.log(image + 1)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

    # Lookup and one-hot encode labels
    outputs = {}
    for label in lookup_tables:
        raw_label = lookup_tables[label].lookup(key)
        outputs[label + "_output"] = tf.one_hot(raw_label, 5, dtype=tf.int64)
    element['input_layer'] = image
    return element, outputs

def get_filter_table(indices):
    indices = indices.copy()
    keys = indices.map(lambda idx: f"{idx[0]}|{idx[1]}")
    values = tf.ones(len(keys), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1
    )
    return table

def filter_dataset(element, lookup_table):
    key = tf.strings.join([tf.as_string(element['droplet_id']), element['experiment_id']], separator='|')
    val = lookup_table.lookup(key)
    return tf.not_equal(val, -1)

def df_to_dataset(df, alt_dataprep=None):
    tfrecord_files = []
    for droplet_id, experiment_id in df.index:
        tfrecord_files.append(os.path.join(os.getenv('DB_DIR'), experiment_id, f'{droplet_id // 1024}.tfrecord'))
    tfrecord_files = list(set(tfrecord_files))
    random.shuffle(tfrecord_files)

    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=5)
    ds = ds.map(DbManager.parse_function)

    # filtering
    filter_table = get_filter_table(df.index)
    ds = ds.filter(partial(filter_dataset, lookup_table=filter_table))

    # adding annotations and preparing model input and output data
    annotation_tables = get_annotation_tables(df)
    if alt_dataprep is None:
        ds = ds.map(partial(prepare_data, lookup_tables=annotation_tables))
    else:
        ds = ds.map(partial(alt_dataprep, lookup_tables=annotation_tables))
    return ds

if __name__ == '__main__':
    pass