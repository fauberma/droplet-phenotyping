import sys
import os
sys.path.append(os.path.join(os.getcwd(), os.pardir))

import numpy as np
import pandas as pd
import tensorflow as tf
from dataset_tools import df_to_dataset
from Tools.exp_tools import Experiment

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy

def cell_count(inputs, cls_label):
    conv1 = Conv2D(32, (3, 3), activation='tanh', name=cls_label + '_conv1')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), name=cls_label + '_pool1')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='tanh', name=cls_label + '_conv2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name=cls_label + '_pool2')(conv2)

    conv3 = Conv2D(128, (3, 3), activation='tanh', name=cls_label + '_conv3')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), name=cls_label + '_pool3')(conv3)

    flatten = Flatten(name=cls_label + '_flatten')(pool3)

    dense1 = Dense(512, activation='tanh', name=cls_label + '_dense1')(flatten)
    dropout1 = Dropout(0.5, name=cls_label + '_dropout1')(dense1)

    dense2 = Dense(256, activation='tanh', name=cls_label + '_dense2')(dropout1)
    dropout2 = Dropout(0.5, name=cls_label + '_dropout2')(dense2)

    dense3 = Dense(128, activation='tanh', name=cls_label + '_dense3')(dropout2)
    dropout3 = Dropout(0.5, name=cls_label + '_dropout3')(dense3)

    output = Dense(5, activation='softmax', name=cls_label + '_output')(dropout3)
    return output
    #model = Model(inputs=inputs, outputs=output, name=cls_label + '_model')

    #return model

def get_model(labels):
    input = Input(shape=(128, 128, 3))
    outputs = [cell_count(input, label) for label in labels]

    model = Model(inputs=input, outputs=outputs)
    model.compile(
        optimizer=Adam(),
        loss={f"{label}_output": CategoricalCrossentropy() for label in labels},
        metrics={f"{label}_output": CategoricalAccuracy() for label in labels}
    )
    return model

def prepare_data(element, lookup_tables):
    droplet_id = element['droplet_id']
    experiment_id = element['experiment_id']

    # Build composite key
    key = tf.strings.join([tf.as_string(droplet_id), experiment_id], separator='|')

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

    return image, outputs

if __name__ == '__main__':



    exp_ids = ['NKIP_FA_052', 'NKIP_FA_053', 'NKIP_FA_055', 'NKIP_FA_056']
    experiments = [Experiment(exp_id) for exp_id in exp_ids]


    target = pd.concat([exp.db.get_annotations(label_type='target') for exp in experiments])
    dead_target = pd.concat([exp.db.get_annotations(label_type='dead_target') for exp in experiments])
    effector = pd.concat([exp.db.get_annotations(label_type='effector') for exp in experiments])
    dead_effector = pd.concat([exp.db.get_annotations(label_type='dead_effector') for exp in experiments])
    master_df = pd.concat([target, effector, dead_target, dead_effector]).pivot(columns='label_type', index=['droplet_id', 'experiment_id', 'source'], values='value')


    manual_data = master_df.query('source == "manual"').copy().reset_index(level=2,drop=True)
    manual_data = manual_data.loc[np.invert(manual_data.isna().any(axis=1))]
    manual_data = manual_data.astype({'target': int, 'effector': int, 'dead_target': int, 'dead_effector': int})
    manual_data.drop(index=manual_data.query('target == 10 | effector == 10 | dead_target == 10 | dead_effector == 10').index, inplace=True)


    cell_count_data = master_df.query('source == "cell_count_v2"').copy().reset_index(level=2,drop=True)
    cell_count_data = cell_count_data.astype({'target': int, 'effector': int, 'dead_target': int, 'dead_effector': int})
    cell_count_data.drop(manual_data.index, inplace=True)


    train_ds = df_to_dataset(cell_count_data, alt_dataprep=prepare_data)
    val_ds = df_to_dataset(manual_data, alt_dataprep=prepare_data)


    train_ds = train_ds.batch(32).prefetch(50)
    val_ds = val_ds.repeat(12).batch(32).prefetch(50)

    with tf.device('/cpu:0'):
        model = get_model(['target', 'effector', 'dead_target', 'dead_effector'])
        model.fit(train_ds, validation_data=val_ds, batch_size=32, steps_per_epoch=500, epochs=30, validation_steps=44)
        model.save(os.path.join(os.getenv('MODEL_DIR'), 'cell_count', 'cell_count_v4_2.keras'))