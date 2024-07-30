__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'

import numpy as np
import pandas as pd
import os
import glob
from readlif.reader import LifFile
import datetime
from Setup.config import config


def parse_lif(file):
    lif = LifFile(file)
    timestamps = _recursive_memblock_is_image(lif.xml_root)
    image_df = []
    for i, im in enumerate(lif.image_list):
        image_ts = timestamps[i]
        merged = im['dims'][-1] == 1
        n_channels = im['channels']
        t_index = im['dims'][3]
        for t in range(t_index):
            image_df.append([i, im['name'], image_ts[t*n_channels], t , n_channels , im['bit_depth'][0], im['scale'][0], merged])
    image_df = pd.DataFrame(image_df, columns=['index', 'name', 'timestamp', 't_index', 'n_channels', 'bit_depth', 'resolution', 'merged'])
    return image_df


def _recursive_memblock_is_image(tree, return_list=None):
    """Creates list of TRUE or FALSE if memblock is image"""
    dmi8_offset = 13353548400 - 1709074800 #2393.02.27 - 2024.02.28
    if return_list is None:
        return_list = []

    children = tree.findall("./Children/Element")
    if len(children) < 1:  # Fix for 'first round'
        children = tree.findall("./Element")
    for item in children:
        has_sub_children = len(item.findall("./Children/Element/Data")) > 0
        is_image = (
                len(item.findall("./Data/Image")) > 0
        )
        # Check to see if the Memblock idnetified in the XML actually has a size,
        # otherwise it won't have an offset
        if int(item.find("./Memory").attrib["Size"]) > 0:
            ts = item.findall("./Data/Image/TimeStampList")[0].text.rstrip(' ').split(' ')
            timestamp = [datetime.datetime.fromtimestamp(int(int(t, 16) / 1e7) - dmi8_offset) for t in ts]
            return_list.append(timestamp)

        if has_sub_children:
            _recursive_memblock_is_image(item, return_list)

    return return_list


def _load_LUTs():
    LUTs = {}
    for file in glob.glob(os.path.join(os.getenv('SCRIPT_DIR'), 'Tools', 'LUTs', '*.npy')):
        name = os.path.basename(file)[:-4]
        LUTs[name] = np.load(file)
    return LUTs


class RawLoader:
    def __init__(self, expID):
        self.expID = expID
        self.an_dir = os.path.join(os.getenv('ANALYSES_DIR'), expID)
        self.param_df = pd.read_excel(os.path.join(self.an_dir, 'setup.xlsx'), sheet_name='parameters', index_col=0)
        self.channel_df = pd.read_excel(os.path.join(self.an_dir, 'setup.xlsx'), sheet_name='channels', index_col=0)
        self.frame_df = pd.read_excel(os.path.join(self.an_dir, 'setup.xlsx'), sheet_name='raw_data', index_col=0)
        if 'annotations' in self.param_df.index:
            _ = self.param_df.loc['annotations', 'Value']
            if isinstance(_, str):
                self.annotations = [_, ]
            else:
                self.annotations = list(_)
        else:
            self.annotations = []
        if 'conditional' in self.param_df.index:
            _ = self.param_df.loc['conditional', 'Value']
            if isinstance(_, str):
                self.conditions = [_, ]
            else:
                self.conditions = list(_)
        else:
            self.conditions = []

    def get_frame(self, frameID):
        meta = self.get_meta(frameID)
        lif = LifFile(meta['path'])
        image = lif.get_image(meta['image_index'])
        t = meta['t_index']
        if 'z_index' in meta.index:
            z = meta['z_index']
        else:
            z = 0

        frame = np.array([np.array(image.get_frame(c=c, t=t, z=z, m=0)) for c in self.channel_df.index])
        return frame, meta

    def get_meta(self, frameID):
        meta = self.frame_df.loc[frameID, :].copy()
        lif = LifFile(meta['path'])
        image = lif.get_image(meta['image_index'])
        meta['frameID'] = frameID
        meta['scale'] = image.scale[0]
        meta['bit_depth'] = image.bit_depth[0]
        meta['channels'] = self.channel_df.shape[0]
        return meta

    def get_LUTs(self):
        LUTs = _load_LUTs()
        return [LUTs[LUT] for LUT in self.channel_df['LUT']]

    def get_dropregister(self, as_multiindex=False):
        csv_path = os.path.join(self.an_dir, 'drop_register.csv')
        if os.path.isfile(csv_path):
            drop_register = pd.read_csv(csv_path, index_col='GlobalID').convert_dtypes()
            if as_multiindex:
                return drop_register.set_index(pd.MultiIndex.from_product([[self.expID], drop_register.index], names=['expID', 'GlobalID']))
            else:
                return drop_register
        else:
            print('No drop register exists yet.')

    def update_dropregister(self, drop_register):
        csv_path = os.path.join(self.an_dir, 'drop_register.csv')
        drop_register.to_csv(csv_path)

