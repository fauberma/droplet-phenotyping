__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'

import numpy as np
import pandas as pd
import os
import glob
from readlif.reader import LifFile
import datetime
import xml.etree.ElementTree as ET

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


class LeicaHandler:
    def __init__(self, frame_df, channel_df=None):
        self.frame_df = frame_df

        if channel_df is None:
            liffile = LifFile(self.frame_df.loc[0, 'path'])
            # Extract unique channel info
            channels = []
            seen = set()
            for elem in liffile.xml_root.iter("WideFieldChannelInfo"):
                name = elem.attrib.get("UserDefName") or "Unnamed"
                if name not in seen:
                    seen.add(name)
                    channels.append({"channel_name": name, "LUT": elem.attrib.get("LUT")})

            # Create DataFrame
            self.channel_df = pd.DataFrame(channels)
            self.channel_df.index.name = "channel_index"
        else:
            self.channel_df = channel_df

    def get_frame(self, frame_id):
        meta = self.get_meta(frame_id)
        lif = LifFile(meta['path'])
        image = lif.get_image(meta['image_index'])
        t = meta['t_index']
        if 'z_index' in meta.index:
            z = meta['z_index']
        else:
            z = 0

        frame = np.array([np.array(image.get_frame(c=c, t=t, z=z, m=0)) for c in self.channel_df.index])
        return frame, meta

    def get_meta(self, frame_id):
        meta = self.frame_df.loc[frame_id, :].copy()
        lif = LifFile(meta['path'])
        image = lif.get_image(meta['image_index'])
        meta['frame_id'] = frame_id
        meta['scale'] = image.scale[0]
        meta['bit_depth'] = image.bit_depth[0]
        meta['channels'] = self.channel_df.index.size
        return meta

    def get_LUTs(self):
        LUTs = {}
        for file in glob.glob(os.path.join(os.path.dirname(__file__), 'LUTs', '*.npy')):
            name = os.path.basename(file)[:-4]
            LUTs[name] = np.load(file)
        return np.array([LUTs[LUT.lower()] for LUT in self.channel_df['LUT']])




