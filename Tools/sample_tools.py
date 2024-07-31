__author__ = 'Florian Aubermann'
__email__ = 'florian.aubermann@mr.mpg.de'
__status__ = 'development'


import numpy as np
import pandas as pd
import os
from skimage import filters
from skimage import color
from PIL import Image, ImageDraw
from skimage.transform import hough_circle, hough_circle_peaks
from PIL import Image
from Tools.leica_tools import RawLoader


class Sample:
    def __init__(self, expID, frameID):
        self.frameID = frameID
        self.expID = expID
        self.rawloader = RawLoader(expID)
        self.droplet_df = pd.DataFrame(columns=['Score', 'x', 'y', 'r', 'x_min', 'y_min', 'x_max', 'y_max'])

    def reload_droplets(self):
        drop_register = self.rawloader.get_dropregister()
        if drop_register is not None:
            selection = drop_register.query(f'frameID == {self.frameID}')
            self.droplet_df = selection.drop(columns='frameID')

    def detect_droplets(self, mode='constant', channel_index=0, return_df=True):
        frame, meta = self.rawloader.get_frame(self.frameID)

        if mode == 'constant':
            radii = [(meta['droplet_size'] * meta['scale']) // 2, ]
        elif mode == 'sweep' and 'size_range' in meta.index:
            mean = (meta['droplet_size'] * meta['scale']) // 2
            span = (meta['size_range'] * meta['scale']) // 2
            radii = np.arange(mean - span, mean + span)

        if 'threshold' in meta.index:
            threshold = meta['threshold']
        else:
            threshold = 0.7

        bg_subtracted = frame[channel_index] - filters.gaussian(frame[channel_index], 3)
        t = filters.threshold_otsu(bg_subtracted)
        binary = bg_subtracted < t

        self.droplet_df[['Score', 'x', 'y', 'r']] = self.hough_trans(binary, radii, threshold)[['Score', 'x', 'y', 'r']]
        self.droplet_df['x_min'] = self.droplet_df['x'] - self.droplet_df['r']
        self.droplet_df['y_min'] = self.droplet_df['y'] - self.droplet_df['r']
        self.droplet_df['x_max'] = self.droplet_df['x'] + self.droplet_df['r']
        self.droplet_df['y_max'] = self.droplet_df['y'] + self.droplet_df['r']
        self.droplet_df['x_shape'] = self.droplet_df['x_max'] - self.droplet_df['x_min']
        self.droplet_df['y_shape'] = self.droplet_df['y_max'] - self.droplet_df['y_min']
        self.droplet_df['frameID'] = self.frameID
        for condition in self.rawloader.conditions:
            self.droplet_df[condition] = meta[condition]
        print(f'{self.droplet_df.index.size} droplets in frame {self.frameID} detected \n')
        if return_df:
            return self.droplet_df

    def get_droplet(self, index):
        x_min, y_min, x_max, y_max = self.droplet_df.loc[index, ['x_min', 'y_min', 'x_max', 'y_max']].astype(int)
        frame, info = self.rawloader.get_frame(self.frameID)
        return frame[:, y_min:y_max, x_min:x_max]

    def visualize_droplets(self, channel, factors=None, save=True):
        if channel == 'composite':
            if factors is None:
                factors = np.ones(self.rawloader.channel_df.index.size)
            frame, meta = self.rawloader.get_frame(self.frameID)
            array_norm = factors.reshape((meta['channels'], 1, 1)) * frame
            array_8bit = array_norm // 256
            array_8bit[array_8bit > 255] = 255
            array_8bit = array_8bit.astype(int)
            rgb = np.zeros((frame.shape[1], frame.shape[2], 3))
            for LUT, channel in zip(self.rawloader.get_LUTs(), array_8bit):
                rgb += LUT[channel]
            rgb[rgb > 255] = 255
            im = Image.fromarray(rgb.astype(np.uint8))
        else:
            frame, meta = self.rawloader.get_frame(self.frameID)
            im = frame[channel].astype(float) / 2 ** (meta['bit_depth'] - 8)
            im = Image.fromarray(color.gray2rgb(im).astype(np.uint8))

        image_draw = ImageDraw.Draw(im)
        if 'outlier' not in self.droplet_df.columns:
            self.droplet_df['outlier'] = False
        colors = {False: (255, 255, 255), True: (255, 0, 0)}
        for dropID, drop in self.droplet_df.iterrows():
            image_draw.rectangle(xy=(drop.x_min, drop.y_min, drop.x_max, drop.y_max), outline=colors[drop['outlier']], width=4)
        if save:
            im.save(os.path.join(self.rawloader.an_dir, 'preview.png'))
        else:
            return im


    @staticmethod
    def hough_trans(bin_im, rad, threshold):
        score, x, y, r = hough_circle_peaks(hough_circle(bin_im, rad), rad,
                                            normalize=True,
                                            min_xdistance=int(rad[0] * 1.4),
                                            min_ydistance=int(rad[0] * 1.4),
                                            threshold=threshold)
        objects = (pd.DataFrame({'Score': score, 'x': x, 'y': y, 'r': r})
                   .astype({'Score': float, 'x': int, 'y': int, 'r': int}))
        objects = objects.query(f'y>r and {bin_im.shape[0]}-y>r and x>r and {bin_im.shape[1]}-x>r').reset_index()
        objects.sort_values(by=['y', 'x'], inplace=True)
        return objects

