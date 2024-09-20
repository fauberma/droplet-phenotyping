import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from PIL import ImageDraw


def prepare_data(droplet_df, CNN_prefix):
    droplet_df.columns = droplet_df.columns.str.replace(CNN_prefix + '_', '')
    droplet_df['Total'] = droplet_df[['Target', 'Effector', 'dead_Target', 'dead_Effector']].sum(axis=1)
    droplet_df['Total_dead'] = droplet_df[['dead_Target', 'dead_Effector']].sum(axis=1)
    droplet_df['Total_Effector'] = droplet_df[['Effector','dead_Effector']].sum(axis=1)
    droplet_df['Total_Target'] = droplet_df[['Target','dead_Target']].sum(axis=1)
    return droplet_df


def get_pie(data, ax, return_legend=True):
    distribution = data.query('Effector == 1 & Total_Target >= 2').groupby('dead_Target').size() / data.index.size
    labels = distribution.index
    cmap = plt.get_cmap("OrRd")

    colors = cmap(np.linspace(0, 1, len(labels)))
    wedges, texts, = ax.pie(distribution.values, startangle=90, colors=colors)
    circle = Circle((0, 0), 1, edgecolor='gray', fill=False, linewidth=1)
    ax.add_artist(circle)
    if return_legend:
        patches = [Line2D([], [], marker="o", ms=10, ls="", mfc=colors[i], mec='gray', label=str(i)) for i in labels]
        return ax, patches
    else:
        return ax


def get_killing(data, groupby, viability='pooled'):
    base_effector = data.query('Total_Target == 0 & 1 <= Total_Effector <= 4').groupby(groupby).apply(lambda x: x['Effector'].sum()/x['Total_Effector'].sum(), include_groups=False)
    base_target = data.query('Total_Effector == 0 & 1 <= Total_Target <=4').groupby(groupby).apply(lambda x: x['Target'].sum()/x['Total_Target'].sum(), include_groups=False)
    if viability == 'single_cell':
        base_effector = data.query('Total_Target == 0 & Total_Effector == 1').groupby(groupby).apply(lambda x: x['Effector'].sum()/x['Total_Effector'].sum(), include_groups=False)
        base_target = data.query('Total_Effector == 0 & Total_Target == 1').groupby(groupby).apply(lambda x: x['Target'].sum()/x['Total_Target'].sum(), include_groups=False)

    raw_target = 1 - data.query('Total_Effector == 1 & 2 <= Total_Target <=4').groupby(groupby).apply(lambda x: x['Target'].sum()/x['Total_Target'].sum(), include_groups=False)
    net_target = raw_target - (1 - base_target)
    return base_effector, base_target, raw_target, net_target


def get_bubble(data, ax, factor=10):
    matrix, x_bins, y_bins = np.histogram2d(data['Total_Target'], data['Total_Effector'], bins=(5, 5), range=((0, 5), (0, 5)), density=True)
    x_coord, y_coord = np.indices(matrix.shape)
    ax.scatter(x_coord.flatten(), y_coord.flatten(), s=factor*100*matrix.flatten(), color='#e3655b', zorder=1)
    ax.set_xlim(-1, 5)
    ax.set_xticks(np.arange(0, 5))
    ax.set_xlabel('Target cells per droplet')

    ax.set_ylim(-1, 5)
    ax.set_yticks(np.arange(0, 5))
    ax.set_ylabel('Effector cells per droplet')
    return ax, matrix


def add_scale_bar(image, scale_factor, scale_length_um):
    # Open the image
    draw = ImageDraw.Draw(image)

    # Calculate the length of the scale bar in pixels
    scale_length_px = scale_length_um * scale_factor

    # Define the position and size of the scale bar
    bar_length = int(scale_length_px)
    bar_height = 20  # You can adjust the height of the scale bar
    padding = 20  # Padding from the edge of the image

    # Position the scale bar in the lower left corner
    image_width, image_height = image.size
    bar_x = padding
    bar_y = image_height - padding - bar_height

    # Draw the scale bar
    draw.rectangle([bar_x, bar_y, bar_x + bar_length, bar_y + bar_height], fill="black")
    return image


def adjust_dpi(image, desired_width_in_inches, output_path):
    # Calculate the DPI to achieve the desired width in inches
    image_width_px, image_height_px = image.size
    dpi = image_width_px / desired_width_in_inches

    # Save the image with the adjusted DPI
    image.save(output_path, dpi=(dpi, dpi))
    print(f"Image DPI adjusted to {dpi} and saved to {output_path}")


def panel_builder(frame, axs, LUTs, factors=None):
    frame = np.moveaxis(frame // 256, 2, 0)
    if factors is not None:
        frame = factors.reshape(4, 1, 1) * frame
        frame[frame > 255] = 255

    rgb_frames = np.array([LUT[channel] for channel, LUT in zip(frame, LUTs)])
    composite = np.sum(rgb_frames, axis=0, keepdims=True)
    rgb_frames = np.vstack([composite, rgb_frames])
    rgb_frames[rgb_frames > 255] = 255

    for channel, ax in zip(rgb_frames, axs.flatten()):
        ax.axis('off')
        ax.imshow(channel.astype(np.uint8))
    return axs