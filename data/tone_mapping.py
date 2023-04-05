import pandas as pd
import h5py
import numpy as np


def tonemapping_single_image(self, rgb_image, render_entity_id):
    assert render_entity_id.any(0) #all(render_entity_id != 0)

    valid_mask = render_entity_id != -1

    if np.count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = 0.3 * rgb_image[:, :, 0] + 0.59 * rgb_image[:, :, 1] + 0.11 * rgb_image[:, :, 2]  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, self.percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:

            # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = np.power(self.brightness_nth_percentile_desired, self.inv_gamma) / brightness_nth_percentile_current

    return np.power(np.maximum(scale * rgb_image, 0), self.gamma)


def tonemapping(self, csv_file="downloads/image_files.csv"):
    #
    # compute brightness according to "CCIR601 YIQ" method, use CGIntrinsics strategy for tonemapping, see [1,2]
    # [1] https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py
    # [2] https://landofinterruptions.co.uk/manyshades
    #

    dataframe = pd.read_csv(csv_file, header=None, skiprows=[0])

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = 90  # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling


    for index, file in dataframe.iterrows():
        rgb_image_name = file[0]
        render_entity_name = file[3]
        rgb_color = h5py.File(rgb_image_name, "r")["dataset"][:].astype(np.float32)
        render_entity_id = h5py.File(render_entity_name, "r")["dataset"][:].astype(np.int32)

        toned_image = tonemapping_single_image(self, rgb_color, render_entity_id)

        output_file_name = rgb_color.replace(".color.hdf5", "tonemap.jpg")

        print("Saving output file: " + output_file_name)

        np.imsave(output_file_name, np.clip(toned_image, 0, 1))


class ToneMap:
    #
    # compute brightness according to "CCIR601 YIQ" method, use CGIntrinsics strategy for tonemapping, see [1,2]
    # [1] https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py
    # [2] https://landofinterruptions.co.uk/manyshades
    #

    def __init__(self):
        self.gamma = 1.0 / 2.2 # standard gamma correction exponent
        self.inv_gamma = 1.0 / self.gamma
        self.percentile = 90  # we want this percentile brightness value in the unmodified image...
        self.brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling


    tonemapping = tonemapping
    tonemapping_single_image = tonemapping_single_image