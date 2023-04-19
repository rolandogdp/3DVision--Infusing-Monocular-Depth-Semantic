import h5py
import numpy
from PIL import Image
import numpy

def createimage(hddepthname, targetpath):
  #hddepthname = '/content/drive/MyDrive/Depth Estimation/ai_001_001/' + 'images/scene_cam_00_geometry_hdf5/frame.0032.depth_meters.hdf5'
  with h5py.File(hddepthname, "r") as f:
    a_group_key = list(f.keys())[0]
    ds_arr = f[a_group_key][()]  # returns as a numpy array

    intWidth = 1024
    intHeight = 768
    fltFocal = 886.81
    npyImageplaneX = numpy.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(numpy.float32)[:, :, None]
    npyImageplaneY = numpy.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(numpy.float32)[:, :, None]
    npyImageplaneZ = numpy.full([intHeight, intWidth, 1], fltFocal, numpy.float32)
    npyImageplane = numpy.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = ds_arr / numpy.linalg.norm(npyImageplane, 2, 2) * fltFocal

    npyDepth = npyDepth/10

    Depthimg = Image.fromarray(numpy.uint8(npyDepth * 255) , 'L')
    Depthimg.save(targetpath, 'PNG')