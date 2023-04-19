import os
import shutil
import csv
from createdistanceimage import createimage

dirtocopy = "ai_001_001"
directory = "C:\\Users\\Kush\\OneDrive\\Desktop\\3D Vision\\Datasets\\Apple\\" + dirtocopy + "\\" + dirtocopy + "\\images"
#directory2 = "C:\\Users\\Kush\\OneDrive\\Desktop\\3D Vision\\Datasets\\Apple\\ai_001_002\\ai_001_002\\images\\scene_cam_00_geometry_preview"
newdirectory = "C:\\Users\\Kush\\OneDrive\\Desktop\\3D Vision\\Datasets\\AppleNew\\" + dirtocopy + "\\images"


l = os.listdir(directory)
lastsubl = l[-1]
num = int(lastsubl[10:12])
for i in range(num+1):
	if i < 10:
		subdir = 'scene_cam_' + str(0) + str(i)
		origsubdir = subdir + '_geometry_hdf5'
	elif i < 100:
		subdir = 'scene_cam_' + str(i)
		origsubdir = subdir + '_geometry_hdf5'
	subdirpath = os.path.join(directory, origsubdir)
	newsubdir = os.path.join(newdirectory, subdir)
	l = os.listdir(subdirpath)
	ldepth = l[0::10]
	for j in range(len(ldepth)):
		targetfile = ldepth[j][:-4] + 'png'
		fdepth = os.path.join(subdirpath, ldepth[j])
		targetpath = os.path.join(newsubdir, targetfile)
		createimage(fdepth, targetpath)
