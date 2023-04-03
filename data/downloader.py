import os
import argparse
import requests
import csv
import zipfile
zipfile.ZipExtFile.MIN_READ_SIZE = 2 ** 20

def get_scenes(list):
    scenes_list=[]
    frames_list = []

    a=True
    for s in list:
        if ".color.hdf5" in s:
            scenes_list.append(s[-35:-33])
            frames_list.append(s[-15:-11])

    
    return set(frames_list),set(scenes_list)


class WebFile:
    def __init__(self, url, session):
        with session.head(url) as response:
            size = int(response.headers["content-length"])

        self.url = url
        self.session = session
        self.offset = 0
        self.size = size

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def available(self):
        return self.size - self.offset

    def seek(self, offset, whence=0):
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset = min(self.offset + offset, self.size)
        elif whence == 2:
            self.offset = max(0, self.size + offset)

    def read(self, n=None):
        if n is None:
            n = self.available()
        else:
            n = min(n, self.available())

        end_inclusive = self.offset + n - 1

        headers = {
            "Range": f"bytes={self.offset}-{end_inclusive}",
        }

        with self.session.get(self.url, headers=headers) as response:
            data = response.content

        self.offset += len(data)

        return data

def download_files_me(url_list,max_file_size_GB,download_path="download/", max_frames=10):
    # Maybe todo, return a list of the files path/names, or a csv, for easier loading afterwards?
    
    #create csv files
    image_files_list = []; 
    
    downloaded_size = 0
    max_file_size_bytes = max_file_size_GB * (10**9)
    session = requests.session()
    for url in url_list:
        f = WebFile(url, session)
        z = zipfile.ZipFile(f)
        frames_list,cam_list = get_scenes(z.namelist())
        ai_name = url[-14:-4]
        for frame in list(frames_list)[:max_frames]:
            for cam in cam_list:
                rgb_file_name = f"{ai_name}/images/scene_cam_{cam}_final_hdf5/frame.{frame}.color.hdf5"
                depth_file_name = f"{ai_name}/images/scene_cam_{cam}_geometry_hdf5/frame.{frame}.depth_meters.hdf5"
                segmentation_file_name = f"{ai_name}/images/scene_cam_{cam}_geometry_hdf5/frame.{frame}.semantic.hdf5"
                files = [rgb_file_name, depth_file_name, segmentation_file_name]   
                image_files_list.append(files); 
                for file_name in files: 
                    try:
                        res = z.extract(file_name, download_path)           
                        path = os.path.join(download_path, file_name) 
                        downloaded_size+=os.path.getsize(path)
                        print(res)

                        if downloaded_size >= max_file_size_bytes:

                            print(f"Maximum download size reached: {downloaded_size/(10**9)} / {max_file_size_bytes/(10**9)}")

                            return 1
                    except KeyError:
                        continue
                    
                    
    with open(os.path.join(download_path,"image_files.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["RGB", "Depth", "Segmentation"])
        writer.writerows(image_files_list)       
