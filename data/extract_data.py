import pandas as pd 

import numpy as np 
import shutil
import os 


dataframe = pd.read_csv("./final_test_data.csv") 

output_dataframe = pd.DataFrame(columns = list(dataframe.columns)) 


destination_folder_name = "Final_Test_Data/" 

destination_folder = os.environ['THREED_VISION_ABSOLUTE_DOWNLOAD_PATH'] + destination_folder_name #need to create the folder 
for row in range(len(dataframe.index)):
	new_row = []
	for index, file in enumerate(dataframe.loc[row]): 
		file_name = file.split("/")[-1] 
		shutil.copy(file, destination_folder+file_name)
		new_row.append(destination_folder_name+file_name)	
	output_dataframe.loc[row] = new_row
		
		

print(output_dataframe)
print(os.environ["THREED_VISION_ABSOLUTE_DOWNLOAD_PATH"] + "/Final_Test_Data.csv")
output_dataframe.to_csv(os.environ["THREED_VISION_ABSOLUTE_DOWNLOAD_PATH"] + "Final_Test_Data.csv")

print("wrote dataframe to csv")

		
