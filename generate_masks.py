import numpy as np
import os
import mrcfile
import cv2
import glob
import pandas as pd
import config

MRC_FILE_LOCATION = "test_dataset/10081/images"
CSV_FILE_LOCATION = "output/csv_files"
MASK_FILE_LOCATION = "Evaluation/General/VMPicker/10081/masks"
STAR_FILE_LOCATION = "output/star_files/10081.star"

coordinates_ = pd.read_csv(f'{STAR_FILE_LOCATION}', skiprows = 6) #Read Star File16
print(coordinates_.columns)

records = coordinates_["_rlnCoordinateY #3"]

records = records.iloc[1:]
# print(records.head())
columns_names = ['Micrographs Filename', 'X-Coordinate', 'Y-Coordinate']

df = pd.DataFrame()
micrograph_filename = []
x_coordinate = []
y_coordinate = []

for i in range(len(records)):
    # try:
    #     values = records[i].split("\t")
    # except:
    #     values = records[i].split(" ")
    values = records.iloc[i].split()
    # print(f"Processed values: {values}")
    micrograph_filename.append(values[0])
    print(values[0])
    x_coordinate.append(int(float(values[1])))
    y_coordinate.append(int(float(values[2])))


# import re
# for i in range(len(records)):
#     line = records.iloc[i]
    
#     match = re.match(r'^(.*?)\s(\d+)\s(\d+)\s(\d+)$', line.strip()) 
    
#     if match:
#         micrograph_filename_ = match.group(1).strip()  
#         x_coordinate_ = int(match.group(2))    
#         y_coordinate_ = int(match.group(3))    
#         diameter = int(match.group(4))         
        
#         micrograph_filename.append(micrograph_filename_)
#         print(micrograph_filename_)
#         x_coordinate.append(x_coordinate_)
#         y_coordinate.append(y_coordinate_)
#     else:
#         print(f"Line {i} does not match expected format: {line}")


# import re

# micrograph_filename = []
# x_coordinate = []
# y_coordinate = []

# for i in range(len(records)):
#     line = records.iloc[i]
#     line = str(line).strip()

#     # sb1_210512 pos 1042 1-2_1.mrc    2593.5    2064.0    -9999    -9999    0.9999869
#     match = re.match(r'^(.*?\.mrc)\s+([\d\.]+)\s+([\d\.]+)', line)

#     if match:
#         micrograph_filename_ = match.group(1).strip()
#         x_coordinate_ = int(float(match.group(2)))
#         y_coordinate_ = int(float(match.group(3)))

#         micrograph_filename.append(micrograph_filename_)
#         x_coordinate.append(x_coordinate_)
#         y_coordinate.append(y_coordinate_)

#         print(micrograph_filename_)
#     else:
#         print(f"Line {i} does not match expected format: {line}")

    
df.insert(0, columns_names[0], micrograph_filename)
df.insert(1, columns_names[1], x_coordinate)
df.insert(2, columns_names[2], y_coordinate)

try:
    os.makedirs(CSV_FILE_LOCATION)
except:
    pass

diameter = int(input("Please, enter diameter of protein in pixels \n"))
files = df['Micrographs Filename'].unique()
for f in files:
    f_name = f[:-4]
    df_box = df[df['Micrographs Filename'] == f]
    df_coord = df_box.loc[:, ['X-Coordinate', 'Y-Coordinate']]
    df_new = pd.DataFrame()
    col_names = ['X-Coordinate', 'Y-Coordinate', 'Diameter']
    x_coordinates = []
    y_coordinates = []
    diameters = []
    for index, row in df_coord.iterrows():
        x, y = row["X-Coordinate"], row["Y-Coordinate"]
        x_coordinates.append(x)
        y_coordinates.append(y)
        diameters.append(diameter)
    df_new.insert(0, col_names[0], x_coordinates)
    df_new.insert(1, col_names[1], y_coordinates)
    df_new.insert(2, col_names[2], diameters)
    df_new.to_csv(f"{CSV_FILE_LOCATION}/{f_name}.csv", index=False)
    
coordinate_files = glob.glob(f"{CSV_FILE_LOCATION}/*.csv")
for cf in coordinate_files:
    f_name = cf.split("/")[-1][:-4]
    micrograph_filename = f"{MRC_FILE_LOCATION}/{f_name}.jpg"
    print(f"Reading image from: {micrograph_filename}")
    
    image = cv2.imread(micrograph_filename, cv2.IMREAD_GRAYSCALE)
    # image = cv2.flip(image, 0)  
    if image is None:
        print(f"Error reading image: {micrograph_filename}")
        continue

    mask = np.zeros_like(image)
    try:
        coordinates = pd.read_csv(cf, usecols=[0, 1, 2])
        print(f"Coordinates read from: {cf}")
        for i, c in coordinates.iterrows():
            x = c['X-Coordinate']
            y = c['Y-Coordinate']
            r = int(c['Diameter'] / 2)
            coords = cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

        # Flip the mask vertically to correct the mirroring issue
        # flipped_mask = cv2.flip(mask, 0)
        
        mask_path = f"{MASK_FILE_LOCATION}/{f_name}_mask.jpg"
        cv2.imwrite(mask_path, coords)
        print(f'Successfully saved mask to: {mask_path}')
    except Exception as e:
        print(f"Error creating mask for {f_name}: {e}")


