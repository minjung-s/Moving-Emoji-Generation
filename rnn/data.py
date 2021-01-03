import os
import glob
import json
import numpy as np
import pandas as pd

file_list = glob.glob('landmark/*.json')

landmark_seq = np.array([0 for _ in range(5032)])
for file_name in file_list:
    f = open(file_name, 'r')
    json_data = json.load(f)
    length = len(json_data)
    if length < 40:
        continue
    sequence_arr = []
    for i in range(length-40):
        sequence = []
        for j in range(i,i+37):
            temp = json_data[str(j)].values()
            dim_1 = []
            for xy in temp:
                dim_1.extend(xy)
            sequence.extend(dim_1)
        sequence_arr.append(sequence)
    landmark_seq = np.vstack([landmark_seq, np.array(sequence_arr)])
    f.close()


df = pd.DataFrame(landmark_seq[1:,:])
df.to_csv("landmarks.csv")
