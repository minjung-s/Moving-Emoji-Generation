import os
import glob
import json
import numpy as np
import pandas as pd

disgust_list = glob.glob('landmark/landmark/disgust/*.json')
happiness_list = glob.glob('landmark/landmark/happiness/*.json')
surprise_list = glob.glob('landmark/landmark/surprise/*.json')


landmark_seq = np.array([0 for _ in range(6936)])
for file_name in surprise_list:
    f = open(file_name, 'r')
    json_data = json.load(f)
    length = len(json_data)
    if length < 53:
        continue
    sequence_arr = []
    for i in range(length-52):
        sequence = []
        for j in range(i,i+51):
            temp = json_data[str(j)].values()
            dim_1 = []
            for xy in temp:
                dim_1.extend(xy)
            sequence.extend(dim_1)
        sequence_arr.append(sequence)
    landmark_seq = np.vstack([landmark_seq, np.array(sequence_arr)])
    f.close()


df_surprise = pd.DataFrame(landmark_seq[1:,:])
df_surprise['condition'] = 2


df.to_csv("condition_landmarks.csv")
