import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

file_name = input() #변화시키고 싶은 랜드마크 json 파일
condition = input() # disgust, happiness, surprise

#landmark 불러오기
f = open(file_name, 'r')
json_data = json.load(f)
data = json_data[0]
input_landmark = []
for i in range(68):
    temp = data[str(i)]
    input_landmark.extend(temp)

t = np.array(input_landmark)
f.close()

#condition 정하기
cond_dict = {'disgust':0, 'happiness':1, 'surprise':2}
c = cond_dict[condition]

#자료형 변환
t = torch.tensor(np.exp(t/1024))
t = t.view(1,136)
t = t.type(torch.FloatTensor)

c = torch.tensor(c)
c = c.type(torch.FloatTensor)

#generator model
generator = torch.load('generator_exp.pytorch', map_location={'cuda:0':'cpu'}) #여기서 warning 생김
generator.eval()
outputs = generator(t,c)

#output 자료형 변환
outputs = outputs.view(50, -1) # 50시점
outputs = outputs.detach().numpy()
outputs = np.log(outputs)
outputs = outputs*1024

#landmark dict로 변환
landmark = dict()
for t in range(50):
    t_landmark = outputs[t]
    temp_dict = dict()
    for i in range(68):
        temp_dict[str(i)] = list(map(lambda x: int(round(x)),t_landmark[2*i:2*(i +1)]))

    landmark[str(t)] = temp_dict

#json 파일로 저장
f = open('save_json.json', 'w') #save_file_path = 'save_json.json'
json.dump(landmark, f)
f.close()
