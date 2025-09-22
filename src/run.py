import sys
import pandas as pd
import numpy as np
import os
import subprocess####

clean_list = pd.read_csv('data_seed_ratio.csv')




# ratio_data = ['yeast-2_vs_4', 'yeast-0-3-5-9_vs_7-8','yeast-0-2-5-6_vs_3-7-8-9', 'yeast-0-2-5-7-9_vs_3-6-8',
#  'yeast-0-5-6-7-9_vs_4', 'vowel0']

# for col, ser in clean_list.iterrows():  
#     dataset_name = f"{ser['dataname']}"
#     dataset_dir = os.path.join('node_vec_시각화', dataset_name)
    


#     if not os.path.exists(dataset_dir):
#         os.makedirs(dataset_dir)
#         print("폴더생성")


for col, ser in clean_list.iterrows():  
    # if ser['dataname'] in ratio_data:
    subprocess.run([sys.executable, 'main.py',
    '-config', f"config/{ser['dataname']}/idgl.yml",
    '-seed1','45',
    # '-fake_samples', 'True',
    # '-prompts', 'True',
    '-hidden_size',str(ser['hidden_size']),
    '-update_adj_ratio',str(ser['adj_ratio']),
    '-knn_size',str(ser['knn_size'])])


# for col, ser in clean_list.iterrows():  
#     if ser['dataname'] in ratio_data:
#         for k in [5,10,20,30,40,50,60,70,80,90,100]:
#             for i in range(10, 0, -1):
#                 subprocess.run([sys.executable, 'main.py',
#                 '-config', f"config/{ser['dataname']}/idgl.yml",
#                 '-seed1','45',
#                 '-ratio',f'{i / 100}',
#                 # '-fake_samples', 'True',
#                 # '-prompts', 'True',
#                 '-hidden_size',str(ser['hidden_size']),
#                 '-update_adj_ratio',str(ser['adj_ratio']),
#                 '-knn_size',f'{k}'])


# for col, ser in clean_list.iterrows():  
#     if ser['dataname'] in ratio_data:
#         for i in range(10, 0, -1):
#             subprocess.run([sys.executable, 'main.py',
#             '-config', f"config/{ser['dataname']}/idgl.yml",
#             '-seed1','45',
#             '-ratio',f'{i / 100}',
#             '-hidden_size',str(ser['hidden_size']),
#             '-update_adj_ratio',str(ser['adj_ratio']),
#             '-knn_size',str(ser['knn_size'])])


# ##GSL 만 실행

# clean_list = pd.read_csv('data_seed.csv')

# for col, ser in clean_list.iterrows():  
#     for hidden_size in [16,32,64]:
#         for adj_ratio in [1,0.9,0.8,0.7,0]:
        
#             subprocess.run([sys.executable, 'main.py',
#             '-config', f"config/{ser['dataname']}/idgl.yml",
#             '-seed1','45',
#             '-hidden_size',f'{hidden_size}',
#             '-update_adj_ratio',f'{adj_ratio}',
#             '-knn_size',str(ser['knn_size'])])


# for col, ser in clean_list.iterrows():  

#     for hidden_size in [16,32,64]:
#         for adj_ratio in [0.5]:
#             subprocess.run([sys.executable, 'main.py',
#             '-config', f"config/{ser['dataname']}/idgl.yml",
#             '-seed1','45',
#             '-fake_samples', 'True',
#             '-prompts', 'True',
#             '-hidden_size',f'{hidden_size}',
#             '-update_adj_ratio',f'{adj_ratio}',
#             '-knn_size',str(ser['knn_size'])])



import requests
text = "Tabular 실험 완료!"
requests.get(f'https://api.telegram.org/bot6106722436:AAE_kPbGzdEUVoKNChDeYQ7cto4pw3wiojo/sendMessage?chat_id=6162493277&text={text}')