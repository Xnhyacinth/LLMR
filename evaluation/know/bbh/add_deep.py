'''
Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
Author: Xnhyacinth, Xnhyacinth@qq.com
Date: 2024-06-23 11:50:27
'''
import json
import re

with open('fewshot0.json', 'r') as json_file:
    data = json.load(json_file)
# Extract prompt



newdata = {}
for d in data.values():
    for item in d:
        item['rational_deep'] = item['rational'].replace("Let's think step by step", "Take a deep breath and work on this question step-by-step")
    
# Save the new dict as a JSON file
with open('fewshot.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("JSON file 'ruin_names.json' has been created.")