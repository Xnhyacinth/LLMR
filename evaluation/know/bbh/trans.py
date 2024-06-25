'''
Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
Author: Xnhyacinth, Xnhyacinth@qq.com
Date: 2024-06-23 11:50:27
'''
import json
import re

with open('test_summary.json', 'r') as json_file:
    data = json.load(json_file)
# Extract prompt

ds = ['dyck_languages', 'multistep_arithmetic_two', 'object_counting', 'word_sorting']

newdata = {}
for d in data:
    if d['task_name'] not in ds:
        d['prefix_prompt'] = 'The following are multiple choice questions (with answers) about {subject}.'.format(subject=d['task_name'])
    elif d['task_name'] == 'multistep_arithmetic_two':
        d['prefix_prompt'] = 'The following are multi-step math questions about {subject}.'.format(subject=d['task_name'])
    elif d['task_name'] == 'word_sorting':
        d['prefix_prompt'] = 'The following are list of words sorting questions about {subject}.'.format(subject=d['task_name'])
    elif d['task_name'] == 'object_counting':
        d['prefix_prompt'] = 'The following are objects mentioned in the statement counting questions about {subject}.'.format(subject=d['task_name'])
    elif d['task_name'] == 'dyck_languages':
        d['prefix_prompt'] = 'The following are bracket matching questions that keeps the brackets closed about {subject}.'.format(subject=d['task_name'])
    
# Save the new dict as a JSON file
with open('test_summary_prefix.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("JSON file 'ruin_names.json' has been created.")