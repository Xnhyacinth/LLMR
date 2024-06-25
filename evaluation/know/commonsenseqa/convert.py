import os
import sys
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    
    file = open('./csqa_wn_filtered_contriever.json', 'r', encoding='utf-8')
    lines = file.read()
    #print(type(lines))
    lines = json.loads(lines)
    #print(type(lines))
    file.close()
    result = []
    for line in lines:
        #line = json.loads(line)
        tmp = {}
        tmp["question"] = line["question"]["stem"]
        key = line["answerKey"]
        answer = None
        choices = []
        for choice in line["question"]["choices"]:
            if choice["label"] == key:
                answer = choice["text"]
            choices.append(choice["text"])
        tmp["choices"] = choices
        tmp["answer"] = answer
        tmp["knowledge"] = line["knowledge"]
        result.append(tmp)
    json_file = open('/data1/xdluo/alpaca-lora-main/data/commonsenseqa/csqa_wn_filtered_contriever2.json', mode='w')
    json.dump(result, json_file, indent=4) 