import argparse
import json
from collections import defaultdict
import numpy as np
from collections import Counter
import re

def get_selfconsistency_res(prediction: list):
    def find_most_common_except_LineAndZero(prediction):
        # 使用 Counter 统计元素出现次数
        counter = Counter(tuple(sublist) for sublist in prediction)
        most_common_elements = counter.most_common()
        # 检查列表长度和第一个元素是否是 '0'
        if len(most_common_elements) == 1 and most_common_elements[0][0] == ('0',):
            return prediction[0], 0
        else:
            if(most_common_elements[0][0] == ('0',)): 
                res = list(most_common_elements[1][0]) #如果最大是0返回次大
            else: 
                res = list(most_common_elements[0][0]) #返回最大值
        return res, prediction.index(res)

    # 得到prediction中出现次数最多的元素，如果有多个，返回第一个
    prediction = [['0'] if item == [] or item == 'None' or item == ['None'] or item == set() or item == [set()] or item == None or item == [None] or item == 'error' or item == ['error'] or item == {} else item for item in prediction]
    prediction = [ str(item) if type(item) == int or type(item) == float else item for item in prediction]
    prediction = [ [item] if type(item) == str else item for item in prediction]
    
    return find_most_common_except_LineAndZero(prediction)

def get_output_file(args):
    output_predict = dict()

    with open(args.ori_path, 'r') as f:
        for _idx, line in enumerate(f):
            line = json.loads(line.strip())
            question = line[list(line.keys())[0]]['question']
            predictions = line[list(line.keys())[0]]['prediction']
            prediction, idx = get_selfconsistency_res(predictions)
            output_predict[_idx] = prediction
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(output_predict, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default="./output/ccks_round1/all_result.txt")
    parser.add_argument('--output_path', type=str, default='./output/ccks_round1/output.txt')
    args = parser.parse_args()
    get_output_file(args)