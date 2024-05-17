import json 
import sys
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
PAD = '[0]'

def print_error(msg):
    print(msg,'  @  ',sys._getframe(1).f_code.co_filename,sys._getframe(1).f_lineno)

def str_to_real(str_num):
    if str_num == '':
        return ''
    if type(str_num) is str:
        str_num=str_num.replace('−','-')
    try:
        return float(str_num)
    except:
        return str_to_real_thou(str_num)
    
# 千分位
def str_to_real_thou(str_num, unit=1):
    # 只能有 -,.和数字
    if not (str_num.replace('-','').replace(',','').replace('.','').replace(' ','')).isdigit():
        return str_num
    
    if str_num[0]=='(' and str_num[-1]==')':
        str_num = str_num.replace('(','').replace(')','')
    try:
        if str_num[0] == '-':
            unit = -unit
            str_num = str_num[1:]
                
        if str_num[0] == '.':
            str_num = '0' + str_num
        
        if '.' in str_num and ',' in str_num:#点在逗号前面，交换 53.870,32
            if str_num.index('.') < str_num.index(','):
                str_num = str_num.replace(',','=')
                str_num = str_num.replace('.',',')
                str_num = str_num.replace('=','.')
        if '.' not in str_num and ',' in str_num:
            if len(str_num.split(',')[-1])!=3:
                # print_error(str_num)
                assert len(str_num.split(',')) == 2
                str_num = str_num.replace(',','.')
        if '.' in str_num:
            num, deci = str_num.split('.')
            deci_len = len(deci)
        else:
            num = str_num
            deci_len = 0
            deci = 0
        

        result = 0
        num = num.split(',')
        for idx, e in enumerate(num):
            if idx>0:
                assert len(e) == 3
            result = result*1000 + int(e)
        
        if deci_len!=0:
            result = result+int(deci)/pow(10, deci_len)
            result = result*unit
        else:
            result = int(result)
        return result
    except Exception as e:
        # print_error(e) 
        return str_num


def csv2CG(path, delimiter="\t"):
    triples = []
    table = np.loadtxt(path, dtype='str', delimiter = delimiter)
    header = table[0]
    
    for rows in table[1:]:
        key = rows[0]
        cols = header[1:]
        vals = rows[1:]
        triples += Table2CG(key, cols, vals)
    return triples

def txt2CG(path, delimiter="\t"):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            h, r, t = line.stripe().split(delimiter)
            triples += KG2CG((h, r, t))
    return triples

def KG2CG(triple):
    h, r, t = triple
    return [(h,r,PAD), (r, t, h)]

def Table2CG(key, cols, vals):
    # key : str 主键  
    # cols : list 一行属性
    # vals : list 对应属性值
    # return triple list
    result = []
    for c,v in zip(cols,vals):
        result.append((key, c, PAD))
        result.append((c, v, key))
    return result

def tb_to_csv():
    file_type = ['train','dev','test'][1]
    
    table_ = f'../../Unified_rep/WikiSQL/data/{file_type}.tables.jsonl'  
    out_dir = f'../dataset/WikiSQL_TB_csv/{file_type}/'
    
    with open(table_,'r') as fp:
        for jsonObj in tqdm(fp):#每一行
            table_data = json.loads(jsonObj)
            # 'id': '1-10015132-11'
            # 必须有个主键 默认第一列
            # 'header': ['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team']
            # 'rows': [['Antonio Lang', '21', 'United States', 'Guard-Forward', '1999-2000', 'Duke'], 
            #    ['Voshon Lenard', '2', 'United States', 'Guard', '2002-03', 'Minnesota'], 
            #    ['Martin Lewis', '32, 44', 'United States', 'Guard-Forward', '1996-97', 'Butler CC (KS)']]
            table_id = table_data['id']
            
            table = np.array([table_data['header']] + table_data['rows'])
            import pdb; pdb.set_trace()
            np.savetxt(out_dir + f"{table_id}.csv", table, fmt="%s", delimiter = "\t")
            
            
if __name__=="__main__":
    # tb_to_csv()
    path = f'../dataset/WikiSQL_TB_csv/dev/1-10015132-11.csv'
    triples = csv2CG(path)
    import pdb; pdb.set_trace()