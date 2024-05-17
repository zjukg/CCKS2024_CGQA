from collections import defaultdict
import structllm as sllm
import json
import os
from tqdm import tqdm


class CronQuestionDataFormat:
    def __init__(self, question: str, relation_list: list, annotation: dict, answer: list = None, answer_type: str = None, _type: str = None):
        self.question = question
        self.relation_list = relation_list
        self.annotation = annotation
        self.answer = answer
        self.answer_type = answer_type
        self._type = _type
        
def _temp2CG_(tempkg_file):
    PAD = '[0]'
    triples_cg = set() 
    relations = set()
    entities_2_line = defaultdict(set)
    all_lines_id = set()
    
    with open(tempkg_file, 'r', )as f:
        table=[]
        for idx, line in enumerate(f.readlines()):                        
            elements = line.strip().split('\t')
            try:
                assert len(elements) == 5                
            except Exception as e:
                raise Exception(f'Fail to read {tempkg_file}, elements in row{idx+1} != 5: {line}')
            
            h, r, t, start_, end_ = elements
            triples_cg.add((h, r, PAD))
            triples_cg.add((r, t, h))
            entities_2_line[(t, r)].add(h)#尾实体+关系 对应的头实体(行) 有哪些
            
            triples_cg.add(('start_time', start_, (h,r,t)))
            triples_cg.add(((h,r,t), 'start_time', PAD))
            triples_cg.add(('end_time', end_, (h,r,t)))
            triples_cg.add(((h,r,t), 'end_time', PAD))
            entities_2_line[(start_, 'start_time')].add((h,r,t))
            entities_2_line[(end_, 'end_time')].add((h,r,t))
            
            for e_time in range(int(start_), int(end_)+1):
                triples_cg.add(('time', str(e_time), (h,r,t)))
                triples_cg.add(((h,r,t), 'time', PAD))
                entities_2_line[(str(e_time), 'time')].add((h,r,t))
            
            all_lines_id.add(h)
            all_lines_id.add((h,r,t))
            relations.add(r)
            relations.add('time')
            relations.add('start_time')
            relations.add('end_time')
            
    entities = list(entities_2_line.keys())
    return {
        'table_id': '1',
        'triples': triples_cg,
        'entities': entities,
        'entities_2_line': entities_2_line,
        'all_lines_id': all_lines_id, 
        'relations': list(relations),
    }

def get_tempKGquestion(data_path):
    
    with open(data_path, 'r') as fp:
        tb_question = json.loads(fp.read())

    # question_list = []
    # with open('output/temp/V6_all_v4prompt/all_result.txt', 'r') as fp:
    #     for line in fp.readlines():
    #         question_list.append(json.loads(line)['question'])
        

    KGQA_data = []
    for KG_id in tb_question.keys():
        question = tb_question[KG_id]['question']
        answer = tb_question[KG_id]['answer'][1]
        relations = tb_question[KG_id]['relations']
        
        answer_type = tb_question[KG_id]['answer_type']
        _type = tb_question[KG_id]['type']

        id2entity = tb_question[KG_id]['entities']
        annotation = tb_question[KG_id]['annotation']

        resEntity = [item for entity_key, item in id2entity.items() if entity_key not in annotation.values() ]
        
        if resEntity != []:
            # import pdb; pdb.set_trace()
            if len(resEntity) > 1 or '{tail2}' not in question:
                assert False, (resEntity, question)
                
            # 用resEntity中的第一个元素替换question中的{tail2}
            question = question.replace('{tail2}', resEntity[0])
            
            # print(f"KG_id:{KG_id}, question:{question}, resEntity:{resEntity}")

        # if _type != 'time_join' and _type != 'before_after' and _type != 'first_last':
        #     continue
        
        # if _type != 'first_last' or '\'s' in question or '\'' not in question:
        #     continue
        
        # if question in question_list:
        #     continue

        # if _type != 'before_after':
        #     continue

        relation_list = [item for key,item in relations.items()]

        for tmp_key in annotation.keys():
            if annotation[tmp_key][0] == 'Q' and annotation[tmp_key][1:].isdigit():
                annotation[tmp_key] = id2entity[annotation[tmp_key]]

        if resEntity == []:
            tmp_data = CronQuestionDataFormat(question, relation_list, annotation, answer, answer_type, _type)
        else:
            tmp_annotation = str(annotation)+", "+str(resEntity)
            tmp_data = CronQuestionDataFormat(question, relation_list, tmp_annotation, answer, answer_type, _type)
            
        KGQA_data.append(tmp_data)
        # print(f"question:{question}\n, relation_list:{relation_list}\n, annotation:{annotation}\n, answer:{answer}\n, answer_type:{answer_type}\n, type:{type}\n")
    
    return KGQA_data

def read_tempKG(args):
    print('read data...')
    test_table_ = _temp2CG_(args.folder_path)
    KG_data = sllm.cg.data(test_table_['triples'], test_table_['entities_2_line'], test_table_['all_lines_id'], if_temp=True)
    
    KGQA_data = get_tempKGquestion(args.data_path)
    if args.retriever is not None: # 需要检索demo
        assert args.train_folder_path != None, "train_folder_path is None"
        train_tkg_data = get_tempKGquestion(args.train_folder_path)
        train_data = dict()
        for item in train_tkg_data:
            train_data[item.question] = {
                "relation_list": item.relation_list,
                "annotation": item.annotation,
                # "answer":answer,
                # "answer_type":answer_type,
                # "_type":_type
            }
        args.train_data = train_data
        
    return KG_data, KGQA_data


class WQSPDataFormat:
    def __init__(self, question, TopicEntityName, First_step, Second_step, TopicEntityID, table_id=None, answer=None):
        self.question = question
        self.TopicEntityName = TopicEntityName
        self.First_step = First_step
        self.Second_step = Second_step
        self.TopicEntityID = TopicEntityID
        self.table_id = table_id
        self.answer = answer

def _kg2CG_(kg_file):
    PAD = '[0]'
    triples_cg = set() 
    relations = set()
    entities_2_line = defaultdict(set)
    all_lines_id = set()
    
    with open(kg_file, 'r', )as f:
        table=[]
        for idx, line in enumerate(f.readlines()):                        
            elements = line.strip().split('\t')
            try:
                assert len(elements) == 3                
            except Exception as e:
                raise Exception(f'Fail to read {kg_file}, elements in row{idx+1} != 3: {line}')
            
            h, r, t = elements
            triples_cg.add((h, r, PAD))
            triples_cg.add((r, t, h))
            entities_2_line[(t, r)].add(h)#尾实体+关系 对应的头实体(行) 有哪些
            all_lines_id.add(h)
            relations.add(r)
            
    entities = list(entities_2_line.keys())
    return {
        'table_id': '1',
        'triples': triples_cg,
        'entities': entities,
        'entities_2_line': entities_2_line,
        'all_lines_id': all_lines_id, 
        'relations': list(relations),
    }

def KG2CG(folder_path):

    KG_data = dict()
    file_names = os.listdir(folder_path) # 使用os.listdir()函数获取文件夹下所有文件和子文件夹的名称

    paths_of_files = [] # 所有csv文件路径
    for file_name in file_names:
        path_of_file = os.path.join(folder_path, file_name)
        paths_of_files.append(path_of_file)
    
    for path_of_file in tqdm(paths_of_files):
        table_name = path_of_file.split('/')[-1].split('.txt')[0]  # 最后一个/和.之间的字符串
        test_table_ = _kg2CG_(path_of_file)
        KG_data[table_name] = sllm.cg.data(test_table_['triples'], test_table_['entities_2_line'], test_table_['all_lines_id'])
    
    return KG_data

def get_wqsp_question(data_path):
    KGQA_data = []
    with open(data_path, 'r') as fp:
        for line in fp:
            obj = json.loads(line)
            
            table_id = obj['ID']
            question = obj['question']
            answers = obj['answers']
            answer = [ item[0][3:] if item[0][:3] == "ns:" else item[0] for item in answers ]
            # kg_tuples = obj['kg_tuples']
            TopicEntityName = obj['entities'][0][0]
            TopicEntityID = obj['entities'][0][1]
            First_step = obj['First_step']
            Second_step = obj['Second_step']
            KGQA_data.append(WQSPDataFormat(question, TopicEntityName, First_step, Second_step, TopicEntityID, table_id, answer))
            # KGQA_data.append((ID, question+'\n'+str([TopicEntityName])+'\n'+str([TopicEntityID]), answer, First_step, Second_step))
    
    return KGQA_data


def read_wqsp(args):
    print('read table data...')
    
    KG_data = KG2CG(args.folder_path)
    if "wqsp" in args.folder_path.lower() and args.retriever is not None:
        train_kg_data = KG2CG(args.folder_path.replace("test", "train"))
        KG_data.update(train_kg_data)

        dev_kg_data = KG2CG(args.folder_path.replace("test", "dev"))
        KG_data.update(dev_kg_data)

    args.table_data = KG_data
    KGQA_data = get_wqsp_question(args.data_path)

    return KG_data, KGQA_data