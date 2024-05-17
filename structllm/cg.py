
from collections import defaultdict, Counter
import argparse
import re, os
import sys 
from .format_ import str_to_real, print_error
from .llmfunction import LLMfunction


class data(object):
    def __init__(self, triples, entities_2_line, all_lines_id, if_temp=False):
        self.triples = triples 
        self.entity_2_line = entities_2_line
        self.all_lines_id = all_lines_id
        self.node2id = {}
        self.id2node = {}
        self.n1n2_to_c = defaultdict(set)
        self.n1c_to_n2 = defaultdict(set)
        
        self.n1_to_n2c0 = defaultdict(set)
        #head_entity --> relation # n1 对应的n2 需要c==[0]
        
        self.n1_to_n2c1 = defaultdict(set)
        # self.n1_to_n2 = defaultdict(set)# relation --> tail entity

        self.nodes = set()
        #relation/key --> tail_entity/value # n1 对应的n2 需要c!=[0]
        
        # self.get_id()
        self._get_triple()
        # n1 n2 c

        self.mapping_function = {
            "get_information": self.get_information_,
            "min":self.Min,
            "max":self.Max,
            "mean":self.Mean,
            "count":self.Count,
            "sum":self.Sum,
            'keep':self.Keep,
            "set_union":self.set_union_,
            "set_intersection":self.set_intersection_,
            "set_difference":self.set_difference_,
            "set_negation":self.set_negation,
            'previous_row':self.previous_row,
            'next_row':self.next_row,
        }
        self.time_rel_mark = ['time', 'start_time', 'end_time']
    # def _get_ddict(self):
    #     def fun():
    #         return defaultdict(set)
    #     return defaultdict(fun)
    def add_hrt_knowledge(self, triple_list_hrt):#list of triples
        
        triple_list = set()
        for h,r,t in triple_list_hrt:            
            triple_list.add((r, t, h))
            triple_list.add((h, r, '[0]'))
            
        self.triples = self.triples.union(triple_list)
        nodes = set(self.nodes)
        for n1, n2, c in triple_list:            
            for ele in [n1, n2, c]:
                if ele not in self.node2id:
                    id_ = str(len(self.node2id))
                    self.node2id[ele] = id_
                    self.id2node[id_] = ele
            nodes.add(n1)
            nodes.add(n2)            
            self.n1n2_to_c[(n1,n2)].add(c)
            self.n1c_to_n2[(n1,c)].add(n2)
            if c =='[0]':
                self.n1_to_n2c0[n1].add(n2)# head 2 relation
            else:
                self.n1_to_n2c1[n1].add((n2,c))# relation 2 tail, head
            # self.entity_2_line[(v, c)].add(key)
        self.nodes = list(nodes)
        
    
    def _get_triple(self):
        
        # print('load data ...')
        nodes = set()
        for n1, n2, c in self.triples:
            for ele in [n1, n2, c]:
                if ele not in self.node2id:
                    id_ = str(len(self.node2id))
                    self.node2id[ele] = id_
                    self.id2node[id_] = ele
            # 转成id
            # n1 = self.node2id[n1]
            # n2 = self.node2id[n2]
            # c = self.node2id[c]
            
            nodes.add(n1)
            nodes.add(n2)
            self.n1n2_to_c[(n1,n2)].add(c)
            self.n1c_to_n2[(n1,c)].add(n2)
            # if self.id2node[c]=='[0]':
            if c =='[0]':
                self.n1_to_n2c0[n1].add(n2)# head 2 relation
            else:
                self.n1_to_n2c1[n1].add((n2, c))# relation 2 tail
        self.nodes = list(nodes)

    def _search_node(self, node, condition=None, scope=None):#node是relation
        # scope 是 condition (头实体) 的集合
        #如果出现head_entity,不是主键,实际上是tail_entity,配合node(realtion/key)现找到行的id作为真的head_entity
        
        if condition:
            if type(condition) is set:
                temp = set()
                for es in condition:
                    if es not in self.all_lines_id:
                        temp |= self.entity_2_line[(es, node)]
                    else:
                        temp.add(es)
                condition = temp
            else:
                if condition not in self.all_lines_id:
                    condition = self.entity_2_line[(condition, node)]#变成集合(tail_entity, relation)                    
                
        if scope: 
            temp = set()          
            for es in scope:
                if es not in self.all_lines_id:
                    temp |= self.entity_2_line[(es, node)]
                else:
                    temp.add(es)
            scope = temp
            
            
        result = set()
        
        def one_n1_to_n2(node):
            if node in self.n1_to_n2c0:
                return self.n1_to_n2c0[node]
            elif node in self.n1_to_n2c1:
                return set([i[0] for i in self.n1_to_n2c1[node]])
            else:
                return {}
                    
        if scope is None:
            if condition is None:# 没给condition head_entity
                if type(node) is set:
                    for n in node:
                        result |= one_n1_to_n2(n)
                else:
                    result = one_n1_to_n2(node)
                
            else: # 给了condition
                if type(condition) is set:
                    for cond in condition:
                        result |= self.n1c_to_n2[(node, cond)]
                else:
                    result = self.n1c_to_n2[(node, condition)]
        else:
            for cond in scope:   
                result |= self.n1c_to_n2[(node, cond)]
                
        return result

    def _compare(self, cmp, target_value, candi_value):
        # 返回True or False
        if cmp == '=':
            #print(target_value, candi_value)
            target_value = str_to_real(target_value)
            candi_value = str_to_real(candi_value)
            #print('转换',target_value, candi_value)
            return candi_value == target_value
        
        try:
            target_value = str_to_real(target_value)
            candi_value = str_to_real(candi_value)
            assert type(target_value) in [int, float]
            assert type(candi_value) in [int, float]
        except Exception as e: # 不能转换成“数值”直接比较的
            raise Exception(f"出现非数值类，需要调用大模型. details: {e}")
            # # 调用gpt_function
        
        try:
            assert cmp in ['>','<','=','>=','<=']
            if cmp == '>':
                return candi_value > target_value
            elif cmp == '>=':
                return candi_value >= target_value
            elif cmp == '<':
                return candi_value < target_value
            elif cmp == '<=':
                return candi_value <= target_value
            
        except Exception as e:
            raise Exception(f"出现预设外的比较符，需要调用大模型. details: {e}")
            
        
        
    def _search_condition(self, node1, node2, cmp):
        result = set()
        # 特判：node2是most/least，直接返回
        if node2 in ['most','least']:
            values_ = [i[0] for i in self.n1_to_n2c1[node1]]
            condis_ = [i[1] for i in self.n1_to_n2c1[node1]]
            cnt_dc = Counter(values_)
            if node2=='most':
                match_val = [i for i in cnt_dc.keys() if cnt_dc[i]==max(cnt_dc.values())]
            elif node2=='least':
                match_val = [i for i in cnt_dc.keys() if cnt_dc[i]==min(cnt_dc.values())]
            
            return set([condis_[i] for i in range(len(condis_)) if values_[i] in match_val])
             

        if type(node2) not in [set, list]:
            node2 = [node2]
            
        for n2 in node2:#合并“=”后，node2可能是个set
            the_result = set()
            try:
                for value_, cond_ in self.n1_to_n2c1[node1]:
                    # print(f'compare {n2} and {value_}')
                    if self._compare(cmp, target_value=n2, candi_value=value_):
                        the_result.add(cond_)
            except Exception as e:
                print_error(e)
                # import pdb; pdb.set_trace()
                # 调用大模型完成比较
                llm_result = LLMfunction(args=self.args, question=self.question, query=self.queries, 
                                         task=self.task, step=self.step, mid_output=self.mid_output,
                                         candidate=self.n1_to_n2c1[node1])
                # todo 需要把llm返回的结果转成一个set
                # llm_result是个string 要处理成set格式
                the_result = set(llm_result)
        
            result |= the_result

        return result
    
    def set_union(self, set1, set2):
        return set1 | set2
    def set_union_(self, parameters):
        set1, set2 = parameters
        return self.set_union(set1, set2)
    
    def set_intersection(self, set1, set2, set3=None, set4=None, set5=None):
        # import pdb; pdb.set_trace()
        result = set1 & set2
        for se in [set3, set4, set5]:
            if se is not None:
                result &= se
        return result
    
    def set_intersection_(self, parameters):
        set1, set2, set3, set4, set5 = parameters
        return self.set_intersection(set1, set2, set3, set4, set5)
    
    def set_difference(self, set1, set2):
        if type(set2) is not set:
            if type(set2) is list:
                set2 = set(set2)
            else:# str int 等
                set2 = set([set2])          
        return set1 - set2
    
    def set_difference_(self, parameters):
        set1, set2 = parameters
        return self.set_difference(set1,set2)
    
    def set_negation(self, set1):
        # set1是condition集合，返回其他所有行
        return set(self.all_lines_id)-set1
    
    def Keep(self, parameters):
        set1, value, cmp = parameters
        return self.Keep_(set1, value, cmp)
    
    def Keep_(self, list1, value, cmp):
        if type(value) in [list, set]:
            value = list(value)[0]

        if cmp == '<':
            list1 = [str_to_real(i) for i in list1 if str_to_real(i) < str_to_real(value)]
        elif cmp == '>':
            list1 = [str_to_real(i) for i in list1 if str_to_real(i) > str_to_real(value)]
        elif cmp == '>=':
            list1 = [str_to_real(i) for i in list1 if str_to_real(i) >= str_to_real(value)]
        elif cmp == '<=':
            list1 = [str_to_real(i) for i in list1 if str_to_real(i) <= str_to_real(value)]
        return list1

    def Max_(self, set1):
        set1 = list(set1)
        number_set = defaultdict(set)
        for i in set1:
            v = str_to_real(self.id2node[i])
            number_set[v].add(i)    
        max_v = max(list(number_set.keys()))
        
        return number_set[max_v]
    
    def previous_row(self, set1):
        set2 = set([str_to_real(i)-1 for i in set1])
        return set2
    
    def next_row(self, set1):
        set2 = set([str_to_real(i)+1 for i in set1])
        return set2
    

    def Max(self, set1):
        set1 = [str_to_real(i) for i in set1]
        # print(set1)  
        # 加入判断 是否要进入llm todo 如果全是int/float 才用max() 否则进入LLM
        try:
            for i in set1:
                assert type(i) in [int, float]
            return max(set1)
        except:
            llm_result = LLMfunction(args=self.args, question=self.question, query=self.queries, 
                                    task=self.task, step=self.step, mid_output=self.mid_output, candidate=set1)
            #求set1的max
            return llm_result
    
    def Min(self, set1):
        #print(set1)
        set1 = [str_to_real(i) for i in set1]   
        #print(set1) 
        try:
            for i in set1:
                assert type(i) in [int, float]
            return min(set1)
        except:
            llm_result = LLMfunction(args=self.args, question=self.question, query=self.queries, 
                                    task=self.task, step=self.step, mid_output=self.mid_output, candidate=set1)
            #求set1的min
            return llm_result
    
    def Min_(self, set1):
        set1 = list(set1)
        number_set = defaultdict(set)
        for i in set1:
            v = str_to_real(self.id2node[i])
            number_set[v].add(i)    
        min_v = min(list(number_set.keys()))
        
        return number_set[min_v]
    
    def Mean(self, set1):
        number_set = [str_to_real(i) for i in set1]
        result = sum(number_set)/len(number_set)        
        return result
    
    def Mean_(self, set1):
        number_set = [str_to_real(self.id2node[i]) for i in set1]
        result = sum(number_set)/len(number_set)        
        return result
    
    def Sum(self, set1):
        number_set = [str_to_real(i) for i in set1]
        result = sum(number_set)
        return result
    
    def Sum_(self, set1):
        number_set = [str_to_real(self.id2node[i]) for i in set1]
        result = sum(number_set)
        return result
    
    def Count(self, set1):
        return len(set1)
    
    def get_information_(self, parameters):
        relation, head_entity, tail_entity, key, value, tail_entity_cmp, value_cmp = parameters
        return self.get_information(relation, head_entity, tail_entity, key, value, tail_entity_cmp, value_cmp)
    
    
    def build_triple_condition_(self, head_entity_set=None, relation_set=None, tail_entity_set=None):
        triple_conditions = set()
        if head_entity_set is not None and type(head_entity_set) not in [set, list]:
            head_entity_set = [head_entity_set]
        if relation_set is not None and type(relation_set) not in [set, list]:
            relation_set = [relation_set]
        if tail_entity_set is not None and type(tail_entity_set) not in [set, list]:
            tail_entity_set = [tail_entity_set]

        for e_n in self.nodes:
            if type(e_n) is tuple:
                flag = True
                if head_entity_set is not None and e_n[0] not in head_entity_set:#是其中一个就行
                    flag = False
                if relation_set is not None and e_n[1] not in  relation_set:
                    flag = False
                if tail_entity_set is not None and e_n[2] not in tail_entity_set:
                    flag = False
                if flag:
                    triple_conditions.add(e_n)
        return triple_conditions
    def get_information(self,relation= None, head_entity = None, tail_entity = None, key = None, value = None, tail_entity_cmp=None, value_cmp=None):
        # import pdb; pdb.set_trace()
        # ----------- temporal -----------
        if (str(key) in self.time_rel_mark) and (value is None) and (sum([i==None for i in [relation, head_entity, tail_entity]])<3):
            # h,r,t 至少有一个不是None
            triple_conditions=self.build_triple_condition_(head_entity, relation, tail_entity)
            result = self._search_node(node=key, condition=triple_conditions)
            return result
        if (str(key) in self.time_rel_mark) and (None not in [value, relation]) and (sum([i==None for i in [head_entity, tail_entity]])==1):
            scope1 = self._search_condition(node1=key, node2=value, cmp=value_cmp)# (h, r, t)
            if None not in [relation, head_entity] and tail_entity is None:
                candi_res = self._search_node(node=relation, condition=head_entity)
                result = set([i for i in candi_res if (head_entity, relation, i) in scope1])
            if None not in [relation, tail_entity] and head_entity is None:
                # import pdb; pdb.set_trace()
                candi_res = self._search_condition(node1=relation, node2=tail_entity, cmp=tail_entity_cmp)
                result = set([i for i in candi_res if (i, relation, tail_entity) in scope1])
            return result
        
        # ----------- original -----------
        if None not in [relation, head_entity] and sum([i==None for i in [tail_entity, key, value]])==3:# --tail entity
            result = self._search_node(node=relation, condition=head_entity)
            
        if None not in [key, head_entity] and sum([i==None for i in [tail_entity, relation, value]])==3:# --tail entity
            result = self._search_node(node=key, condition=head_entity)
            
        if None not in [relation, tail_entity] and sum([i==None for i in [head_entity, key, value]])==3:# --head entity 
            # print(relation, tail_entity, tail_entity_cmp)
            result = self._search_condition(node1=relation, node2=tail_entity, cmp=tail_entity_cmp)
        
        if None not in [relation, value] and sum([i==None for i in [head_entity, key, tail_entity]])==3:# --head entity
            result = self._search_condition(node1=relation, node2=value, cmp=value_cmp)
        
        if None not in [key, value] and sum([i==None for i in [relation, head_entity, tail_entity]])==3:# --condition / head entity
            result = self._search_condition(node1=key, node2=value, cmp=value_cmp)
            
        if None not in [key, tail_entity] and sum([i==None for i in [relation, head_entity, value]])==3:# --condition / head entity
            result = self._search_condition(node1=key, node2=tail_entity, cmp=tail_entity_cmp)
        
        if None not in [head_entity] and sum([i==None for i in [relation, tail_entity, key, value]])==4:# --relation
            result = self._search_node(node=head_entity)
        
        if None not in [relation] and sum([i==None for i in [head_entity, tail_entity, key, value]])==4:#--tail entities, values
            
            result = self._search_node(node=relation)
        
        # table里 head entity 是唯一的 可以 先找head, 再通过relation找tail
        if None not in [relation, key, value] and sum([i==None for i in [head_entity, tail_entity]])==2:
            
            scope1 = self._search_condition(node1=key, node2=value, cmp=value_cmp) #condition 头实体/ hkg_triple[0] in (k,v, hkg_triple[0])
            print('\t 1st -- get scope1:',scope1)
            result = self._search_node(node=relation, condition=None, scope=scope1)
            print('\t 2nd -- get result:',result)
        # # relation tail key
        
        if None not in [relation, key, tail_entity] and sum([i==None for i in [head_entity, value]])==2:
            scope1 = self._search_condition(node1=relation, node2=tail_entity, cmp=tail_entity_cmp) #condition 头实体/ hkg_triple[0] in (k,v, hkg_triple[0])
            print('\t 1st -- get scope1:',scope1)
            result = self._search_node(node=key, condition=None, scope=scope1)
            print('\t 2nd -- get result:',result)
            
        # 同时满足 找head可行 head是唯一的
        if None not in [relation, tail_entity, key, value] and sum([i==None for i in [head_entity]])==1:
            scope1 = self._search_condition(node1=key, node2=value, cmp=value_cmp) # condition 头实体/ hkg_triple[0] in (k,v, hkg_triple[0]) 
            print('\t 1st -- get scope1:', scope1)
            scope2 = self._search_condition(node1=relation, node2=tail_entity, cmp=tail_entity_cmp) 
            print('\t 2nd -- get scope2:', scope2)
            result = self.set_intersection(scope1, scope2)
            print('\t 3rd -- get result by set_intersection:', result)
            
        # relation head key
        if None not in [relation, key, head_entity] and sum([i==None for i in [tail_entity, value]])==2:
            # 实际上head_entity是 tail entity
            scope1 = self._search_node(node=relation, condition=head_entity) 
            print('\t 1st -- get scope1:', scope1)
            # tail entity当head entity 去找key的value???
            result = self._search_node(node=key, condition=None, scope = scope1)
            print('\t 2nd -- get result:',result)
        return result
    
    
    def parse_query(self, text_query):
        """
        text_query example: get_information(relation="No.", entity="3")  
        """
        legel_para_name = ['relation', 'head_entity', 'tail_entity', 'key', 'value', 'set', 'set1', 'set2', 'set3', 'set4']
                
        def deal_para(para_str, para_values, para_names):
            for o in para_str:
                o_name, o_value, o_cmp = o
                if (o_value[0] == "'" and o_value[-1] == "'") or (o_value[0] == '"' and o_value[-1] == '"'):
                    o_value = o_value[1:-1]

                if o_name not in legel_para_name:
                    print_error(f'parameter name {o_name} is illegal, must in {legel_para_name}')
                else:
                    if o_value.startswith('output_of_') or o_cmp in ['<', '>', '<=', '>=']:
                        para_values[para_names.index(o_name)] = o_value#比较大小/output_of 不用转成id
                    else:
                        try:
                            para_values[para_names.index(o_name)] = self.id2node[o_value]
                        except Exception as e:
                            print_error(f'{o_value} not a node id ! {e}')
                            # para_values[para_names.index(o_name)] = o_value
                    
        fun_name = text_query.split('(')[0].strip()
        
        if fun_name not in self.mapping_function:
            return fun_name, [None], [None]
        
        para = re.findall(r'\(.*\)',text_query)[0][1:-1]
        para = para.replace(' ','')
        para_str = [i.strip() for i in para.split(',')]#这里的问题，
        # para_str = [i.strip() for i in para.split(', ')]# bug3 #这里的问题，但是不确定会不会影响其他query
        print(para_str)
        tail_entity_cmp, value_cmp = '=', '='
        for idx, e_str in enumerate(para_str):
            if 'tail_entity' in e_str or 'value' in e_str:                
                cmp_flag = '='
                for cmp in ['>=', '<=', '<', '>']:
                    if cmp in e_str:
                        cmp_flag = cmp
                        break                
                if 'tail_entity' in e_str:
                    tail_entity_cmp = cmp_flag                    
                elif 'value' in e_str:
                    value_cmp = cmp_flag
                para_str[idx] = (e_str.split(cmp_flag)[0], e_str.split(cmp_flag)[1], cmp_flag)
            else:
                para_str[idx] = (e_str.split('=')[0], e_str.split('=')[1], None)
        
        if fun_name == 'get_information':
            para_names = ['relation', 'head_entity', 'tail_entity', 'key', 'value', 'tail_entity_cmp', 'value_cmp']
            para_values = [None]*5 + [tail_entity_cmp, value_cmp]
            
        elif fun_name in ['set_intersection']:
            para_names = ['set1', 'set2', 'set3', 'set4' ,'set5']
            para_values = [None]*5
            
        elif fun_name in ['set_union', 'set_difference']:
            para_names = ['set1', 'set2']
            para_values = [None]*2
            
        elif fun_name in ['min', 'max', 'mean', 'sum', 'count', 'set_negation']:
            para_names = ['set']
            para_values = [None]*1
        elif fun_name in ['previous_row', 'next_row']:
            para_names = ['set']
            para_values = [None]*1
        elif fun_name in ['keep']:
            para_names = ['set', 'value', 'value_cmp']
            para_values = [None]*2 + [value_cmp]

        deal_para(para_str, para_values, para_names)
        # print(fun_name, para_names, para_values)
        return fun_name, para_names, para_values
         
    def excute_single_query(self, text_query, target_type):
        try:
            fun_name, para_names, parameters = self.parse_query(text_query)
            if fun_name not in self.mapping_function:
                # 函数名不在预设列表中
                llm_result = LLMfunction(args=self.args, question=self.question, query=self.queries, 
                                         task=self.task, step=self.step, mid_output=self.mid_output)
                # todo 需要把llm返回的结果转成一个set
                res = set(llm_result)
                
            else:
                query = Query(fun_name, parameters)
                res = self.mapping_function[query.name](query.parameters)
            
            #最后一步
            if type(res) is set:
                final_res=set()
                for er in res:
                    if er in self.all_lines_id:
                        final_res |= self._search_node(node=target_type, condition=er)
                    else:
                        final_res.add(er)
            else:
                final_res = res
                
            return final_res
        except Exception as e:
            print_error(e)
            return None
    
    def excute_query(self, args, text_queries, target_type=None, node_query=None, task=None, question=None):
        self.args = args
        self.mid_output = defaultdict(dict)
        self.task = task
        self.question = question
        """
        text_query: [query1, query2, query3,...]
        """
        self.queries = node_query
        # import pdb; pdb.set_trace()
        try:
            for q_id, q in enumerate(text_queries):
                self.step = q_id+1                
                para_string_ = ''
                fun_name, para_names, parameters = self.parse_query(q)
                
                self.mid_output[f'query{self.step}']['fun_name'] = fun_name
                
                if fun_name not in self.mapping_function:
                    # 函数名不在预设列表中
                    llm_result = LLMfunction(args=self.args, question=self.question, query=self.queries, 
                                         task=self.task, step=self.step, mid_output=self.mid_output)
                    # todo 需要把llm返回的结果转成一个set
                    res = set(llm_result)
                else:
                    for pn, pv in zip(para_names, parameters):
                        if pv is not None:
                            self.mid_output[f'query{self.step}'][pn] = pv
                            if pn.endswith('cmp') and pv=='=':
                                continue
                            elif pn.endswith('cmp'):
                                para_string_ += f'{pn}:{pv}, '                        
                            else:
                                para_string_ += f'{pn}={pv}, '
                                    
                    for idx, para in enumerate(parameters):
                        if str(para).startswith('output_of_'):
                            parameters[idx] = self.mid_output[para.replace('output_of_','')]['output']
                    # print('parameters', parameters)        
                    if len(parameters) == 1:
                        parameters = parameters[0]
                
                    query = Query(fun_name, parameters)
                    res = self.mapping_function[query.name](query.parameters)
                    # if type(res) is set:
                    #     res2id = set()
                    #     for er in res:
                    #         if er in self.node2id:
                    #             res2id.add(self.node2id[er]) 
                    # print(res2id)
                    # import pdb; pdb.set_trace()
                self.mid_output[f'query{self.step}']['output'] = res# 0-->1
                print(f'\t output_of_query{self.step}:{res}', f'\t query{self.step}：{query.name}({para_string_})')
                
                #最后一步
                if q_id == len(text_queries)-1:
                    if (target_type is not None) and (type(res) in [set, list]):
                        final_res=set()
                        for er in res:
                            if er in self.all_lines_id:
                                # import pdb; pdb.set_trace()
                                final_res |= self._search_node(node=target_type, condition=er)
                            else:
                                final_res.add(er)
                    else:# 没有target_type kgqa/ res直接是数值等
                        final_res = res
            # if type(res) is set: #不是mean和sum的其他所有
            #     res = set([self.id2node[i] for i in res])

            return final_res, self.mid_output #update by jl
        except Exception as e:
            print_error(e)
            return None, None #update by jl

# 执行过程中 完成一个 改一个 例如scope1 
class Query():
    def __init__(self, name, parameters):
        self.name=name
        self.parameters = parameters
        self.check_query_correctness()
    
    def check_query_correctness(self):
        if self.name=='get_information':
            # self.parameters : h,r,t,p,v
            if len(self.parameters) != 7: 
                print_error(f'parameter number of {self.name}() must be 5')
        elif self.name in ['set_intersection']:
            if len(self.parameters) != 5: 
                print_error(f'parameter number of {self.name}() must be 5')
            for e_p in self.parameters:
                if e_p is not None and type(e_p) is not set:
                    print_error(f'all parameters\' type of {self.name}() must be set')
        elif self.name in ['set_union', 'set_difference', 'keep']:   
            if  self.name == 'kepp':
                if len(self.parameters) != 3:
                    print_error(f'parameter number of {self.name}() must be 2')
            else:
                if len(self.parameters) != 2:
                    print_error(f'parameter number of {self.name}() must be 2')
            if type(self.parameters[0]) not in [set, list]:
                print_error(f'parameter1 type of {self.name}() must be set')
        elif self.name in ['min', 'max', 'mean', 'sum', 'count', 'set_negation', 'previous_row', 'next_row']:
            if type(self.parameters) not in [set, list]:
                print_error(f'parameter\'s type of {self.name}() must be set or list')
        

