import structllm as sllm
import re


def LLMfunction(args, question, query, task, step, mid_output, candidate=None):
    '''
        Use LLM to solve the problem of the current step.

        Input:
            args: the args of the program

            question (str): the origin question
                - example: "how many times was a position of at least 10th place or better earned?"
            
            query (list): the query list of the question
                - example: ["get_information(relation='Position', tail_entity>='10th place')","count(set1='output_of_query1')"]
            
            task (list): the task list of the question
                - example: ["Step1: Find the entity with a position of at least 10th place or better", "Step2: Get the count of output_of_query1"]
            
            step (int): the current step of the problem
                - example: 2
            
            mid_output (dict): the mid_output of the previous step
                - example: {"output_of_query1": ['1st place', '2nd place', ...]}

        Output:
            result: list
    '''
    print("现在在大模型中")
    llm = sllm.llm.gpt(args)


    # 有额外数据提供
    if candidate != None:
        if f"query{step-1}" in mid_output.keys() and list(mid_output[f'query{step-1}']['output'])==candidate: # 1. max min 特殊符号
            prompt=[
                {
                  "role": "system",
                  "content": "You are a helpful assistant. If the data format provided is complex, use common sense.\nPlease enclose the final answer in quotation marks for ease of extraction."
                },
                {
                  "role": "user",
                  "content": "Question: Get the min output_of_query1 And Output_query1: {'12.15 (87)', '14.16 (100)', '8.5 (53)', '10.9 (69)', '6.7 (43)', '2.9 (21)', '11.12 (78)', '8.9 (57)', '9.11 (65)', '13.6 (84)', '11.8 (74)', '9.4 (58)', '14.11 (95)', '14.14 (98)', '11.2 (68)', '4.11 (35)', '19.14 (128)', '17.13 (115)', '8.10 (58)', '17.6 (108)', '6.9 (45)', '11.11 (77)'}.\nPlease tell me the only data who satisfies the condition."
                },
                {
                  "role": "assistant",
                  "content": "Based on the given data, it appears that the min data is '2.9 (21)'."
                },
                {
                  "role": "user",
                  "content": "Get the best position of output_of_query1 And  Output_query1: {'13th (q)', '5th', '7th', '–'}.\nPlease tell me the only data who satisfies the condition."
                },
                {
                  "role": "assistant",
                  "content": "Based on the given data, it appears that the best data is '5th'."
                },
            ]
            mid_output_str = str()
            for idx in range(step-1):
                query_d = f'query{idx+1}'
                mid_output_str += f"Output_query{idx+1}: {mid_output[query_d]['output']}."

            mid_output_str = f"{task[step-1][7:]} And {mid_output_str}\nPlease tell me the only data who satisfies the condition."
            print(mid_output_str)
            tmp = {
                "role": "user",
                "content": mid_output_str
            }
            prompt.append(tmp)
            
            response = llm.get_response(prompt)
            print(f"response:{response}")
            # 提取出最后一个\n后的所有字符串
            last_part = response[response.rfind('\n')+1:]

            # 使用正则表达式找到第一个单引号或双引号内的内容
            result = re.findall(r'["\'](.*?)["\']', last_part)
            result = [item for item in result if item in candidate]
            if result: return result
            else: return ["None"]

        # if "times" in query[step-1] or "most" in query[step-1]: # 1. 问题中的实体出现次数
        #     print("1. 问题中的实体出现次数")
        #     data_list = [tup[0] for tup in candidate]

        #     prompt = [
        #         {
        #           "role": "system",
        #           "content": "You are a helpful assistant. Find the number of occurrences of all entities and determine who has the most."
        #         },
        #         {
        #           "role": "user",
        #           "content": "Question: Find the Label that appears the most times.\nHere are all the entities:\n['Disky', 'Capitol', 'Capitol', 'Label', 'EMI', 'Edel', 'Arcade', 'Wagram', 'Disky', 'Atlantic', 'EMI', 'EMI', 'EMI', 'Disky, Raven', 'Arcade', 'Wagram']\nFind the entity that appears the most times."
        #         },
        #         {
        #           "role": "assistant",
        #           "content": "To find the entity that appears the most times, we need to count the occurrences of each entity. Here is the count of occurrences for each entity:\n\n- 'Disky': 3 occurrences\n- 'EMI': 4 occurrences\n- 'Capitol': 2 occurrences\n- 'Edel': 1 occurrence\n- 'Wagram': 2 occurrences\n- 'Atlantic': 1 occurrence\n- 'Arcade': 2 occurrences\n- 'Raven': 1 occurrence\n- 'Label': 1 occurrence\n\nFrom the list above, we can see that 'EMI' appears the most times with 4 occurrences.\nTherefore, the entity that appears the most times is 'EMI'."
        #         },
        #         {
        #           "role": "user",
        #           "content": "Question: Find the Pole Position that appears the most times.\nHere are all the entities:\n['Carlos Checa', 'Troy Bayliss', 'Troy Bayliss', 'Max Neukirchner', 'Troy Corser', 'Noriyuki Haga', 'Max Neukirchner', 'Troy Bayliss', 'Troy Bayliss', 'Carlos Checa', 'Noriyuki Haga', 'Max Neukirchner', 'Troy Corser', 'Max Neukirchner', 'Troy Bayliss', 'Troy Bayliss', 'Troy Corser', 'Troy Corser']\nFind the entity that appears the most times."
        #         },
        #         {
        #           "role": "assistant",
        #           "content": "To find the entity that appears the most times, we need to count the occurrences of each entity. Here is the count of occurrences for each entity:\n\n- 'Troy Bayliss': 6 occurrences\n- 'Noriyuki Haga': 2 occurrences\n- 'Troy Corser': 4 occurrences\n- 'Carlos Checa': 2 occurrences\n- 'Max Neukirchner': 4 occurrences\n\nFrom the list above, we can see that 'Troy Bayliss' appears the most times with 6 occurrences.\nTherefore, the Pole Position that appears the most times is 'Troy Bayliss'."
        #         },
        #         {
        #           "role": "user",
        #           "content": "Find the Winning team that appears the most.\nHere are all the entities:\n['Team Penske', 'Galles-Kraco Racing', 'Team Penske', 'Team Penske', 'Galles-Kraco Racing', 'Galles-Kraco Racing', 'Galles-Kraco Racing', 'Galles-Kraco Racing', 'Newman/Haas Racing', 'Newman/Haas Racing', 'Newman/Haas Racing', 'Newman/Haas Racing', 'Team Penske', 'Doug Shierson Racing', 'Galles-Kraco Racing', 'Newman/Haas Racing', 'Team Penske']\nFind the entity that appears the most times."
        #         },
        #         {
        #           "role": "assistant",
        #           "content": "To find the entity that appears the most times, we need to count the occurrences of each entity. Here is the count of occurrences for each entity:\n\n- 'Team Penske': 5 occurrences\n- 'Galles-Kraco Racing': 6 occurrences\n- 'Newman/Haas Racing': 5 occurrences\n- 'Doug Shierson Racing': 1 occurrence\n\nFrom the list above, we can see that both 'Galles-Kraco Racing' appears the most times with 6 occurrences.\nTherefore, the Winning team that appears the most times is 'Galles-Kraco Racing'."
        #         },
        #     ]
            
        #     mid_output_str = f"Question: {task[step-1][7:]}\nHere are all the entities:\n{data_list}\nFind the entity that appears the most times."
        #     print(mid_output_str)
        #     tmp = {
        #         "role": "user",
        #         "content": mid_output_str
        #     }
        #     prompt.append(tmp)
        #     print(f"tmp:f{tmp['content']}")

        #     response = llm.get_response(prompt)
        #     print(f"response:{response}")
            
        #     # 提取出最后一个\n后的所有字符串
        #     last_part = response[response.rfind('\n')+1:]

        #     # 使用正则表达式找到第一个单引号或双引号内的内容
        #     result = re.findall(r'["\'](.*?)["\']', last_part)
        #     result = [item for item in result if item in data_list]
        #     if result: return result
        #     else: return ["None"]
            
        else: # 2. 带符号比大小
            print("2. 带符号比大小")
            prompt = [
                {
                  "role": "system",
                  "content": "You are a helpful assistant."
                },
                {
                  "role": "user",
                  "content": "Find the players who are taller than output_of_query2 And Output_query1: {'[line_8]'}.Output_query2: {'6–4'}.\nAll data are given in the format of (data, line_number):\n\"{('6–3', '[line_3]'), ('6–9', '[line_6]'), ('6–8', '[line_1]'), ('6–5', '[line_4]'), ('6–5', '[line_7]'), ('5–11', '[line_5]'), ('5–10', '[line_2]')\".\nPlease check it step by step and tell me the line_number whose data satisfies the condition."
                },
                {
                  "role": "assistant",
                  "content": "To find the players who are taller than '6-4', we need to check the values given in the format ('height', 'line_number') one by one.\n\nHere are the step-by-step results:\n\n1. ('6–3', '[line_3]') - Not taller than '6-4'.\n2. ('6–9', '[line_6]') - Taller than '6-4'.\n3. ('6–8', '[line_1]') - Taller than '6-4'.\n4. ('6–5', '[line_4]') - Taller than '6-4'.\n5. ('6–5', '[line_7]') - Taller than '6-4'.\n6. ('5–11', '[line_5]') - Not taller than '6-4'.\n7. ('5–10', '[line_2]') - Not taller than '6-4'.\n\nThe line numbers whose heights satisfy the condition (taller than '6-4') are [line_6], [line_1], [line_4], and [line_7]."
                },
                {
                  "role": "user",
                  "content": "Find the line_number which has more strokes than output_of_query2 And Output_query1: {'[line_3]'}.Output_query2: {'3 strokes'}.\nAll data are given in the format of (data, line_number):\n{('1 stroke', '[line_5]'), ('Playoff', '[line_2]'), ('5 strokes', '[line_4]'), ('1 stroke', '[line_1]'), ('2 strokes', '[line_2]')}\nPlease check it step by step and tell me the line_number whose data satisfies the condition."
                },
                {
                  "role": "assistant",
                  "content": "To find the line_number which has more strokes than {'3 strokes'}, we need to check the values given in the format ('strokes', 'line_number')  one by one.\n\n\n1. ('1 stroke', '[line_5]') - Less strokes than '3 strokes'.\n2. ('Playoff', '[line_2]') - Not applicable for stroke comparison.\n3. ('5 strokes', '[line_4]') - More strokes than '3 strokes'.\n4. ('1 stroke', '[line_1]') - Less strokes than '3 strokes'.\n5. ('2 strokes', '[line_2]') - Less strokes than '3 strokes'.\n\nThe line number(s) whose number of strokes satisfies the condition (more than '3 strokes') is: [line_4]."
                },
                {
                  "role": "user",
                  "content": "Find the positions that are at least 10th place or better And \nAll data are given in the format of (data, line_number):\n\"{('13th (q)', '[line_1]'), ('5th', '[line_3]') , ('7th', '[line_2]'), ('–', '[line_4]')}\".\nPlease check it step by step and tell me the line_number whose data satisfies the condition."
                },
                {
                  "role": "assistant",
                  "content": "To find the positions that are at least 10th place or better, we need to compare the positions given in the format ('position', 'line_number') with the condition of being 10th place or better. Here are the step-by-step results:\n\n1. ('13th (q)', '[line_1]') - Not 10th place or better.\n2. ('5th', '[line_3]') - 10th place or better.\n3. ('7th', '[line_2]') - 10th place or better.\n4. ('–', '[line_4]') - Not applicable for position comparison.\n\nThe line numbers whose positions satisfy the condition (10th place or better) are: [line_2], and [line_3]."
                }
            ]

            mid_output_str = str()
            for idx in range(step-1):
                query_d = f'query{idx+1}'
                mid_output_str += f"Output_query{idx+1}: {mid_output[query_d]['output']}."
            tmp = {
                "role": "user",
                "content": f"{task[step-1][7:]} And {mid_output_str}\nAll data are given in the format of (data, line_number):\n\"{candidate}\".\nPlease check it step by step and tell me the line_number whose data satisfies the condition."
            }
            prompt.append(tmp)
            print(f"tmp:{tmp['content']}")
            response = llm.get_response(prompt)
            print(f"response:{response}")

            # 使用正则表达式找到所有的line_(数字)
            list_line = re.findall(r'line_\d+', response[response.rfind('\n')+1:])
            result = ['[' + line_d + ']' for line_d in list_line]
            if result: return result
            else: return ["None"]
    
    # 无额外数据提供 3. 选择题
    else:
        print("3. 选择题")
        flag = 0
        for idx in range(step-1):
            query_d = f'query{idx+1}'
            tmp = mid_output[query_d]['output']
            if tmp != '' and tmp != set() : flag = 1
        if flag == 0: return ["None"]
        
        prompt = [
            {
              "role": "system",
              "content": "Answer questions based on the information provided."
            },
            {
              "role": "user",
              "content": "Question: who won more silvers, venezuela or chile?\nInformation provided:\nStep1: Find the Nation of Venezuela. output_of_query1: {'[line_1]'}\nStep2: Find the Nation of Chile. output_of_query2: {'[line_4]'}\nStep3: Find the number of silvers of output_of_query1. output_of_query3: {'8'}\nStep4: Find the number of silvers of output_of_query2. output_of_query4: {'4'}\nStep5: Compare the number of silvers of output_of_query3 and output_of_query4.\nJust tell me the answer in a word or phrase without spaces or newlines."
            },
            {
              "role": "assistant",
              "content": "Venezuela"
            }
        ]
        input = f"Question: {question}\nInformation provided:\n"
        for idx in range(step-1):
            query_d = f'query{idx+1}'
            input += f"{task[idx]} And output_of_query{idx+1}: {mid_output[query_d]['output']}.\n"
        
        input += f"{task[step-1]}\n"
        tmp = {
              "role": "user",
              "content": f"{input}Just tell me the answer in a word or phrase without spaces or newlines."
        }
        print(f"tmp:f{tmp['content']}")
        prompt.append(tmp)
        response = llm.get_response(prompt)
        print(f"response:{response}")
        if response: return [response]
        else: return ["None"]