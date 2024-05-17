import structllm as sllm
import structllm.translate2CGdata as trans2CGdata
import json

class kgqa_query_prompt():
    def __init__(self, args, question, table_data, relations, collection = None):
        self.question = question
        self.model = args.model
        
        # retrieve demonstrations
        if collection is not None:
            # self.retrieve_dynamic_prompt(args, question, table_data, relations, collection)   
            pass                 
        else:
            with open(args.prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
        
        self.naive_prompt.append(
            {
                "role": "user",
                "content": self.kgqa_schema_Prompt()
            }
        )

    def kgqa_schema_Prompt(self):
        
        prompt = f"Question: {self.question}"
        return prompt