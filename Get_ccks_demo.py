# 导入所需的库
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import torch
import json
from collections import defaultdict
import structllm as sllm
import CGdata_for_ccks

# 示例数据，模拟包含 1000 条自然语言问题的数据集
def kmeans_clustering():
    with open('dataset/CCKS_round1/train_qa.json', 'r', encoding='utf-8') as f:
        qa_data = json.loads(f.read())
        sample_data = [qa_data[i]['question'] for i in qa_data.keys()]

    # 加载 Sentence-BERT 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SentenceTransformer('msmarco-distilbert-base-tas-b').to(device)

    # 使用 Sentence-BERT 将问题转换为向量表示
    question_embeddings = model.encode(sample_data)

    # 定义要聚类的簇数
    num_clusters = 8

    # 使用 K 均值聚类算法进行聚类
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(question_embeddings)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # 从每个簇中选择代表性的问题
    representative_questions = []
    for cluster_id in range(num_clusters):
        cluster_questions = [sample_data[i] for i in range(len(sample_data)) if cluster_labels[i] == cluster_id]
        cluster_center = cluster_centers[cluster_id]
        distances = [np.linalg.norm(model.encode([question])[0] - cluster_center) for question in cluster_questions]
        representative_question_index = np.argmin(distances)
        representative_question = cluster_questions[representative_question_index]
        representative_questions.append(representative_question)

    # 打印代表性问题
    for i, question in enumerate(representative_questions):
        print(f"代表性问题 {i+1}: {question}")

    # 代表性问题 1: 陈望道的学生中，有多少人的毕业院校是广西师范大学？
    # 代表性问题 2: 黄晓君和罗文裕共同创作过的音乐作品有哪些？
    # 代表性问题 3: 胡淑仪的儿子是谁？
    # 代表性问题 4: 王喆的主要作品和音乐作品总共有多少首？答案
    # 代表性问题 5: 杜并是谁的曾祖父？
    # 代表性问题 6: 破晓之战的主要演员有多少人？
    # 代表性问题 7: 李建复的音乐作品中，有几首是他的代表作品？
    # 代表性问题 8: 参演过顶楼的演员中，有哪些人的代表作品是熔炉？

if __name__ == '__main__':
    # Step1: 选择代表性问题
    # kmeans_clustering()
    
    args = CGdata_for_ccks.parse_args()

    args.folder_path = 'dataset/CCKS_round1/kg(utf8).txt'
    args.data_path = 'dataset/CCKS_round1/train_qa.json'
    KG_data, KGQA_data, relations = CGdata_for_ccks.kg2CG(args)
    print(len(KG_data), len(KGQA_data), len(relations))

    node2id = KG_data['main'].node2id

    # Step2: 构建demo

    question = "陈望道的学生中，有多少人的毕业院校是广西师范大学？"
    label = ["0"]
    # Step1: 找到陈望道的学生
    # Query1: get_information(relation="学生", head_entity="陈望道")
    # Step2: 找到广西师范大学的学生
    # Query2: get_information(relation="毕业院校", tail_entity="广西师范大学")
    # Step3: 求两个集合的交集
    # Query3: set_intersection(set1="output_of_query2", set2="output_of_query1")
    # Step4: 计算集合的大小
    # Query4: count(set="output_of_query3")
    query1 = f"get_information(relation={node2id['学生']}, head_entity={node2id['陈望道']})"
    query2 = f"get_information(relation={node2id['毕业院校']}, tail_entity={node2id['广西师范大学']})"
    query3 = f"set_intersection(set1='output_of_query2', set2='output_of_query1')"
    query4 = f"count(set='output_of_query3')"

    text_queries = [query1, query2, query3, query4]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res}\n')


    question = "黄晓君和罗文裕共同创作过的音乐作品有哪些？"
    label = ["0"]
    # Step1: 找到黄晓君的音乐作品
    # Query1: get_information(relation="音乐作品", head_entity="黄晓君")
    # Step2: 找到罗文裕的音乐作品
    # Query2: get_information(relation="音乐作品", head_entity="罗文裕")
    # Step3: 求两个集合的交集
    # Query3: set_intersection(set1="output_of_query1", set2="output_of_query2")
    query1 = f"get_information(relation={node2id['音乐作品']}, head_entity={node2id['黄晓君']})"
    query2 = f"get_information(relation={node2id['音乐作品']}, head_entity={node2id['罗文裕']})"
    query3 = f"set_intersection(set1='output_of_query1', set2='output_of_query2')"
    text_queries = [query1, query2, query3]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res[0]}\n')

    question = "胡淑仪的儿子是谁？"
    label = ["司马朱生","司马郁"]
    # Step1: 找到胡淑仪的儿子
    # Query1: get_information(relation="儿子", head_entity="胡淑仪")
    query1 = f"get_information(relation={node2id['儿子']}, head_entity={node2id['胡淑仪']})"
    text_queries = [query1]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res[0]}\n')

    question = "王喆的主要作品和音乐作品总共有多少首？"
    label = ["6"]
    # Step1: 找到王喆的主要作品
    # Query1: get_information(relation="主要作品", head_entity="王喆")
    # Step2: 找到王喆的音乐作品
    # Query2: get_information(relation="音乐作品", head_entity="王喆")
    # Step3: 求两个集合的并集
    # Query3: set_union(set1="output_of_query1", set2="output_of_query2")
    # Step4: 计算集合的大小
    # Query4: count(set="output_of_query3")
    query1 = f"get_information(relation={node2id['主要作品']}, head_entity={node2id['王喆']})"
    query2 = f"get_information(relation={node2id['音乐作品']}, head_entity={node2id['王喆']})"
    query3 = f"set_union(set1='output_of_query1', set2='output_of_query2')"
    query4 = f"count(set='output_of_query3')"
    text_queries = [query1, query2, query3, query4]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res}\n')

    question = "杜并是谁的曾祖父？"
    label = ["0"]
    # Step1: 找到杜并的曾祖父
    # Query1: get_information(relation="曾祖父", head_entity="杜并")
    query1 = f"get_information(relation={node2id['曾祖父']}, head_entity={node2id['杜并']})"
    text_queries = [query1]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res[0]}\n')

    question = "破晓之战的主要演员有多少人？"
    label = ["8"]
    # Step1: 找到破晓之战的主要演员
    # Query1: get_information(relation="主要演员", head_entity="破晓之战")
    # Step2: 计算集合的大小
    # Query2: count(set="output_of_query1")
    query1 = f"get_information(relation={node2id['主要演员']}, head_entity={node2id['破晓之战']})"
    query2 = f"count(set='output_of_query1')"
    text_queries = [query1, query2]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res}\n')

    question = "李建复的音乐作品中，有几首是他的代表作品？"
    label = ["2"]
    # Step1: 找到李建复的音乐作品
    # Query1: get_information(relation="音乐作品", head_entity="李建复")
    # Step2: 找到李建复的代表作品
    # Query2: get_information(relation="代表作品", head_entity="李建复")
    # Step3: 求两个集合的交集
    # Query3: set_intersection(set1="output_of_query2", set2="output_of_query1")
    # Step4: 计算集合的大小
    # Query4: count(set="output_of_query3")
    query1 = f"get_information(relation={node2id['音乐作品']}, head_entity={node2id['李建复']})"
    query2 = f"get_information(relation={node2id['代表作品']}, head_entity={node2id['李建复']})"
    query3 = f"set_intersection(set1='output_of_query2', set2='output_of_query1')"
    query4 = f"count(set='output_of_query3')"
    text_queries = [query1, query2, query3, query4]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res}\n')


    question = "参演过顶楼的演员中，有哪些人的代表作品是熔炉？"
    label = ["金贤秀"]
    # Step1: 找到参演过顶楼的演员
    # Query1: get_information(relation="主要演员", head_entity="顶楼")
    # Step2: 找到参演熔炉的演员
    # Query2: get_information(relation="主要演员", head_entity="熔炉")
    # Step3: 求两个集合的交集
    # Query3: set_intersection(set1="output_of_query2", set2="output_of_query1")
    query1 = f"get_information(relation={node2id['主要演员']}, head_entity={node2id['顶楼']})"
    query2 = f"get_information(relation={node2id['主要演员']}, head_entity={node2id['熔炉']})"
    query3 = f"set_intersection(set1='output_of_query2', set2='output_of_query1')"
    text_queries = [query1, query2, query3]
    res = KG_data['main'].excute_query(args, text_queries)
    print(f'{question}\t a={res[0]}\n')