import os
import json
import chromadb
import torch
import openai
import time
import re
from torch import Tensor
from openai import OpenAI
from tqdm.autonotebook import trange
from typing import List, Union, TypeVar, Dict
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from tenacity import retry, wait_random_exponential, stop_after_attempt
import fnmatch

Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)

@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(6))
def get_embedding_openai(text, model="text-embedding-ada-002") -> List[float]:
    client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
    if isinstance(text, str):
        max_retries = 20
        for i in range(max_retries):
            try:
                # 尝试执行可能会引发错误的代码
                result = client.embeddings.create(input=[text], model=model).data[0].embedding
                # 如果代码成功执行，那么跳出循环
                break
            except openai.APIConnectionError as e:
                # 如果引发了APIConnectionError，那么等待一秒然后重试
                if i < max_retries - 1:  # 如果不是最后一次重试
                    time.sleep(1)  # 等待一秒
                    continue
                else:  # 如果是最后一次重试，那么重新引发错误
                    raise
        return result
    elif isinstance(text, list):
        max_retries = 20
        for i in range(max_retries):
            try:
                # 尝试执行可能会引发错误的代码
                result = client.embeddings.create(input=text, model=model)
                # 如果代码成功执行，那么跳出循环
                break
            except openai.APIConnectionError as e:
                # 如果引发了APIConnectionError，那么等待一秒然后重试
                if i < max_retries - 1:  # 如果不是最后一次重试
                    time.sleep(1)  # 等待一秒
                    continue
                else:  # 如果是最后一次重试，那么重新引发错误
                    raise
        return result

class NewEmbeddingFunction(EmbeddingFunction):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def __call__(self, input: D) -> Embeddings:
        embeddings = self.encoder.encode(input)
        return embeddings


class EncoderAda002:
    def encode(
        self,
        text: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[Tensor]:
        text_embeddings = []
        for batch_start in trange(
            0, len(text), batch_size, disable=not show_progress_bar
        ):
            batch_end = batch_start + batch_size
            batch_text = text[batch_start:batch_end]
            # print(f"Batch {batch_start} to {batch_end-1}")
            assert "" not in batch_text
            resp = get_embedding_openai(batch_text)
            for i, be in enumerate(resp.data):
                assert (
                    i == be.index
                )  # double check embeddings are in same order as input
            batch_text_embeddings = [e.embedding for e in resp.data]
            text_embeddings.extend(batch_text_embeddings)

        return text_embeddings


class OpenaiAda002:
    def __init__(self) -> None:
        self.q_model = EncoderAda002()
        self.doc_model = self.q_model

    def encode(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> List[Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)


class Encoder:

    def __init__(self, encoder_name: str) -> None:
        self.encoder_name = encoder_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        # self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        
        if encoder_name == "text-embedding-ada-002":
            self.encoder = OpenaiAda002()
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-ada-002",
            )
        else:
            pass


def _get_embedding_and_save_to_chroma(
    data: List[Dict[str, str]],
    collection: Collection,
    encoder: Encoder,
    batch_size: int = 64,
):
    encoder_ = encoder.encoder

    docs = [item["question"] for item in data]
    meta_keys = list(data[0].keys())
    del meta_keys[meta_keys.index("question")]

    embeddings = encoder_.encode(docs, batch_size=batch_size, show_progress_bar=True)
    # else:
    #     embeddings = encoder_.doc_model.encode(
    #         docs, batch_size=batch_size, show_progress_bar=True
    #     )
    if not isinstance(embeddings, list):
        embeddings = embeddings.tolist()
    if len(embeddings) > 20000:
        for i in range(0, len(embeddings), 20000):
            collection.add(
                embeddings=embeddings[i : i + 20000],
                documents=docs[i : i + 20000],
                metadatas=[
                    {key: data[i][key] for key in meta_keys}
                    for i in range(i, min(len(embeddings), i + 20000))
                ],
                ids=[str(i) for i in range(i, min(len(embeddings), i + 20000))],
            )
    else:
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=[
                {key: data[i][key] for key in meta_keys} for i in range(len(docs))
            ],
            ids=[str(i) for i in range(len(docs))],
        )
    return collection

def get_embedding_align(dataset_path: str, retriever: str, chroma_dir: str, name: str = None):
    dataset_name = "ccks"
    chroma_path = os.path.join(chroma_dir, retriever, dataset_name)

    encoder = Encoder(retriever)
    if name == None:
        name = "main"
    
    embedding_function = encoder.ef
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,
        get_or_create=True,
    )
    if not collection.count():
        retrieve_data = []
        with open(dataset_path, 'r', )as f:
            for idx, line in enumerate(f.readlines()):                        
                elements = line.strip().split('\t')
                h, r, t = elements
                data_relation = {
                    "question": r,
                    "type": "relation"
                }
                if(data_relation in retrieve_data): continue
                retrieve_data.append(data_relation)

        _get_embedding_and_save_to_chroma(retrieve_data, collection, encoder)


    return collection