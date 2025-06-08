from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from langchain import PromptTemplate
from langchain_core.runnables import RunnableSequence
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.cluster import DBSCAN
from langchain.embeddings.base import Embeddings
from core.clova_inference import ClovaInference
from model.sbert_wrapper import SBERTWrapper
from chromadb import HttpClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from sqlalchemy.orm import Session
# from db.db import SessionLocal
# from db.db_models import Missions
import time, torch, re, random, os
import numpy as np
import pandas as pd

load_dotenv()

# 래퍼 생성
sbert_model = SentenceTransformer('./kr-sbert', device='cpu')
sbert_wrapper = SBERTWrapper(sbert_model)

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chroma embedding function
def embedding_func(texts):
    return sbert_model.encode(texts, convert_to_numpy=True).tolist()

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, ssl=False)

# 컬렉션 가져오기
mission_collection = chroma_client.get_or_create_collection(
    name="mission_collection"
)
hated_mission_collection = chroma_client.get_or_create_collection(
    name="hated_mission_collection"
)

# LLM 파이프라인 (HyperCLOVA-X)
MODEL_PATH = os.getenv("MODEL_PATH")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

clova_llm = ClovaInference(model_path=MODEL_PATH, sbert_model=sbert_model, mission_collection=mission_collection, hated_mission_collection=hated_mission_collection, user_query=None)
llm_missions = clova_llm.infer()
print(llm_missions)

# db = SessionLocal()

# # Missions 객체를 담을 리스트
# missions_to_add = []

# for key, value_list in llm_missions.items():
#     for emoji_value, raw_value in value_list:
#         mission = Missions(
#             title=raw_value,
#             description=emoji_value,
#             difficulty=key
#         )
#         missions_to_add.append(mission)

# # 리스트를 add_all로 한번에 넣기
# db.add_all(missions_to_add)
# db.commit()
# db.close()

if __name__ == 'main':
  print('main.py 실행 완료!')