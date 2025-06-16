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
from db.db import SessionLocal
from db.db_models import Missions
from peft import PeftModel
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

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# base 모델 이름 (huggingface hub or local)
base_model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

# adapter가 저장된 로컬 경로 (.env에서 가져오기)
MODEL_PATH = os.getenv("MODEL_PATH")  # ex: "./lora_adapter"

# base model과 tokenizer 불러오기
tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=True)

# LoRA adapter 적용
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model = model.to(device)

clova_llm = ClovaInference(model=model, tokenizer=tokenizer, sbert_model=sbert_model, mission_collection=mission_collection, hated_mission_collection=hated_mission_collection, user_query=None)
llm_missions = clova_llm.infer()
# print(llm_missions)

db = SessionLocal()

# Missions 객체를 담을 리스트
missions_to_add = []

for key, value_list in llm_missions.items():
    for emoji_value, summarized_value in value_list:
        mission = Missions(
            title=summarized_value,
            description=emoji_value,
            difficulty=key
        )
        missions_to_add.append(mission)

# 리스트를 add_all로 한번에 넣기
db.add_all(missions_to_add)
db.commit()
db.close()

if __name__ == '__main__':
  print('main.py 실행 완료!')