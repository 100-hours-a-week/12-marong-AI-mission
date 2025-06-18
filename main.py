from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import DBSCAN
from core.clova_inference import ClovaInference
from model.sbert_wrapper import SBERTWrapper
from chromadb import HttpClient
from dotenv import load_dotenv
from db.db import SessionLocal
from db.db_models import Missions
from peft import PeftModel
from main_tool import get_top_posts
import torch, os

# sbert_wrapper 생성
sbert_model = SentenceTransformer('./kr-sbert', device='cpu')
sbert_wrapper = SBERTWrapper(sbert_model)

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chroma embedding function
def embedding_func(texts):
    return sbert_model.encode(texts, convert_to_numpy=True).tolist()

load_dotenv()

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

# base 모델 이름 (huggingface hub or local)
base_model_name = "models/hyperclovax-1.5b-instruct"

# adapter가 저장된 로컬 경로
MODEL_PATH = os.getenv("MODEL_PATH")

# base model과 tokenizer 불러오기
tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=True)

# LoRA adapter 적용
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model = model.to(device)

db = SessionLocal()
contents_high = get_top_posts(db, "상", 2)
contents_middle = get_top_posts(db, "중", 3)
contents_low = get_top_posts(db, "하", 3)

contents = [contents_high, contents_middle, contents_low]
print(contents, "피드 데이터 DB 추출 테스트 완료!")

clova_llm = ClovaInference(model=model, contents=contents, tokenizer=tokenizer, sbert_model=sbert_model, mission_collection=mission_collection, hated_mission_collection=hated_mission_collection, user_query=None)
llm_missions = clova_llm.infer()

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