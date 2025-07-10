from sklearn.metrics.pairwise import cosine_similarity
from postprocess.config import BLOCKED_KEYWORDS, BLOCKED_PHRASES, BLOCKED_SUFFIXES, FOOD_PATTERNS
import numpy as np
import re

class CleanMission:
  def __init__(self):    
    self.blocked_keywords = BLOCKED_KEYWORDS
    
  def is_valid_mission(self, mission):    
    if len(mission) < 7:
        return False
    if len(mission) > 30:  # 너무 길면 제거
        return False
    if mission[-1:] != "기":
        return False
    if mission[-2:] == " 기":
        return False
    if any(phrase in mission for phrase in BLOCKED_PHRASES):
        return False
    if re.search(r'[^\uAC00-\uD7A3\u3131-\u318E\u1100-\u11FF\u0020-\u007E]', mission):
        # 한글, 영문, 숫자, 공백 외의 이상한 문자 포함 시
        return False
    # 조사 + 동사 형태 제거
    if re.search(r"제목에\s*[가-힣]+\s*기", mission):
        return False
    if re.search(r"마니띠가\s*좋아하는\s*\S+\s*추천하기", mission):
        return False
    if re.search(r"마니띠가\s*좋아하는\s*(\S+?인)", mission):
        return False
    if re.search(r"마니띠\s*\S+\s*불러주기", mission):
        return False
    if re.search(r"나의 음악 취향을\s*\S+\s*칭찬해주기", mission):
        return False
    if re.search(r"마니띠\s*\S+\s*만들기", mission):
        return False
    if re.search(r"마니띠\s*\S+\s*만들어주기", mission):
        return False
    if re.search(r"마니띠\s*본", mission):
        return False
    if re.search(r"별점\s*(주기|남기기)", mission):
        return False
    if re.search(r"표정\s*\S+\s*지목", mission):
        return False
    if not mission.endswith("기"):
        return False
    if mission.strip().endswith("기"):  # "기" 단독 끝나는 문장 제거
        if len(mission.strip().split()) <= 3:  # 너무 짧은 문장인 경우
            return False
    if mission.startswith("- "):
        mission = mission[2:].strip()
    if any(bad in mission for bad in self.blocked_keywords):
        return False
    if mission.endswith(BLOCKED_SUFFIXES):
        return False
    for pattern in FOOD_PATTERNS:
        if re.search(pattern, mission):
            return False
          
    # 동어 반복 3회 이상 필터링
    verbs = re.findall(r'(\w+기)', mission)
    verb_counts = {}
    for v in verbs:
        verb_counts[v] = verb_counts.get(v, 0) + 1
    if any(count >= 3 for count in verb_counts.values()):
        return False
    
    return True
  
  def is_in_hated_collection(self, sbert_model, mission, hated_mission_collection, SIM_THRESHOLD):
    try:
        # 임베딩 직접 생성
        mission_emb = sbert_model.encode(mission, convert_to_numpy=True).tolist()

        # hated_mission_collection에서 임베딩 검색 (top-1)
        results = hated_mission_collection.query(
            query_embeddings=[mission_emb],
            n_results=1
        )

        # 결과에서 유사도 계산
        if results and results['embeddings'] and results['embeddings'][0]:
            db_emb = np.array(results['embeddings'][0][0])  # top-1
            similarity = cosine_similarity([mission_emb], [db_emb])[0][0]

            # ChromaDB는 cosine distance 기반 → 유사도는 1 - distance
            return similarity >= SIM_THRESHOLD
        else:
            return False
    except Exception as e:
        print(f"Error during burden check: {e}")
        return False