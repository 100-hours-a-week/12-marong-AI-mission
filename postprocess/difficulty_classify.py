from postprocess.config import VERB_MAP, VERB_DIFFICULTY_MAP
import re

class DiffiClassify:
  def __init__(self):      
    self.verb_map = VERB_MAP
    self.verb_difficulty_map = VERB_DIFFICULTY_MAP
      
  # 난이도 판별 함수
  def classify(self, sentence):
      sentence_lower = sentence.lower()
      
      # 피드 포함 시 무조건 하
      if "피드" in sentence_lower:
          return "하"

      # 동사 추출 및 표준화
      verbs = re.findall(r'(\w+기)', sentence)
      verbs_mapped = [self.verb_map.get(v, v) for v in verbs]

      if not verbs_mapped:
          return "중"  # 동사 없으면 중 기본값

      # 동사별 난이도 추출
      levels = [self.verb_difficulty_map.get(v, "중") for v in verbs_mapped]

      # 상 포함 시 상
      if "상" in levels:
          return "상"
      elif "중" in levels:
          return "중"
      else:
          return "하"