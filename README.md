# Marong AI 미션 (Marong AI Mission)

**마롱(Marong)**은 마니또 기반 SNS 서비스이며, 이 저장소는 마니또 게임에 사용되는 **AI 기반 미션 생성기**를 구현한 프로젝트입니다.

![마롱](https://github.com/user-attachments/assets/eaf515d0-b8c8-4522-a22a-77e18d729853)

## 주요 기능

- **LangChain + HuggingFace** 기반 마니또 미션 자동 생성
- **SBERT 임베딩** + **ChromaDB RAG 검색** 기반 유사 예시 제공
- **미션 필터링**, **중복 제거**, **이모지 부착**, **난이도 분류** 등 다양한 후처리 포함
- **GPU 기반 최적화**된 파이프라인

---

## 아키텍처 개요

```
[사용자 쿼리 or 랜덤 쿼리 or 피드(좋아요 수 많은 컨텐츠)]
         ↓
    유사 예시 검색 (ChromaDB)
         ↓
LLM 텍스트 생성 (LangChain + HuggingFace)
         ↓
    문장 정제 / 난이도 분류
         ↓
    중복 제거 / 이모지 부착
         ↓
     최종 미션 리스트 반환
```

---

## 설치 방법

```bash
# 초기 설정
python clova_down.py
pip install "chromadb[server]"
python scripts/sbert_down.py
# 필요시 실행
pip install langchain_huggingface
pip install pymysql
ps aux | grep chroma
kill 포트번호
python scripts/run_chroma.py
python main.py
```

---

## 사용 예시

```python
from clova_inference import ClovaInference
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

model = AutoModelForCausalLM.from_pretrained("your-llm")
tokenizer = AutoTokenizer.from_pretrained("your-llm")
sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

clova = ClovaInference(
    model=model,
    tokenizer=tokenizer,
    sbert_model=sbert_model,
    contents=[[], user_posts, []],  # 난이도별 사용자 피드
    mission_collection=mission_chroma_collection,
    hated_mission_collection=hated_mission_collection,
    user_query="마니띠에게 몰래 선물 주는 미션 추천해줘"
)

output = clova.infer(target_counts={'상': 0, '중': 3, '하': 0})
print(output)
```

---

## 디렉토리 구조

```
12-marong-AI-mission/
├── clova_inference.py
├── postprocess/
│   ├── clean_mission.py         # 미션 유효성 필터
│   ├── emoji_gen.py             # 이모지 추가 도구
│   ├── difficulty_classify.py   # 난이도 판별기
│   └── config.py                # 랜덤 쿼리 등 설정값
└── README.md
```

---

## 핵심 모듈 설명

| 모듈                              | 설명                       |
| --------------------------------- | -------------------------- |
| `ClovaInference`                  | 미션 생성 전체 파이프라인  |
| `CleanMission`                    | 부적절한 미션 내용 필터링  |
| `EmojiGen`                        | 랜덤 이모지 부착기         |
| `DiffiClassify`                   | SBERT 기반 난이도 분류     |
| `DBSCAN`                          | 유사도 기반 중복 미션 제거 |
| `LangChain + HuggingFacePipeline` | 자연어 미션 생성기         |

---

## 미션 생성 규칙 (Prompt 기반)

- 각 미션은 반드시 `~기`로 끝나는 한 문장
- **구체적 정보 금지** (예: 이름, 색상, 노래 제목 등)
- **IT 관련 키워드**는 반드시 포함 (깃허브, 프로그래밍 등)
- 마니띠의 **집/방 관련 행위 금지**
- 비밀스럽고 마니또다운 행동 유도 (쪽지, 피드, 몰래 도움 등)

---

## 출력 예시

```python
{
  '중': [
    ('마니띠의 디스코드 메시지에 귀여운 이모지 달기 😺', '마니또 미션: ⭐️ 달기'),
    ('마니띠가 쓴 개발 용어에 리액션 달기 💻', '마니또 미션: ⭐️ 달기'),
    ('마니띠가 올린 글을 조용히 북마크 하기 📌', '마니또 미션: ⭐️ 하기')
  ]
}
```

---

## 향후 확장 방향

- 미션 피드백 기반 난이도/테마 자동 조정
- 그룹 설명 텍스트 기반 미션 생성
- 미션 성공 여부 데이터와 사용자 성향 분석 연동

---
