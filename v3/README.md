# Marong AI 미션 (Marong AI Mission)

마롱(Marong)은 마니또 기반 SNS 서비스이며, 이 저장소는 마니또 게임에 사용되는 **AI 기반 미션 생성기**를 구현한 프로젝트입니다.

![마롱](https://github.com/user-attachments/assets/eaf515d0-b8c8-4522-a22a-77e18d729853)

---

## 주요 기능

- **LangChain + HuggingFace** 기반 마니또 미션 자동 생성
- **LangGraph 기반 파이프라인 구성**으로 유연하고 재사용 가능한 모듈화 구현
- **그룹별 좋아요 피드가 많은** 콘텐츠 내용 반영하여 미션 생성
- **그룹별 설명 텍스트를 LLM 프롬프트에 반영**하여 미션 생성
- **SBERT 임베딩** + **ChromaDB RAG 검색** 기반 유사 예시 제공
- **미션 필터링**, **중복 제거**, **이모지 부착**, **난이도 분류** 등 다양한 후처리 포함
- **EXAONE 평가 LLM 기반 미션 평가** (일관성, 적절성, 창의성, 수행가능성 기준)
- **GPU 기반 최적화**된 파이프라인

---

## 아키텍처 개요

```
[사용자 쿼리 or 랜덤 쿼리 or 피드 기반 콘텐츠]
        + 그룹별 설명 텍스트 (LLM 프롬프트 포함)
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
git clone https://github.com/100-hours-a-week/12-marong-AI-mission.git
cd 12-marong-AI-mission

# 의존성 설치
pip install -r requirements.txt

# 모델 다운로드
python clova_down.py
python scripts/sbert_down.py

# ChromaDB 서버 설치
pip install "chromadb[server]"

# 필요시 수동 설치
pip install langchain_huggingface pymysql
```

---

## 사용 예시(방법)

```bash
# 1. ChromaDB 서버 실행
python scripts/run_chroma.py

# 2. 메인 실행 (기본 그룹 ID는 1)
python main.py
```

---

## 디렉토리 구조

```
12-marong-AI-mission/
├── main.py                      # 메인 실행 파일
│
├── core/
│   ├── llm_tools.py             # 미션 생성 핵심 파이프라인
│   └── data_tools.py            # Backend DB 기반 미션 생성 정보 조회 도구
│
├── db/
│   ├── db.py                    # Backend DB 연결
│   └── db_models.py             # DB 모델 정의
│
├── postprocess/
│   ├── clean_mission.py         # 미션 유효성 필터링
│   ├── config.py                # 랜덤 쿼리 등 설정값
│   ├── difficulty_classify.py   # 난이도 판별기
│   └── emoji_gen.py             # 이모지 추가 도구
│
├── scripts/
│   ├── run_chroma.py            # ChromaDB 서버 실행 스크립트
│   └── sbert_down.py            # SBERT 모델 다운로드 스크립트
│
└── README.md
```

---

## 핵심 모듈 설명

| 파일 경로                            | 모듈명/구성                       | 역할 설명                                   |
| ------------------------------------ | --------------------------------- | ------------------------------------------- |
| `core/llm_tools.py`                  | `ClovaInference`                  | 미션 생성 전체 파이프라인 관리              |
| `postprocess/clean_mission.py`       | `CleanMission`                    | 부적절하거나 비자연스러운 미션 필터링       |
| `postprocess/emoji_gen.py`           | `EmojiGen`                        | 미션 문장에 난이도 기반 이모지 부착         |
| `postprocess/difficulty_classify.py` | `DiffiClassify`                   | SBERT 기반 난이도 분류기                    |
| _(내부 모듈)_                        | `DBSCAN`                          | SBERT 임베딩 유사도 기반 중복 미션 제거     |
| _(통합 구성)_                        | `LangChain + HuggingFacePipeline` | 자연어 기반 미션 생성 엔진 (LLM 파이프라인) |

---

## 미션 생성 규칙 (Prompt 기반)

- 각 미션은 반드시 `~기`로 끝나는 한 문장
- **구체적 정보 금지** (예: 이름, 색상, 노래 제목 등)
- **IT 관련 키워드**는 반드시 포함 (예: 깃허브, 코딩, 디스코드 등)
- 마니띠의 **집/방 관련 행위 금지**
- **비밀스럽고 마니또다운** 행동 유도 (쪽지, 피드 반응, 몰래 도움 등)

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
