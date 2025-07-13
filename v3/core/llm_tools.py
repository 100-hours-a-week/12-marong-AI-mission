from langchain_core.runnables import RunnableLambda
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from postprocess.config import RANDOM_QUERIES
from langchain import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import random
import re
import numpy as np

def create_gen_llm_chain(model, tokenizer, device):
    """
    - HuggingFace pipeline과 PromptTemplate을 조합해 생성용 LangChain LLMChain 생성
    - rag_context, query, group_description을 입력값으로 받을 수 있는 구조
    """
    
    template = """
    아래는 기존의 마니또 미션 예시야:
    {rag_context}

    위 예시와 사용자 질문 '{query}', 그리고 마니또 게임 수행 그룹의 설명 '{group_description}'에 어울리는 '{difficulty}' 난이도의 마니또 미션 5개를 작성해줘.

    조건:
    - 각 미션은 '미션 1:', '미션 2:' 등으로 시작해.
    - 각 미션은 반드시 '~기'로 끝나는 한 문장으로 작성해.
    - 구체적 고유명사나 장소명은 포함하지 마.
    - 출력은 번호가 붙은 4개의 문장만 작성해. 다른 내용은 절대 포함하지 마.

    미션:
    """

    prompt = PromptTemplate.from_template(template)

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=120,
        temperature=0.4,
        top_p=0.8,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        batch_size=4
    )

    gen_llm = HuggingFacePipeline(pipeline=gen_pipe)

    # LangChain LLMChain: prompt | llm 구조
    gen_llm_chain = prompt | gen_llm
    return gen_llm_chain
  
def create_eval_llm_chain(model, tokenizer, device):
    """
    - HuggingFace pipeline과 PromptTemplate을 조합해 평가용 LangChain LLMChain 생성
    - query, mission을 입력값으로 받을 수 있는 구조
    """
    
    template = """
    당신은 창의적인 콘텐츠 평가 전문가입니다. 아래 조건에 따라 마니또 미션을 평가해주세요.

    ### 역할
    - 콘텐츠 평가 전문가로서, 마니또 미션의 질(일관성·적절성·창의성·실행가능성)을 점수화합니다.
    - 각 기준에 대해 1점(낮음)부터 5점(높음) 사이에서 평가해 주세요.

    ### 입력
    프롬프트: `{query}`
    생성된 미션: `{mission}`

    ### 출력 포맷
    - **답변은 반드시 쉼표로 구분된 4개의 숫자**로만 구성하세요.
    - **이유도 출력하세요.**
    - 형식 예시:

    4,5,4,5  
    일관성: 프롬프트의 의도와 유사한 표현을 잘 따랐음.  
    적절성: 부적절한 표현 없이 자연스럽게 구성됨.  
    창의성: 기존 아이디어와 유사하지만 맥락은 유지됨.  
    실행 가능성: 실제 상황에서 수행하기 쉬움.
    """

    prompt = PromptTemplate.from_template(template)

    eval_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256,
        temperature=0.4,
        top_p=0.8,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        trust_remote_code=True,
        batch_size=4
    )

    eval_llm = HuggingFacePipeline(pipeline=eval_pipe)

    # LangChain LLMChain: prompt | llm 구조
    eval_llm_chain = prompt | eval_llm
    return eval_llm_chain

def parse_eval_output(raw_output: str) -> dict:
    # 1. 점수 패턴 찾기 (쉼표로 구분된 4개의 숫자)
    score_matches = list(re.finditer(r"\b([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\b", raw_output))
    
    if len(score_matches) < 2:
        return {
            "error": "2개 이상의 점수를 찾을 수 없음",
            "raw_output": raw_output
        }
    
    # 2. 두 번째 점수 기준으로 추출
    match = score_matches[1]
    score1, score2, score3, score4 = map(int, match.groups())
    reason_text = raw_output[match.end():].strip()

    return {
        "일관성": score1,
        "적절성": score2,
        "창의성": score3,
        "수행가능성": score4,
        "총합": score1 + score2 + score3 + score4,
        "이유": reason_text
    }

def select_query_node(state: dict) -> dict:
  attempt = state.get("attempt", 0)
  difficulty_idx = state.get("difficulty_idx", 0)
  contents = state.get("contents", [[], [], []])
  sbert_model = state.get("sbert_model")
  user_query = state.get("user_query")
  random_queries = state["random_queries"]
  used_user_query = state.get("used_user_query", False)
  
  def get_avg_embedding(texts):
    if not texts:
      return None
    return np.mean(sbert_model.encode(texts, convert_to_numpy=True), axis=0)
  
  if attempt % 2 == 0 and contents[difficulty_idx]:
    print("임베딩 기반 쿼리 사용")
    query_embedding = get_avg_embedding(contents[difficulty_idx])
    query_text = "별도 없음"
  else:
    if user_query and not used_user_query:
      query_text = user_query
      query_embedding = sbert_model.encode(user_query, convert_to_numpy=True)
      used_user_query = True
    else:
      query_text = random.choice([user_query] + random_queries) if user_query else random.choice(random_queries)
      query_embedding = sbert_model.encode(query_text, convert_to_numpy=True)
      
  return {
    **state,
    "query": query_text,
    "query_embedding": query_embedding,
    "used_user_query": used_user_query
  }
  
def rag_node(state: dict) -> dict:
  query_embedding = state["query_embedding"]
  difficulty_idx = state["difficulty_idx"]
  mission_collection = state["mission_collection"]
  difficulty_list = [["상"], ["중"], ["하"]]
  
  results = mission_collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"난이도": {"$in": difficulty_list[difficulty_idx]}}
  )
  
  documents = results.get("documents", [[]])[0] if results else []
  rag_context_list = [f"- {doc}" for doc in documents] if documents else ["- (예시 없음)"]
  rag_context = "\n".join(rag_context_list)
  
  return {
    **state,
    "rag_context": rag_context,
    "rag_context_list": rag_context_list
  }
  
def generate_mission_node(state):
  """
  - rag_context, query, group_description을 기반으로 LangChain LLMChain을 실행
  - PromptTemplate와 HuggingFacePipeline으로 미션 텍스트 생성
  - 생성된 텍스트를 state에 추가
  """
  
  difficulty_keywords = {"상": "어려운", "중": "보통", "하": "쉬운"}
  
  gen_llm_chain = state["gen_llm_chain"]
  query = state["query"]
  rag_context = state["rag_context"]
  group_description = state["group_description"]
  
  input_dict = {
    "rag_context": rag_context,
    "query": query,
    "group_description": group_description,
    "difficulty": difficulty_keywords[state["current_diff"]]
  }
  
  result = gen_llm_chain.invoke(input_dict)
  print(f"Generated mission for difficulty '{state['current_diff']}': {result}")
  
  return {**state, "generated": result}

def postprocess_node(state):
  raw_output = state["generated"]  # rag_context_list에서 상위 3개만 사용
  clean_tool = state["clean_tool"]
  sbert_model = state["sbert_model"]
  hated_collection = state["hated_mission_collection"]
  final_output = state.get("final_output", {})
  
  # 1. 텍스트 전처리 및 정규식 추출
  if "미션:" in raw_output:
      text = raw_output.split("미션:")[-1].strip()
  else:
      text = raw_output.strip()

  missions = re.findall(r'(?:미션\s*\d+:)?\s*(마니띠[^\n]*?기)', text, re.MULTILINE)
  missions = list(dict.fromkeys([m.strip() for m in missions]))  # 중복 제거
  print(f"Raw missions extracted: {missions}")
  print(f"Extracted {len(missions)} missions from generated text.")

  # 2. 유효성 체크 및 혐오 미션 필터링
  cleaned = []
  for m in missions:
      if clean_tool.is_valid_mission(m) and not clean_tool.is_in_hated_collection(sbert_model, m, hated_collection, 200):
          m = re.sub(r'^\s*-\s*', '', m)
          cleaned.append(m)
  print(f"Cleaned {len(cleaned)} valid missions after filtering.")
  
  cleaned = cleaned + state["rag_context_list"][:3]

  # 3. DBSCAN 중복 제거
  if cleaned:
      embs = sbert_model.encode(cleaned)
      clustering = DBSCAN(eps=0.2, min_samples=1, metric="cosine", n_jobs=-1).fit(embs)
      unique = {}
      for idx, label in enumerate(clustering.labels_):
          if label not in unique:
              unique[label] = cleaned[idx]
      cleaned = list(unique.values())
  print(f"Deduplicated to {len(cleaned)} missions after clustering.")

  # 4. 기존 결과와 유사도 비교해서 중복 제거
  deduped = []
  for m in cleaned:
      m_emb = sbert_model.encode(m, convert_to_numpy=True)

      existing = [
          mission[0]
          for mission_list in final_output.values()
          for mission in mission_list
          if isinstance(mission, list) and len(mission) > 1 and isinstance(mission[1], str)
      ]

      if existing:
          existing_embs = sbert_model.encode(existing, convert_to_numpy=True)
          sims = cosine_similarity([m_emb], existing_embs)[0]
          if np.max(sims) < 0.8:
              deduped.append(m)
      else:
          deduped.append(m)

  return {**state, "mid_output": deduped}

def eval_node(state):
    eval_llm_chain = state["eval_llm_chain"]       # LangChain LLMChain 인스턴스
    query = state["query"]                          # 프롬프트 (생성 요청 내용)
    result = state["mid_output"]                    # 미션 리스트
    current_diff = state["current_diff"]
    emoji_gen = state["emoji_generator"]
    final_output = state.get("final_output", {})    # 누적 저장 딕셔너리
    
    for mission in result:
        input_dict = {
            "query": query,
            "mission": mission
        }
        raw_output = eval_llm_chain.invoke(input_dict)
        print("raw_output", raw_output)

        parsed_result = parse_eval_output(raw_output)
        print("parsed_result", parsed_result)
        
        # 평가 결과 출력
        print(f"Mission '{mission}' 평가 점수: {parsed_result.get('총합')}")
        print(f"Mission '{mission}' 평가 결과: {parsed_result.get('이유')}")
        print(f"Mission '{mission}' 평가 세부사항: "
              f"일관성={parsed_result.get('일관성')}, "
              f"적절성={parsed_result.get('적절성')}, "
              f"창의성={parsed_result.get('창의성')}, "
              f"수행가능성={parsed_result.get('수행가능성')}, ")
        
        if parsed_result.get("총합") < 12:
            continue

        if current_diff not in final_output:
            final_output[current_diff] = []
        
        # 결과 저장
        final_output[current_diff].append([
            emoji_gen.add_emojis(mission),
            f"마니또 미션: 🫢 {mission.strip().split()[-1]}",
            parsed_result.get("일관성", None),
            parsed_result.get("적절성", None),
            parsed_result.get("창의성", None),
            parsed_result.get("수행가능성", None),
            parsed_result.get("총합", 0),
            parsed_result.get("이유", None)
        ])
        
    return {**state, "final_output": final_output}

def check_completion_node(state):
    target_counts = state.get("target_counts")
    final_output = state["final_output"]
    current_diff = state["current_diff"]

    is_done = all(
        len(final_output.get(diff, [])) >= target_counts.get(diff, 0)
        for diff in target_counts
    )
    
    partially_done = len(final_output.get(current_diff, [])) >= target_counts.get(current_diff, 0)
    
    need_process = None
    
    if is_done:
      need_process = "end"
    elif partially_done:
      need_process = "update"
    else:
      need_process = "generate"
      
    print(f"Check completion: {need_process} for difficulty '{current_diff}'")
    
    return {**state, "done": need_process}

def update_state_node(state):
    difficulty_order = state["difficulty_order"]
    difficulty_idx = state["difficulty_idx"]
    attempt = state["attempt"]
    done = state["done"]

    # 시도 수 증가
    attempt += 1

    # 난이도 변경
    if done == "update" and difficulty_idx + 1 < len(difficulty_order):
        difficulty_idx += 1
        current_diff = difficulty_order[difficulty_idx]
    else:
        current_diff = state["current_diff"]
    
    print(f"Attempt {attempt}, Difficulty Index {difficulty_idx}, Current Difficulty: {current_diff}")

    return {
        **state,
        "attempt": attempt,
        "difficulty_idx": difficulty_idx,
        "current_diff": current_diff
    }