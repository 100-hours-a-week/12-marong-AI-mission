# GPU-Only
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain import PromptTemplate
from postprocess.clean_mission import CleanMission
from postprocess.emoji_gen import EmojiGen
from postprocess.difficulty_classify import DiffiClassify
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from postprocess.config import RANDOM_QUERIES
import torch, re, random
import numpy as np

class ClovaInference:
  def __init__(self, model, contents, tokenizer, sbert_model, mission_collection, hated_mission_collection, group_description, user_query=None):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = tokenizer
    self.model = model
    self.sbert_model = sbert_model
    self.difficulty_list = [["상"], ["중"], ["하"]]
    self.mission_collection = mission_collection
    self.hated_mission_collection = hated_mission_collection
    self.clean_tool = CleanMission()
    self.emoji_generator = EmojiGen()
    self.difficulty_classifier = DiffiClassify()
    self.contents = contents
    self.group_description = group_description

    self.pipe = pipeline(
        "text-generation",
        model=self.model,
        tokenizer=self.tokenizer,
        device=0 if self.device == "cuda" else -1,
        max_new_tokens=120,
        temperature=0.4,
        top_p=0.8,
        do_sample=True,
        eos_token_id=self.tokenizer.eos_token_id,
        batch_size=4
    )
    self.llm = HuggingFacePipeline(pipeline=self.pipe)
    self.user_query = user_query
    self.random_queries = RANDOM_QUERIES
      
    # 프롬프트 작성
    self.template = """
    아래는 기존의 마니또 미션 예시야:
    {rag_context}

    위 예시와 사용자 질문 '{query}', 그리고 마니또 게임 수행 그룹의 설명 '{group_description}'에 어울리는 쉬운 난이도의 마니또 미션 5개를 작성해줘.

    조건:
    - 각 미션은 '미션 1:', '미션 2:', '미션 3:', '미션 4:', '미션 5:' 으로 시작해.
    - 각 미션은 반드시 '~기'로 끝나는 한 문장으로 작성해.
    - 마니띠의 취향, 이름, 색깔, 노래 제목 등 구체적인 정보는 포함하지 마. 예를 들어 '분홍색', '아로하' 같은 특정 정보는 사용하지 말고, 포괄적인 표현으로 작성해.
    - 위 예시를 참고해서 마니띠에게 쪽지를 보내거나, 피드를 보내거나 몰래 도와주는 것과 같이 비밀스럽게 그 사람을 위해 미션을 수행하는 내용이어야 해.
    - 위 예시에서 컴퓨터 개발, 프로그래밍, 깃허브, 노션 등 IT와 관련된 내용이 있으면 반드시 해당 내용도 포함해줘.
    - 마니띠의 집이나 방에서 수행하는 미션은 절대 포함하지 마.
    - 출력은 번호를 붙인 4개의 문장만 작성해. 다른 내용은 절대 포함하지 마.
    - 예시: 미션 1: 마니띠의 디스코드 메시지에 귀여운 이모지 달기

    미션:
    """
    
    self.prompt = PromptTemplate.from_template(self.template)
    self.llm_chain = self.prompt | self.llm

  def infer(self, target_counts={'상': 4, '중': 12, '하': 20}, k=5):
    difficulty_classifier = DiffiClassify()
      
    def get_average_post_embedding(contents: list[str], model):
        if not contents:
            return []
            
        model = model        
        embeddings = model.encode(contents, convert_to_numpy=True)
        return np.mean(embeddings, axis=0)

    high_embedding = None if not self.contents[0] else get_average_post_embedding(self.contents[0], self.sbert_model)
    middle_embedding = None if not self.contents[1] else get_average_post_embedding(self.contents[1], self.sbert_model)
    low_embedding = None if not self.contents[2] else get_average_post_embedding(self.contents[2], self.sbert_model)
    embedding_vectors = [high_embedding, middle_embedding, low_embedding]
      
    # 미션 생성 루프
    max_attempts = 200
    attempt = 0
    final_output = {'상': [], '중': [], '하': []}
    
    difficulty_idx = 0
    difficulty_order = ["상", "중", "하"]
    Flag = False

    while (any(len(final_output[level]) < target_counts[level] for level in target_counts)) and attempt < max_attempts:
        attempt += 1
        current_diff = difficulty_order[difficulty_idx]
        
        if len(final_output[current_diff]) >= target_counts[current_diff]:
            if difficulty_idx < 2:
                difficulty_idx += 1
            continue
        
        if attempt % 2 == 0 and embedding_vectors[difficulty_idx] is not None:
            query_emb = embedding_vectors[difficulty_idx]
            query = "별도 없음."
            print("피드 좋아요 수 기반 미션 선택!")
        else:
            if not self.user_query:
                query = random.choice(self.random_queries)
                print(f" 랜덤 선택된 쿼리: {query}")
            # user_query가 별도로 있는 경우
            elif self.user_query and not Flag:
                query = self.user_query
                Flag = True
            elif self.user_query and Flag:
                query = random.choice([self.user_query] + random.sample(self.random_queries, 5))
            else:
                query = random.choice(self.random_queries)
                print(f" 랜덤 선택된 쿼리: {query}")
                
            query_emb = self.sbert_model.encode(query, convert_to_numpy=True).tolist()
            
        print(f"\n 시도 {attempt}번째... (현재 난이도: {current_diff})")
        
        results = self.mission_collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        where={"난이도": {"$in": self.difficulty_list[difficulty_idx]}}
        )
        
        rag_context_list = [f"- {doc}" for doc in results['documents'][0]] if results and results['documents'] else ["- (예시 없음)"]
        rag_context = "\n".join(rag_context_list)
        print(rag_context)

        response_raw = self.llm_chain.invoke({"rag_context": rag_context, "query": query, "group_description": self.group_description})

        # 후처리
        if "미션:" in response_raw:
            after_mission = response_raw.split("미션:")[-1].strip()
        else:
            after_mission = response_raw.strip()

        matches = re.findall(r'(?:미션\s*\d+:|^\d+:)\s*([^\n]*?기)', after_mission, re.MULTILINE)

        cleaned_output = []
        for m in dict.fromkeys([m.strip() for m in matches]):
            if self.clean_tool.is_valid_mission(m) and not self.clean_tool.is_in_hated_collection(self.sbert_model, m, self.hated_mission_collection, 200):
                cleaned_output.append(m)

        print(" rag_context 예시:", rag_context_list)
        print()
        
        sample_size = min(4, len(rag_context_list))
        cleaned_output = cleaned_output + random.sample(rag_context_list, sample_size)

        if cleaned_output:
            # SBERT 임베딩 + DBSCAN 중복 제거
            embeddings = self.sbert_model.encode(cleaned_output)
            clustering = DBSCAN(eps=0.2, min_samples=1, metric="cosine", n_jobs=-1).fit(embeddings)

            clusters = {}
            for idx, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = cleaned_output[idx]

            cleaned_output = list(clusters.values())

        deduped_cleaned = []
        for m in cleaned_output:
            # "-" 접두사 제거
            if m.startswith("- "):
                m = m[2:].strip()

            m_emb = self.sbert_model.encode(m, convert_to_numpy=True)
            flat_final_output = [m for sublist in final_output.values() for m in sublist]

            if flat_final_output:
                final_embs = self.sbert_model.encode(flat_final_output, convert_to_numpy=True)
                sims = cosine_similarity([m_emb], final_embs)[0]
                if np.max(sims) < 0.8:
                    deduped_cleaned.append(m)
            else:
                deduped_cleaned.append(m.strip())

        for m in deduped_cleaned:
            if len(final_output[current_diff]) < target_counts[current_diff]:
                last_word = m.strip().split()[-1]
                modified_m = f"마니또 미션: ⭐️ {last_word}"
                
                if current_diff == difficulty_classifier.classify(m):
                    final_output[current_diff].append((self.emoji_generator.add_emojis(m), modified_m))
        
        if len(final_output[current_diff]) >= target_counts[current_diff]:
            if difficulty_idx < 2:
                difficulty_idx += 1

        print(final_output)
        print()

    return final_output