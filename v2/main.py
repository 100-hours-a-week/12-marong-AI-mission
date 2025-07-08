from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, END
from postprocess.clean_mission import CleanMission
from postprocess.emoji_gen import EmojiGen
from postprocess.difficulty_classify import DiffiClassify
from model.sbert_wrapper import SBERTWrapper
from chromadb import HttpClient
from dotenv import load_dotenv
from db.db import SessionLocal
from db.db_models import Missions, GroupMissions
from peft import PeftModel
from datetime import datetime
from main_tool import get_top_posts, get_group_info, largest_mission_id
from tool.get_week_index import GetWeekIndex
from core.clova_tools import (
    create_llm_chain,
    select_query_node,
    rag_node,
    generate_mission_node,
    postprocess_node,
    check_completion_node,
    update_state_node
)
from postprocess.config import RANDOM_QUERIES
import os
import torch

# 초기 환경 설정
base_date = datetime(2025, 1, 6)
today = datetime.today()
week_index = GetWeekIndex(today, base_date).get()

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('./kr-sbert', device='cpu')
sbert_wrapper = SBERTWrapper(sbert_model)

# Chroma DB 클라이언트 설정
load_dotenv()
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, ssl=False)
mission_collection = chroma_client.get_or_create_collection(name="mission_collection")
hated_mission_collection = chroma_client.get_or_create_collection(name="hated_mission_collection")

# 모델 로드
MODEL_PATH = os.getenv("MODEL_PATH")
base_model_name = "models/hyperclovax-1.5b-instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=True)
model = PeftModel.from_pretrained(base_model, MODEL_PATH).to(device)

# LLMChain 생성
llm_chain = create_llm_chain(model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Graph 정의
graph = StateGraph(dict)
graph.add_node("select_query", select_query_node)
graph.add_node("rag", rag_node)
graph.add_node("generate", generate_mission_node)
graph.add_node("postprocess", postprocess_node)
graph.add_node("check_completion", check_completion_node)
graph.add_node("update_state", update_state_node)

graph.set_entry_point("check_completion")

graph.add_conditional_edges("check_completion", lambda state: "END" if state["done"] == "end" else "update_state" if state["done"] == "update" else "select_query", {
    "update_state": "update_state",
    "select_query": "select_query",
    "END": END
})
graph.add_edge("update_state", "check_completion")
graph.add_edge("select_query", "rag")
graph.add_edge("rag", "generate")
graph.add_edge("generate", "postprocess")
graph.add_edge("postprocess", "check_completion")

mission_graph = graph.compile()

# DB 작업
db = SessionLocal()
group_info = get_group_info(db)

for g_id, g_desc in group_info.items():
    try:
        m_id = largest_mission_id(db) + 1
        contents = [
            get_top_posts(db, "상", g_id, 3),
            get_top_posts(db, "중", g_id, 3),
            get_top_posts(db, "하", g_id, 3)
        ]

        # 초기 state 구성
        initial_state = {
            "attempt": 0,
            "difficulty_idx": 0,
            "difficulty_order": ["상", "중", "하"],
            "current_diff": "상",
            "final_output": {"상": [], "중": [], "하": []},
            "target_counts": {"상": 0, "중": 3, "하": 0},
            "clean_tool": CleanMission(),
            "emoji_generator": EmojiGen(),
            "sbert_model": sbert_model,
            "hated_mission_collection": hated_mission_collection,
            "mission_collection": mission_collection,
            "contents": contents,
            "group_description": g_desc,
            "user_query": None,
            "random_queries": RANDOM_QUERIES,
            "llm_chain": llm_chain
        }

        result = mission_graph.invoke(initial_state, config={"recursion_limit": 200})
        final_output = result["final_output"]
        print(f"[성공] final_output: {final_output}")

        missions_to_add = []
        group_missions_to_add = []

        for level, missions in final_output.items():
            for emoji_mission, summary_mission in missions:
                mission = Missions(
                    id=m_id,
                    title=summary_mission,
                    description=emoji_mission,
                    difficulty=level
                )
                group_mission = GroupMissions(
                    group_id=g_id,
                    mission_id=m_id,
                    week=week_index,
                    max_assignable=5,
                    remaining_count=5
                )
                missions_to_add.append(mission)
                group_missions_to_add.append(group_mission)
                m_id += 1

        db.add_all(missions_to_add)
        db.add_all(group_missions_to_add)
        db.commit()
        print(f"[성공] {g_id}, {g_desc} 그룹 미션 커밋 완료")
    except Exception as e:
        db.rollback()
        print(f"[오류] 그룹 ID {g_id} 미션 생성 중 오류 발생: {e}")

db.close()

if __name__ == "__main__":
    print("LangGraph 기반 미션 생성 완료")