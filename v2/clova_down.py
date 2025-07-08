import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

print("시작: .env 로드")
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
print(f"HF_TOKEN 로드: {hf_token}")

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
save_path = "models/hyperclovax-1.5b-instruct"

os.makedirs(save_path, exist_ok=True)
print(f"디렉토리 생성 완료: {save_path}")

print("Config 다운로드 중...")
config = AutoConfig.from_pretrained(model_name, token=hf_token)
config.save_pretrained(save_path)
print("Config 다운로드 완료")

print("Tokenizer 다운로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.save_pretrained(save_path)
print("Tokenizer 다운로드 완료")

print("Model 다운로드 중...")
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
model.save_pretrained(save_path)
print("Model 다운로드 완료")