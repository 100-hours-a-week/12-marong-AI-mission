import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드 (HyperCLOVA-X 1.5B)
model_path = "./models/hyperclovax-1.5b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)

# LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)

# JSONL 파일 로드 및 변환
json_path = "./manitto_gitmission_fin.jsonl"
df = pd.read_json(json_path, lines=True)
df["text"] = df["prompt"] + "\n" + df["completion"]
dataset = Dataset.from_pandas(df[["text"]])

# 토큰화
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 데이터 Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="no",
    fp16=True if device == "cuda" else False,
    gradient_accumulation_steps=4,
    report_to="none",
    remove_unused_columns=False
)

# Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 학습 시작
trainer.train()

# LoRA 모델 저장
model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")