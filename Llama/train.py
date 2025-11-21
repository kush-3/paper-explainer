import torch 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers.data import data_collator

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# load model
model_name = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"Model loaded! Parameters: {model.num_parameters():,}")


# lora configuration 
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# load training data 
print("Loading training data .....")
dataset = load_dataset("json", data_files='training_data.json', split='train')
print(f"loaded {len(dataset)} examples")

# tokenize function 
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

print(f"Tokenized each example is {256} tokens .")

# traininng configuration
training_args = TrainingArguments(
    output_dir="./llama-instruct-finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=5,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    report_to="none",
    use_cpu=False,
)
print("Training arguments cinfigured .")

# data collatort -handles batching and padding of the data 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
print("Trainer is ready")


# model training
print("\n" + "="*50)
print("starting fine-tuning")
print("="*50 + "\n")

trainer.train()

# save the fine-tuined model
print("saving model ...")
model.save_pretrained("./llama-finetuned")
tokenizer.save_pretrained("./llama-finetuned")

print("\n" + "="*50)
print("Fine tuning complete")
print("model saved to ./llama-finetuned")
print("="*50)

