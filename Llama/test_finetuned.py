import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading base model ....")
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading fine tuned adapters ....")
finetuned_model = PeftModel.from_pretrained(base_model, "./llama-instruct-finetuned/checkpoint-40")
print("Models loaded \n")

def generate(model, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response 


test_prompts = [
    "Explain gradient descent like I'm 5:",
    "Describe overfitting like you're talking to a friend:",
    "What is backpropagation in simple terms?",
]


print("="*60)
print("COMPARING BASE vs FINE-TUNED")
print("="*60)

for prompt in test_prompts:
    print(f"\n📝 PROMPT: {prompt}\n")
    
    print("🔴 BASE MODEL:")
    print(generate(base_model, prompt))
    print()
    
    print("🟢 FINE-TUNED MODEL:")
    print(generate(finetuned_model, prompt))
    print("-"*60)