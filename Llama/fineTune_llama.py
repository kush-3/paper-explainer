import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if MPS (Apple Silicon) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading Llama 3.2 1B... (this may take a minute)")
model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded!")

# Test the base model
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test prompts - we'll compare these before and after fine-tuning
test_prompts = [
    "Explain backpropagation in simple terms:",
    "What is a neural network?",
    "Explain gradient descent like I'm 5:",
]

print("\n" + "="*50)
print("BASE MODEL RESPONSES (Before Fine-tuning)")
print("="*50 + "\n")

for prompt in test_prompts:
    print(f"PROMPT: {prompt}")
    print(f"RESPONSE: {generate_response(prompt)}")
    print("-"*50 + "\n")