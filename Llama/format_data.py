import json
from training_data import training_examples

def format_for_llama(examples):
    """
    Format training data into Llama's instruction format.
    
    Llama expects:
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {response}<|eot_id|>
    """
    
    formatted_data = []
    
    for example in examples:
        # Create the formatted prompt-response pair
        formatted_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['response']}<|eot_id|>"""
        
        formatted_data.append({"text": formatted_text})
    
    return formatted_data

# Format the data
formatted_examples = format_for_llama(training_examples)

# Save to JSON file
with open("training_data.json", "w") as f:
    json.dump(formatted_examples, f, indent=2)

print(f"Formatted {len(formatted_examples)} examples")
print(f"Saved to training_data.json")

# Show one example
print("\n" + "="*50)
print("EXAMPLE FORMATTED DATA:")
print("="*50)
print(formatted_examples[0]["text"])