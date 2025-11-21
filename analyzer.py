import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import MODEL_PATH, ADAPTER_PATH


def load_model():
    """
    Load the fine tuned model
    """
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # load model with adapter on top
    adapter = ADAPTER_PATH
    model = PeftModel.from_pretrained(base_model, adapter)

    return model, tokenizer


def analyse_section(text, section_name, model, tokenizer):
    """
    Input: takes section text, section name(abstract or discussion), model, tokenizer
    Returns: Simple explaination of the section .
    """

    # right prompt for th e llm
    prompt = get_prompt(section_name, text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # decode/
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # remove prompt from the full_response and convert to strinf 
    response = full_response[len(prompt):].strip()

    return response 





PROMPTS = {
    "abstract": """Read this abstract and give me:
1. ELI5 (Explain Like I'm 5) - 2 sentences max
2. Key contribution - What's new? (1 sentence)

Abstract:
{text}

Response:""",
    "introduction": """Read this introduction and tell me:
1. What problem does this paper solve? (1 sentence)
2. Why does this problem matter? (1 sentence)

Introduction:
{text}

Response:""",
    "method": """Read this methods section and explain:
1. How does their approach work? (2-3 simple sentences)
2. What makes it different from other approaches? (1 sentence)

Methods:
{text}

Response:""",
    "methodology": """Read this methodology section and explain:
1. How does their approach work? (2-3 simple sentences)
2. What makes it different from other approaches? (1 sentence)

Methodology:
{text}

Response:""",
    "approach": """Read this section and explain:
1. How does their approach work? (2-3 simple sentences)
2. What's the key idea? (1 sentence)

Approach:
{text}

Response:""",
    "experiment": """Read this experiments section and tell me:
1. What did they test? (1 sentence)
2. How did they test it? (1 sentence)

Experiments:
{text}

Response:""",
    "results": """Read these results and tell me:
1. What were the main findings? (2 sentences)
2. Did it work well? (1 sentence)

Results:
{text}

Response:""",
    "discussion": """Read this discussion and tell me:
1. What are the limitations they mention? (1-2 sentences)
2. What limitations do they NOT mention? (1 sentence)

Discussion:
{text}

Response:""",
    "conclusion": """Read this conclusion and give me:
1. Main takeaway (1 sentence)
2. Should I read the full paper? Why or why not? (1 sentence)

Conclusion:
{text}

Response:""",
    "related work": """Read this related work section and tell me:
1. What previous approaches existed? (1-2 sentences)
2. How is this paper different? (1 sentence)

Related Work:
{text}

Response:""",
}


def get_prompt(section_name, text):
    """Get the right prompt for a section, truncate text if too long"""
    # Truncate to avoid token limits
    text = text[:2000]

    # Get prompt template (default to generic if section not found)
    template = PROMPTS.get(
        section_name, f"Explain this {section_name} simply:\n\n{{text}}\n\nResponse:"
    )

    return template.format(text=text)



if __name__ == "__main__":
    model, tokenizer = load_model()

    test_abstract = """
    Retrieval-Augmented Generation(RAG) has emerged as a key approach for 
    enhancing large language models with external knowledge...
    """

    result = analyse_section(test_abstract, "abstract", model, tokenizer)
    print(result)