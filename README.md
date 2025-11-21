# 📄 Paper Explainer

AI-powered tool that transforms complex arXiv research papers into simple, digestible explanations.

> Paste an arXiv link → Get ELI5 explanations of every section

## 🎯 What It Does
```bash
python main.py https://arxiv.org/abs/2406.04744
```

**Output:**
```
ABSTRACT:
--------------------
1. ELI5: RAG helps LLMs answer questions better by adding new knowledge.
2. Key contribution: CRAG dataset provides a comprehensive benchmark for evaluating RAG solutions.

INTRODUCTION:
--------------------
1. Problem: LLM answers lack factual accuracy or grounding.
2. Why it matters: Impacts reliability of QA systems used in fact-checking and customer service.

RESULTS:
--------------------
1. Main findings: RAG system can learn from web pages and APIs to answer questions.
2. Yes, it worked well - answers were accurate.

... (continues for all sections)
```

## 🛠️ Tech Stack

- **LLM**: Fine-tuned Llama 3.2 1B Instruct (LoRA)
- **PDF Processing**: PyMuPDF (fitz)
- **ML Framework**: Hugging Face Transformers + PEFT
- **Fine-tuning**: LoRA (0.07% of parameters trained)

## 📁 Project Structure
```
paper_explainer/
├── main.py              # CLI entry point
├── downloader.py        # Fetch PDFs from arXiv
├── extractor.py         # Extract text from PDFs
├── chunker.py           # Split into sections (Abstract, Methods, etc.)
├── analyzer.py          # LLM analysis with custom prompts
├── config.py            # Configuration paths
├── requirements.txt     # Dependencies
│
└── Llama/               # Fine-tuning code
    ├── train.py         # LoRA fine-tuning script
    ├── training_data.py # Training examples
    ├── format_data.py   # Data formatting for Llama
    └── test_finetuned.py# Compare base vs fine-tuned
```

## 🚀 Installation

**1. Clone the repo:**
```bash
git clone https://github.com/kush-3/paper-explainer.git
cd paper-explainer
```

**2. Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Login to Hugging Face (required for Llama):**
```bash
huggingface-cli login
```

**5. Update config.py with your adapter path:**
```python
ADAPTER_PATH = "./Llama/llama-instruct-finetuned/checkpoint-40"
```

## 💻 Usage

**Analyze any arXiv paper:**
```bash
python main.py https://arxiv.org/abs/2406.04744
```

**Run fine-tuning yourself (optional):**
```bash
cd Llama
python format_data.py      # Format training data
python train.py            # Train with LoRA
python test_finetuned.py   # Compare results
```

## 🧠 How It Works

1. **Download** - Fetches PDF from arXiv using paper ID
2. **Extract** - Converts PDF to raw text using PyMuPDF
3. **Chunk** - Splits text into sections (Abstract, Introduction, Methods, etc.)
4. **Analyze** - Sends each section to fine-tuned LLM with section-specific prompts
5. **Output** - Returns simple explanations for each section

## 🎓 Fine-Tuning Details

- **Base Model**: Llama 3.2 1B Instruct
- **Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: 851,968 (0.07% of total)
- **Training Data**: 30 examples of simple ML explanations
- **Epochs**: 10
- **Final Loss**: 2.64

## 🔮 Future Improvements

- [ ] Better section detection with regex patterns
- [ ] Handle tables and figures
- [ ] Web interface (Streamlit)
- [ ] Support for non-arXiv PDFs
- [ ] Batch processing multiple papers
- [ ] Citation extraction

## 📚 What I Learned

This project was built as part of learning ML/AI from scratch:
- Built neural networks from scratch (no frameworks)
- Learned tokenization, embeddings, and attention
- Implemented RAG (Retrieval-Augmented Generation)
- Fine-tuned LLMs using LoRA
- Built end-to-end ML pipelines

## 📝 License

MIT

---

**Built by [Kush Patel](https://github.com/kush-3)**