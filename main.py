import sys 
from downloader import download_paper
from extractor import extract_text
from chunker import chunk_paper
from analyzer import load_model, analyse_section

def main():
    # get url 
    url = sys.argv[1]
    print("Downloading URL")

    # download
    file_path = download_paper(url)

    # text extraction
    text = extract_text(file_path)
    print("Extracting Text ...")


    # chunk
    print("splitting into sections ...")
    sections = chunk_paper(text)
    print(f"Found {len(sections)} sections .")

    # load model and tokenizer
    print("Loading model and tokeniser ....")
    model, tokenizer = load_model()

    # analyze each section
    print("\n" + "="*50)
    print("PAPER ANALYSIS")
    print("="*50)
    for section_name, section_text in sections.items():
        if section_name == "references":
            continue
        print(f"\n{section_name.upper()}:")
        print("-"*20)
        result = analyse_section(section_text, section_name, model, tokenizer)        
        print(result)



if __name__ == "__main__":
    main()
