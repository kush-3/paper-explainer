from downloader import download_paper
import fitz

def extract_text(pdf_path):
    """
    Takes pdf_path as input and returns a full text content as string 
    """
    doc = fitz.open(pdf_path)
    out_text = ""
    for page in doc:
        text = page.get_text()
        out_text += text

    return out_text

if __name__ == "__main__":
    text = extract_text("papers/2406.04744.pdf")
    print(f"Extracted {len(text)} characters")
    print(f"First 500 characters: \n{text[:500]}")

