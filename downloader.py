import requests 
import os

def download_paper(url):
    # Extract paper ID
    paper_id = url.split('/')[-1]
    
    # Build PDF URL
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    # Create folder
    papers_dir = "./papers"
    os.makedirs(papers_dir, exist_ok=True)
    
    # Download
    response = requests.get(pdf_url)
    
    if response.status_code == 200:
        file_path = f"{papers_dir}/{paper_id}.pdf"
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_path}")
        return file_path
    else:
        raise Exception(f"Failed to download. Status: {response.status_code}")

# Test it
if __name__ == "__main__":
    url = "https://arxiv.org/abs/2406.04744"
    path = download_paper(url)
    print(f"Saved to: {path}")