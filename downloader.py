import requests
import os


def download_paper(url):
    papers_dir = "./papers"
    os.makedirs(papers_dir, exist_ok=True)

    # Check if it's an arXiv abstract page
    if "arxiv.org/abs/" in url:
        paper_id = url.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        filename = f"{paper_id}.pdf"

    # Check if it's already a direct PDF link
    elif url.endswith(".pdf"):
        pdf_url = url
        filename = url.split("/")[-1]

    # Otherwise, assume it's an arXiv PDF page
    elif "arxiv.org/pdf/" in url:
        pdf_url = url
        paper_id = url.split("/")[-1].replace(".pdf", "")
        filename = f"{paper_id}.pdf"

    else:
        raise Exception("Unsupported URL. Use arXiv link or direct PDF URL.")

    # Download
    response = requests.get(pdf_url)

    if response.status_code == 200:
        file_path = f"{papers_dir}/{filename}"
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {file_path}")
        return file_path
    else:
        raise Exception(f"Failed to download. Status: {response.status_code}")


if __name__ == "__main__":
    url = "https://arxiv.org/abs/2406.04744"
    path = download_paper(url)
    print(f"Saved to: {path}")
