SECTIONS = [
    "abstract", 
    "introduction",
    "related work",
    "method",
    "methodology",
    "approach",
    "experiment",
    "results",
    "discussion",
    "conclusion",
    "references"
]

def chunk_paper(text):
    """
    Takes: Full paper text
    Returns: Dictionary with section names as key, content as values
    """
    text_lower = text.lower() 
    sections = {}

    positions = []
    for section_name in SECTIONS:
        pos = text_lower.find(section_name)
        if pos != -1:
            positions.append((pos, section_name))
    positions.sort()

    for i, (pos, name) in enumerate(positions):
        start = pos

        if i + 1 < len(positions):
            end = positions[i + 1][0]
        else:
            end = len(text)
        sections[name] = text[start:end]

    return sections 



if __name__ == "__main__":
    from extractor import extract_text

    text = extract_text("./papers/2406.04744.pdf")
    sections = chunk_paper(text)

    print(f"Found {len(sections)} sections:")
    for name, content in sections.items():
        print(f"  - {name}: {len(content)} characters .")