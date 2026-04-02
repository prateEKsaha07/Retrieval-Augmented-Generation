import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_data():
    with open("data/data.txt", "r") as f:
        return f.read()

def split_data(data):
    # split by sections (## headings)
    sections = re.split(r'\n##\s+', data)

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for section in sections:
        if not section.strip():
            continue

        # optional: keep section title
        section_chunks = splitter.split_text(section)
        all_chunks.extend(section_chunks)

    return all_chunks