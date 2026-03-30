from langchain_text_splitters import RecursiveCharacterTextSplitter

# data loader
def load_data():
    with open("data/data.txt", "r") as f:
        data = f.read()
    return data

def split_data(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        separators = ["\n\n", "\n", ".", " ",""]
    )
    chunks = splitter.split_text(data)
    return chunks
