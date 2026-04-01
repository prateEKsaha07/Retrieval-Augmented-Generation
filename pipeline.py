from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os

def create_vector_store(chunks):
    Embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks,Embeddings)
    
    # saving the db
    os.makedirs("db", exist_ok=True)
    vector_db.save_local("db/FAISS_index") 
    print("vectors stored successfully")


def retrieve_chunks(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.load_local(
        "db/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = db.similarity_search_with_score(query, k=5)

    filtered = []

    for doc, score in results:
        # relaxed threshold
        if score < 0.7:
            filtered.append(doc.page_content)

    if not filtered:
        filtered = [doc.page_content for doc, _ in results[:3]]

    return filtered[:3]

  
# load the model
def generate_answer(query, chunks):
    from transformers import pipeline

    context = "\n".join(chunks)

    prompt = f"""
Answer the question based only on the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
    generator = pipeline(
        "text-generation",
        model="distilgpt2"
    )

    response = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        temperature=None  # explicitely diseabled because its causing a way too much problem in run time
    )

    full_text = response[0]["generated_text"]
    answer = full_text.split("Answer:")[-1].strip()

    return answer