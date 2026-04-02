from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re


# GLOBAL MODELS
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./models"
)
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

DB_PATH = "db/faiss_index"

# CREATING VECTOR DATABASE

def create_vector_store(chunks):
    if not chunks:
        raise ValueError("No chunks provided to create vector store.")

    print("Creating FAISS vector store...")
    vector_db = FAISS.from_texts(chunks, EMBEDDINGS)

    os.makedirs("db", exist_ok=True)
    vector_db.save_local(DB_PATH)

    print("Vector DB created successfully!")

# RETRIEVING RELEVANT CHUNKS
def retrieve_chunks(query, k=5):
    if not os.path.exists(DB_PATH):
        raise ValueError("FAISS DB not found. Run create_vector_store first.")

    db = FAISS.load_local(
        DB_PATH,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    results = db.similarity_search_with_score(query, k=2)

    # basic keyword filtering
    stopwords = {"who", "are", "is", "the", "what", "which"}
    query_words = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords]

    scored_chunks = []

    for doc, score in results:
        text = doc.page_content.lower()
        keyword_match = sum(word in text for word in query_words)

        # relaxed threshold
        if score < 1.5:
            scored_chunks.append((doc.page_content, score, keyword_match))

    # fallback if nothing passes filter
    if not scored_chunks:
        return [doc.page_content for doc, _ in results[:3]]

    # sort: best similarity + keyword relevance
    scored_chunks.sort(key=lambda x: (x[1], -x[2]))

    return [chunk[0] for chunk in scored_chunks[:2]]


# extraction function 
def extract_relevant_sentences(chunks, query):
    query_words = query.lower().split()
    best_sentence = []
    max_score = 0

    for chunk in chunks:
        sentences = chunk.split(".")
        for sentence in sentences:
            score = sum(word in sentence.lower() for word in query_words)
            if score > max_score:
                max_score = score
                best_sentence = sentence.strip()
    return best_sentence

# def get_chunks(query):
#     docs = retriever.get_relevant_document(query)
#     return [doc.page_content for doc in docs]

# generate answers
def generate_answer(query,chunks):
    # will now try to extract only the relavant sentances
    context = extract_relevant_sentences(chunks, query)

    # prompt for LLm
    prompt = f"""
answer the following question using the sentence

sentence :{context}

question :{query}

answer in clear one sentence 
"""
    inputs = tokenizer(prompt,return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens = 50,
        do_sample = False,
        num_beams = 4
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# GENERATE ANSWER
# im scapping this fution because there were some problem with the model while generating answers so im shifting to hybrid rag for some exprimentation
# def generate_answer(query, chunks):
#     TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-small")
#     MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
#     if not chunks:
#         return "I don't have enough information to answer that."

#     context = "\n".join(chunks)

#     prompt = f"""
# Answer the question using only the given context.

# Return the answer as ONE complete sentence.
# Start with the term asked in the question.

# Do not list phrases. Do not repeat words.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

#     inputs = TOKENIZER(prompt, return_tensors="pt")

#     output = MODEL.generate(
#         **inputs,
#         max_new_tokens=60,
#         min_length=30,
#         do_sample=False,
#         num_beams=4,
#         early_stopping=True
#     )

#     answer = TOKENIZER.decode(output[0], skip_special_tokens=True)
#     return answer