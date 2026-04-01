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

    stopwords = {"who", "are", "is", "the", "what", "which"}
    query_words = [w for w in query.lower().split() if w not in stopwords]

    scored_chunks = []

    for doc, score in results:
        text = doc.page_content.lower()
        keyword_match = sum(word in text for word in query_words)

        if score < 0.7:
            scored_chunks.append((doc.page_content, score, keyword_match))

    if not scored_chunks:
        return [doc.page_content for doc, _ in results[:3]]

    # sort by similarity + keyword relevance
    scored_chunks.sort(key=lambda x: (x[1], -x[2]))

    return [chunk[0] for chunk in scored_chunks[:2]]

  
# load the model
def generate_answer(query, chunks):
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    context = "\n".join(chunks)

    prompt = f"""You are a helpful AI assistant.

    Answer the question using the context below.
    Explain the answer in 2-4 complete sentences.
    Do not give short phrases or incomplete answers.
    Include important details from the context.

    Context:{context}

    Question:{query}

    Answer:
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    inputs = tokenizer(prompt,return_tensors= "pt")

    # loading new model
    output = model.generate(
        **inputs,
        max_new_tokens = 200,
        min_length = 60, #incresing the answer's size
        do_sample = False,
        repetition_penalty = 1.2, # stopping from repeating same line agin and again
        no_repeat_ngram_size = 3  #  blocks phrase loops
    )
    
    answer = tokenizer.decode(output[0],skip_special_tokens = True)
    return answer