from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re
import torch

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
def extract_relevant_sentence(chunks, query):
    stopwords = {"what","is","are","which","how","the","of","in"}

    query_words = [
        w for w in re.findall(r'\b\w+\b', query.lower())
        if w not in stopwords
    ]

    best_sentence = ""

    query_lower = query.lower()
    is_definition = query_lower.startswith("what is") or query_lower.startswith("define")

    for chunk in chunks:
        sentences = re.split(r'\.\s+|\n+', chunk)
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            words = re.findall(r'\b\w+\b', sentence_lower)

            # will check each word properly(correction // exact match only)
            if any(word == w for word in query_words for w in words):

                # avoid similar word confusion confusion in my case it was confusing with TCP with TCP/IP 
                if any(f"{word}/" in sentence_lower for word in query_words):
                    continue

                # returns 2 sentences 
                # next_sentence = sentences[i+1].strip() if i+1 < len(sentence) else ""

                # robust 2 sentence logic(it will try ever line not just 2nd line and select the most maning full line as the partner line)
                next_sentence = ""

                for j in range(i+1, len(sentences)):
                    candidate = sentences[j].strip()
                    candidate_lower = candidate.lower()

                    #must contain same keyword (DNS in this case)
                    if any(word in candidate_lower for word in query_words):
                        next_sentence = candidate
                        break

                    # allows example lines in most defination cases it contains defination then examples so this will also consider that example
                    if "for example" in candidate_lower:
                        next_sentence = candidate
                        break

                if is_definition:
                    return sentence.strip()
                else:
                    return(sentence.strip()+ ". "+ next_sentence.strip()).strip()
                # return sentence.strip() + ". " + next_sentence.strip()

                # return sentence.strip() # it returns only one sentence
    return ""

# generate answers
def generate_answer(query,chunks):
    # will now try to extract only the relavant sentances
    context = extract_relevant_sentence(chunks, query)

    query_lower = query.lower()
    is_definition = query_lower.startswith("what is") or query_lower.startswith("define")

    if not is_definition:
        return context # skipping completely if its not an definition

    # prompt for LLm
    prompt = f"""
Use ALL the information given below.

Do NOT shorten the answer.
Do NOT summarize.
Return full explanation.

Text:
{context}

Question: {query}
Answer:
"""
    inputs = tokenizer(prompt,return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens = 50,
        do_sample = False,
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