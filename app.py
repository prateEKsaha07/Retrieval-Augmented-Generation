from utils import load_data, split_data
from pipeline import create_vector_store, retrieve_chunks, generate_answer

# original_data = load_data()
# print(len(original_data))

# create DB 
# original_data = load_data()
# chunks = split_data(original_data)
# create_vector_store(chunks)


query =input("ask: ")

chunks = retrieve_chunks(query)
print(chunks)
answer = generate_answer(query,chunks)
print(answer)







# query
# query = input("ask something: ")
# chunks = retriever.get_relevent_documents(query)
# answers = generate_answer(query, chunks)

# chunks = retrieve_chunks(query)
# print("these are chunks \n")

# print(chunks)

# answers = generate_answer(query, chunks)
# print(f"the output of the query: {query} is ", "\n", {answers})