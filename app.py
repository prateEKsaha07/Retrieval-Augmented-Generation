from utils import load_data, split_data
from pipeline import create_vector_store, retrieve_chunks, generate_answer

# original_data = load_data()
# print(len(original_data))

# # STEP 1: create DB again (run only once)
# original_data = load_data()
# chunks = split_data(original_data)
# create_vector_store(chunks)

# # STEP 2: query
query = input("ask something: ")
chunks = retrieve_chunks(query)
print("these are chunks \n")

print(chunks)

answers = generate_answer(query, chunks)
print(f"the output of the query: {query} is ", "\n", {answers})