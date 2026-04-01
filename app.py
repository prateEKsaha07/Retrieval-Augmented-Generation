from utils import load_data,split_data
from pipeline import create_vector_store , retrieve_chunks , generate_answer

# create_vector_store(chunks)
def create_new_db():
    original_data = load_data()
    chunks = split_data(original_data)
    print(original_data[:200])
    print(len(chunks))
    print(chunks[0])

# create_new_db()

# query = input("ask something:")
# results = retrieve_chunks(query)
# print("\n Top results")
# for r in results:
#     print("-", r[:200],"\n")

# changing to new version for more natural human like answers
query = input("ask something:")
chunks = retrieve_chunks(query)
answers = generate_answer(query,chunks)