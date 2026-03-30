from utils import load_data,split_data

original_data = load_data()
print(original_data[:200])

chunks = split_data(original_data)
print(len(chunks))
print(chunks[0])