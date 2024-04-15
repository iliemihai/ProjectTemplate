import pickle


train_filename = "train.pkl"
with open(train_filename, "rb") as f:
    train_data = pickle.load(f)

test_filename = "test.pkl"
with open(test_filename, "rb") as f:
    test_data = pickle.load(f)

print(f"There are around: {len(train_data)} train examples...")
print(f"There are around: {len(test_data)} test examples...")
#for data in train_data:
#    print("Training dataset: ", data)

#for data in test_data:
#    print("Testing dataset: ", data)
