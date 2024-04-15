from tqdm import tqdm
import itertools
import random
import json
import pickle
from thefuzz import fuzz
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


questions_dict = json.load(open("filtered_citations_with_questions_final.json", "r"))

THRESHOLD = 97
NUM_SELECTED = 10
data = []

for cit in tqdm(questions_dict.keys()):
    for d in questions_dict[cit]:
        question = d["question"]
        response = d["text"].replace("\n", "").replace("\r", "").replace("\t", "")
        doc_id = d["doc_id"]
        pos_context = d["pos_context"]
        neg_context = d["neg_context"]

        key_to_remove = None
        for key, text in zip(pos_context.keys(), pos_context.values()):
            if fuzz.ratio(text, response) > THRESHOLD:
                contained = True
                key_to_remove = key
            else:
                contained = False

        if contained:
            pos_context.pop(key_to_remove)

 
        pos_context = [[key, val] for key, val in zip(pos_context.keys(), pos_context.values())]
        neg_context = [[key, val] for key, val in zip(neg_context.keys(), neg_context.values())]

        pos_context = pos_context[:NUM_SELECTED]
        neg_context = neg_context[:NUM_SELECTED]

        pos_context.append([doc_id, response])

        random.shuffle(pos_context)
        random.shuffle(neg_context)
        
        full_pos_context = ""
        for val in pos_context:
            d = val[0].split("_")[0]
            p = val[1].replace("\n", "").replace("\r", "").replace("\t", "")
            full_pos_context += f"Document:  {d}  Paragraph:  {p}\n"

        full_neg_context = ""
        for val in neg_context:
            d = val[0].split("_")[0]
            p = val[1].replace("\n", "").replace("\r", "").replace("\t", "")
            full_neg_context += f"Document:  {d}  Paragraph:  {p}\n"

        POS_PROMPT = f"""
### Instruction: 
Examine the provided Input and Context from bellow containing a question and paragraphs with their corresponding documents carefully. If at least one paragraph addresses the question from Input, respond with YES and specify Document: followed by the document name and Paragraph: followed by the most relevant paragraph. Extract only one Document with Paragraph from Context. Do not use information from outside the Context. If no paragraph address the question, respond with NO.

### Input:
{question}

### Context: 
{full_pos_context}

### Response: 
YES
Document: {doc_id}
Paragraph: {response}
        """
        
        if len(tokenizer(POS_PROMPT, return_tensors='pt')["input_ids"][0]) > 8192:
            print(len(tokenizer(POS_PROMPT, return_tensors='pt')["input_ids"][0]))

        NEG_PROMPT = f"""
### Instruction: 
Examine the provided Input and Context from bellow containing a question and paragraphs with their corresponding documents carefully. If at least one paragraph addresses the question from Input, respond with YES and specify Document: followed by the document name and Paragraph: followed by the most relevant paragraph. Extract only one Document with Paragraph from Context. Do not use information from outside the Context. If no paragraph address the question, respond with NO.


### Input:
{question}

### Context:
{full_neg_context}

### Response: 
NO
Document: None
Paragraph: None
        """

        if len(tokenizer(NEG_PROMPT, return_tensors='pt')["input_ids"][0]) > 8192:
            print(len(tokenizer(NEG_PROMPT, return_tensors='pt')["input_ids"][0]))


        data.append([POS_PROMPT, NEG_PROMPT])


train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

train_data = list(itertools.chain.from_iterable(train_data))
test_data = list(itertools.chain.from_iterable(test_data))
print("TRAIN LEN: ", len(train_data), train_data[:5])
print("TEST LEN: ", len(test_data))

with open('train.pkl', 'wb') as file:
    pickle.dump(train_data, file)

with open('test.pkl', 'wb') as file:
    pickle.dump(test_data, file)
