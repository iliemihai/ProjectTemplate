from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import itertools
import threading
import requests
import random
import json
import ast

url = "https://y4bo327ayb.execute-api.eu-west-1.amazonaws.com/dev/rag_request"
NUM_COMP = 250
NUM_SELECTED = 50

headers = {
    "Content-Type": "application/json",
    "authToken": "uZbWndsrqpMoZGiKpnen"
}

questions_dict = json.load(open("filtered_citations_with_questions.json", "r"))

new_question_dict = {}

lock = threading.Lock()

def select_first_n_elements(d, n):
    if type(d) != dict:
        print(d)
        d = ast.literal_eval(d)
    return dict(itertools.islice(d.items(), n))

def select_last_n_elements(d, n):
    if type(d) != dict:
        print(d)
        d = ast.literal_eval(d)
    return dict(list(d.items())[-n:])

def process_citation(cit):

    global new_question_dict

    for d in tqdm(questions_dict[cit]):
        try:
            comp_id = str(d["company_id"])
            question = [d["question"]]

            data = {
                "company_name": comp_id,
                "questions": question,
                "num_returned_docs": NUM_COMP
            }

            response = requests.post(url, headers=headers, json=data)
            text = ast.literal_eval(response.text)
            context = text["body"]
            d["pos_context"] = select_first_n_elements(context, NUM_SELECTED)
            d["neg_context"] = select_last_n_elements(context, NUM_SELECTED)
 
            with lock:
                if cit not in new_question_dict.keys():
                    new_question_dict[cit] = [d]
                else:
                    new_question_dict[cit].append(d)
        except Exception as e:
            print(e)

# Use ThreadPoolExecutor to run the function in parallel
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_citation, questions_dict.keys()), total=len(questions_dict)))


with open("filtered_citations_with_questions_final.json", "w") as f:
    json.dump(new_question_dict, f, indent=4)
