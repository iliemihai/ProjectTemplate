from tqdm import tqdm
import requests
import openai
import random
import json
import time


url = "http://iliemihai92.go.ro:9000/run-cli/"
headers = {
        "Content-Type": "application/json"
}
KEY = "6e430085ad944097b08a2d758bcefedd"
ENDPOINT = "https://sustai-chatgpt-poc.openai.azure.com/"

openai.api_key = KEY
openai.api_base = ENDPOINT
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
deployment_name = 'sustai-poc-3'


descriptions = { 
    "CH-G.1.1-TPV-T4-GS" : "Definition of conflicts of interest and commitment to minimize these",
    "CH-G.1.1-TPV-T4-1K" : "Prohibition of bribery",
    "CH-G.1.1-TPV-T4-OE" :  "Definition of bribery or corruption",
    "CH-G.1.1-TPV-T4-EJ" : "Guidelines of what is considered acceptable behaviour",
    "CH-G.1.1-TPV-T4-UH" : "Definition and prohibition of facilitation payments",
    "CH-S.3.1.3.3-TPV-T4-X0" : "Operational measures to monitor and respond to data breaches and cyberattacks",
    "CH-E.2.1.1-TPV-T4-UH" : "Monitoring of suppliers' environmental performance",
    "CH-G.1.1-TPV-T4-EN" : "There is no evidence of a formal policy but the company has a general statement addressing the issue",
    "CH-S.3.1.3.3-TPV-T4-FA" : "Governance structures in place for cybersecurity management",
    "CH-S.3.1.3.3-TPV-T4-RT" : "Regular employee training on cybersecurity issues",
    "CH-E.2.1.1-TPV-T4-GS" : "Compliance with environmental standards included in legally binding agreements with suppliers",
    "CH-S.3.1.3-TPV-T4-0E" : "Commitment to require third parties with whom the data is shared to comply with the company's policy",
    "CH-S.3.1.3.3-TPV-T4-1A" : "Regular internal security audits or vulnerability assessments or penetration testing of the company's systems, products and practices affecting user data",
    "CH-E.2.1.1-TPV-T4-J7" : "Reporting on environmental issues in the supply chain",
    "CH-E.2.1.1-TPV-T4-OE" : "Systematic consideration of suppliers environmental performance during procurement",
    "CH-S.3.1.3.3-TPV-T4-O1" : "Regular external security audits or vulnerability assessments of the company's systems, products and practices affecting user data",
    "CH-S.1.3.1-TPV-T4-9D" : "Global gender pay gap audit or compensation review",
    "CH-E.2.1.1-TPV-T4-EJ" : "Engagement with suppliers to address non-compliance or improve their environmental performance",
    "CH-E.2.1.1-TPV-T4-4O" : "Engagement with NGOs or industry peers to address environmental issues in the supply chain",
    "CH-E.2.1.1-TPV-T4-LT" : "Targets and deadlines for the environmental improvement of suppliers",
    "CH-S.3.1.3.3-TPV-T4-RT" : "Regular employee training on cybersecurity issues",
    "CH-S.1.3.1-TPV-T4-FJ" : "Commitment to gender pay equality",
    "CH-S.1.3.1-TPV-T4-DR" : "Initiatives to close the gender pay gap",
    "XX-S.1.3.3-TPV-T4-7A" : "The company has established partnerships (incl. joint ventures) with other companies or independent (research) institutions or NGO's supporting its commitment to gender equality",
    "XX-E.4.1.2-TPV-T4-C5" : "The company states  that it utilizes an internal carbon price in decision making",
    "XX-S.1.3.3-TPV-T4-F1" : "The company has a target for gender equality aligned with international standards or industry best-practice in the workforce ",
    "XX-E.1.2.7.4-TPV-T4-17" : "The company reports on progress against key milestones or interim targets, or is committed to report going forward",
    "XX-S.1.3.3-TPV-T4-A3" : "Responsibility of senior management or the board of directors towards gender equality",
    "XX-G.3.1.2-TPV-T4-85" : "The company discloses how it engages with policy makers on pro-climate policy",
    "XX-S.1.3.3-TPV-T4-C1" : "The company has a quantitative target for gender equality  aligned with international standards on the board of directors ",
    "XX-S.1.3.3-TPV-T4-86" : "The company reports on progress against key milestones or interim targets, or is committed to report going forward."
}


data = json.load(open("filtered_citations.json", "r"))

final_dict = {}

for tick in tqdm(list(data.keys())):
    for line in data[tick]:
        #print(line["text"].replace("\n", "").replace("\r", "").replace("\t", ""), descriptions[tick])
        cit = line["text"].replace("\n", "").replace("\r", "").replace("\t", "")
        des = descriptions[tick]
        PROMPT = f"Please generate a question as general as possible without specifying companies names or groups names or entities names  for which the response is DESCRIPTION and it aligns with the CITATION:\nCITATION:{cit} \nDESCRIPTION:{des} \nQUESTION:"

        response = openai.ChatCompletion.create(engine=deployment_name, model="gpt-3.5-turbo", temperature=0,  messages=[{"role": "user", "content": PROMPT}])
        text = response["choices"][0]["message"]["content"].replace('\n', '').replace(' .', '.').strip()
        print("QUEST: ", text)
        #final_dict[cit] = [tick, text]
        line["question"] = text

        if tick not in final_dict.keys():
            final_dict[tick] = [line]
        else:
            final_dict[tick].append(line)

        time.sleep(5)

with open("filtered_citations_with_questions.json", "w") as f:
    json.dump(final_dict, f, indent=4)
