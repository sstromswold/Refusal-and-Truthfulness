import json
import random

with open('/Users/sarah/NN_Homework4/dev-v2.0.json', 'r') as file:
  data = json.load(file)
  print(len(data))

impossible = []
possible = []

possible_counter=0

for article in data["data"]:
    for paragraph in article["paragraphs"]:
      context = paragraph["context"]
      for qa in paragraph["qas"]:
        # Keep only Q&A pairs where is_impossible == True
        if qa.get("is_impossible", False):
          impossible.append({
            "instruction": "Context: " + context + "\nQuestion: " + qa["question"] + "",
            "category": "1"
          })
        else: 
          #if possible_counter%1 == 0:
          impossible.append({
            "instruction": "Context: " + context + "\nQuestion: " + qa["question"] + "",
            "category": "0"
          })
      #possible_counter+=1

with open("test.json", "w") as f:
    json.dump(impossible, f, indent=4)

# with open("possible_val.json", "w") as f:
#     json.dump(possible, f, indent=4)
