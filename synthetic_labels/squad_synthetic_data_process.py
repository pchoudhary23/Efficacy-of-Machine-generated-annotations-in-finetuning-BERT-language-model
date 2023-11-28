import pandas as pd
import json

def clean_data(data):
  ## Splitting on newline
  temp = {}
  for key in data.keys():
    temp[key] = data[key].split('\n')

  ## Splitting on '. ' as the list is numbered
  final = {}
  for key in temp.keys():
    try:
      final[int(key)] = [x.split('. ')[1] if x != 'Answers: ' else x for x in temp[key] if x != 'Answers:']
    except:
      ## log the keys with error
      print(key)
      final[int(key)] = temp[key]
    
  ## Cleaning keys with prefixes
  for key in final.keys():
    final[key] = [x.replace('Answer: ', '') for x in final[key]]
  
  ## Saving processed synthetic data
  with open('../data/synthetic_processed.json', 'w', encoding='utf-8') as f:
      json.dump(final, f, ensure_ascii=False, indent=4)
  clean_data = final
  return clean_data
  

