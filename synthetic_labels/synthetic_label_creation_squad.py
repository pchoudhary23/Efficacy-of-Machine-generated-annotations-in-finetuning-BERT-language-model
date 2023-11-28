from openai import OpenAI
import time
import pandas as pd
from datasets import load_dataset
import json
from collections import defaultdict

OPENAI_API_KEY = ''

client = OpenAI(
  api_key = OPENAI_API_KEY,
)  

# validation_dataset = squad_dataset["validation"]

# Question: What is the Grotto at Notre Dame?
# Answer: a Marian place of prayer and reflection
# Question: What sits on top of the Main Building at Notre Dame?
# Answer: a golden statue of the Virgin Mary

def preprocess_data(dataset):
    context_dict = {}
    question_dict = defaultdict(list)
    answer_dict = defaultdict(list)
    
    start = 0
    
    for end in range(len(dataset['question'])):
        batch_questions = []
        batch_answers = []
        if end == 0 or dataset["context"][end-1] == dataset["context"][end]:
            continue
        else:
            batch_questions.extend(dataset['question'][start:end])
            batch_answers.extend(dataset["answers"][start:end])
            start = end
            batch_answers = [i['text'][0] for i in batch_answers]
            context_dict[len(context_dict.keys())] = dataset["context"][start]
            question_dict[len(question_dict.keys())] = batch_questions
            answer_dict[len(answer_dict.keys())] = batch_answers
        
    return context_dict, question_dict, answer_dict
    
def generate_answer(ques_str, context):
    
    messages=[
                {"role": "system", "content": '''You are an assistant which can search for answers within a reading passage (context) for a given question. 
                    The answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 
                    Following are a few examples of answers. Generate the entity only and not the complete sentence in the answer for the given questions and context and return answers in a list.
                    Question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
                    Answer: Saint Bernadette Soubirous
                    Question: What is in front of the Notre Dame Main Building?
                    Answer: a copper statue of Christ
                    Question: The Basilica of the Sacred heart at Notre Dame is beside to which structure?
                    Answer: the Main Building
                    '''},
                {"role": "user", "content": f"Context: {context}\n{ques_str}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    # Extracting the answer from the GPT-3 response
    answer = response.choices[0].message.content
    return answer

def main(context_dict, questions_dict):
    synthetic_dict = defaultdict(list)
    
    # Example usage with a few questions from SQuAD
    for i in range(len(context_dict.keys())):
        print(f"Context --> {i+1} processed\n")
        context = context_dict[i]
        
        ques_str = ''
        for ques in questions_dict[i]:
            ques_str += f'Question: {ques}\n'
        
        # # Extract the context and question and actual answer
        # print(f"Example {i + 1} processed\n")
        # context = train_dataset["context"][i]
        # question = train_dataset["question"][i]
        # actual_answer = train_dataset["answers"][i]["text"][0]

        # Generate a synthetic answer
        sythetic_str = generate_answer(ques_str, context)
        
        # synthetic_answers = st
        
        # if i%20 == 0:
        #     print(f"Question: {question}\nActual Answer: {actual_answer}\nSynthetic Answer: {sythetic_answer}\n{'='*50}\n")

        ## Appedmnding to the lists
        # contexts.append(context)
        # questions.append(question)
        # answers.append(actual_answer)
        synthetic_dict[i] = sythetic_str
        
        # Save intermediate results in json format to avoid losing progress
        if i%20 == 0:
            with open('../data/synthetic_answers_squad.txt', 'w') as json_file:
                json.dump(synthetic_dict, json_file)
                
    return synthetic_dict
            
    

if __name__ == '__main__':
    # Load the SQuAD dataset
    squad_dataset = load_dataset("squad")

    # Access the training and validation sets
    train_dataset = squad_dataset["train"][:10]
    
    context_dict, questions_dict, answers_dict = preprocess_data(train_dataset)
    
    syntetic_dict = main(context_dict, questions_dict)

    # try:
    #     final_df = pd.DataFrame({'Context': contexts, 'Question': questions, 'Actual Answer': answers, 'Synthetic Answer': synthetic_answers})
    #     final_df.to_csv('../data/synthetic_answers_squad.csv', index=False)
    # except:
    #     print('Error in saving the dataframe')