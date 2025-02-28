import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
import random
import pickle

warnings.filterwarnings("ignore")

colorama.init()
from colorama import Fore, Style, Back

# Load dataset
with open('final_fixed_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
with open('intents1.json', 'r', encoding='utf8') as file2:
    data_new = json.load(file2) 

# Load both models
model1 = keras.models.load_model('chat-model')
model2 = keras.models.load_model('chat-model-new')

# Load both tokenizers
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)
with open('tokenizer_new.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)

# Load both label encoders
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder1 = pickle.load(enc)
with open('label_encoder_new.pickle', 'rb') as enc:
    lbl_encoder2 = pickle.load(enc)

# Parameters
max_len = 50

def get_response(model, tokenizer, lbl_encoder, user_input):
    #result = model.predict(keras.preprocessing.sequence.pad_sequences(
     #   tokenizer.texts_to_sequences([user_input]), #truncating='post', maxlen=max_len))
    #predicted_index = np.argmax(result)
    #return lbl_encoder.inverse_transform([predicted_index])[0]
    sequence = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([user_input]), truncating = 'post', maxlen = max_len
    )
    prediction = model.predict(sequence)
    predicted_index = np.argmax(prediction)
    predicted_tag = lbl_encoder.inverse_transform([predicted_index])[0]

    print(Fore.YELLOW + f"DEBUG: Predicted tag -> {predicted_tag}" + Style.RESET_ALL)

    for entry in data:
        if entry["patterns"].strip().lower() == predicted_tag.strip().lower():
            return entry["responses"]
        elif entry["Context"].strip().lower() == predicted_tag.strip().lower():
            return entry["Response"]
        
    return "I'm not sure how to respond."

def determine_model(user_input):
    # Example logic: If input contains certain keywords, use a specific model
    keywords_model1 = ["greeting", "morning", "greeting","morning","afternoon","evening","night","goodbye","thanks","no-response","neutral-response","about","skill","creation","name","help","sad","stressed","worthless","depressed","happy","casual","anxious","not-talking","sleep","scared","death","understand","done","suicide","hate-you","hate-me","default","jokes","repeat","wrong","stupid","location","something-else","friends","ask","problem","no-approach","learn-more","user-agree","meditation","user-meditation","Calm-useful","user-advice","learn-mental-health","mental-health-fact","fact-1","fact-2","fact-3","fact-5","fact-6","fact-7","fact-8","fact-9","fact-10","fact-11","fact-12","fact-13","fact-14","fact-15","fact-16","fact-17","fact-18","fact-19","fact-20","fact-21","fact-22","fact-23","fact-24","fact-25","fact-26","fact-27","fact-28","fact-29","fact-30","fact-31","fact-32"
]  
    # Model1 handles mental health
    keywords_model2 = ["career", "motivation", "success"]  # Model2 handles personal growth
    
    for word in keywords_model1:
        if word in user_input.lower():
            return model1, tokenizer1, lbl_encoder1
    for word in keywords_model2:
        if word in user_input.lower():
            return model2, tokenizer2, lbl_encoder2
    
    # Default fallback (choose the most confident model later if needed)
    return model1, tokenizer1, lbl_encoder1

def chat():
    print(Fore.YELLOW + 'Start talking with Calm, your Personal Therapeutic AI Assistant. (Type quit to stop talking)' + Style.RESET_ALL)
    
    while True:
        print(Fore.LIGHTBLUE_EX + 'User: ' + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == 'quit':
            print(Fore.GREEN + 'Calm:' + Style.RESET_ALL, "Take care. See you soon.")
            break
        
        # Select the appropriate model based on input context
        selected_model, selected_tokenizer, selected_lbl_encoder = determine_model(inp)
        response = get_response(selected_model, selected_tokenizer, selected_lbl_encoder, inp)
        
        print(Fore.GREEN + 'Calm:' + Style.RESET_ALL, response)

chat()
