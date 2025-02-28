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

# Load datasets properly
with open('final_fixed_dataset.json', 'r', encoding='utf-8') as file:
    data_new = json.load(file)
with open('intents1.json', 'r', encoding='utf-8') as file2:
    data_old = json.load(file2)["intents"]  # Extract intents list

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

def get_response(model, tokenizer, lbl_encoder, user_input, dataset, dataset_type):
    """Predict response using the selected model and dataset."""
    sequence = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len
    )

    prediction = model.predict(sequence)
    predicted_index = np.argmax(prediction)
    predicted_tag = lbl_encoder.inverse_transform([predicted_index])[0]

    print(Fore.YELLOW + f"DEBUG: Predicted tag -> {predicted_tag}" + Style.RESET_ALL)  # Debugging

    # Search for the correct response
    if dataset_type == "old":  # Handling intents1.json structure
        for entry in dataset:
            if entry["tag"] == predicted_tag:
                return random.choice(entry["responses"])  # Pick a random response
    elif dataset_type == "new":  # Handling final_fixed_dataset.json structure
        for entry in dataset:
            if entry["Context"].strip().lower() == predicted_tag.strip().lower():
                return entry["Response"]

    return "I'm not sure how to respond."

def determine_model(user_input):
    """Determine which model should be used based on user input."""
    keywords_model1 = ["sad", "stressed", "worthless", "depressed", "anxious", "suicide", "death"]
    keywords_model2 = ["career", "motivation", "success"]  # Model2 handles personal growth

    for word in keywords_model1:
        if word in user_input.lower():
            return model1, tokenizer1, lbl_encoder1, data_old, "old"
    for word in keywords_model2:
        if word in user_input.lower():
            return model2, tokenizer2, lbl_encoder2, data_new, "new"
    
    # Default fallback
    return model1, tokenizer1, lbl_encoder1, data_old, "old"

def chat():
    print(Fore.YELLOW + 'Start talking with Calm, your Personal Therapeutic AI Assistant. (Type quit to stop talking)' + Style.RESET_ALL)
    
    while True:
        print(Fore.LIGHTBLUE_EX + 'User: ' + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == 'quit':
            print(Fore.GREEN + 'Calm:' + Style.RESET_ALL, "Take care. See you soon.")
            break
        
        # Select the appropriate model based on input context
        selected_model, selected_tokenizer, selected_lbl_encoder, selected_dataset, dataset_type = determine_model(inp)
        response = get_response(selected_model, selected_tokenizer, selected_lbl_encoder, inp, selected_dataset, dataset_type)
        
        print(Fore.GREEN + 'Calm:' + Style.RESET_ALL, response)

chat()
