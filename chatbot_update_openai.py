import json
import numpy as np
import warnings
import openai
import os
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
import random
import pickle
import subprocess


warnings.filterwarnings("ignore")

colorama.init()
from colorama import Fore, Style, Back

# Load OpenAI API Key securely
openai.api_key = "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"

# Load datasets
with open('final_fixed_dataset.json', 'r', encoding='utf-8') as file:
    data_new = json.load(file)
with open('intents1.json', 'r', encoding='utf-8') as file2:
    data_old = json.load(file2)["intents"]

# Load models
model1 = keras.models.load_model('chat-model')
model2 = keras.models.load_model('chat-model-new')

# Load tokenizers
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)
with open('tokenizer_new.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)

# Load label encoders
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
    confidence = np.max(prediction)

    print(Fore.YELLOW + f"DEBUG: Predicted tag -> {predicted_tag}, Confidence -> {confidence:.2f}" + Style.RESET_ALL)

    # Use dataset response if confidence is high
    if confidence > 0.7:
        if dataset_type == "old":
            for entry in dataset:
                if entry["tag"] == predicted_tag:
                    return random.choice(entry["responses"])
        elif dataset_type == "new":
            for entry in dataset:
                if entry["Context"].strip().lower() == predicted_tag.strip().lower():
                    return entry["Response"]

    # If confidence is low, use GPT response
    return get_gpt_response(user_input)

#def get_gpt_response(user_input):
    """Generate a response using OpenAI GPT with the new API format."""
    try:
        client = openai.OpenAI(api_key=openai.api_key)  # Initialize OpenAI client
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Calm, an AI therapist who provides empathetic, supportive, and caring responses."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content  # Correct way to get GPT response
    except Exception as e:
        print(Fore.RED + f"GPT ERROR: {e}" + Style.RESET_ALL)
        return "I'm having trouble thinking right now. Can you try again later?"
    
def get_gpt_response(user_input):
    """Generate a response using Ollama's local LLM instead of OpenAI GPT."""
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", user_input],  # Change "mistral" to "llama2" if using LLaMA
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error using Ollama: {e}"

def determine_model(user_input):
    """Determine which model should be used based on user input."""
    keywords_model1 = ["sad", "stressed", "worthless", "depressed", "anxious", "suicide", "death"]
    keywords_model2 = ["career", "motivation", "success"]

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
