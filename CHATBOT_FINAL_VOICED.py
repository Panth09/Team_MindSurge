import json
import numpy as np
import warnings
import speech_recognition as sr
import sounddevice as sd
from kokoro import KPipeline
import os
import random
import pickle
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama

warnings.filterwarnings("ignore")

colorama.init()
from colorama import Fore, Style, Back

# Set eSpeak path for phonemizer
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak\command_line\espeak.exe"

# Initialize Kokoro TTS pipeline
pipeline = KPipeline(lang_code='a')  # 'a' for American English

# Load datasets
with open('final_fixed_dataset.json', 'r', encoding='utf-8') as file:
    data_new = json.load(file)
with open('intents1.json', 'r', encoding='utf-8') as file2:
    data_old = json.load(file2)["intents"]

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

def speak(response_text):
    """Convert AI response to speech and play it."""
    generator = pipeline(response_text, voice='af_heart', speed=1, split_pattern=r'\n+')

    for _, _, audio in generator:
        sd.play(audio, samplerate=24000)
        sd.wait()  # Wait until audio finishes

def recognize_speech():
    """Recognize speech from the user using microphone."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(Fore.LIGHTBLUE_EX + "Listening..." + Style.RESET_ALL)
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, text)
            return text.lower()
        except sr.UnknownValueError:
            print(Fore.RED + "Could not understand, please try again!" + Style.RESET_ALL)
            return ""
        except sr.RequestError:
            print(Fore.RED + "Speech Recognition API request failed" + Style.RESET_ALL)
            return ""

def get_response(model, tokenizer, lbl_encoder, user_input, dataset, dataset_type):
    """Predict response using the selected model and dataset."""
    sequence = keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences([user_input]), truncating='post', maxlen=max_len
    )
    prediction = model.predict(sequence)
    predicted_index = np.argmax(prediction)
    predicted_tag = lbl_encoder.inverse_transform([predicted_index])[0]

    print(Fore.YELLOW + f"DEBUG: Predicted tag -> {predicted_tag}" + Style.RESET_ALL)

    if dataset_type == "old":
        for entry in dataset:
            if entry["tag"] == predicted_tag:
                return random.choice(entry["responses"])
    elif dataset_type == "new":
        for entry in dataset:
            if entry["Context"].strip().lower() == predicted_tag.strip().lower():
                return entry["Response"]

    return "I'm not sure how to respond."

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

    return model1, tokenizer1, lbl_encoder1, data_old, "old"

def chat():
    print(Fore.YELLOW + "Start talking with Calm, your Personal Therapeutic AI Assistant." + Style.RESET_ALL)
    print("Type 'quit' to exit. Type 'speak' if you want to use your microphone.\n")

    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input("Type your message or say 'speak': ")

        if inp.lower() == "quit":
            print(Fore.GREEN + "Calm:" + Style.RESET_ALL, "Take care. See you soon.")
            speak("Take care. See you soon.")
            break
        elif inp.lower() == "speak":
            inp = recognize_speech()
            if not inp:
                continue  # If no speech detected, ask again

        selected_model, selected_tokenizer, selected_lbl_encoder, selected_dataset, dataset_type = determine_model(inp)
        response = get_response(selected_model, selected_tokenizer, selected_lbl_encoder, inp, selected_dataset, dataset_type)

        print(Fore.GREEN + "Calm:" + Style.RESET_ALL, response)
        speak(response)

chat()
