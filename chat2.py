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

# Load trained model
model = keras.models.load_model('chat-model-new')

# Load tokenizer and label encoder
with open('tokenizer_new.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder_new.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Parameters
max_len = 50

def chat():
    print(Fore.YELLOW + 'Start talking with Pandora, your Personal Therapeutic AI Assistant. (Type quit to stop talking)' + Style.RESET_ALL)
    
    while True:
        print(Fore.LIGHTBLUE_EX + 'User: ' + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == 'quit':
            print(Fore.GREEN + 'Pandora:' + Style.RESET_ALL, "Take care. See you soon.")
            break
        
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))
        predicted_index = np.argmax(result)
        response = lbl_encoder.inverse_transform([predicted_index])[0]
        
        print(Fore.GREEN + 'Pandora:' + Style.RESET_ALL, response)

chat()
