import random
import string
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('wordnet')

# Initialize
lemmatizer = WordNetLemmatizer()

# Define intents and examples
training_data = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "see you later", "goodbye"],
    "thanks": ["thanks", "thank you", "I appreciate it"],
    "name": ["what is your name?", "who are you?"],
    "help": ["can you help me?", "I need assistance", "help me please"],
    "service": ["what services do you offer?what services do you offer?", "tell me about your services"],
    "billing": ["I have a billing issue", "how do I pay?", "my payment failed"]
}

# Bot responses
responses = {
    "greeting": ["Hello! How can I help you?", "Hi there! Need any assistance?"],
    "goodbye": ["Goodbye! Have a great day!", "See you soon!"],
    "thanks": ["You're welcome!", "Glad I could help!"],
    "name": ["I'm BRG Assist, your service chatbot."],
    "help": ["Sure! Please tell me what you need help with."],
    "service": ["We offer 24/7 customer support, billing help, and more."],
    "billing": ["You can pay using credit card or UPI. If you're facing issues, contact our billing team."]
}

# Prepare the corpus
corpus = []
labels = []
for intent, examples in training_data.items():
    for example in examples:
        corpus.append(example.lower())
        labels.append(intent)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y = labels

# Train classifier
model = LogisticRegression()
model.fit(X, y)

# Preprocess input
def clean_input(text):
    tokens = text.lower().split()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return " ".join(lemmatized)

# Chatbot response function
def get_bot_response(user_input):
    cleaned = clean_input(user_input)
    vector = vectorizer.transform([cleaned])
    intent = model.predict(vector)[0]
    return random.choice(responses[intent])

# ---------------- TKINTER GUI ----------------

# GUI Setup
root = tk.Tk()
root.title("🤖 BRG Assist - Customer Service Chatbot")
root.geometry("500x500")

# Chat history
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Arial", 12))
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Entry field
entry_field = tk.Entry(root, font=("Arial", 12))
entry_field.pack(padx=10, pady=5, fill=tk.X)

# Send button function
def send_message():
    user_text = entry_field.get()
    if user_text.strip() == "":
        return

    chat_area.config(state='normal')
    chat_area.insert(tk.END, f"You: {user_text}\n")
    bot_response = get_bot_response(user_text)
    chat_area.insert(tk.END, f"BRG Assist: {bot_response}\n\n")
    chat_area.config(state='disabled')
    chat_area.yview(tk.END)
    entry_field.delete(0, tk.END)

# Send button
send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 12), bg="lightblue")
send_button.pack(pady=5)

# Enter key binding
root.bind('<Return>', lambda event: send_message())

# Start GUI
chat_area.config(state='normal')
chat_area.insert(tk.END, "BRG Assist: Hi! I’m your customer service assistant. How can I help?\n\n")
chat_area.config(state='disabled')

root.mainloop()
