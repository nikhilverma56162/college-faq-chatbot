import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the faq data
with open("intents.json", "r") as f:
    file = json.load(f)

questions = []
labels = []

for intent in file["intents"]:
    for q in intent["patterns"]:
        questions.append(q)
        labels.append(intent["tag"])

# vectorizing the text
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(questions)


def reply(user_msg):
    user_tfidf = tfidf.transform([user_msg])
    score = cosine_similarity(user_tfidf, tfidf_matrix)
    index = score.argmax()
    confidence = score[0][index]

    if confidence < 0.3:
        return "Sorry, I didn't understand that. Try asking something else 🙂"

    tag = labels[index]

    for intent in file["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])


# ---------------- Streamlit UI ---------------- #

st.title("College FAQ Chatbot 🎓")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask your question")

if user_input:
    bot_answer = reply(user_input)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", bot_answer))

for sender, msg in st.session_state.chat:
    st.write(f"**{sender}:** {msg}")