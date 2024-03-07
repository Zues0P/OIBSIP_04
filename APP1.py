import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)  # removing special characters
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  # removing stopwords

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # performing stemming

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title("Email Classifier")
st.subheader("Enter Email Message:")
input_message = st.text_area('Message Body:', height=200)

if st.button('Classify Email'):
    transformed_message = transform_text(input_message)
    vector_input = tfidf.transform([transformed_message])
    result = model.predict(vector_input)[0]
    probability = model.predict_proba(vector_input)[0][1]

    st.subheader("Classification Result:")
    if result == 1:
        st.write("This email is classified as SPAM with a probability of {:.2f}%".format(probability*100))
    else:
        st.write("This email is classified as NOT SPAM with a probability of {:.2f}%".format((1-probability)*100))
