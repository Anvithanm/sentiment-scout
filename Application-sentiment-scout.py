"""
========================================================================================================================
Author = Anvitha Hiriadka
Created on 04/21/2024
========================================================================================================================
This file contains the code for the UI of "Sentiment Scout" Application.
This Application loads the pre-trained model file sentiment_analysis.pkl, for sentiment analysis.
The UI of this Application contains following elements:
- Text field : For the user to enter the reviews/comments.
- Analyse Button : On clicking this button , it triggers the model and gives the best predicted Sentiment Result

Input : User can input review, just the way we give our review for the products/services
Output : This Application analyses the input and displayes the predicted sentiment and main aspects of the review
"""

#Importing necessary Libraries
import streamlit as st
import pickle
from streamlit.logger import get_logger
import nltk
from nltk import pos_tag
from nltk import word_tokenize
import spacy
#Setting up NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Getting logger object for logging
LOGGER = get_logger(__name__)

# Loading the pre-trained model
with open("sentiment_analysis.pkl", 'rb') as file:
    model = pickle.load(file)

#Function to extract the aspects from the review
def extract_aspect(review):
    aspects = []
    for word, pos in pos_tag(review):
        if pos.startswith('NN') or pos in ['NNP', 'NNPS']:
            aspects.append(word)
    result = ', '.join(aspects)
    return result

#Function to perform aspect-based sentiment analysis
def aspect_based_sentiment_analysis(input_data):
    tokenize_words = word_tokenize(input_data)
    get_aspect = extract_aspect(tokenize_words)
    review_aspect = list(get_aspect.split(" "))
    predicted_sentiment = model.predict([input_data])
    return predicted_sentiment[0], review_aspect

#Function to run the streamlit application
def run():
    #Setting the title
    st.title(":blue[_Sentiment Scout_]")
    st.divider()
    #Displaying the image for Application
    st.image('Image_app.png', width=550)
    #Adding sub-header for Input review
    st.subheader('Input Review', divider="rainbow")
    #Text area for user input
    input_review = st.text_area("Enter a review")
    #Button to trigger sentiment analysis
    if st.button('Analyse'):
        #Calling the sentiment analysis function
        sentiment, aspect = aspect_based_sentiment_analysis(input_review)
        #Displaying the predicted sentiment
        st.write('Prediction of Sentiment : ',sentiment)
        print (type(aspect))
        #Displaying the extracted Aspects
        st.write('Aspects detected : ', aspect)
    

#Main Function
if __name__ == "__main__":
    run()
