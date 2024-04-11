import streamlit as st
import tensorflow as tf
from utils.bot import ChatBot
import pickle

# Load the model 
# model = tf.keras.models.load_model('../chatbot')

chatbot_model = pickle.load(open('../chatbot.pkl', 'rb'))

chatbot = ChatBot(chatbot_model.encoder, chatbot_model.decoder, name='chatbot')

st.title('Chatbot')

user_input = st.text_input('You:', '')

if st.button('Send'):
    user_input = chatbot.preprocess(user_input)
    user_input = tf.convert_to_tensor(user_input)
    user_input = tf.expand_dims(user_input, 0)
    # Assuming the preprocessing ensures user_input is integer type, no need to cast to float
    encoder_state_h, encoder_state_c = chatbot.encoder(user_input)
    decoder_input = tf.convert_to_tensor([[chatbot.word2id['<start>']]])
    # Keep decoder_input as integers if your model expects indices
    decoder_input = tf.expand_dims(decoder_input, 0)
    decoder_input_state_h = encoder_state_h
    decoder_input_state_c = encoder_state_c
    response = []
    for i in range(20):  # Limiting the response length to 20 tokens
        decoder_output, [decoder_input_state_h, decoder_input_state_c] = chatbot.decoder([decoder_input, [decoder_input_state_h, decoder_input_state_c]])
        decoder_output = tf.squeeze(decoder_output)
        token_id = chatbot.sample(decoder_output)
        if chatbot.id2word[token_id] == '<end>':
            break
        response.append(chatbot.id2word[token_id])
        decoder_input = tf.convert_to_tensor([[token_id]])
        # Keep as integers
        decoder_input = tf.expand_dims(decoder_input, 0)
    response = ' '.join(response)
    st.text_area('Bot:', value=response, height=100)

text = st.text_input('You:', '')

message = st.chat_message('Bot:', 'Hello! How can I help you?')