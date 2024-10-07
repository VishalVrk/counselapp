from flask import Flask, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from gradio_client import Client  # Import the Gradio Client
from gradio_client import handle_file
from flask_cors import CORS
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire Flask app

# Load your dataset and models
counselchat_data_cleaned = pd.read_csv('counselchat_data_cleaned.csv')

# Load the fine-tuned retriever model
retriever_model = SentenceTransformer('fine_tuned_retriever_model')

# Build FAISS Index for fast retrieval
question_embeddings = retriever_model.encode(counselchat_data_cleaned['questionText'].tolist(), convert_to_tensor=True)
d = question_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(question_embeddings.cpu().numpy())

# Initialize the Gradio Client for speech-to-text
gradio_client = Client("https://jefercania-speech-to-text-whisper.hf.space/--replicas/qgf0y/")
emotional_client = Client("https://ertugruldemir-speechemotionrecognition.hf.space/")
chat_client = Client("Qwen/Qwen2.5-72B-Instruct")

# Function to read JSON from the returned filepath
def read_json_from_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Function to retrieve the most relevant question and its answer (truncated to 100 words)
def retrieve_answer(query):
    query_embedding = retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, 1)
    
    # Retrieve the most relevant question and corresponding answer
    relevant_question = counselchat_data_cleaned['questionText'].iloc[indices[0][0]]
    relevant_answer = counselchat_data_cleaned['answerText'].iloc[indices[0][0]]
    
    # Truncate the answer to 100 words
    truncated_answer = " ".join(relevant_answer.split()[:100])
    
    return relevant_question, truncated_answer

# Function to process the audio URL, convert it to text, and retrieve the answer
def process_audio_and_get_answer(audio_url):
    try:
        # Use the Gradio Client to transcribe the audio
        result = gradio_client.predict(handle_file(audio_url), api_name="/predict")
        emotion =emotional_client.predict(audio_url, api_name="/predict")
         # Read the transcribed text from the JSON file
        transcribed_result = read_json_from_file(emotion)
        print(audio_url)
        print(result)
        print(transcribed_result['label'])
        # Assume that the result is the transcribed text (question)
        transcribed_question = result
        emotion_results = transcribed_result['label']
        
        # Retrieve the most relevant answer based on the transcribed question
        relevant_question, answer = retrieve_answer(transcribed_question)
        message="Rephrase the following answer to align with the emotion of the question\n"+"the person is in "+emotion_results+"\n"+"Original Answer: "+answer
        print(message)      
        counsel_chat = chat_client.predict(
		query=message,
		history=[],
		system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
		api_name="/model_chat"
)
        desired_phrase = counsel_chat[1][0][1]
        print(desired_phrase)
        return {
            'transcribed_question': transcribed_question,
            'most_relevant_question': relevant_question,
            'answer': answer,
            'emotion':emotion_results,
            'counsel_chat':desired_phrase
        }
    
    except Exception as e:
        # Capture any errors raised by the Gradio app
        return {'error': f'Error during audio processing: {str(e)}'}


# Define the API endpoint
@app.route('/ask_audio', methods=['POST'])
def ask_audio_question():
    data = request.get_json()  # Get the input audio URL from the request body
    audio_url = data.get('audio_url', '')  # Extract the audio URL from the request data
    print(audio_url)
    # If an audio URL is provided
    if audio_url:
        # response=audio_url
        # Process the audio and retrieve the answer
        response = process_audio_and_get_answer(audio_url)
    else:
        response = {'error': 'Please provide a valid audio URL.'}
    
    return jsonify(response)

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)