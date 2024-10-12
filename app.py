from flask import Flask, request, jsonify,send_from_directory
import faiss
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
from gradio_client import Client  # Import the Gradio Client
from gradio_client import handle_file
from flask_cors import CORS
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire Flask app

# Configure public folder for storing and serving files
PUBLIC_FOLDER = 'public'
if not os.path.exists(PUBLIC_FOLDER):
    os.makedirs(PUBLIC_FOLDER)

# Load dataset and model once during app initialization
counselchat_data_cleaned = pd.read_csv('counselchat_data_cleaned.csv')

# Lazy loading for the retriever model
retriever_model = None

def load_retriever_model():
    global retriever_model
    if retriever_model is None:
        retriever_model = SentenceTransformer('fine_tuned_retriever_model')

# Initialize FAISS Index once
question_embeddings = None
index = None

def build_faiss_index():
    global question_embeddings, index
    if question_embeddings is None:
        question_embeddings = retriever_model.encode(counselchat_data_cleaned['questionText'].tolist(), convert_to_tensor=True)
        d = question_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(question_embeddings.cpu().numpy())

# Initialize the Gradio Clients once during app initialization
gradio_client = Client("https://jefercania-speech-to-text-whisper.hf.space/--replicas/qgf0y/")
emotional_client = Client("https://ertugruldemir-speechemotionrecognition.hf.space/")
chat_client = Client("Qwen/Qwen2.5-72B-Instruct")
tts_client=client = Client("vrkforever/Text-to-Speech")

# Function to read JSON from the returned filepath
def read_json_from_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Function to retrieve the most relevant question and its answer (truncated to 100 words)
def retrieve_answer(query):
    load_retriever_model()  # Ensure model is loaded
    build_faiss_index()  # Ensure FAISS index is built
    
    query_embedding = retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, 1)
    
    relevant_question = counselchat_data_cleaned['questionText'].iloc[indices[0][0]]
    relevant_answer = counselchat_data_cleaned['answerText'].iloc[indices[0][0]]
    
    truncated_answer = " ".join(relevant_answer.split()[:100])
    
    return relevant_question, truncated_answer

# Function to process the audio URL, convert it to text, and retrieve the answer
def process_audio_and_get_answer(audio_url):
    try:
        # Use Gradio Client to transcribe the audio
        result = gradio_client.predict(handle_file(audio_url), api_name="/predict")
        emotion = emotional_client.predict(audio_url, api_name="/predict")
        
        transcribed_result = read_json_from_file(emotion)
        transcribed_question = result  # Assuming the result contains the transcribed question
        emotion_results = transcribed_result['label']
        
        relevant_question, answer = retrieve_answer(transcribed_question)
        message = f"Rephrase the following answer to align with the emotion of the question\n within 300 characters" \
                  f"The person is in {emotion_results}\n" \
                  f"Original Answer: {answer}"
        
        counsel_chat = chat_client.predict(
            query=message,
            history=[],
            system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            api_name="/model_chat"
        )
        desired_phrase = counsel_chat[1][0][1]

        filepath = tts_client.predict(
            text=desired_phrase,
            voice="bn-IN-TanishaaNeural - bn-IN (Female)",
            rate=0,
            pitch=0,
            api_name="/predict"
        )
        
        # Get the file and save it in the public folder
        filename = os.path.basename(filepath[0])
        saved_path = os.path.join(PUBLIC_FOLDER, filename)
        
        # Move the file to the public folder
        os.rename(filepath[0], saved_path)

        # Generate a URL to access the file
        file_url = f"/files/{filename}"

        
        
        return {
            'transcribed_question': transcribed_question,
            'most_relevant_question': relevant_question,
            'answer': answer,
            'emotion': emotion_results,
            'counsel_chat': desired_phrase,
            "filepath":file_url
        }
    
    except Exception as e:
        return {'error': f'Error during audio processing: {str(e)}'}

# Define the API endpoint
@app.route('/ask_audio', methods=['POST'])
def ask_audio_question():
    data = request.get_json()
    audio_url = data.get('audio_url', '')
    
    if audio_url:
        response = process_audio_and_get_answer(audio_url)
    else:
        response = {'error': 'Please provide a valid audio URL.'}
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "Hello we are live"

@app.route('/files/<filename>', methods=['GET'])
def serve_file(filename):
    return send_from_directory(PUBLIC_FOLDER, filename)

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
