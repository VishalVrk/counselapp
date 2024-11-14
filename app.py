from flask import Flask, request, jsonify, send_from_directory
import faiss
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
from gradio_client import Client  # Import the Gradio Client
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
print("Loaded dataset successfully.")

# Lazy loading for the retriever model
retriever_model = None

def load_retriever_model():
    global retriever_model
    if retriever_model is None:
        print("Loading retriever model...")
        retriever_model = SentenceTransformer('fine_tuned_retriever_model')
    else:
        print("Retriever model already loaded.")

# Initialize FAISS Index once
question_embeddings = None
index = None

def build_faiss_index():
    global question_embeddings, index
    if question_embeddings is None:
        print("Building FAISS index...")
        question_embeddings = retriever_model.encode(counselchat_data_cleaned['questionText'].tolist(), convert_to_tensor=True)
        d = question_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(question_embeddings.cpu().numpy())
    else:
        print("FAISS index already built.")

# Initialize the Gradio Clients once during app initialization
emotional_client = Client("https://ertugruldemir-speechemotionrecognition.hf.space/")
chat_client = Client("Qwen/Qwen2.5-72B-Instruct")
tts_client = Client("vrkforever/Text-to-Speech")
print("Initialized Gradio clients.")

# Function to retrieve the most relevant question and its answer (truncated to 100 words)
def retrieve_answer(query):
    load_retriever_model()  # Ensure model is loaded
    build_faiss_index()  # Ensure FAISS index is built
    
    query_embedding = retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, 1)
    
    relevant_question = counselchat_data_cleaned['questionText'].iloc[indices[0][0]]
    relevant_answer = counselchat_data_cleaned['answerText'].iloc[indices[0][0]]
    
    truncated_answer = " ".join(relevant_answer.split()[:100])
    
    print(f"Query: {query}")
    print(f"Relevant Question: {relevant_question}")
    print(f"Answer (truncated): {truncated_answer}")
    
    return relevant_question, truncated_answer

def process_audio_and_get_answer(audio_url, transcribed_text):
    try:
        print(f"Processing audio for emotion and text for answer retrieval.")
        
        # Emotion analysis using Gradio client with audio URL
        emotion_file_path = emotional_client.predict(audio_url, api_name="/predict")
        print("Emotion file path:", emotion_file_path)  # Debug print to check the file path
        
        # Read the JSON file to extract the emotion label
        with open(emotion_file_path, 'r') as f:
            emotion_data = json.load(f)
        
        # Check if the label exists in the JSON data
        if 'label' in emotion_data:
            emotion = emotion_data['label']
            print(f"Detected emotion: {emotion}")
        else:
            raise ValueError("Unexpected emotion data format: 'label' key not found.")
        
        # Proceed with response generation using the transcribed text
        relevant_question, answer = retrieve_answer(transcribed_text)
        message = f"Rephrase the following answer to align with the emotion of the question within 300 characters. The person is in {emotion}. Original Answer: {answer}"
        
        # Generate the counseling perspective
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
            'transcribed_question': transcribed_text,
            'most_relevant_question': relevant_question,
            'answer': answer,
            'emotion': emotion,
            'counsel_chat': desired_phrase,
            "filepath": file_url
        }
    
    except Exception as e:
        print(f"Error during audio and text processing: {str(e)}")
        return {'error': f'Error during audio and text processing: {str(e)}'}



# Define the API endpoint
@app.route('/ask_audio', methods=['POST'])
def ask_audio_question():
    data = request.get_json()
    audio_url = data.get('audio_url', '')
    transcribed_text = data.get('transcribed_text', '')
    
    if audio_url and transcribed_text:
        print("Received audio URL and transcribed text in request.")
        response = process_audio_and_get_answer(audio_url, transcribed_text)
    else:
        print("Audio URL or transcribed text missing in request.")
        response = {'error': 'Please provide both audio URL and transcribed text.'}
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return "Hello, we are live!"

@app.route('/files/<filename>', methods=['GET'])
def serve_file(filename):
    print(f"Serving file: {filename}")
    return send_from_directory(PUBLIC_FOLDER, filename)

# Start the Flask app
if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5005)