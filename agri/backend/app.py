# This Flask application serves as the backend for the AI Farmer Assistant Bot.
# It is designed to handle all business logic, including API calls to the
# Gemini API and integrating with external data sources via Google Search grounding.

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import base64
import os
import time

app = Flask(__name__)
# CORS is enabled to allow the frontend (running on a different port) to communicate with the backend.
CORS(app)

# --- Gemini API Configuration ---
# In a production environment, this should be loaded from environment variables.
# The platform provides the API key at runtime, so we leave it empty here.
GEMINI_API_KEY = ""
GEMINI_API_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_API_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- Utility Function for Gemini API Calls ---
def call_gemini_api(payload):
    """
    Makes a POST request to the Gemini API with a given payload.
    This function handles the common logic for all API calls.
    It includes exponential backoff for a more robust connection.
    """
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, json=payload, timeout=60)
            response.raise_for_status() # Raise an exception for bad status codes
            api_response = response.json()
            # Check for a valid response structure
            if 'candidates' in api_response and len(api_response['candidates']) > 0:
                return api_response['candidates'][0]['content']['parts'][0]['text']
            return "Sorry, I couldn't generate a response. Please try again."
        except requests.exceptions.RequestException as e:
            print(f"API call failed on attempt {i+1}: {e}")
            if i < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** i)
            else:
                return "I'm sorry, I am unable to connect to the AI service right now. Please check your network and try again."
        except (KeyError, IndexError) as e:
            print(f"Error parsing Gemini API response: {e}")
            return "An unexpected error occurred while processing the AI response."
    return "I'm sorry, the service is currently unavailable. Please try again later."


# --- Main API Endpoint ---
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    This is the central endpoint for all user queries.
    It intelligently routes the request based on whether an image is present.
    """
    data = request.json
    user_message = data.get('message', '')
    image_data = data.get('imageData', None)

    try:
        if image_data:
            # If an image is provided, handle a multi-modal request for pest/disease diagnosis.
            return handle_image_query(user_message, image_data)
        else:
            # If no image, handle a text-only query using Gemini with Google Search grounding.
            return handle_text_query(user_message)
    except Exception as e:
        print(f"Unhandled error in chat endpoint: {e}")
        return jsonify({"response": "An unexpected error occurred. Please try again."}), 500


# --- Helper Function for Image Queries ---
def handle_image_query(user_message, image_data):
    """
    Constructs a payload for an image analysis request to the Gemini API.
    The Gemini model can analyze an image and provide a diagnosis.
    """
    # Create the payload with both text and image parts.
    payload = {
        "contents": [{
            "parts": [
                {
                    "text": "Analyze this image for signs of crop disease or pests. Identify the issue and provide a concise, actionable remedy. Be specific about the diagnosis and potential treatments."
                },
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": image_data.split(',')[1]  # Remove the data URL header
                    }
                },
                {"text": user_message}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2, # Lower temperature for more focused, factual responses.
        },
    }
    
    # Send the request to the Gemini API and get the text response.
    gemini_response_text = call_gemini_api(payload)
    return jsonify({"response": gemini_response_text})


# --- Helper Function for Text Queries with Grounding ---
def handle_text_query(user_message):
    """
    Constructs a payload for a text query using Gemini with Google Search grounding.
    This allows the model to access real-time, external information.
    """
    payload = {
        "contents": [{"parts": [{"text": user_message}]}],
        "tools": [{"google_search": {}}],  # Enables Google Search for grounding.
        "systemInstruction": {
            "parts": [{
                "text": "You are a helpful and knowledgeable AI Farmer Assistant. "
                        "Answer questions about crops, soil, weather, fertilizers, "
                        "pests, and market rates. Use Google Search to get the most "
                        "up-to-date information, especially for weather and market prices. "
                        "Provide concise, easy-to-understand advice for a farmer."
            }]
        },
        "generationConfig": {
            "temperature": 0.7, # A slightly higher temperature for more natural conversation.
        },
    }

    # Send the request to the Gemini API with grounding enabled.
    gemini_response_text = call_gemini_api(payload)
    return jsonify({"response": gemini_response_text})


if __name__ == '__main__':
    # This runs the Flask server. It will be listening for requests from your frontend.
    app.run(debug=True, port=5000)
