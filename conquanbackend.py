# conquanbackend.py - Updated Flask Backend for Project Risk Analysis
# Handles 16x16 eigenmatrix extraction and PERT-based distributions

from flask import Flask, request, jsonify
from flask_cors import CORS 
import integrationapi 
import os
import threading
import time

app = Flask(__name__)
# Enable CORS for the Streamlit frontend
CORS(app) 

# System state
models_are_loaded = False
models_loading_lock = threading.Lock()
model_loading_thread = None

@app.route('/status', methods=['GET'])
def get_status():
    """Returns real-time status including model loading progress."""
    with integrationapi._status_lock:
        status_data = integrationapi._internal_analysis_status.copy()
    status_data['models_loaded'] = models_are_loaded
    return jsonify(status_data)

@app.route('/load_models', methods=['POST'])
def load_models_endpoint():
    """Triggers asynchronous model loading."""
    global models_are_loaded, model_loading_thread

    with models_loading_lock:
        if models_are_loaded:
            return jsonify({"message": "AI models already loaded."}), 200
        if model_loading_thread and model_loading_thread.is_alive():
            return jsonify({"message": "AI models are currently loading..."}), 202

        def _load():
            global models_are_loaded
            try:
                integrationapi.load_all_models()
                models_are_loaded = True
            except Exception as e:
                print(f"Error loading models: {e}")
                integrationapi._update_internal_status(f"Error loading models: {e}", 0, error=str(e))
                models_are_loaded = False

        model_loading_thread = threading.Thread(target=_load)
        model_loading_thread.start()
        return jsonify({"message": "AI models started loading asynchronously."}), 202

@app.route('/analyze_project', methods=['POST'])
def analyze_project_route():
    """
    Main analysis endpoint. 
    Receives project description and returns PERT results and 16x16 eigenmatrix.
    """
    if not models_are_loaded:
        return jsonify({"error": "AI models are not loaded. Please load models first."}), 503

    data = request.json
    project_description = data.get('project_description', '')

    if not project_description:
        return jsonify({"error": "No project description provided."}), 400

    try:
        # Calls the updated pipeline that returns 16x16 debug information
        analysis_results = integrationapi.run_analysis_pipeline_for_api(project_description)
        
        if analysis_results.get('error'):
            return jsonify({"error": analysis_results['error']}), 500
            
        return jsonify(analysis_results), 200
    except Exception as e:
        print(f"Flask backend analyze_project_route error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/perform_safety_analysis', methods=['POST'])
def perform_safety_analysis_route():
    if not models_are_loaded:
        return jsonify({"error": "AI models are not loaded."}), 503

    data = request.json
    try:
        safety_results = integrationapi.run_safety_analysis_pipeline(
            data.get('project_description'),
            data.get('initial_analysis_results', {})
        )
        if safety_results.get('error'):
            return jsonify({"error": safety_results['error']}), 500
        return jsonify(safety_results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/procurement_analysis', methods=['POST'])
def procurement_analysis_route():
    if not models_are_loaded:
        return jsonify({"error": "AI models are not loaded."}), 503

    data = request.json
    try:
        results = integrationapi.run_procurement_analysis_pipeline(
            data.get('initial_analysis_results', {}),
            data.get('project_description', '')
        )
        if results.get('error'):
            return jsonify({"error": results['error']}), 500
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_schedule_xml', methods=['POST'])
def generate_schedule_xml_route():
    if not models_are_loaded:
        return jsonify({"error": "AI models are not loaded."}), 503
    data = request.json
    try:
        results = integrationapi.generate_project_schedule_and_xml(data.get('project_description', ''))
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_with_ai():
    if not models_are_loaded:
        return jsonify({"error": "AI models are not loaded."}), 503
    data = request.json
    try:
        ai_response = integrationapi._generate_llm_response(data.get('message'))
        return jsonify({"response": ai_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)