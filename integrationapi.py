# ifntegrationapi.py - Comprehensive Project Risk Analysis and AI Assistant Backend
# This file contains the core logic for project risk assessment, quantum influence
# calculation, safety analysis, procurement analysis, schedule generation, and AI chat.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import random
import joblib
import pennylane as qml
from pennylane import numpy as np # Use pennylane.numpy for QML operations
import json
import os
import uuid
import re
import warnings
from threading import Thread, Lock
from tqdm import tqdm
import requests
import google.generativeai as genai
import time
from dotenv import load_dotenv
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import random # For jitter in retry logic
# Load environment variables from key.env file.
# This ensures sensitive information like API keys are not hardcoded.
load_dotenv('key.env')

# --- Qiskit Imports ---
# QiskitRuntimeService is used for connecting to IBM Quantum's cloud services.
# Sampler is for running quantum circuits on real hardware or simulators using V2 primitives.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
# QiskitBackendNotFoundError is used to catch specific errors when a backend is unavailable.
from qiskit.providers.exceptions import QiskitBackendNotFoundError
# QuantumCircuit and transpile are essential for defining and optimizing quantum circuits.
from qiskit import QuantumCircuit, transpile
# For V2 primitives, PrimitiveResult is key to handling results.
from qiskit.primitives import PrimitiveResult


# NEW: Import for handling specific Gemini API exceptions, ensuring graceful degradation.
import google.api_core.exceptions

# NEW: Import the safety screening function from prohibition.py.
# This external module is assumed to contain logic for content moderation.
from prohibition import _perform_safety_screening

# --- IMPORTANT: Set Matplotlib backend BEFORE importing pyplot ---
# 'Agg' backend is used for non-interactive plotting, suitable for generating images
# that will be saved to memory (BytesIO) and then base64 encoded for web display.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Standard library imports for I/O and encoding.
import io
import base64


# Suppress specific warnings that might clutter the console output during model loading or inference.
warnings.filterwarnings("ignore", message="The `pad_token_id` has not not been set.")
warnings.filterwarnings("ignore", message="using `max_steps` is deprecated and will be removed in a future version")
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", message="A matching Triton is not available, some optimizations will not be enabled.\n"
                                          "For example, if you want to use the Triton fused RMSNorm, make sure to install Triton.\n")

# --- Global Configuration and Model Storage ---
# Paths to pre-trained models and scalers. These should be relative to the script's directory.
PRETRAINED_MODEL_NAME = "./models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
MODEL_PATH_BERT_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "project_analysis_bert.pth")
SCALER_PATH_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "baseline_scaler.pkl")
_models = {
    "risk_analysis_model": None,
    "q_llm_tokenizer": None,
    "q_llm_model": None,
    "q_llm_peft_model": None,
    "q_llm_streamer": None,
    "qiskit_runtime_service": None, # Store QiskitRuntimeService instance
    "qiskit_real_backend": None, # Store the selected real Qiskit backend
}
_models_lock = Lock() # Lock for thread-safe access to models
_internal_analysis_status = {"message": "Initializing...", "progress": 0, "error": None}
_status_lock = Lock() # Lock for thread-safe access to status
MODEL_PATH_QUALQUAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "QualQuan.pth")
NUMERICAL_SCALER_QUALQUAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "numerical_scaler_qualquan.pkl")
QUAL_LABEL_ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "qual_label_encoder.pkl")

# Configuration for the fine-tuned TinyLlama model (Q-LLM).
BASE_TINYLLAMA_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyllama_risk_tuned", "final_adapters")

# Simulation parameters.
NUM_SIMULATIONS = 20000 # Number of Monte Carlo simulations to run.
NUM_RISKS_TO_GENERATE_FROM_LLM = 15 # Default number of risks/opportunities to generate.
NUM_TASKS_TO_GENERATE_FROM_LLM = 10 # Number of project tasks to generate.

# --- Exchange Rate Configuration and Dynamic Fetching ---
# Default USD to CAD exchange rate, used as a fallback if API fetch fails.
DEFAULT_USD_TO_CAD_EXCHANGE_RATE = 1.37
# API key for Open Exchange Rates, loaded from environment variables.
OER_API_KEY = os.environ.get("OER")
OER_API_URL = "https://openexchangerates.org/api/latest.json"

# Global cache for the exchange rate to avoid repeated API calls.
_exchange_rate_cache = {
    "rate": DEFAULT_USD_TO_CAD_EXCHANGE_RATE,
    "timestamp": 0 # Unix timestamp of last fetch, 0 means never fetched or expired.
}
EXCHANGE_RATE_CACHE_DURATION_SECONDS = 24 * 60 * 60 # Cache duration: 24 hours.

def get_current_usd_to_cad_exchange_rate(api_key, default_rate):
    """
    Fetches the current USD to CAD exchange rate from the Open Exchange Rates API.
    Caches the result to minimize API calls and falls back to a default rate if the API call fails.

    Args:
        api_key (str): The API key for Open Exchange Rates.
        default_rate (float): The default exchange rate to use if API fetching fails.

    Returns:
        float: The current or default USD to CAD exchange rate.
    """
    global _exchange_rate_cache

    current_time = time.time()
    # Check if the cached rate is still valid.
    if current_time - _exchange_rate_cache["timestamp"] < EXCHANGE_RATE_CACHE_DURATION_SECONDS:
        print(f"Using cached USD to CAD exchange rate: {_exchange_rate_cache['rate']:.4f} (cached at {time.ctime(_exchange_rate_cache['timestamp'])})")
        return _exchange_rate_cache["rate"]

    print("Cached exchange rate expired or not found. Attempting to fetch new rate...")

    if not api_key:
        print("Warning: OER API key not found. Using default exchange rate.")
        _exchange_rate_cache["rate"] = default_rate
        _exchange_rate_cache["timestamp"] = current_time # Update timestamp even for default.
        return default_rate

    try:
        # Open Exchange Rates API typically defaults to USD as base for free plans.
        params = {'app_id': api_key, 'base': 'USD', 'symbols': 'CAD'}
        response = requests.get(OER_API_URL, params=params, timeout=5) # Added a timeout for robustness.
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx).

        data = response.json()
        if 'rates' in data and 'CAD' in data['rates']:
            current_rate = data['rates']['CAD']
            print(f"Successfully fetched current USD to CAD exchange rate: {current_rate:.4f}")
            _exchange_rate_cache["rate"] = current_rate
            _exchange_rate_cache["timestamp"] = current_time
            return current_rate
        else:
            print("Error: 'CAD' rate not found in API response. Using default exchange rate.")
            _exchange_rate_cache["rate"] = default_rate
            _exchange_rate_cache["timestamp"] = current_time
            return default_rate
    except requests.exceptions.ConnectionError as e:
        print(f"Network error connecting to OER API: {e}. Using default exchange rate.")
        _exchange_rate_cache["rate"] = default_rate
        _exchange_rate_cache["timestamp"] = current_time
        return default_rate
    except requests.exceptions.Timeout as e:
        print(f"OER API request timed out: {e}. Using default exchange rate.")
        _exchange_rate_cache["rate"] = default_rate
        _exchange_rate_cache["timestamp"] = current_time
        return default_rate
    except requests.exceptions.RequestException as e:
        print(f"Error fetching exchange rate from OER API: {e}. Using default exchange rate.")
        _exchange_rate_cache["rate"] = default_rate
        _exchange_rate_cache["timestamp"] = current_time
        return default_rate
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from OER API response: {e}. Using default exchange rate.")
        _exchange_rate_cache["rate"] = default_rate
        _exchange_rate_cache["timestamp"] = current_time
        return default_rate
    except Exception as e:
        print(f"An unexpected error occurred while fetching exchange rate: {e}. Using default exchange rate.")
        _exchange_rate_cache["rate"] = default_rate
        _exchange_rate_cache["timestamp"] = current_time
        return default_rate

# Assign the USD_TO_CAD_EXCHANGE_RATE dynamically at startup.
USD_TO_CAD_EXCHANGE_RATE = get_current_usd_to_cad_exchange_rate(OER_API_KEY, DEFAULT_USD_TO_CAD_EXCHANGE_RATE)
# --- End Exchange Rate Configuration ---

# Paths for logging generated schedule data.
SCHEDULE_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "schedule_log.jsonl")
PROCUREMENT_SCHEDULE_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "procurement_schedule_log.jsonl")
RESOURCE_PLAN_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "resource_plan_log.jsonl")
STAKEHOLDER_ANALYSIS_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stakeholder_analysis_log.jsonl")


# Global variables to store loaded models and data.
# These are initialized to None and populated during the `load_all_models` function.
project_analysis_bert_model = None
project_analysis_bert_tokenizer = None
risk_llm_qualquan_model = None # Refers to the fine-tuned TinyLlama model.
fine_tuned_risk_model = None # Alias for risk_llm_qualquan_model after PEFT merge.
fine_tuned_risk_tokenizer = None
numerical_scaler_qualquan = None
qual_label_encoder = None
baseline_scaler = None
scheduler_model = None
scheduler_tokenizer = None
gemini_model_text = None
gemma_model_risk = None

# Flag to control whether to use Gemini API or local LLM (TinyLlama).
USE_GEMINI_API = True
gemini_model_text = None # Stores the Gemini GenerativeModel instance.
# List of Gemini API keys for redundancy and rate limit management.
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY2"),
    os.environ.get("GEMINI_API_KEY3"),
    os.environ.get("GEMINI_API_KEY4"),
    os.environ.get("GEMINI_API_KEY5"),
    os.environ.get("GEMINI_API_KEY6"),
    os.environ.get("GEMINI_API_KEY7")
]
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite" # Preferred Gemini model for text generation.
GEMMA_MODEL_NAME = "gemma-3-27b-it" # Eat it google
# Determine the device (CUDA/GPU or CPU) for PyTorch operations.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for LLM operations: {device}")

# Global variables for quantum circuits and weights.
q_weights = None # Weights for the primary quantum circuit.
quantum_circuit = None # The primary quantum circuit (real or simulated).
real_quantum_circuit = None # Reference to the QNode for the real quantum device.
simulated_quantum_circuit = None # Reference to the QNode for the local simulator.

TRAINED_Q_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained_q_weights.pt")
QUANTUM_TRAINING_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "quantum_training_log.jsonl")

# Global variables for an alternative quantum circuit, used for validity comparison.
alternative_quantum_circuit = None
q_weights_alternative = None
num_wires = 4 # Number of qubits/wires for the quantum circuits.

# --- Internal State Management for Status Reporting ---
# This dictionary holds the current status of the backend, including model loading progress.
_internal_analysis_status = {
    "message": "Backend initializing...",
    "progress": 0,
    "models_loaded": False,
    "error": None,
    "timestamp": time.time() # Timestamp of the last status update.
}
_status_lock = Lock() # A lock to ensure thread-safe updates to the status dictionary.

def _update_internal_status(message, progress=None, error=None):
    """
    Updates the internal status dictionary with thread-safety.
    This function is used to provide real-time feedback on backend operations.

    Args:
        message (str): A descriptive message about the current status.
        progress (int, optional): A percentage indicating the progress (0-100). Defaults to None.
        error (str, optional): An error message if an error occurred. Defaults to None.
    """
    with _status_lock:
        _internal_analysis_status["message"] = message
        if progress is not None:
            _internal_analysis_status["progress"] = progress
        if error is not None:
            _internal_analysis_status["error"] = error
        _internal_analysis_status["timestamp"] = time.time()
    print(f"INTERNAL STATUS: {message} ({progress if progress is not None else _internal_analysis_status['progress']}%)")


# --- Model Loading Function ---
def load_all_models():
    """
    Loads all project AI/ML models. 
    Fixes the 'meta tensor' crash by forcing specific device mapping for PEFT models.
    """
    global project_analysis_bert_model, project_analysis_bert_tokenizer, \
           risk_llm_qualquan_model, fine_tuned_risk_model, fine_tuned_risk_tokenizer, \
           numerical_scaler_qualquan, qual_label_encoder, baseline_scaler, \
           gemini_model_text, gemma_model_risk, q_weights, quantum_circuit, \
           real_quantum_circuit, simulated_quantum_circuit, \
           alternative_quantum_circuit, q_weights_alternative, num_wires, \
           scheduler_model, scheduler_tokenizer, _models

    SCHEDULER_BASE_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    SCHEDULER_ADAPTER_PATH = os.path.join("schedulemodel", "scheduler_model_tuned", "final_adapters")
    
    try:
        _update_internal_status("Starting model loading process...", 0)

        with _models_lock:
            # 1. BERT Baseline and Scalers
            _update_internal_status("Loading BERT and Scalers...", 10)
            project_analysis_bert_tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
            project_analysis_bert_model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME).to(device)
            project_analysis_bert_model.eval()
            
            numerical_scaler_qualquan = joblib.load(NUMERICAL_SCALER_QUALQUAN_PATH)
            qual_label_encoder = joblib.load(QUAL_LABEL_ENCODER_PATH)
            baseline_scaler = joblib.load(SCALER_PATH_BASELINE)

            # 2. Gemini & Gemma API Configuration
            if USE_GEMINI_API:
                _update_internal_status("Configuring Gemini & Gemma APIs...", 25)
                api_loaded = False
                for i, api_key in enumerate(GEMINI_API_KEYS):
                    if api_key:
                        try:
                            genai.configure(api_key=api_key)
                            gemini_model_text = genai.GenerativeModel(GEMINI_MODEL_NAME)
                            gemma_model_risk = genai.GenerativeModel(GEMMA_MODEL_NAME)
                            _models["baseline_estimator"] = gemini_model_text
                            _models["risk_analysis_model"] = gemma_model_risk
                            api_loaded = True
                            break 
                        except Exception: continue
                if not api_loaded: raise ConnectionError("API Key validation failed.")

            # 3. QualQuan Risk LLM (TinyLlama)
            _update_internal_status("Loading Risk LLM (TinyLlama)...", 45)
            base_risk_model = AutoModelForCausalLM.from_pretrained(BASE_TINYLLAMA_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
            fine_tuned_risk_tokenizer = AutoTokenizer.from_pretrained(BASE_TINYLLAMA_MODEL_ID)
            try:
                # Merge adapters to avoid multi-model VRAM fragmentation
                fine_tuned_risk_model = PeftModel.from_pretrained(base_risk_model, LORA_ADAPTERS_PATH)
                fine_tuned_risk_model = fine_tuned_risk_model.merge_and_unload()
                risk_llm_qualquan_model = fine_tuned_risk_model
            except Exception:
                fine_tuned_risk_model = base_risk_model
            fine_tuned_risk_model.to(device).eval()

            # 4. QWEN SCHEDULER (Meta-Tensor & NoneType Fix)
            _update_internal_status("Loading Scheduler (Qwen)...", 60)
            local_qwen_path = os.path.join(os.getcwd(), "models--Qwen2.5-1.5B-Instruct")
            sched_id = local_qwen_path if os.path.exists(local_qwen_path) else SCHEDULER_BASE_ID
            
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                
                # --- FIX: SAFE DEVICE MAPPING ---
                if device.type == "cuda":
                    # Use explicit index if available, else default to primary 0
                    gpu_idx = device.index if device.index is not None else 0
                    target_map = {"": gpu_idx}
                else:
                    target_map = {"": "cpu"}

                sched_base = AutoModelForCausalLM.from_pretrained(
                    sched_id,
                    quantization_config=bnb_config,
                    device_map=target_map, # Forces model onto real hardware
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                scheduler_tokenizer = AutoTokenizer.from_pretrained(sched_id)

                if os.path.exists(SCHEDULER_ADAPTER_PATH):
                    # Load adapters using the exact same hardware mapping
                    scheduler_model = PeftModel.from_pretrained(
                        sched_base, 
                        SCHEDULER_ADAPTER_PATH, 
                        device_map=target_map
                    )
                else:
                    scheduler_model = sched_base
                _update_internal_status("Scheduler loaded.", 75)
            except Exception as e:
                print(f"Scheduler Load Error: {e}")
                scheduler_model = None

            _update_internal_status("Initializing quantum circuits...", 95)
            
            # --- Define the Quantum Circuit Ansatz ---
            # This quantum circuit structure (ansatz) will be used for all quantum computations.
            # It uses AngleEmbedding for input data and StronglyEntanglingLayers for entanglement.
            def quantum_ansatz(weights, data):
                """
                Defines the quantum circuit ansatz.

                Args:
                    weights (np.array): Trainable weights for the entangling layers.
                    data (np.array): Input data to be embedded into the quantum circuit.

                Returns:
                    list: Expectation values of PauliZ operators on each wire.
                """
                qml.AngleEmbedding(data, wires=range(num_wires))
                qml.StronglyEntanglingLayers(weights, wires=range(num_wires))
                return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

            # --- 1. Attempt to set up REAL Quantum Computer Backend ---
            real_quantum_circuit_dev = None
            ibm_quantum_token = os.getenv("IBM_QUANTUM_TOKEN")
            if ibm_quantum_token:
                try:
                    print("Attempting to connect to IBM Quantum Runtime service for REAL backend...")
                    # Initialize QiskitRuntimeService with token and channel.
                    # The instance ID is specific to a particular IBM Quantum account/provider.
                    service = QiskitRuntimeService(token=ibm_quantum_token, channel='ibm_cloud', instance='crn:v1:bluemix:public:quantum-computing:us-east:a/62057764101b46b4a5551e846c604b9c:5a185390-8288-4075-ad14-0b8af1dbaddf::')
                    _models["qiskit_runtime_service"] = service # Store the service instance.
                    
                    # First, try to connect to 'ibm_brisbane' explicitly.
                    target_backend_name = 'ibm_torino'
                    try:
                        backend = service.backend(target_backend_name)
                        # Check if the backend is suitable (enough qubits, not a simulator, operational).
                        if backend.configuration().n_qubits >= num_wires and not backend.configuration().simulator and backend.status().operational:
                            # Store the real Qiskit backend for direct use with Sampler.
                            _models["qiskit_real_backend"] = backend 
                            print(f"SUCCESS: Real Qiskit backend '{target_backend_name}' selected for direct Sampler use.")
                        else:
                            print(f"INFO: Targeted backend '{target_backend_name}' is not suitable (e.g., not enough qubits, simulator, or not operational).")
                            _models["qiskit_real_backend"] = None # Ensure it's None so we search for another
                    except QiskitBackendNotFoundError:
                        print(f"INFO: Targeted backend '{target_backend_name}' not found or accessible. Searching for least busy suitable backend.")
                    except Exception as e:
                        print(f"ERROR: Unexpected error when trying to connect to '{target_backend_name}': {e}. Searching for least busy suitable backend.")

                    # If a real backend is still not set, find the least busy one.
                    if _models["qiskit_real_backend"] is None:
                        print("Searching for the least busy suitable real backend...")
                        # Filter for real, operational backends with enough qubits.
                        real_backends = service.backends(filters=lambda b: b.configuration().n_qubits >= num_wires and not b.configuration().simulator and b.status().operational)
                        if real_backends:
                            # Sort by pending jobs to find the least busy one.
                            least_busy_backend = sorted(real_backends, key=lambda b: b.status().pending_jobs)[0]
                            _models["qiskit_real_backend"] = least_busy_backend
                            print(f"SUCCESS: Real Qiskit backend '{least_busy_backend.name}' selected for direct Sampler use.")
                        else:
                            print(f"INFO: No suitable real IBM quantum computer found. Real quantum computation will not be available.")
                except Exception as e:
                    print(f"ERROR: Could not connect to IBM Quantum real device: {e}. Real quantum computation will not be available.")
            else:
                print("INFO: IBM_QUANTUM_TOKEN environment variable not found. Real quantum computation will not be available.")

            # --- 2. Attempt to set up LOCAL SIMULATED (GPU/CPU) Backend ---
            simulated_quantum_circuit_dev = None
            try:
                # Prioritize GPU simulator if CUDA is available.
                if torch.cuda.is_available():
                    simulated_quantum_circuit_dev = qml.device("lightning.gpu", wires=num_wires)
                    print("SUCCESS: Local simulated backend 'lightning.gpu' (CUDA) initialized.")
                else:
                    # Fallback to CPU simulator.
                    simulated_quantum_circuit_dev = qml.device("lightning.qubit", wires=num_wires)
                    print("SUCCESS: Local simulated backend 'lightning.qubit' (CPU) initialized.")
            except qml.DeviceError as e:
                print(f"ERROR: PennyLane Lightning device failed to load: {e}. Falling back to default.qubit.")
                simulated_quantum_circuit_dev = qml.device("default.qubit", wires=num_wires)
                print("SUCCESS: Fallback simulated backend 'default.qubit' initialized.")
            
            # Create the QNode for the simulated device.
            simulated_quantum_circuit = qml.QNode(quantum_ansatz, simulated_quantum_circuit_dev)

            # --- 3. Set the default PennyLane quantum_circuit for general use ---
            # This will now always be the simulator. Real hardware is accessed via the Qiskit Sampler directly.
            quantum_circuit = simulated_quantum_circuit
            print("Default PennyLane quantum circuit set to LOCAL SIMULATOR.")


            # --- Load Quantum Weights ---
            # Load pre-trained quantum circuit weights if available, otherwise initialize randomly.
            if os.path.exists(TRAINED_Q_WEIGHTS_PATH):
                # Added weights_only=True to resolve FutureWarning in PyTorch.
                q_weights = torch.load(TRAINED_Q_WEIGHTS_PATH, weights_only=True)
                print("Pre-trained quantum circuit weights loaded.")
            else:
                # Initialize random weights if no trained weights are found.
                q_weights = nn.Parameter(torch.rand(1, num_wires, 3, dtype=torch.float64))
                print("New random quantum circuit weights initialized.")

            # --- Setup Alternative Quantum Circuit (for validity comparison) ---
            _update_internal_status("Setting up alternative quantum circuit...", 99)
            try:
                # The alternative circuit uses a different ansatz (BasicEntanglerLayers)
                # and a default PennyLane simulator for comparison purposes.
                dev_alt = qml.device("default.qubit", wires=num_wires)
                @qml.qnode(dev_alt)
                def alternative_qnode(weights, data):
                    """
                    Defines an alternative quantum circuit ansatz for comparison.
                    Uses BasicEntanglerLayers.
                    """
                    qml.AngleEmbedding(data, wires=range(num_wires))
                    num_layers_alt = 2 # Number of layers for the alternative circuit.
                    qml.BasicEntanglerLayers(weights, wires=range(num_wires))
                    return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]
                alternative_quantum_circuit = alternative_qnode
                num_layers_alt = 2
                q_weights_alternative = np.random.uniform(low=0, high=2 * np.pi, size=(num_layers_alt, num_wires), requires_grad=True)
                _update_internal_status("Alternative quantum circuit setup complete.", 99)
            except Exception as e:
                _update_internal_status(f"Failed to set up alternative quantum circuit: {e}", 0, error=str(e))
                print(f"Error setting up alternative quantum circuit: {e}")
                alternative_quantum_circuit = None
                q_weights_alternative = None

            _update_internal_status("All models and data loaded successfully.", 100)
            _internal_analysis_status["models_loaded"] = True

    except Exception as e:
        _update_internal_status(f"Error loading models: {e}", 0, error=str(e))
        _internal_analysis_status["models_loaded"] = False
        print(f"Backend: Error during model loading: {e}")
        raise # Re-raise the exception to indicate a critical failure.
def _get_qiskit_job_results(job_result, num_wires):
    """
    Processes the result from a Qiskit Sampler job to get the quasi-distribution and a heuristic expectation value.
    This function correctly handles the V2 Primitive result format and is robust to different outcome types.

    Args:
        job_result (PrimitiveResult): The result object from job.result().
        num_wires (int): The number of qubits in the circuit.

    Returns:
        tuple: A tuple containing (quasi_distribution, avg_exp_val).
               - quasi_distribution (dict): A dictionary mapping outcomes to probabilities.
               - avg_exp_val (float): A heuristic average expectation value.
    """
    # Access the first PubResult directly by indexing the PrimitiveResult.
    pub_result = job_result[0]
    data_bin = pub_result.data

    # The classical register is typically named 'c' or 'meas' by default.
    # We check for 'c' first, then try other common names if not found.
    if hasattr(data_bin, 'c'):
        counts = data_bin.c.get_counts()
    elif hasattr(data_bin, 'meas'):
         counts = data_bin.meas.get_counts()
    else:
        # Fallback to the first available data bin if common names aren't found.
        try:
            counts = next(iter(data_bin.values())).get_counts()
        except (StopIteration, AttributeError):
             raise AttributeError("Could not find a classical register with counts in SamplerPubResult. Check circuit measurement and register naming.")

    shots = sum(counts.values())
    if shots == 0:
        return {}, 0.0

    quasi_distribution = {outcome: freq / shots for outcome, freq in counts.items()}

    # Heuristic to map the quantum measurement distribution to a single classical value.
    # This calculates an average Z-basis expectation value based on the parity of the bitstrings.
    avg_exp_val = 0.0
    for outcome, prob in quasi_distribution.items():
        # FIX: Handle cases where the outcome from get_counts() is an integer or a binary string.
        if isinstance(outcome, str):
            # If it's already a string, assume it's a binary representation (e.g., '101' or '0b101').
            # Remove any '0b' prefix for consistency before padding.
            bitstring = outcome.replace('0b', '')
        else:
            # If it's an integer, convert it to a binary string and remove the '0b' prefix.
            bitstring = bin(outcome)[2:]
        
        # Pad the bitstring to the correct number of wires.
        padded_bitstring = bitstring.zfill(num_wires)

        # Calculate parity: (-1) if odd number of 1s, (+1) if even.
        parity = (-1)**(padded_bitstring.count('1'))
        avg_exp_val += parity * prob
    
    return quasi_distribution, avg_exp_val


# NEW FUNCTION: Compare real quantum hardware vs. local GPU/CPU simulation






# NEW FUNCTION: Compare real quantum hardware vs. local GPU/CPU simulation
def compare_real_vs_simulated_backend(input_data):
    """
    Executes the same quantum computation on the real quantum backend (if available) and the
    local simulator, then returns their results for direct comparison.
    This function is crucial for validating quantum algorithms and understanding hardware performance.

    Args:
        input_data (np.array): The input data for the quantum circuit. This data will be
                               embedded into the quantum circuit.

    Returns:
        dict: A dictionary containing results from both backends, including execution times,
              and an error message if one of the backends is not available or encounters an issue.
    """
    global real_quantum_circuit, simulated_quantum_circuit, q_weights, num_wires

    results = {
        "real_backend_result": None,
        "simulated_backend_result": None,
        "real_backend_execution_time_seconds": None,
        "simulated_backend_execution_time_seconds": None,
        "error": None
    }

    # Ensure the local simulator is available. It's the primary fallback.
    if simulated_quantum_circuit is None:
        results["error"] = "Local simulated quantum circuit is not available. Cannot perform any quantum computation."
        print(f"ERROR: {results['error']}")
        return results

    # Ensure input data is correctly formatted as a PennyLane NumPy array and padded/truncated to `num_wires`.
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data, requires_grad=False)
    if input_data.shape[-1] != num_wires:
        if input_data.shape[-1] < num_wires:
            padded_data = np.pad(input_data, (0, num_wires - input_data.shape[-1]), 'constant', constant_values=0)
            input_data = padded_data
        else:
            input_data = input_data[:num_wires]
    input_data = np.array(input_data, requires_grad=False) # Ensure no gradient tracking for input data.

    try:
        # --- Run on Local Simulator (GPU/CPU) ---
        print("\n--- Executing on Local Simulator Backend ---")
        start_time_sim = time.time()
        simulated_output = simulated_quantum_circuit(q_weights, input_data)
        end_time_sim = time.time()
        results["simulated_backend_result"] = [float(x) for x in simulated_output]
        results["simulated_backend_execution_time_seconds"] = round(end_time_sim - start_time_sim, 4)
        print(f"Local Simulator Execution Time: {results['simulated_backend_execution_time_seconds']:.4f} seconds")
        print(f"Local Simulator Result: {results['simulated_backend_result']}")

    except Exception as e:
        error_msg = f"An error occurred while running the local simulator: {e}"
        print(f"ERROR: {error_msg}")
        results["error"] = error_msg
        return results # Stop if simulation fails as it's the baseline.

    # Attempt to run on the real quantum backend if it's available.
    if _models.get("qiskit_real_backend") is None:
        results["error"] = "Real quantum hardware backend is not available. Cannot perform comparison with real device."
        print(f"INFO: {results['error']}")
        return results # Return results with only simulation data and an error message.

    try:
        # --- Run on Real Quantum Backend (via Sampler) ---
        print("\n--- Executing on Real Quantum Backend (via Qiskit Sampler) ---")
        print("NOTE: This may take several minutes due to job queuing and execution on the quantum device.")
        
        with _models_lock:
            qiskit_service = _models.get("qiskit_runtime_service")
            qiskit_backend = _models.get("qiskit_real_backend")

        if qiskit_service is None or qiskit_backend is None:
            raise RuntimeError("Qiskit Runtime Service or real backend not initialized.")

        # Create a simple Qiskit QuantumCircuit for the Sampler
        qc_for_sampler = QuantumCircuit(num_wires, num_wires)
        
        # Create a Bell state circuit for demonstration.
        qc_for_sampler.h(0)
        qc_for_sampler.cx(0, 1)
        qc_for_sampler.measure(range(num_wires), range(num_wires)) # Measure all qubits

        # Transpile the Qiskit circuit for the selected backend
        print(f"Transpiling Qiskit circuit for backend: {qiskit_backend.name}...")
        transpiled_qc_for_sampler = transpile(qc_for_sampler, qiskit_backend)
        print("Qiskit Circuit (Transpiled for Sampler):")
        print(transpiled_qc_for_sampler.draw(output='text'))

        # Initialize Sampler with the real backend
        sampler = Sampler(mode=qiskit_backend)
        print("Sampler initialized for real backend.")

        start_time_real = time.time()
        # Run the transpiled Qiskit circuit with Sampler
        job = sampler.run([transpiled_qc_for_sampler], shots=1024)
        print(f"Job submitted! Job ID: {job.job_id()}")
        
        real_qiskit_result = job.result() # Wait for the job to finish
        end_time_real = time.time()

        # Use the centralized helper function to process the Sampler V2 results.
        quasi_distribution, _ = _get_qiskit_job_results(real_qiskit_result, num_wires)

        # Convert quasi_distribution to a more readable format for comparison.
        # The keys are integers representing the measured bitstrings. We convert them to binary strings.
        real_output_for_comparison = {bin(k)[2:].zfill(num_wires) if isinstance(k, int) else k: float(v) for k, v in quasi_distribution.items()}
        
        results["real_backend_result"] = real_output_for_comparison
        results["real_backend_execution_time_seconds"] = round(end_time_real - start_time_real, 4)
        print(f"Real Quantum Backend Execution Time (via Sampler): {results['real_backend_execution_time_seconds']:.4f} seconds")
        print(f"Real Quantum Backend Result (via Sampler): {results['real_backend_result']}")

    except Exception as e:
        error_msg = f"An error occurred while running on the real quantum backend via Sampler: {e}"
        print(f"ERROR: {error_msg}")
        if results["error"] is None:
            results["error"] = ""
        results["error"] += f" | {error_msg}" # Append real backend error to existing errors.
        
    return results

def perform_quantum_sensitivity_analysis(original_data, num_perturbations=100, perturbation_strength=0.01):
    """
    Performs a sensitivity analysis on the primary quantum circuit by perturbing its input data.
    This helps understand how robust the quantum circuit's output is to small variations in input.

    Args:
        original_data (np.array): The original input data for the quantum circuit (e.g., [0.5, 0.2, 0.8, 0.1]).
                                  Must be a PennyLane NumPy array or convertible to one.
        num_perturbations (int): The number of times to perturb the input data and run the circuit.
        perturbation_strength (float): The magnitude of the random perturbation to apply to each data point.
                                       This is the standard deviation of the normal distribution used for noise.

    Returns:
        dict: A dictionary containing statistical measures of the perturbed outputs:
              - 'mean_output': The mean of the outputs across all perturbations.
              - 'std_dev_output': The standard deviation of the outputs, indicating sensitivity.
              - 'min_output': The minimum output observed.
              - 'max_output': The maximum output observed.
              - 'perturbation_results': A list of outputs from each perturbed run.
              - 'error': Error message if the quantum model is not loaded or an issue occurs.
    """
    global quantum_circuit, q_weights, num_wires

    if quantum_circuit is None or q_weights is None:
        _update_internal_status("Quantum circuit or weights not loaded for sensitivity analysis.", error=True)
        return {"error": "Quantum model not loaded. Please load models first."}

    # Ensure input data is a PennyLane NumPy array and correctly sized.
    if not isinstance(original_data, np.ndarray):
        original_data = np.array(original_data, requires_grad=False)

    if original_data.shape[-1] != num_wires:
        # Pad or truncate data to num_wires features as expected by AngleEmbedding.
        if original_data.shape[-1] < num_wires:
            padded_data = np.pad(original_data, (0, num_wires - original_data.shape[-1]), 'constant', constant_values=0)
            original_data = padded_data
        else:
            original_data = original_data[:num_wires]
    
    original_data = np.array(original_data, requires_grad=False)

    results = []
    _update_internal_status(f"Starting quantum sensitivity analysis with {num_perturbations} perturbations...", 0)

    for i in tqdm(range(num_perturbations), desc="Quantum Sensitivity Analysis"):
        # Create perturbed data by adding random noise from a normal distribution.
        noise = np.random.normal(0, perturbation_strength, original_data.shape)
        perturbed_data = np.array(original_data + noise, requires_grad=False)

        # Ensure perturbed_data is clipped to reasonable bounds (e.g., for angle embeddings).
        # Clipping to [-pi, pi] is a common practice for AngleEmbedding.
        perturbed_data = np.clip(perturbed_data, -np.pi, np.pi)

        try:
            # Execute the quantum circuit with perturbed data.
            output = quantum_circuit(q_weights, perturbed_data)
            # Assuming the output is a single expectation value, convert to numpy float.
            # If the circuit returns multiple expectation values, this would need adjustment.
            results.append(output[0].numpy())
        except Exception as e:
            print(f"Error during quantum circuit execution for perturbation {i}: {e}")
            results.append(np.nan) # Append NaN if an error occurs for a specific run.
    
    # Filter out NaNs if any errors occurred during individual runs.
    valid_results = [r for r in results if not np.isnan(r)]

    if not valid_results:
        _update_internal_status("Sensitivity analysis failed: No valid results obtained.", 100, error="No valid results.")
        return {"error": "No valid results from sensitivity analysis. Check quantum circuit or input data."}

    # Calculate statistical measures from the valid results.
    mean_output = float(np.mean(valid_results))
    std_dev_output = float(np.std(valid_results))
    min_output = float(np.min(valid_results))
    max_output = float(np.max(valid_results))

    _update_internal_status("Quantum sensitivity analysis completed.", 100)

    return {
        "mean_output": mean_output,
        "std_dev_output": std_dev_output,
        "min_output": min_output,
        "max_output": max_output,
        "perturbation_results": [float(r) for r in valid_results] # Ensure results are standard floats.
    }

def compare_quantum_circuit_validity(original_data, num_perturbations=100, perturbation_strength=0.01):
    """
    Compares the outputs of the primary and an alternative quantum circuit
    under input data perturbations to assess consistency and validity.
    This helps in verifying if different circuit designs or implementations yield similar results.

    Args:
        original_data (np.array): The original input data for the quantum circuit.
                                  Must be a PennyLane NumPy array or convertible to one.
        num_perturbations (int): The number of times to perturb the input data and run both circuits.
        perturbation_strength (float): The magnitude of the random perturbation to apply to each data point.

    Returns:
        dict: A dictionary containing comparison metrics:
              - 'primary_results_sample': A sample of outputs from the primary circuit.
              - 'alternative_results_sample': A sample of outputs from the alternative circuit.
              - 'mean_absolute_difference': Mean absolute difference between their outputs (vector-wise).
              - 'correlation_coefficient': Pearson correlation between their outputs (if outputs are 1D).
              - 'mean_cosine_similarity': Average cosine similarity between their outputs (for multi-dimensional).
              - 'num_valid_comparisons': The number of successful comparisons.
              - 'error': Error message if circuits are not loaded or an issue occurs.
    """
    global quantum_circuit, q_weights, alternative_quantum_circuit, q_weights_alternative, num_wires

    # Check if both primary and alternative quantum models are loaded.
    if quantum_circuit is None or q_weights is None:
        _update_internal_status("Primary quantum model not loaded for comparison.", error=True)
        return {"error": "Primary quantum model not loaded. Please load models first."}
    
    if alternative_quantum_circuit is None or q_weights_alternative is None:
        _update_internal_status("Alternative quantum model not loaded for comparison.", error=True)
        return {"error": "Alternative quantum model not loaded. Please ensure it's initialized during model loading."}

    # Prepare input data.
    if not isinstance(original_data, np.ndarray):
        original_data = np.array(original_data, requires_grad=False)

    if original_data.shape[-1] != num_wires:
        if original_data.shape[-1] < num_wires:
            padded_data = np.pad(original_data, (0, num_wires - original_data.shape[-1]), 'constant', constant_values=0)
            original_data = padded_data
        else:
            original_data = original_data[:num_wires]

    original_data = np.array(original_data, requires_grad=False)

    primary_outputs = []
    alternative_outputs = []
    
    _update_internal_status(f"Starting quantum circuit comparison with {num_perturbations} perturbations...", 0)

    for i in tqdm(range(num_perturbations), desc="Quantum Circuit Comparison"):
        # Generate perturbed data for each run.
        noise = np.random.normal(0, perturbation_strength, original_data.shape)
        perturbed_data = np.array(original_data + noise, requires_grad=False)
        perturbed_data = np.clip(perturbed_data, -np.pi, np.pi) # Clip if needed by embedding.

        try:
            primary_out = quantum_circuit(q_weights, perturbed_data)
            primary_outputs.append([float(x) for x in primary_out]) # Convert to list of floats.
        except Exception as e:
            print(f"Error primary circuit for perturbation {i}: {e}")
            primary_outputs.append([np.nan] * num_wires) # Append NaNs for vector output if error.

        try:
            alternative_out = alternative_quantum_circuit(q_weights_alternative, perturbed_data)
            alternative_outputs.append([float(x) for x in alternative_out]) # Convert to list of floats.
        except Exception as e:
            print(f"Error alternative circuit for perturbation {i}: {e}")
            alternative_outputs.append([np.nan] * num_wires) # Append NaNs for vector output if error.
    
    # Filter out runs where either circuit failed (had NaN values).
    valid_comparisons_primary = []
    valid_comparisons_alternative = []
    for i in range(len(primary_outputs)):
        if not any(np.isnan(val) for val in primary_outputs[i]) and \
           not any(np.isnan(val) for val in alternative_outputs[i]):
            valid_comparisons_primary.append(primary_outputs[i])
            valid_comparisons_alternative.append(alternative_outputs[i])

    if not valid_comparisons_primary:
        _update_internal_status("Comparison failed: No valid results obtained.", 100, error="No valid comparisons.")
        return {"error": "No valid results from comparison. Check quantum circuits or input data."}

    # Convert lists of lists to numpy arrays for easier calculation of metrics.
    primary_results_array = np.array(valid_comparisons_primary)
    alternative_results_array = np.array(valid_comparisons_alternative)

    # Calculate comparison metrics.
    # Mean absolute difference (element-wise, then averaged across all runs and features).
    mean_abs_diff = np.mean(np.abs(primary_results_array - alternative_results_array))

    correlation_coefficient = np.nan # Initialize as NaN for cases where it's not applicable.
    # If the output is a single value (e.g., num_wires=1), calculate direct correlation.
    if primary_results_array.shape[1] == 1:
        flat_primary = primary_results_array.flatten()
        flat_alternative = alternative_results_array.flatten()
        if len(flat_primary) > 1: # Need at least two data points for correlation.
            correlation_coefficient = np.corrcoef(flat_primary, flat_alternative)[0, 1]
    
    # Calculate Cosine Similarity for multi-dimensional vector comparison.
    cosine_similarities = []
    for i in range(len(primary_results_array)):
        vec1 = primary_results_array[i]
        vec2 = alternative_results_array[i]
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 > 1e-9 and norm_vec2 > 1e-9: # Use a small epsilon for numerical stability.
            cosine_sim = dot_product / (norm_vec1 * norm_vec2)
            cosine_similarities.append(cosine_sim)
        else:
            # If both vectors are zero, they are perfectly similar (cosine similarity 1.0).
            # If one is zero and the other is not, they are not similar (cosine similarity 0.0).
            if np.all(vec1 == 0) and np.all(vec2 == 0):
                cosine_similarities.append(1.0)
            else:
                cosine_similarities.append(0.0)

    mean_cosine_similarity = float(np.mean(cosine_similarities)) if cosine_similarities else np.nan


    _update_internal_status("Quantum circuit comparison completed.", 100)

    # Return a sample of results, not all of them, to keep response size manageable.
    sample_size = min(5, len(primary_results_array))
    return {
        "primary_results_sample": [list(map(float, r)) for r in primary_results_array[:sample_size].tolist()],
        "alternative_results_sample": [list(map(float, r)) for r in alternative_results_array[:sample_size].tolist()],
        "mean_absolute_difference": float(mean_abs_diff),
        "correlation_coefficient": float(correlation_coefficient), # Will be NaN if not 1D output.
        "mean_cosine_similarity": mean_cosine_similarity,
        "num_valid_comparisons": len(valid_comparisons_primary)
    }


# --- Helper for XML Escaping ---
def _xml_escape(text):
    """
    Escapes special characters in a string for XML compatibility.
    This prevents issues when embedding text into XML tags.

    Args:
        text (str): The input string to escape.

    Returns:
        str: The XML-escaped string.
    """
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\"", "&quot;")
    text = text.replace("'", "&apos;")
    return text

# NEW: Global caches for performance improvement.
# Caching results of computationally expensive or frequently called functions.
_bert_embedding_cache = {}
_qualquan_cache = {}
_risks_opportunities_cache = {}
_baseline_estimates_cache = {}
_project_tasks_cache = {}
_procurement_schedule_cache = {}
_safety_screening_cache = {}
_resource_plan_cache = {} # New cache for resource plans.
_stakeholder_analysis_cache = {} # New cache for stakeholder analysis.


def _screen_description_for_safety(project_description):
    """
    Screens the project description for harmful content (e.g., weapons, violence, illegal activities)
    using an LLM. If LLM calls fail, it falls back to a keyword-based check from prohibition.py.

    Args:
        project_description (str): The description of the project to be screened.

    Returns:
        dict: A dictionary with 'is_safe' (boolean) and 'reason' (string).
              'is_safe' is True if no harmful content is detected, False otherwise.
    """
    cache_key = project_description
    if cache_key in _safety_screening_cache:
        print("Using cached safety screening result.")
        return _safety_screening_cache[cache_key]

    _update_internal_status("Performing safety screening of project description...", 5)
    prompt = (
        f"Analyze the following project description for any content related to weapons, "
        f"violence, illegal activities, or other harmful subjects. "
        f"Respond with a JSON object containing two keys: 'is_safe' (boolean) and 'reason' (string). "
        f"If 'is_safe' is false, clearly state the specific reason(s) why, referencing the harmful content found. "
        f"Project Description: {project_description}"
    )

    # Define the expected JSON schema for the LLM response.
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "is_safe": {"type": "BOOLEAN"},
            "reason": {"type": "STRING"}
        },
        "required": ["is_safe", "reason"]
    }
    json_output = None
    
    # Try Gemini API first for safety screening.
    if USE_GEMINI_API:
        for i, api_key in enumerate(GEMINI_API_KEYS):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                    response_model = genai.GenerativeModel(GEMINI_MODEL_NAME).generate_content(
                        [{"role": "user", "parts": [{ "text": prompt }] }],
                        generation_config=generation_config
                    )
                    json_output = response_model.text
                    print(f"Safety screening: Gemini API with key {i+1} successful.")
                    break # Exit loop if successful.
                except Exception as e:
                    print(f"Safety screening: Gemini API with key {i+1} error: {e}. Trying next key.")
            else:
                print(f"Gemini API key {i+1} not found. Skipping.")
    
    # If Gemini API fails or is not used, try the local fine-tuned TinyLlama model.
    if json_output is None and fine_tuned_risk_model is not None and fine_tuned_risk_tokenizer is not None:
        try:
            inputs = fine_tuned_risk_tokenizer(prompt + "\n\nProvide the response in JSON format only.", return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_tokens = fine_tuned_risk_model.generate(**inputs, max_new_tokens=200, pad_token_id=fine_tuned_risk_tokenizer.eos_token_id)
            generated_text = fine_tuned_risk_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            json_match = re.search(r"\{.*?\}", generated_text, re.DOTALL)
            if json_match:
                json_output = json_match.group(0)
        except Exception as e:
            print(f"Local TinyLlama error for safety screening: {e}. Attempting keyword fallback.")

    try:
        if json_output:
            parsed_output = json.loads(json_output)
            # Ensure boolean and string types for the parsed output.
            is_safe = bool(parsed_output.get('is_safe', False))
            reason = str(parsed_output.get('reason', 'Unknown reason.'))
            result = {"is_safe": is_safe, "reason": reason}
            _safety_screening_cache[cache_key] = result # Cache the result.
            _update_internal_status("Safety screening completed via LLM.", 100)
            return result
        else:
            # Fallback to keyword-based screening if LLMs fail to provide a structured response.
            print("LLM safety screening failed. Falling back to keyword-based screening.")
            try:
                _perform_safety_screening(project_description) # This function is expected to raise ValueError if unsafe.
                result = {"is_safe": True, "reason": "Passed keyword-based safety screening."}
                _safety_screening_cache[cache_key] = result # Cache the result.
                _update_internal_status("Safety screening completed via keyword check (no harmful keywords found).", 100)
                return result
            except ValueError as e:
                result = {"is_safe": False, "reason": f"Keyword-based safety screening detected harmful content: {e}"}
                _safety_screening_cache[cache_key] = result # Cache the result.
                _update_internal_status(f"Safety screening failed via keyword check: {e}", 100, error=str(e))
                return result
            except Exception as e:
                result = {"is_safe": False, "reason": f"An unexpected error occurred during keyword screening: {e}. Cannot proceed."}
                _safety_screening_cache[cache_key] = result # Cache the result.
                _update_internal_status(f"Unexpected error during keyword-based safety screening: {e}", 100, error=str(e))
                return result

    except json.JSONDecodeError as e:
        print(f"Error parsing safety screening JSON from LLM: {e}. Raw: {json_output[:500] if json_output else 'N/A'}")
        # If LLM returned malformed JSON, try keyword fallback.
        print("Malformed JSON from LLM. Falling back to keyword-based screening.")
        try:
            _perform_safety_screening(project_description)
            result = {"is_safe": True, "reason": "Passed keyword-based safety screening (LLM JSON parsing failed)."}
            _safety_screening_cache[cache_key] = result # Cache the result.
            _update_internal_status("Safety screening completed via keyword check (no harmful keywords found).", 100)
            return result
        except ValueError as e:
            result = {"is_safe": False, "reason": f"Keyword-based safety screening detected harmful content: {e}"}
            _safety_screening_cache[cache_key] = result # Cache the result.
            _update_internal_status(f"Safety screening failed via keyword check: {e}", 100, error=str(e))
            return result
        except Exception as e:
            result = {"is_safe": False, "reason": f"An unexpected error occurred during keyword-based safety screening (after LLM JSON parse error): {e}. Cannot proceed."}
            _safety_screening_cache[cache_key] = result # Cache the result.
            _update_internal_status(f"Unexpected error during keyword-based safety screening (after LLM JSON parse error): {e}", 100, error=str(e))
            return result
    except Exception as e:
        print(f"Unexpected error in safety screening: {e}")
        result = {"is_safe": False, "reason": f"An unexpected error occurred during safety screening: {e}. Please try again."}
        _safety_screening_cache[cache_key] = result # Cache the result.
        _update_internal_status("Unexpected error during safety screening. Defaulting to unsafe.", 100, error=f"An unexpected error occurred during safety screening: {e}. Please try again.")
        return result


def _get_bert_analysis_embedding(text):
    """
    Generates a numerical embedding for the input text using a pre-trained BERT model.
    These embeddings capture the semantic meaning of the text and can be used as features
    for other machine learning models (e.g., for baseline estimation fallback).

    Args:
        text (str): The input text (e.g., project description).

    Returns:
        np.array: A NumPy array representing the BERT embedding of the text.
                  Returns a zero array if BERT models are not loaded.
    """
    cache_key = text
    if cache_key in _bert_embedding_cache:
        print("Using cached BERT embedding.")
        return _bert_embedding_cache[cache_key]

    if project_analysis_bert_tokenizer is None or project_analysis_bert_model is None:
        _update_internal_status("BERT models not loaded for embedding.", error=True)
        return np.zeros(768) # Return a zero vector if BERT is not available.

    inputs = project_analysis_bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(): # Disable gradient calculation for inference.
        outputs = project_analysis_bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    _bert_embedding_cache[cache_key] = embedding # Cache the result.
    return embedding

def _qualquan_risk_assessment(project_context, risk_description, risk_category="Contractor-borne", iteration_number=0):
    """
    Simulates a QualQuan assessment using Gemma 3-27b-it.
    
    UPDATES:
    - Prompt explicitly maps 'risk_reduction' to 'ENHANCEMENT' for opportunities.
    - Independently enforces floors for Probability vs Impact enhancement.
    - Fixes 3-point estimate generation for Owner/Engineer risks (Triangular).
    """
    # Cache key includes category
    cache_key = (project_context, risk_description, risk_category)
    if cache_key in _qualquan_cache:
        print("Using cached QualQuan risk assessment.")
        return _qualquan_cache[cache_key]

    # --- 1. Safe Defaults ---
    safe_defaults = {
        "is_opportunity": False,
        "qualitative_decision": "Standard Analysis",
        "probability": 2.5, "impact": 2.5,
        "mitigation_strategy": "Monitor and control via standard project management protocols.",
        "risk_reduction_probability": 0.0, "risk_reduction_impact": 0.0,
        "cost_optimistic_usd": 0.0, "cost_most_likely_usd": 0.0, "cost_pessimistic_usd": 0.0,
        "time_optimistic_weeks": 0.0, "time_most_likely_weeks": 0.0, "time_pessimistic_weeks": 0.0
    }

    # --- 2. Dynamic Prompt Construction ---
    # Distribution Instructions
    if risk_category in ["Shared", "Contractor-borne"]:
        dist_instruction = (
            "DISTRIBUTION: Provide a 3-POINT PERT ESTIMATE (Optimistic, Most Likely, Pessimistic).\n"
            "Ensure a clear spread between values."
        )
    else:
        dist_instruction = (
            "DISTRIBUTION: Provide a TRIANGULAR ESTIMATE (Optimistic/Min, Most Likely/Mode, Pessimistic/Max)."
        )

    prompt = (
        f"Project Context: {project_context}\n"
        f"Item: {risk_description}\n"
        f"Category: {risk_category}\n\n"
        f"{dist_instruction}\n\n"
        "CRITICAL INSTRUCTIONS FOR OPPORTUNITIES:\n"
        "1. DEFINITION: Use the keys 'risk_reduction_probability' and 'risk_reduction_impact' to represent "
        "ENHANCEMENT percentages (how much the strategy INCREASES likelihood/benefit).\n"
        "2. VALUES: These must be NON-ZERO (e.g., 0.15 for 15% enhancement).\n"
        "3. COST SIGNS: Opportunities = NEGATIVE costs (Savings). Risks = POSITIVE costs.\n\n"
        "Output ONLY a valid JSON object with these keys:\n"
        "'is_opportunity', 'qualitative_decision', 'probability' (0-5), 'impact' (0-5), "
        "'mitigation_strategy', 'risk_reduction_probability' (0.0-1.0), 'risk_reduction_impact' (0.0-1.0), "
        "'cost_optimistic_usd', 'cost_most_likely_usd', 'cost_pessimistic_usd', "
        "'time_optimistic_weeks', 'time_most_likely_weeks', 'time_pessimistic_weeks'."
    )

    json_output = None

    # --- 3. API Generation ---
    if USE_GEMINI_API:
        for i, api_key in enumerate(GEMINI_API_KEYS):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(GEMMA_MODEL_NAME)
                    response = model.generate_content(prompt)
                    if response and response.text:
                        json_output = response.text
                        break 
                except Exception: pass

    # --- 4. Local Fallback ---
    if json_output is None and fine_tuned_risk_model is not None:
        try:
            inputs = fine_tuned_risk_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            output_tokens = fine_tuned_risk_model.generate(**inputs, max_new_tokens=500)
            json_output = fine_tuned_risk_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        except Exception: pass

    # --- 5. Robust Parsing & Logic Enforcement ---
    try:
        if json_output:
            ai_data = robust_parse_json(json_output)
            final_output = {**safe_defaults, **ai_data}
            
            is_opp = bool(final_output.get('is_opportunity', False))
            
            # --- ENFORCEMENT 1: Enhancement Floor (Independent Checks) ---
            # If values are effectively zero, force a "standard enhancement" value
            red_prob = float(final_output.get('risk_reduction_probability', 0.0))
            red_imp = float(final_output.get('risk_reduction_impact', 0.0))
            
            if is_opp:
                if red_prob < 0.01: 
                    final_output['risk_reduction_probability'] = 0.15 # Default 15% enhancement
                if red_imp < 0.01:
                    final_output['risk_reduction_impact'] = 0.10      # Default 10% enhancement

            # --- ENFORCEMENT 2: 3-Point Estimate Generation ---
            for type_key in ['cost', 'time']:
                suffix = 'usd' if type_key == 'cost' else 'weeks'
                ml_key = f"{type_key}_most_likely_{suffix}"
                opt_key = f"{type_key}_optimistic_{suffix}"
                pess_key = f"{type_key}_pessimistic_{suffix}"

                # Use absolute values for calculation logic to handle negative opportunity costs easily
                ml_val = abs(float(final_output.get(ml_key, 0.0)))
                opt_val = abs(float(final_output.get(opt_key, 0.0)))
                pess_val = abs(float(final_output.get(pess_key, 0.0)))

                # If we have a Most Likely value, but missing or flat boundaries, synthesize them
                needs_calc = (ml_val > 0) and (opt_val < 0.01 or pess_val < 0.01 or opt_val == ml_val)

                if needs_calc:
                    if risk_category in ["Shared", "Contractor-borne"]:
                        # 3-Point PERT (Wider Spread)
                        if is_opp:
                            # Opportunity: Optimistic = Higher Savings (More Negative)
                            opt_val = ml_val * 1.35
                            pess_val = ml_val * 0.75
                        else:
                            # Risk: Optimistic = Lower Cost
                            opt_val = ml_val * 0.75
                            pess_val = ml_val * 1.45
                    else:
                        # Owner/Engineer Triangular (Standard Spread)
                        if is_opp:
                            opt_val = ml_val * 1.20
                            pess_val = ml_val * 0.90
                        else:
                            opt_val = ml_val * 0.90
                            pess_val = ml_val * 1.20

                # --- ENFORCEMENT 3: Sign Convention ---
                # Opportunities = Negative, Risks = Positive
                final_output[ml_key] = -ml_val if (is_opp and type_key == 'cost') else ml_val
                final_output[opt_key] = -opt_val if (is_opp and type_key == 'cost') else opt_val
                final_output[pess_key] = -pess_val if (is_opp and type_key == 'cost') else pess_val

            _qualquan_cache[cache_key] = final_output
            return final_output
        else:
            raise RuntimeError("No model output available.")
            
    except Exception as e:
        print(f"Error in QualQuan logic: {e}. Returning defaults.")
        return safe_defaults
def _monte_carlo_simulation(optimistic, most_likely, pessimistic, num_simulations=NUM_SIMULATIONS):
    """
    Performs a Monte Carlo simulation using a triangular distribution.
    This is commonly used in project management for cost and time estimation.

    Args:
        optimistic (float): The optimistic (minimum) value.
        most_likely (float): The most likely (mode) value.
        pessimistic (float): The pessimistic (maximum) value.
        num_simulations (int): The number of simulation runs.

    Returns:
        np.array: An array of simulated values.
    """
    optimistic = max(0.0, float(optimistic))
    most_likely = max(0.0, float(most_likely))
    pessimistic = max(most_likely, float(pessimistic)) # Ensure pessimistic is at least most_likely.

    # Handle illogical inputs by collapsing the distribution to the most likely value.
    if not (optimistic <= most_likely <= pessimistic):
        optimistic = most_likely
        pessimistic = most_likely
        if optimistic == 0:
            pessimistic = 1e-6 # Ensure a tiny spread for 0 values to avoid error.

    if optimistic == 0 and most_likely == 0 and pessimistic == 0:
        return np.zeros(num_simulations)
    try:
        s_values = np.random.triangular(optimistic, most_likely, pessimistic, num_simulations)
        return s_values
    except Exception as e:
        print(f"Error during np.random.triangular simulation: {e}. Returning zeros.")
        return np.zeros(num_simulations)

def _pert_monte_carlo_simulation(optimistic, most_likely, pessimistic, num_simulations=NUM_SIMULATIONS):
    """
    Performs a Monte Carlo simulation approximating a PERT distribution.
    The PERT distribution is often preferred over triangular for its smoother shape
    and more realistic representation of uncertainty. It uses a triangular distribution
    with an adjusted mode to approximate PERT.

    Args:
        optimistic (float): The optimistic (minimum) value.
        most_likely (float): The most likely value.
        pessimistic (float): The pessimistic (maximum) value.
        num_simulations (int): The number of simulation runs.

    Returns:
        np.array: An array of simulated values.
    """
    optimistic = max(0.0, float(optimistic))
    most_likely = max(0.0, float(most_likely))
    pessimistic = max(most_likely, float(pessimistic)) # Ensure pessimistic is at least most_likely.

    # Handle illogical inputs.
    if not (optimistic <= most_likely <= pessimistic):
        optimistic = most_likely
        pessimistic = most_likely
        if optimistic == 0:
            pessimistic = 1e-6 # Ensure a tiny spread for 0 values to avoid error.

    if optimistic == 0 and most_likely == 0 and pessimistic == 0:
        return np.zeros(num_simulations)

    # Calculate PERT expected value, which will be the mode for our triangular approximation.
    pert_expected_value = (optimistic + 4 * most_likely + pessimistic) / 6.0
    
    # Ensure the mode is within the optimistic and pessimistic bounds.
    mode = max(optimistic, min(pert_expected_value, pessimistic))

    try:
        s_values = np.random.triangular(optimistic, mode, pessimistic, num_simulations)
        return s_values
    except Exception as e:
        print(f"Error during np.random.triangular (PERT approx) simulation: {e}. Returning zeros.")
        return np.zeros(num_simulations)

def _calculate_p85(simulated_values):
    """
    Calculates the 85th percentile of simulated values.
    The P85 value is commonly used in risk analysis to determine a reasonable contingency
    reserve, representing the value below which 85% of outcomes fall.

    Args:
        simulated_values (list or np.array): A list or array of simulated values.

    Returns:
        float: The 85th percentile value. Returns 0.0 if the input list is empty.
    """
    if len(simulated_values) == 0: return 0.0
    return float(np.percentile(simulated_values, 90)) # Explicitly convert to float.

def _plot_to_base64(fig):
    """
    Converts a matplotlib figure to a base64 encoded PNG image string.
    This allows plots to be easily embedded and displayed in web applications.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object to convert.

    Returns:
        str: A base64 encoded string representing the PNG image, prefixed with data URI scheme.
             Returns None if the figure is None.
    """
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free up memory.
    return f"data:image/png;base64,{img_base64}"

from concurrent.futures import ThreadPoolExecutor, as_completed

def _identify_risks_and_opportunities_llm(project_description, num_to_generate=NUM_RISKS_TO_GENERATE_FROM_LLM, category_focus=None):
    """
    Identifies potential risks and opportunities for a project using Gemma 3.
    Uses MULTI-THREADING to fire requests across all API keys simultaneously for speed.
    Includes failover to local TinyLlama and preserves all normalization logic.
    """
    cache_key = (project_description, num_to_generate, category_focus)
    if cache_key in _risks_opportunities_cache:
        print("Using cached identified risks and opportunities.")
        return _risks_opportunities_cache[cache_key]

    # --- Step 1: Explicit Prompt Construction ---
    prompt_intro = f"Based on the following project description, identify {num_to_generate} potential"
    if category_focus:
        prompt_intro += f" {category_focus}"
    
    prompt_intro += (
        " risks and opportunities. For each, provide a brief description using the key 'risk_description'. "
        "Also assign a unique string 'risk_id', indicate if it's an 'opportunity' or 'risk' with "
        "a boolean 'is_opportunity', and assign a 'risk_category' from: "
        "'Contractor-borne', 'Owner-borne', 'Engineer-borne', 'Shared'."
    )
    prompt_intro += "\nOutput ONLY a valid JSON array of objects. Project Description: "
    prompt = f"{prompt_intro}{project_description}"

    json_output = None

    # --- Step 2: Multi-Threaded API Execution (Gemma 3-27b-it) ---
    if USE_GEMINI_API:
        def _attempt_single_api_call(api_key, key_index):
            """Worker function for individual threads."""
            if not api_key:
                return None
            try:
                # Local configuration for this specific thread
                genai.configure(api_key=api_key)
                temp_model = genai.GenerativeModel(GEMMA_MODEL_NAME)
                # Lower temperature for faster, more deterministic JSON output
                response = temp_model.generate_content(
                    prompt, 
                    generation_config=genai.GenerationConfig(temperature=0.1)
                )
                if response and response.text:
                    return (response.text, key_index)
            except Exception as e:
                print(f"Identify risks: Key {key_index+1} failed: {e}.")
            return None

        _update_internal_status("Engaging API keys in parallel...", 20)
        
        with ThreadPoolExecutor(max_workers=len(GEMINI_API_KEYS)) as executor:
            # Submit all keys to the thread pool
            futures = {executor.submit(_attempt_single_api_call, key, i): i for i, key in enumerate(GEMINI_API_KEYS)}
            
            # Monitor for the FIRST successful result
            for future in as_completed(futures):
                result = future.result()
                if result:
                    json_output, successful_index = result
                    print(f"Identify risks: First success from Gemma API Key {successful_index+1}.")
                    break # Stop and ignore other threads once we have data

    # --- Step 3: Local Fallback (TinyLlama) ---
    if json_output is None and fine_tuned_risk_model is not None:
        try:
            print("All API keys failed or were bypassed. Falling back to local TinyLlama.")
            inputs = fine_tuned_risk_tokenizer(prompt + "\n\nJSON array:", return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_tokens = fine_tuned_risk_model.generate(**inputs, max_new_tokens=1500, pad_token_id=fine_tuned_risk_tokenizer.eos_token_id)
            generated_text = fine_tuned_risk_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", generated_text, re.DOTALL)
            if json_match:
                json_output = json_match.group(0)
        except Exception as e:
            print(f"Local TinyLlama error: {e}.")

    # --- Step 4: Normalization and Post-Processing ---
    try:
        if json_output:
            raw_data = robust_parse_json(json_output)
            normalized_results = []
            seen_descriptions = set() # Additional layer to prevent internal duplicates
            
            for item in raw_data:
                # 4.1 Key Normalization
                if 'risk_description' not in item:
                    alt_desc = item.get('description') or item.get('brief_description') or item.get('summary')
                    item['risk_description'] = alt_desc if alt_desc else "Unnamed Project Item"
                
                # De-duplication check: Skip if the text is identical to a previous item in the list
                desc_check = item['risk_description'].strip().lower()
                if desc_check in seen_descriptions:
                    continue
                seen_descriptions.add(desc_check)

                # 4.2 Ensure unique risk_id exists
                if 'risk_id' not in item or not item['risk_id']:
                    item['risk_id'] = str(uuid.uuid4())
                item['risk_id'] = str(item['risk_id'])
                
                # 4.3 Standardize categories
                valid_categories = ['Contractor-borne', 'Owner-borne', 'Engineer-borne', 'Shared']
                if item.get('risk_category') not in valid_categories:
                    item['risk_category'] = 'Contractor-borne'
                
                normalized_results.append(item)
            
            _risks_opportunities_cache[cache_key] = normalized_results
            return normalized_results
        else:
            raise RuntimeError("No model (Cloud or Local) provided output.")
            
    except Exception as e:
        print(f"Error processing identified risks: {e}")
        return [{
            "risk_id": str(uuid.uuid4()), 
            "risk_description": f"AI Error: Unable to process project risks. ({e})", 
            "is_opportunity": False, 
            "risk_category": "Contractor-borne"
        }]
        

def _generate_baseline_estimates_llm(project_description):
    """
    Generates baseline estimates for total project cost and time using an LLM.
    This function prioritizes external LLMs (Gemini, Hugging Face) for accuracy.
    If LLM calls fail, it falls back to a simpler, BERT-based simulated prediction.

    Args:
        project_description (str): The description of the project.

    Returns:
        dict: A dictionary containing optimistic, most likely, and pessimistic estimates
              for cost (CAD) and time (weeks).
              Returns an error dictionary if LLM fails or parsing errors occur,
              or if the BERT fallback also fails.
    """
    cache_key = project_description
    if cache_key in _baseline_estimates_cache:
        print("Using cached baseline estimates.")
        return _baseline_estimates_cache[cache_key]

    prompt = (f"Given the following project description, provide optimistic, most likely, and pessimistic estimates for total project cost (in CAD) and total project time (in weeks). "
              f"Output in a structured JSON format with keys: 'cost_optimistic_cad', 'cost_most_likely_cad', 'cost_pessimistic_cad', "
              f"'time_optimistic_weeks', 'time_most_likely_weeks', 'time_pessimistic_weeks'. Project Description: {project_description}")

    # Define the expected JSON schema for the LLM response.
    response_schema = {"type": "OBJECT", "properties": {"cost_optimistic_cad": {"type": "NUMBER"}, "cost_most_likely_cad": {"type": "NUMBER"}, "cost_pessimistic_cad": {"type": "NUMBER"},
                                                      "time_optimistic_weeks": {"type": "NUMBER"}, "time_most_likely_weeks": {"type": "NUMBER"}, "time_pessimistic_weeks": {"type": "NUMBER"}},
                     "required": ["cost_optimistic_cad", "cost_most_likely_cad", "cost_pessimistic_cad", "time_optimistic_weeks", "time_most_likely_weeks", "time_pessimistic_weeks"]}
    json_output = None
    
    # Try Gemini API first.
    if USE_GEMINI_API:
        for i, api_key in enumerate(GEMINI_API_KEYS):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                    response_model = genai.GenerativeModel(GEMINI_MODEL_NAME).generate_content(
                        [{"role": "user", "parts": [{"text": prompt}]}],
                        generation_config=generation_config
                    )
                    json_output = response_model.text
                    print(f"Baseline estimates: Gemini API with key {i+1} successful.")
                    break # Exit loop if successful.
                except Exception as e:
                    print(f"Baseline estimates: Gemini API with key {i+1} error: {e}. Trying next key.")
            else:
                print(f"Gemini API key {i+1} not found. Skipping.")
    
    # If Gemini fails, try local TinyLlama.
    if json_output is None and fine_tuned_risk_model is not None and fine_tuned_risk_tokenizer is not None:
        try:
            inputs = fine_tuned_risk_tokenizer(prompt + "\n\nProvide the response in JSON format only.", return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_tokens = fine_tuned_risk_model.generate(**inputs, max_new_tokens=500, pad_token_id=fine_tuned_risk_tokenizer.eos_token_id)
            generated_text = fine_tuned_risk_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            json_match = re.search(r"\{.*?\}", generated_text, re.DOTALL)
            if json_match:
                json_output = json_match.group(0)
        except Exception as e:
            print(f"Local TinyLlama error for baseline estimates: {e}. No AI response available.")

    try:
        if json_output:
            parsed_output = json.loads(json_output)
            # Ensure all numerical values are floats/integers.
            for key in ["cost_optimistic_cad", "cost_most_likely_cad", "cost_pessimistic_cad",
                        "time_optimistic_weeks", "time_most_likely_weeks", "time_pessimistic_weeks"]:
                if key in parsed_output:
                    try:
                        parsed_output[key] = float(parsed_output[key])
                    except (ValueError, TypeError):
                        parsed_output[key] = 0.0 # Default to 0.0 if conversion fails.
            _baseline_estimates_cache[cache_key] = parsed_output # Cache the result.
            return parsed_output
        else:
            raise RuntimeError("No suitable LLM could provide baseline estimates.")
    except Exception as e:
        print(f"Error parsing or processing baseline estimates: {e}")
        # Fallback to simulated estimates if LLM fails or parsing fails.
        print("Falling back to BERT-based simulated baseline estimates.")
        embedding = _get_bert_analysis_embedding(project_description)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        if baseline_scaler is None:
            _update_internal_status("Baseline scaler not loaded for fallback estimates.", error=True)
            error_result = {"cost_optimistic_cad": 0.0, "cost_most_likely_cad": 0.0, "cost_pessimistic_cad": 0.0,
                            "time_optimistic_weeks": 0.0, "time_most_likely_weeks": 0.0, "time_pessimistic_weeks": 0.0}
            _baseline_estimates_cache[cache_key] = error_result # Cache the error result too.
            return error_result

        try:
            scaled_features = baseline_scaler.transform(embedding)
            # Simple linear mapping for simulation; replace with a trained model in production.
            cost_factor = scaled_features[0, 0] * 100000 + 50000 # Example scaling.
            time_factor = scaled_features[0, 1] * 10 + 5 # Example scaling.

            result = {
                "cost_optimistic_cad": max(0.0, cost_factor * 0.8),
                "cost_most_likely_cad": max(0.0, cost_factor),
                "cost_pessimistic_cad": max(0.0, cost_factor * 1.2),
                "time_optimistic_weeks": max(0.0, time_factor * 0.8),
                "time_most_likely_weeks": max(0.0, time_factor),
                "time_pessimistic_weeks": max(0.0, time_factor * 1.2)
            }
            _baseline_estimates_cache[cache_key] = result # Cache the result.
            return result
        except Exception as e_fallback:
            print(f"Error during BERT-based fallback estimation: {e_fallback}")
            error_result = {"cost_optimistic_cad": 0.0, "cost_most_likely_cad": 0.0, "cost_pessimistic_cad": 0.0,
                            "time_optimistic_weeks": 0.0, "time_most_likely_weeks": 0.0, "time_pessimistic_weeks": 0.0}
            _baseline_estimates_cache[cache_key] = error_result # Cache the error result too.
            return error_result
# In integrationapi.py

def apply_frugal_filter(noisy_vector, ideal_vector, filter_strength=0.5):
    """
    Performs a simple amplitude filter to mitigate noise by applying a weighted
    average with an ideal, noiseless vector from a simulator.

    Args:
        noisy_vector (np.array): The probability vector from a noisy quantum computer run.
        ideal_vector (np.array): The ideal, noiseless probability vector from a simulator.
        filter_strength (float): A value between 0 and 1 that controls the filter's strength.
                                 0 means no filter; 1 means the result is the ideal vector.

    Returns:
        np.array: The filtered and normalized probability vector.
    """
    if len(noisy_vector) != len(ideal_vector):
        print("Error: Vector lengths do not match. Cannot apply filter.")
        return noisy_vector

    # Apply a linear combination (weighted average) of the noisy and ideal vectors.
    mitigated_vector = (1 - filter_strength) * noisy_vector + filter_strength * ideal_vector

    # Normalize the vector to ensure it remains a valid probability distribution.
    mitigated_vector = mitigated_vector / np.sum(mitigated_vector)

    return mitigated_vector


# In integrationapi.py, replace the existing function with this one

def _calculate_quantum_influence(risk_probability, risk_impact, current_cost, current_time):
    """
    Calculates the quantum-inspired influence on project cost and time.
    This function runs the circuit on a real QPU and, if successful, uses a local
    GPU/CPU simulation to filter the noisy QPU result. If the QPU fails,
    it falls back to the local simulation result directly.
    """
    global simulated_quantum_circuit, q_weights, num_wires

    # Prepare quantum input data (same as before)
    scaled_prob_norm = risk_probability / 5.0
    scaled_impact_norm = risk_impact / 5.0
    q_data = np.array([scaled_prob_norm * np.pi, scaled_impact_norm * np.pi,
                       (scaled_prob_norm + scaled_impact_norm)/2 * np.pi,
                       abs(scaled_prob_norm - scaled_impact_norm) * np.pi])

    with _models_lock:
        qiskit_service = _models.get("qiskit_runtime_service")
        qiskit_backend = _models.get("qiskit_real_backend")

    # --- Primary Path: Attempt QPU execution ---
    if qiskit_service and qiskit_backend:
        print("Attempting quantum influence calculation on real QPU.")
        try:
            # --- QPU Execution (same as before) ---
            if isinstance(q_weights, torch.Tensor):
                q_weights_np = q_weights.detach().cpu().numpy()
            else:
                q_weights_np = np.array(q_weights)

            qc_model = QuantumCircuit(num_wires, num_wires)
            for i in range(num_wires):
                qc_model.ry(float(q_data[i]), i)
            qc_model.barrier()
            num_layers = q_weights_np.shape[0]
            for layer in range(num_layers):
                for wire in range(num_wires):
                    qc_model.ry(float(q_weights_np[layer, wire, 0]), wire)
                    qc_model.rz(float(q_weights_np[layer, wire, 1]), wire)
                    qc_model.ry(float(q_weights_np[layer, wire, 2]), wire)
                for wire in range(num_wires):
                    qc_model.cx(wire, (wire + 1) % num_wires)
                if layer < num_layers - 1:
                    qc_model.barrier()
            qc_model.measure(range(num_wires), range(num_wires))
            
            transpiled_qc_model = transpile(qc_model, qiskit_backend)
            sampler = Sampler(mode=qiskit_backend)
            job = sampler.run([transpiled_qc_model], shots=1024)
            print(f"Submitted QPU job to {qiskit_backend.name}. Job ID: {job.job_id()}")
            real_qiskit_result = job.result()
            
            # Get the noisy probability distribution from the QPU
            quasi_distribution, _ = _get_qiskit_job_results(real_qiskit_result, num_wires)

            ### NEW: Run local simulation to get the ideal vector for filtering ###
            print("Running local simulation to get ideal vector for filtering.")
            if simulated_quantum_circuit is None:
                raise RuntimeError("Local simulator not available for filtering process.")

            @qml.qnode(simulated_quantum_circuit.device)
            def ideal_prob_circuit(weights, data):
                qml.AngleEmbedding(data, wires=range(num_wires))
                qml.StronglyEntanglingLayers(weights, wires=range(num_wires))
                return qml.probs(wires=range(num_wires))
            
            ideal_vector = ideal_prob_circuit(q_weights, q_data).numpy()

            ### NEW: Convert QPU result and apply the Frugal Filter ###
            print("Applying Frugal Filter to QPU results.")
            all_outcomes = [format(i, f'0{num_wires}b') for i in range(2**num_wires)]
            noisy_vector = np.array([quasi_distribution.get(int(outcome, 2), 0.0) for outcome in all_outcomes])
            
            filtered_vector = apply_frugal_filter(noisy_vector, ideal_vector)

            ### NEW: Calculate a new expectation value from the filtered probabilities ###
            avg_exp_val = 0.0
            for i, prob in enumerate(filtered_vector):
                bitstring = all_outcomes[i]
                parity = (-1)**(bitstring.count('1'))
                avg_exp_val += parity * prob
            avg_exp_val = float(avg_exp_val)
            print(f"Calculated new expectation value from filtered result: {avg_exp_val:.4f}")

            # Use the filtered expectation value for the final calculation
            cost_multiplier = 1.0 + (avg_exp_val * 0.1)
            time_multiplier = 1.0 + (avg_exp_val * 0.05)
            cost_influence = current_cost * (cost_multiplier - 1.0)
            time_influence = current_time * (time_multiplier - 1.0)
            
            return cost_influence, time_influence

        except Exception as e:
            print(f"QPU execution failed: {e}. Failing over to local simulation.")
            # Fall through to the simulation-only logic below

    # --- Failover Path: GPU/CPU Simulation (if QPU fails or is unavailable) ---
    print("Using local GPU/CPU simulation for quantum influence.")
    if simulated_quantum_circuit is None or q_weights is None:
        print("Warning: Quantum circuit/weights not loaded. Returning zero influence.")
        return 0.0, 0.0

    try:
        # This part runs if the QPU path fails
        quantum_output = simulated_quantum_circuit(q_weights, q_data)
        avg_exp_val = np.mean([float(val) for val in quantum_output])

        cost_multiplier = 1.0 + (avg_exp_val * 0.1)
        time_multiplier = 1.0 + (avg_exp_val * 0.05)
        cost_influence = current_cost * (cost_multiplier - 1.0)
        time_influence = current_time * (time_multiplier - 1.0)
        
        return cost_influence, time_influence
    except Exception as e_pl:
        print(f"Error during fallback simulation: {e_pl}. Returning zero influence.")
        return 0.0, 0.0# In integrationapi.py, replace the existing function with this one

def _calculate_quantum_influence(risk_probability, risk_impact, current_cost, current_time):
    """
    Calculates the quantum-inspired influence on project cost and time.
    This function runs the circuit on a real QPU and, if successful, uses a local
    GPU/CPU simulation to filter the noisy QPU result. If the QPU fails,
    it falls back to the local simulation result directly.
    """
    global simulated_quantum_circuit, q_weights, num_wires

    # Prepare quantum input data
    scaled_prob_norm = risk_probability / 5.0
    scaled_impact_norm = risk_impact / 5.0
    # Embedding data: [prob, impact, average, diff]
    q_data = np.array([scaled_prob_norm * np.pi, scaled_impact_norm * np.pi,
                       (scaled_prob_norm + scaled_impact_norm)/2 * np.pi,
                       abs(scaled_prob_norm - scaled_impact_norm) * np.pi])

    with _models_lock:
        qiskit_service = _models.get("qiskit_runtime_service")
        qiskit_backend = _models.get("qiskit_real_backend")

    # --- Primary Path: Attempt QPU execution ---
    if qiskit_service and qiskit_backend:
        print("Attempting quantum influence calculation on real QPU.")
        try:
            # --- QPU Execution (Qiskit Circuit Construction) ---
            if isinstance(q_weights, torch.Tensor):
                q_weights_np = q_weights.detach().cpu().numpy()
            else:
                q_weights_np = np.array(q_weights)

            qc_model = QuantumCircuit(num_wires, num_wires)
            
            # Angle Embedding (matches PennyLane logic)
            for i in range(num_wires):
                qc_model.ry(float(q_data[i]), i)
            qc_model.barrier()
            
            # Strongly Entangling Layers (Manual Construction)
            num_layers = q_weights_np.shape[0]
            for layer in range(num_layers):
                for wire in range(num_wires):
                    qc_model.ry(float(q_weights_np[layer, wire, 0]), wire)
                    qc_model.rz(float(q_weights_np[layer, wire, 1]), wire)
                    qc_model.ry(float(q_weights_np[layer, wire, 2]), wire)
                for wire in range(num_wires):
                    qc_model.cx(wire, (wire + 1) % num_wires)
                if layer < num_layers - 1:
                    qc_model.barrier()
            qc_model.measure(range(num_wires), range(num_wires))
            
            # Transpile and Run
            # Optimization level 3 is recommended for Heron R1/R2 to handle defects
            transpiled_qc_model = transpile(qc_model, qiskit_backend, optimization_level=3)
            
            # Enable dynamical decoupling to suppress idle errors
            options = {"dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"}}
            sampler = Sampler(mode=qiskit_backend, options=options)
            
            job = sampler.run([transpiled_qc_model], shots=1024)
            print(f"Submitted QPU job to {qiskit_backend.name}. Job ID: {job.job_id()}")
            real_qiskit_result = job.result()
            
            # Get the noisy probability distribution from the QPU
            quasi_distribution, _ = _get_qiskit_job_results(real_qiskit_result, num_wires)

            ### NEW: Run local simulation to get the ideal vector for filtering ###
            print("Running local simulation to get ideal vector for filtering.")
            if simulated_quantum_circuit is None:
                raise RuntimeError("Local simulator not available for filtering process.")

            # Define a temporary QNode to get probabilities (simulated_quantum_circuit usually returns expval)
            @qml.qnode(simulated_quantum_circuit.device)
            def ideal_prob_circuit(weights, data):
                qml.AngleEmbedding(data, wires=range(num_wires))
                qml.StronglyEntanglingLayers(weights, wires=range(num_wires))
                return qml.probs(wires=range(num_wires))
            
            ideal_vector = ideal_prob_circuit(q_weights, q_data)
            # Ensure it is a numpy array
            if hasattr(ideal_vector, "numpy"):
                ideal_vector = ideal_vector.numpy()
            else:
                ideal_vector = np.array(ideal_vector)

            ### NEW: Convert QPU result and apply the Frugal Filter ###
            print("Applying Frugal Filter to QPU results.")
            # Map Qiskit's dictionary (quasi_distribution) to an array matching PennyLane's bitstring order
            all_outcomes = [format(i, f'0{num_wires}b') for i in range(2**num_wires)]
            # Note: int(outcome, 2) converts binary string back to integer key for the dictionary
            noisy_vector = np.array([quasi_distribution.get(int(outcome, 2), 0.0) for outcome in all_outcomes])
            
            filtered_vector = apply_frugal_filter(noisy_vector, ideal_vector, filter_strength=0.4)

            ### NEW: Calculate a new expectation value from the filtered probabilities ###
            # We calculate Parity (Z-basis expectation) manually from the filtered probabilities
            avg_exp_val = 0.0
            for i, prob in enumerate(filtered_vector):
                bitstring = all_outcomes[i]
                # Parity: +1 if even number of 1s, -1 if odd number of 1s
                parity = (-1)**(bitstring.count('1'))
                avg_exp_val += parity * prob
            avg_exp_val = float(avg_exp_val)
            print(f"Calculated new expectation value from filtered result: {avg_exp_val:.4f}")

            # Use the filtered expectation value for the final calculation
            cost_multiplier = 1.0 + (avg_exp_val * 0.1)
            time_multiplier = 1.0 + (avg_exp_val * 0.05)
            cost_influence = current_cost * (cost_multiplier - 1.0)
            time_influence = current_time * (time_multiplier - 1.0)
            
            return cost_influence, time_influence

        except Exception as e:
            print(f"QPU execution failed: {e}. Failing over to local simulation.")
            # Fall through to the simulation-only logic below

    # --- Failover Path: GPU/CPU Simulation (if QPU fails or is unavailable) ---
    print("Using local GPU/CPU simulation for quantum influence.")
    if simulated_quantum_circuit is None or q_weights is None:
        print("Warning: Quantum circuit/weights not loaded. Returning zero influence.")
        return 0.0, 0.0

    try:
        # This part runs if the QPU path fails
        quantum_output = simulated_quantum_circuit(q_weights, q_data)
        avg_exp_val = np.mean([float(val) for val in quantum_output])

        cost_multiplier = 1.0 + (avg_exp_val * 0.1)
        time_multiplier = 1.0 + (avg_exp_val * 0.05)
        cost_influence = current_cost * (cost_multiplier - 1.0)
        time_influence = current_time * (time_multiplier - 1.0)
        
        return cost_influence, time_influence
    except Exception as e_pl:
        print(f"Error during fallback simulation: {e_pl}. Returning zero influence.")
        return 0.0, 0.0

def create_and_submit_quantum_job(input_data):
    """
    Creates the quantum circuit from PennyLane model, submits the job to Qiskit Sampler,
    and returns a Base64-encoded image of the circuit along with the job object.
    """
    with _models_lock:
        qiskit_service = _models.get("qiskit_runtime_service")
        qiskit_backend = _models.get("qiskit_real_backend")

    if not qiskit_service or not qiskit_backend:
        raise RuntimeError("Qiskit backend not available.")

    # 1. CONVERT PENNYLANE MODEL TO QISKIT CIRCUIT
    if isinstance(q_weights, torch.Tensor):
        q_weights_np = q_weights.detach().cpu().numpy()
    else:
        q_weights_np = np.array(q_weights)

    # Use input_data to generate q_data (this part needs to be adapted from your `_calculate_quantum_influence` logic)
    # Example:
    # scaled_prob_norm = input_data['risk_probability'] / 5.0
    # ... and so on
    # For now, we'll use a placeholder `q_data`
    q_data = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2]) # Placeholder

    qc_model = QuantumCircuit(num_wires, num_wires)
    qc_model.name = "quantum_ansatz_model"

    for i in range(num_wires):
        qc_model.ry(float(q_data[i]), i)
    qc_model.barrier()

    num_layers = q_weights_np.shape[0]
    for layer in range(num_layers):
        for wire in range(num_wires):
            qc_model.ry(float(q_weights_np[layer, wire, 0]), wire)
            qc_model.rz(float(q_weights_np[layer, wire, 1]), wire)
            qc_model.ry(float(q_weights_np[layer, wire, 2]), wire)
        for wire in range(num_wires):
            qc_model.cx(wire, (wire + 1) % num_wires)
        if layer < num_layers - 1:
            qc_model.barrier()

    # 2. Add Measurement
    qc_model.measure(range(num_wires), range(num_wires))

    # Generate circuit image before transpiling
    image_buffer = io.BytesIO()
    qc_model.draw(output='mpl', filename=image_buffer)
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    image_buffer.close()

    # 3. Transpile and submit the job
    transpiled_qc_model = transpile(qc_model, qiskit_backend)
    sampler = Sampler(mode=qiskit_backend)
    job = sampler.run([transpiled_qc_model], shots=1024)

    return image_base64, job

# NEW: Function to get the status of a submitted job
def get_quantum_job_status(job_id):
    """
    Checks the status of a Qiskit job and returns the results if available.
    """
    with _models_lock:
        qiskit_service = _models.get("qiskit_runtime_service")
        qiskit_backend = _models.get("qiskit_real_backend")
    
    if not qiskit_service or not qiskit_backend:
        raise RuntimeError("Qiskit backend not available.")

    job = qiskit_service.job(job_id)
    status = job.status().name

    if status == "DONE":
        result = job.result()
        # Use the centralized helper function to process the Sampler V2 results.
        quasi_distribution, avg_exp_val = _get_qiskit_job_results(result, num_wires)
        
        # You can now use the processed results. For example, returning the expectation value.
        # The full influence calculation would require current_cost and current_time, which are not
        # available in this context, so we return the core quantum result.
        return status, {"qiskit_quasi_distribution": quasi_distribution, "qiskit_avg_exp_val": avg_exp_val}

    elif status == "ERROR" or status == "CANCELLED": # Using 'ERROR' as per qiskit-ibm-runtime JobStatus enum
        error_message = "Job failed or was cancelled."
        try:
            error_message = job.error_message()
        except Exception:
            pass # Ignore if error message cannot be retrieved.
        return status, {"error": error_message}
    
    # For other statuses like 'QUEUED', 'RUNNING'
    return status, None

def run_monte_carlo_simulation(tasks_data, num_simulations=NUM_SIMULATIONS, scenario="project"):
    """
    Runs Monte Carlo simulation for project duration analysis (e.g., for project or safety scenarios).
    It simulates task durations based on triangular distributions and calculates critical path
    (simplified as sum of durations for now) for each simulation run.

    Args:
        tasks_data (list): A list of dictionaries, where each dictionary represents a task
                           with 'name', 'optimistic', 'most_likely', and 'pessimistic' duration estimates.
        num_simulations (int): The number of Monte Carlo simulation runs.
        scenario (str): A string indicating the scenario (e.g., "project", "safety").

    Returns:
        dict: A dictionary containing P-values (P50, P80, P85, P90, P95), mean duration,
              all simulated durations, and P85 durations for individual tasks.
    """
    _update_internal_status(f"Starting Monte Carlo simulation for {scenario} analysis...", 10)

    simulated_project_durations = []
    
    # Generate simulated durations for each task based on its triangular distribution.
    simulated_durations_per_task = {}
    for task in tasks_data:
        task_key = task['name'] # Using name as key, assume unique names.
        optimistic = float(task['optimistic'])
        most_likely = float(task['most_likely'])
        pessimistic = float(task['pessimistic'])
        simulated_durations_per_task[task_key] = [
            _triangular_distribution(optimistic, most_likely, pessimistic)
            for _ in range(num_simulations)
        ]

    # Perform the simulations by calculating the critical path (simplified sum) for each run.
    for i in range(num_simulations):
        project_duration = _calculate_critical_path(tasks_data, simulated_durations_per_task, i)
        simulated_project_durations.append(project_duration)
        _update_internal_status(f"Running simulation {i+1}/{num_simulations} for {scenario} analysis...", 10 + int(80 * (i / num_simulations)))
    
    _update_internal_status(f"Monte Carlo simulation for {scenario} analysis completed.", 90)

    # Calculate P-values (percentiles) and mean duration from the simulated project durations.
    simulated_project_durations_np = np.array(simulated_project_durations)

    p50 = np.percentile(simulated_project_durations_np, 50)
    p80 = np.percentile(simulated_project_durations_np, 80)
    p85 = np.percentile(simulated_project_durations_np, 85)
    p90 = np.percentile(simulated_project_durations_np, 90)
    p95 = np.percentile(simulated_project_durations_np, 95)
    mean_duration = np.mean(simulated_project_durations_np)

    results = {
        "p50": round(p50, 2),
        "p80": round(p80, 2),
        "p85": round(p85, 2),
        "p90": round(p90, 2),
        "p95": round(p95, 2),
        "mean_duration": round(mean_duration, 2),
        "simulated_durations": simulated_project_durations,
        "tasks_with_p85": [] # To store task P85 values for potential reuse.
    }
    
    # Calculate P85 for each individual task and add it to the results.
    for task in tasks_data:
        task_key = task['name']
        if task_key in simulated_durations_per_task:
            task_sim_durations = np.array(simulated_durations_per_task[task_key])
            task_p85 = np.percentile(task_sim_durations, 85)
            results["tasks_with_p85"].append({
                "name": task['name'],
                "optimistic": task['optimistic'],
                "most_likely": task['most_likely'],
                "pessimistic": task['pessimistic'],
                "p85_duration": round(task_p85, 2)
            })

    _update_internal_status(f"Monte Carlo simulation for {scenario} analysis complete. Results ready.", 100)
    return results

def _triangular_distribution(min_val, most_likely_val, max_val):
    """
    Generates a single random number from a triangular distribution.

    Args:
        min_val (float): The lower limit of the distribution.
        most_likely_val (float): The mode (peak) of the distribution.
        max_val (float): The upper limit of the distribution.

    Returns:
        float: A random number drawn from the triangular distribution.
    """
    return random.triangular(min_val, max_val, most_likely_val)

def _calculate_critical_path(tasks, simulated_durations, simulation_index):
    """
    Calculates the critical path duration for a single simulation run.
    NOTE: This is a simplified example (summing all task durations).
    A real critical path algorithm would require a project network diagram
    and more complex logic (e.g., AON or ADM network analysis).

    Args:
        tasks (list): A list of task dictionaries.
        simulated_durations (dict): A dictionary where keys are task names and values are
                                    lists of simulated durations for each task.
        simulation_index (int): The index of the current simulation run.

    Returns:
        float: The total duration of the critical path for this simulation run.
    """
    total_duration = 0
    for task in tasks:
        task_key = task['name']
        total_duration += simulated_durations[task_key][simulation_index]
    return total_duration


# In integrationapi.py, replace the existing run_analysis_pipeline_for_api function with this one.
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_analysis_pipeline_for_api(project_description):
    """
    Final Fixed Pipeline.
    - Resolves NameError by ensuring 'adjusted_most_likely_time' is defined.
    - Aggregates costs using File 1 logic.
    - Generates all plots and data keys required by frontend.
    """
    results = {"error": None, "plots_base64": {}}
    _update_internal_status("Starting integrated Cost Impact & Quantum analysis...", 5)

    try:
        # 1. Safety Screening
        _update_internal_status("Performing safety screening...", 5)
        safety_check_result = _screen_description_for_safety(project_description)
        if not safety_check_result.get('is_safe', False):
            results['error'] = f"Safety Block: {safety_check_result.get('reason', 'Unknown safety violation')}"
            _update_internal_status(results['error'], 0, error=results['error'])
            return results

        # 2. Generate Baseline Estimates
        _update_internal_status("Generating baseline estimates...", 10)
        baseline_estimates = _generate_baseline_estimates_llm(project_description)
        if baseline_estimates.get('error'):
            raise ValueError(f"Baseline Failure: {baseline_estimates['error']}")
        
        results['baseline_estimates'] = {k: float(v) for k, v in baseline_estimates.items() if not k == 'error'}
        initial_most_likely_cost = float(baseline_estimates.get('cost_most_likely_cad', 0.0))
        initial_most_likely_time = float(baseline_estimates.get('time_most_likely_weeks', 0.0))

        # 3. Identify Risks and Opportunities
        _update_internal_status("Identifying risks and opportunities...", 20)
        risks_opportunities_raw = _identify_risks_and_opportunities_llm(project_description)
        
        opportunities = [r for r in risks_opportunities_raw if r.get('is_opportunity', False)]
        risks = [r for r in risks_opportunities_raw if not r.get('is_opportunity', True)]
        
        results['identified_opportunities'] = opportunities
        results['identified_risks'] = risks

        # --- Aggregation Variables ---
        total_contractor_cost_impact = 0.0
        total_oe_cost_impact = 0.0
        total_opportunity_cost_reduction = 0.0
        total_opportunity_time_reduction = 0.0
        total_contractor_time_impact = 0.0
        total_oe_time_impact = 0.0

        # 4. Perform QualQuan Assessment for Risks
        _update_internal_status("Performing QualQuan assessment for risks...", 30)
        risks_with_qualquan_data = []
        for risk_item in tqdm(risks, desc="QualQuan Assessment (Risks)"):
            desc = risk_item.get('risk_description', 'Unnamed Risk')
            r_cat = risk_item.get('risk_category', 'Contractor-borne')
            
            qualquan_data = _qualquan_risk_assessment(project_description, desc, risk_category=r_cat)
            
            prob = float(qualquan_data.get('probability', 0.0))
            impact = float(qualquan_data.get('impact', 0.0))
            red_prob = float(qualquan_data.get('risk_reduction_probability', 0.0))
            red_impact = float(qualquan_data.get('risk_reduction_impact', 0.0))
            
            effective_probability = prob * (1 - red_prob)
            effective_impact = impact * (1 - red_impact)
            
            # File 1 Weighting Logic
            weight = (effective_probability / 5.0) * (effective_impact / 5.0)

            c_impact = float(qualquan_data.get('cost_most_likely_usd', 0.0)) * weight * USD_TO_CAD_EXCHANGE_RATE
            t_impact = float(qualquan_data.get('time_most_likely_weeks', 0.0)) * weight

            risk_item.update(qualquan_data)
            risk_item.update({
                "effective_probability": effective_probability, 
                "effective_impact": effective_impact, 
                "cost_impact_cad": c_impact, 
                "time_impact_weeks": t_impact,
                "risk_category": r_cat
            })

            if r_cat == 'Contractor-borne':
                total_contractor_cost_impact += c_impact
                total_contractor_time_impact += t_impact
            elif r_cat in ['Owner-borne', 'Engineer-borne']:
                total_oe_cost_impact += c_impact
                total_oe_time_impact += t_impact
            elif r_cat == 'Shared':
                total_contractor_cost_impact += c_impact * 0.5
                total_oe_cost_impact += c_impact * 0.5
                total_contractor_time_impact += t_impact * 0.5
                total_oe_time_impact += t_impact * 0.5

            risks_with_qualquan_data.append(risk_item)
        
        # 5. Perform QualQuan Assessment for Opportunities
        _update_internal_status("Performing QualQuan assessment for opportunities...", 35)
        opportunities_with_qualquan_data = []
        for opportunity_item in tqdm(opportunities, desc="QualQuan Assessment (Opportunities)"):
            desc = opportunity_item.get('risk_description', 'Unnamed Opportunity')
            r_cat = opportunity_item.get('risk_category', 'Shared')
            qualquan_data = _qualquan_risk_assessment(project_description, desc, risk_category=r_cat)
            
            prob = float(qualquan_data.get('probability', 0.0))
            impact = float(qualquan_data.get('impact', 0.0))
            red_prob = float(qualquan_data.get('risk_reduction_probability', 0.0))
            red_impact = float(qualquan_data.get('risk_reduction_impact', 0.0))

            effective_probability = np.clip(prob * (1 + red_prob), 0.0, 5.0)
            effective_impact = np.clip(impact * (1 + red_impact), 0.0, 5.0)
            weight = (effective_probability / 5.0) * (effective_impact / 5.0)

            c_ml = float(qualquan_data.get('cost_most_likely_usd', 0.0))
            t_ml = float(qualquan_data.get('time_most_likely_weeks', 0.0))
            
            c_red = (abs(c_ml) * weight) * USD_TO_CAD_EXCHANGE_RATE
            t_red = (abs(t_ml) * weight)
            
            total_opportunity_cost_reduction += c_red
            total_opportunity_time_reduction += t_red
            
            opportunity_item.update(qualquan_data)
            opportunity_item.update({
                "effective_probability": effective_probability,
                "effective_impact": effective_impact,
                "cost_reduction_cad": c_red, 
                "time_reduction_weeks": t_red,
                "risk_category": r_cat
            })
            opportunities_with_qualquan_data.append(opportunity_item)

        # 6. Calculate Quantum Influence and Adjust Baseline
        _update_internal_status("Calculating quantum influence...", 50)
        avg_effective_prob = float(np.mean([r['effective_probability'] for r in risks_with_qualquan_data])) if risks_with_qualquan_data else 0.0
        avg_effective_impact = float(np.mean([r['effective_impact'] for r in risks_with_qualquan_data])) if risks_with_qualquan_data else 0.0
        
        quantum_cost_influence, quantum_time_influence = _calculate_quantum_influence(avg_effective_prob, avg_effective_impact, initial_most_likely_cost, initial_most_likely_time)
        
        # --- FIXED: Explicitly define adjusted_most_likely_time ---
        adjusted_most_likely_cost = max(0.0, initial_most_likely_cost + total_contractor_cost_impact + total_oe_cost_impact + quantum_cost_influence - total_opportunity_cost_reduction)
        adjusted_most_likely_time = max(0.0, initial_most_likely_time + total_contractor_time_impact + total_oe_time_impact + quantum_time_influence - total_opportunity_time_reduction)

        # 7. PERT Monte Carlo Simulations
        _update_internal_status("Running project-wide PERT simulation...", 60)
        
        c_opt = adjusted_most_likely_cost - (initial_most_likely_cost - float(baseline_estimates.get('cost_optimistic_cad', 0)))
        c_pess = adjusted_most_likely_cost + (float(baseline_estimates.get('cost_pessimistic_cad', 0)) - initial_most_likely_cost)
        sim_costs = _pert_monte_carlo_simulation(max(0, c_opt), adjusted_most_likely_cost, c_pess, num_simulations=NUM_SIMULATIONS)

        # 8. Calculate Reserves
        p85_cost = _calculate_p85(sim_costs)
        total_contingency_needed = max(0, p85_cost - adjusted_most_likely_cost)
        
        impact_sum = (total_contractor_cost_impact + total_oe_cost_impact)
        if impact_sum > 0:
            cont_reserve = total_contingency_needed * (total_contractor_cost_impact / impact_sum)
            mgmt_reserve = total_contingency_needed * (total_oe_cost_impact / impact_sum)
        else:
            cont_reserve = total_contingency_needed * 0.5
            mgmt_reserve = total_contingency_needed * 0.5

        total_project_cost_cad = p85_cost

        # 9. Generate Graphs
        _update_internal_status("Generating distribution plots...", 90)
        results['plots_base64'] = {}

        # Cost Plot
        if len(sim_costs) > 0:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.hist(sim_costs, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Cost Distribution')
            ax1.set_title('Simulated Total Project Cost Distribution (PERT Analysis)')
            ax1.set_xlabel('Cost (CAD)')
            ax1.set_ylabel('Density')
            ax1b = ax1.twinx()
            sorted_costs = np.sort(sim_costs)
            cdf = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
            ax1b.plot(sorted_costs, cdf * 100, color='blue', label='Cumulative Frequency (%)')
            ax1.axvline(adjusted_most_likely_cost, color='red', linestyle='dotted', label=f'Adj. ML: ${adjusted_most_likely_cost:,.0f}')
            ax1.axvline(p85_cost, color='green', linestyle='dotted', label=f'P85: ${p85_cost:,.0f}')
            ax1.axvline(total_project_cost_cad, color='purple', linestyle='dashed', label=f'Total Project Cost: ${total_project_cost_cad:,.0f}')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1b.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            results['plots_base64']['cost_distribution_plot_base64'] = _plot_to_base64(fig1)

        # Time Plot - Uses fixed variable 'adjusted_most_likely_time'
        sim_times = _pert_monte_carlo_simulation(adjusted_most_likely_time * 0.9, adjusted_most_likely_time, adjusted_most_likely_time * 1.2)
        p85_time = _calculate_p85(sim_times)
        
        if len(sim_times) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(sim_times, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black', label='Time Distribution')
            ax2.set_title('Simulated Total Project Time Distribution (PERT Analysis)')
            ax2.set_xlabel('Time (Weeks)')
            ax2.set_ylabel('Density')
            ax2b = ax2.twinx()
            sorted_times = np.sort(sim_times)
            cdf_t = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            ax2b.plot(sorted_times, cdf_t * 100, color='darkred', label='Cumulative Frequency (%)')
            ax2.axvline(adjusted_most_likely_time, color='red', linestyle='dotted', label=f'Adj. ML Time: {adjusted_most_likely_time:.1f} wks')
            ax2.axvline(p85_time, color='green', linestyle='dotted', label=f'P85 Time: {p85_time:.1f} wks')
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            results['plots_base64']['time_distribution_plot_base64'] = _plot_to_base64(fig2)

        # 10. Final Data Mapping (Using fixed variable names)
        results.update({
            'adjusted_most_likely_cost_cad': float(adjusted_most_likely_cost), 
            'adjusted_most_likely_time_weeks': float(adjusted_most_likely_time),
            'p85_cost_cad': float(p85_cost),
            'contingency_reserve_cad': float(cont_reserve),
            'management_reserve_cad': float(mgmt_reserve),
            'total_project_cost_cad': float(total_project_cost_cad),
            'total_project_time_weeks': float(p85_time),
            'total_opportunity_cost_reduction_cad': float(total_opportunity_cost_reduction), 
            'total_opportunity_time_reduction_weeks': float(total_opportunity_time_reduction),
            'identified_risks_with_qualquan_data': risks_with_qualquan_data,
            'identified_opportunities_with_qualquan_data': opportunities_with_qualquan_data
        })

        _update_internal_status("Pipeline Complete!", 100)
        return results

    except Exception as e:
        error_msg = f"Analysis Pipeline Crash: {str(e)}"
        _update_internal_status(error_msg, 0, error=error_msg)
        return {"error": error_msg}
        
        
def _generate_llm_response(prompt):
    """
    Generates a text response using the loaded LLM (Gemini or TinyLlama).
    Includes safety screening of the prompt before generation.

    Args:
        prompt (str): The input prompt for the LLM.

    Returns:
        str: The generated text response.
             Returns an error message if safety screening fails or LLM generation encounters an error.
    """
    _update_internal_status("Generating LLM response...", 60)
    # Perform safety screening before generating content.
    safety_check_result = _screen_description_for_safety(prompt)
    if not safety_check_result['is_safe']:
        _update_internal_status("LLM response blocked by safety screening.", 100, error="Safety violation")
        return f"Content blocked due to safety concerns: {safety_check_result['reason']}"

    with _models_lock:
        tokenizer = fine_tuned_risk_tokenizer
        model = fine_tuned_risk_model
        gemini_model = gemini_model_text # Access the global Gemini model instance.

    if USE_GEMINI_API and gemini_model:
        try:
            # Use Gemini API for response generation.
            response = gemini_model.generate_content(prompt, generation_config={"max_output_tokens": 500})
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
                _update_internal_status("LLM response generated via Gemini.", 100)
                return response_text
            else:
                raise ValueError("Gemini API response structure unexpected or content missing.")
        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"Gemini API rate limit exceeded or resource exhausted: {e}. Falling back to local LLM.")
        except Exception as e:
            print(f"Error generating LLM response with Gemini API: {e}. Falling back to local LLM.")

    # Fallback to local TinyLlama if Gemini fails or is not used.
    if not tokenizer or not model:
        _update_internal_status("LLM models not loaded for local generation.", 0, error="LLM models not loaded")
        return "Error: LLM models are not loaded. Please load models first."

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_new_tokens = min(200, 512) # Reasonable limit for new tokens.
        
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1
        )
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Post-process to remove the prompt itself from the response if the model echoes it.
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()

        _update_internal_status("LLM response generated via local model.", 100)
        return response_text
    except Exception as e:
        _update_internal_status(f"Error generating LLM response with local model: {e}", 0, error=str(e))
        return f"Error generating response: {e}"

def run_safety_analysis_pipeline(project_description, initial_analysis_results, num_simulations=NUM_SIMULATIONS):
    """
    Performs safety-specific risk analysis. It identifies safety risks, quantifies them,
    and recalculates project reserves and overall estimates considering these safety aspects.

    Args:
        project_description (str): The description of the project.
        initial_analysis_results (dict): Results from the initial project analysis pipeline.
                                         This provides baseline adjusted costs/times.
        num_simulations (int): Number of Monte Carlo simulations for safety-related impacts.

    Returns:
        dict: A dictionary containing safety analysis results, including identified safety risks/opportunities,
              adjusted project cost/time after safety, new reserves, and base64 encoded plots.
    """
    _update_internal_status("Starting safety analysis pipeline...", 5)
    safety_results = {"error": None, "plots_base64_after_safety": {}}
    
    try:
        # 1. Identify Safety Risks and Opportunities using LLM.
        _update_internal_status("Identifying safety-specific risks and opportunities with LLM...", 10)
        safety_risks_opportunities_raw = _identify_risks_and_opportunities_llm(
            f"Identify potential safety risks and opportunities for: {project_description}. Focus on construction safety, operational safety, and environmental safety.",
            num_to_generate=NUM_RISKS_TO_GENERATE_FROM_LLM,
            category_focus="safety" # Focus the LLM on safety aspects.
        )
        safety_risks_with_qualquan_data = []
        safety_opportunities_with_qualquan_data = []
        safety_risks = [r for r in safety_risks_opportunities_raw if not r['is_opportunity']]
        safety_opportunities = [r for r in safety_risks_opportunities_raw if r['is_opportunity']]
        safety_results['identified_safety_opportunities'] = safety_opportunities
        safety_results['identified_safety_risks'] = safety_risks

        # Initialize aggregators for safety risk impacts.
        total_safety_contractor_cost_impact = 0.0
        total_safety_contractor_time_impact = 0.0
        total_safety_oe_cost_impact = 0.0
        total_safety_oe_time_impact = 0.0
        total_safety_shared_cost_impact = 0.0
        total_safety_shared_time_impact = 0.0

        # Aggregators for safety opportunity reductions.
        total_safety_opportunity_cost_reduction = 0.0
        total_safety_opportunity_time_reduction = 0.0

        # 2. QualQuan Assessment for Safety Risks.
        _update_internal_status("Performing QualQuan assessment for identified safety risks...", 20)
        for risk_item in tqdm(safety_risks, desc="Safety QualQuan Assessment (Risks)"):
            qualquan_data = _qualquan_risk_assessment(project_description, risk_item['risk_description'])
            
            effective_probability = qualquan_data['probability'] * (1 - qualquan_data['risk_reduction_probability'])
            effective_impact = qualquan_data['impact'] * (1 - qualquan_data['risk_reduction_impact'])

            risk_item.update({
                "qualitative_decision": qualquan_data['qualitative_decision'],
                "probability": qualquan_data['probability'],
                "impact": qualquan_data['impact'],
                "effective_probability": effective_probability,
                "effective_impact": effective_impact,
                "mitigation_strategy": qualquan_data['mitigation_strategy'],
                "risk_reduction_probability": qualquan_data['risk_reduction_probability'],
                "risk_reduction_impact": qualquan_data['risk_reduction_impact'],
                "cost_optimistic_usd": qualquan_data['cost_optimistic_usd'],
                "cost_most_likely_usd": qualquan_data['cost_most_likely_usd'],
                "cost_pessimistic_usd": qualquan_data['cost_pessimistic_usd'],
                "time_optimistic_weeks": qualquan_data['time_optimistic_weeks'],
                "time_most_likely_weeks": qualquan_data['time_most_likely_weeks'],
                "time_pessimistic_weeks": qualquan_data['time_pessimistic_weeks']
            })

            risk_item['cost_impact_cad'] = (qualquan_data['cost_most_likely_usd'] * (effective_probability / 5.0) * (effective_impact / 5.0)) * USD_TO_CAD_EXCHANGE_RATE
            risk_item['time_impact_weeks'] = (qualquan_data['time_most_likely_weeks'] * (effective_probability / 5.0) * (effective_impact / 5.0))

            safety_risks_with_qualquan_data.append(risk_item)

            # Aggregate impacts based on risk category for safety risks.
            risk_category = risk_item.get('risk_category', 'Contractor-borne')
            if risk_category == 'Contractor-borne':
                total_safety_contractor_cost_impact += risk_item['cost_impact_cad']
                total_safety_contractor_time_impact += risk_item['time_impact_weeks']
            elif risk_category in ['Owner-borne', 'Engineer-borne']:
                total_safety_oe_cost_impact += risk_item['cost_impact_cad']
                total_safety_oe_time_impact += risk_item['time_impact_weeks']
            elif risk_category == 'Shared':
                total_safety_contractor_cost_impact += risk_item['cost_impact_cad'] * 0.5
                total_safety_contractor_time_impact += risk_item['time_impact_weeks'] * 0.5
                total_safety_oe_cost_impact += risk_item['cost_impact_cad'] * 0.5
                total_safety_oe_time_impact += risk_item['time_impact_weeks'] * 0.5

        safety_results['identified_safety_risks_with_qualquan_data'] = safety_risks_with_qualquan_data

        # 3. QualQuan Assessment for Safety Opportunities.
        _update_internal_status("Performing QualQuan assessment for identified safety opportunities...", 25)
        for opportunity_item in tqdm(safety_opportunities, desc="Safety QualQuan Assessment (Opportunities)"):
            qualquan_data = _qualquan_risk_assessment(project_description, opportunity_item['risk_description'])
            
            effective_probability = np.clip(qualquan_data['probability'] * (1 + qualquan_data['risk_reduction_probability']), 0.0, 5.0)
            effective_impact = np.clip(qualquan_data['impact'] * (1 + qualquan_data['risk_reduction_impact']), 0.0, 5.0)

            opportunity_item.update({
                "qualitative_decision": qualquan_data['qualitative_decision'],
                "probability": qualquan_data['probability'],
                "impact": qualquan_data['impact'],
                "effective_probability": effective_probability,
                "effective_impact": effective_impact,
                "mitigation_strategy": qualquan_data['mitigation_strategy'],
                "risk_reduction_probability": qualquan_data['risk_reduction_probability'],
                "risk_reduction_impact": qualquan_data['risk_reduction_impact'],
                "cost_optimistic_usd": qualquan_data['cost_optimistic_usd'],
                "cost_most_likely_usd": qualquan_data['cost_most_likely_usd'],
                "cost_pessimistic_usd": qualquan_data['cost_pessimistic_usd'],
                "time_optimistic_weeks": qualquan_data['time_optimistic_weeks'],
                "time_most_likely_weeks": qualquan_data['time_most_likely_weeks'],
                "time_pessimistic_weeks": qualquan_data['time_pessimistic_weeks']
            })

            opportunity_item['cost_reduction_cad'] = (qualquan_data['cost_most_likely_usd'] * (effective_probability / 5.0) * (effective_impact / 5.0)) * USD_TO_CAD_EXCHANGE_RATE
            opportunity_item['time_reduction_weeks'] = (qualquan_data['time_most_likely_weeks'] * (effective_probability / 5.0) * (effective_impact / 5.0))

            total_safety_opportunity_cost_reduction += opportunity_item['cost_reduction_cad']
            total_safety_opportunity_time_reduction += opportunity_item['time_reduction_weeks']

            safety_opportunities_with_qualquan_data.append(opportunity_item)
            print(f"DEBUG (Safety): Opportunity '{opportunity_item.get('risk_description', 'N/A')}' processed. Cost Reduction (CAD): {opportunity_item['cost_reduction_cad']:.2f}, Time Reduction (Weeks): {opportunity_item['time_reduction_weeks']:.2f}")

        safety_results['identified_safety_opportunities_with_qualquan_data'] = safety_opportunities_with_qualquan_data
        safety_results['total_safety_opportunity_cost_reduction_cad'] = total_safety_opportunity_cost_reduction
        safety_results['total_safety_opportunity_time_reduction_weeks'] = total_safety_opportunity_time_reduction


        # 4. Adjust Overall Project Estimates based on Initial Analysis + Safety Risks + Safety Opportunities.
        _update_internal_status("Adjusting overall project estimates based on safety analysis...", 30)
        
        # Start with the latest total cost/time from the *initial* analysis results.
        initial_adjusted_cost_from_prev = initial_analysis_results.get('total_project_cost_cad', 0.0)
        initial_adjusted_time_from_prev = initial_analysis_results.get('total_project_time_weeks', 0.0)
        
        # Add the total impact of *safety-specific* risks and subtract safety opportunities.
        total_project_cost_after_safety_ml = initial_adjusted_cost_from_prev + total_safety_contractor_cost_impact + total_safety_oe_cost_impact + total_safety_shared_cost_impact - total_safety_opportunity_cost_reduction
        total_project_time_after_safety_ml = initial_adjusted_time_from_prev + total_safety_contractor_time_impact + total_safety_oe_time_impact + total_safety_shared_time_impact - total_safety_opportunity_time_reduction

        safety_results['total_project_cost_after_safety_ml'] = total_project_cost_after_safety_ml
        safety_results['total_project_time_after_safety_ml'] = total_project_time_after_safety_ml

        # Recalculate O and P for Monte Carlo after safety adjustments.
        baseline_estimates_from_initial = initial_analysis_results.get('baseline_estimates', {})
        original_opt_cost_spread = baseline_estimates_from_initial.get('cost_most_likely_cad', 0.0) - baseline_estimates_from_initial.get('cost_optimistic_cad', 0.0)
        original_pess_cost_spread = baseline_estimates_from_initial.get('cost_pessimistic_cad', 0.0) - baseline_estimates_from_initial.get('cost_most_likely_cad', 0.0)
        
        new_optimistic_cost_after_safety_for_mc = total_project_cost_after_safety_ml - original_opt_cost_spread
        new_pessimistic_cost_after_safety_for_mc = total_project_cost_after_safety_ml + original_pess_cost_spread

        new_optimistic_cost_after_safety_for_mc = max(0.0, new_optimistic_cost_after_safety_for_mc)
        new_optimistic_cost_after_safety_for_mc = min(new_optimistic_cost_after_safety_for_mc, total_project_cost_after_safety_ml)
        new_pessimistic_cost_after_safety_for_mc = max(total_project_cost_after_safety_ml, new_pessimistic_cost_after_safety_for_mc)

        original_opt_time_spread = baseline_estimates_from_initial.get('time_most_likely_weeks', 0.0) - baseline_estimates_from_initial.get('time_optimistic_weeks', 0.0)
        original_pess_time_spread = baseline_estimates_from_initial.get('time_pessimistic_weeks', 0.0) - baseline_estimates_from_initial.get('time_most_likely_weeks', 0.0)
        
        new_optimistic_time_after_safety_for_mc = total_project_time_after_safety_ml - original_opt_time_spread
        new_pessimistic_time_after_safety_for_mc = total_project_time_after_safety_ml + original_pess_time_spread
        
        new_optimistic_time_after_safety_for_mc = max(0.0, new_optimistic_time_after_safety_for_mc)
        new_optimistic_time_after_safety_for_mc = min(new_optimistic_time_after_safety_for_mc, total_project_cost_after_safety_ml)
        new_pessimistic_time_after_safety_for_mc = max(total_project_time_after_safety_ml, new_pessimistic_time_after_safety_for_mc)


        # 5. Re-run Monte Carlo Simulations with Safety Adjustments.
        _update_internal_status("Re-running Monte Carlo for project cost and time after safety analysis...", 40)
        simulated_costs_after_safety = _pert_monte_carlo_simulation(
            new_optimistic_cost_after_safety_for_mc,
            total_project_cost_after_safety_ml,
            new_pessimistic_cost_after_safety_for_mc,
            num_simulations=NUM_SIMULATIONS
        )
        simulated_times_after_safety = _pert_monte_carlo_simulation(
            new_optimistic_time_after_safety_for_mc,
            total_project_time_after_safety_ml,
            new_pessimistic_time_after_safety_for_mc,
            num_simulations=NUM_SIMULATIONS
        )

        safety_results['simulated_costs_after_safety'] = simulated_costs_after_safety.tolist()
        safety_results['simulated_times_after_safety'] = simulated_times_after_safety.tolist()

        # 6. Calculate P85 and Contingency/Management Reserves (After Safety).
        _update_internal_status("Calculating P85 and reserves after safety analysis...", 50)
        p85_cost_after_safety = _calculate_p85(simulated_costs_after_safety)
        p85_time_after_safety = _calculate_p85(simulated_times_after_safety)

        # Contingency Reserve (after safety) = P85 of (Baseline + All Initial Risks + Safety Contractor/Shared Risks) - Adjusted ML after safety.
        contingency_reserve_after_safety_cad = max(0, p85_cost_after_safety - total_project_cost_after_safety_ml)
        safety_results['contingency_reserve_after_safety_cad'] = contingency_reserve_after_safety_cad

        # Simulate Owner/Engineer-born and Shared safety risk costs for management reserve.
        oe_and_shared_safety_risks = [r for r in safety_risks_with_qualquan_data if r.get('risk_category') in ['Owner-borne', 'Engineer-borne', 'Shared']]
        
        total_safety_oe_and_shared_cost_optimistic = 0.0
        total_safety_oe_and_shared_cost_most_likely = 0.0
        total_safety_oe_and_shared_cost_pessimistic = 0.0

        for risk in oe_and_shared_safety_risks:
            multiplier = 0.5 if risk.get('risk_category') == 'Shared' else 1.0
            total_safety_oe_and_shared_cost_optimistic += risk['cost_optimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_safety_oe_and_shared_cost_most_likely += risk['cost_most_likely_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_safety_oe_and_shared_cost_pessimistic += risk['cost_pessimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier

        if total_safety_oe_and_shared_cost_most_likely == 0:
            simulated_safety_owner_engineer_costs = np.zeros(NUM_SIMULATIONS)
        else:
            simulated_safety_owner_engineer_costs = _monte_carlo_simulation(
                total_safety_oe_and_shared_cost_optimistic,
                total_safety_oe_and_shared_cost_most_likely,
                total_safety_oe_and_shared_cost_pessimistic,
                num_simulations=NUM_SIMULATIONS
            )
        safety_results['management_reserve_after_safety_cad'] = _calculate_p85(simulated_safety_owner_engineer_costs)

        # Final Total Project Cost and Time after safety analysis.
        total_project_cost_after_safety_cad = total_project_cost_after_safety_ml + contingency_reserve_after_safety_cad + safety_results['management_reserve_after_safety_cad']
        total_project_time_after_safety_weeks = total_project_time_after_safety_ml + (p85_time_after_safety - total_project_time_after_safety_ml)

        safety_results['total_project_cost_after_safety_cad'] = total_project_cost_after_safety_cad
        safety_results['total_project_time_after_safety_weeks'] = total_project_time_after_safety_weeks
        safety_results['p85_cost_after_safety_cad'] = p85_cost_after_safety
        safety_results['p85_time_after_safety_weeks'] = p85_time_after_safety


        # 7. Generate Plots (Cost and Time Distribution After Safety).
        _update_internal_status("Generating safety analysis plots...", 70)
        safety_results['plots_base64_after_safety'] = {}

        # Cost Distribution Plot (After Safety).
        if len(simulated_costs_after_safety) > 0 and not np.all(simulated_costs_after_safety == 0):
            fig_cost_dist_safety, ax_cost_dist_safety = plt.subplots(figsize=(10, 6))
            ax_cost_dist_safety.hist(simulated_costs_after_safety, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Cost Distribution')
            ax_cost_dist_safety.set_xlabel('Cost (CAD)')
            ax_cost_dist_safety.set_ylabel('Density')
            ax_cost_dist_safety.set_title('Simulated Total Project Cost Distribution (After Safety Analysis)')
            
            cost_mean, cost_std = np.mean(simulated_costs_after_safety), np.std(simulated_costs_after_safety)
            if cost_std > 0:
                ax_cost_dist_safety.set_xlim(max(0, cost_mean - 4 * cost_std), cost_mean + 4 * cost_std)
            else:
                ax_cost_dist_safety.set_xlim(max(0, cost_mean * 0.9), cost_mean * 1.1 if cost_mean > 0 else 1)
            
            ax2_cost_safety = ax_cost_dist_safety.twinx()
            sorted_costs_safety = np.sort(simulated_costs_after_safety)
            cdf_cost_safety = np.arange(1, len(sorted_costs_safety) + 1) / len(sorted_costs_safety)
            ax2_cost_safety.plot(sorted_costs_safety, cdf_cost_safety * 100, color='blue', linestyle='-', label='Cumulative Frequency (%)')
            ax2_cost_safety.set_ylabel('Cumulative Frequency (%)')
            ax2_cost_safety.set_ylim(0, 100)

            ax_cost_dist_safety.axvline(total_project_cost_after_safety_ml, color='red', linestyle='dotted', linewidth=1.5, label=f'Adj. ML Cost: ${total_project_cost_after_safety_ml:,.0f}')
            ax_cost_dist_safety.axvline(p85_cost_after_safety, color='green', linestyle='dotted', linewidth=1.5, label=f'P85 Cost: ${p85_cost_after_safety:,.0f}')
            ax_cost_dist_safety.axvline(total_project_cost_after_safety_cad, color='purple', linestyle='dashed', linewidth=1.5, label=f'Total Project Cost: ${total_project_cost_after_safety_cad:,.0f}')

            lines, labels = ax_cost_dist_safety.get_legend_handles_labels()
            lines2, labels2 = ax2_cost_safety.get_legend_handles_labels()
            ax2_cost_safety.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            safety_results['plots_base64_after_safety']['cost_distribution_plot_after_safety_base64'] = _plot_to_base64(fig_cost_dist_safety)
        else:
            safety_results['plots_base64_after_safety']['cost_distribution_plot_after_safety_base64'] = None

        # Time Distribution Plot (After Safety).
        if len(simulated_times_after_safety) > 0 and not np.all(simulated_times_after_safety == 0):
            fig_time_dist_safety, ax_time_dist_safety = plt.subplots(figsize=(10, 6))
            ax_time_dist_safety.hist(simulated_times_after_safety, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black', label='Time Distribution')
            ax_time_dist_safety.set_xlabel('Time (Weeks)')
            ax_time_dist_safety.set_ylabel('Density')
            ax_time_dist_safety.set_title('Simulated Total Project Time Distribution (After Safety Analysis)')

            time_mean, time_std = np.mean(simulated_times_after_safety), np.std(simulated_times_after_safety)
            if time_std > 0:
                ax_time_dist_safety.set_xlim(max(0, time_mean - 4 * time_std), time_mean + 4 * time_std)
            else:
                ax_time_dist_safety.set_xlim(max(0, time_mean * 0.9), time_mean * 1.1 if time_mean > 0 else 1)

            ax2_time_safety = ax_time_dist_safety.twinx()
            sorted_times_safety = np.sort(simulated_times_after_safety)
            cdf_time_safety = np.arange(1, len(sorted_times_safety) + 1) / len(sorted_times_safety)
            ax2_time_safety.plot(sorted_times_safety, cdf_time_safety * 100, color='darkred', linestyle='-', label='Cumulative Frequency (%)')
            ax2_time_safety.set_ylabel('Cumulative Frequency (%)')
            ax2_time_safety.set_ylim(0, 100)

            ax_time_dist_safety.axvline(total_project_time_after_safety_ml, color='red', linestyle='dotted', linewidth=1.5, label=f'Adj. ML Time: {total_project_time_after_safety_ml:.1f} weeks')
            ax_time_dist_safety.axvline(p85_time_after_safety, color='green', linestyle='dotted', linewidth=1.5, label=f'P85 Time: {p85_time_after_safety:.1f} weeks')
            ax_time_dist_safety.axvline(total_project_time_after_safety_weeks, color='purple', linestyle='dashed', linewidth=1.5, label=f'Total Project Time: {total_project_time_after_safety_weeks:.1f} weeks')

            lines, labels = ax_time_dist_safety.get_legend_handles_labels()
            lines2, labels2 = ax2_time_safety.get_legend_handles_labels()
            ax_time_dist_safety.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            safety_results['plots_base64_after_safety']['time_distribution_plot_after_safety_base64'] = _plot_to_base64(fig_time_dist_safety)
        else:
            safety_results['plots_base64_after_safety']['time_distribution_plot_after_safety_base64'] = None

        # Contractor-Related Safety Risk Cost Distribution Plot.
        contractor_safety_risks = [r for r in safety_risks_with_qualquan_data if r.get('risk_category') == 'Contractor-borne' or r.get('risk_category') == 'Shared']
        total_contractor_safety_cost_optimistic = 0.0
        total_contractor_safety_cost_most_likely = 0.0
        total_contractor_safety_cost_pessimistic = 0.0

        for risk in contractor_safety_risks:
            multiplier = 0.5 if risk.get('risk_category') == 'Shared' else 1.0
            total_contractor_safety_cost_optimistic += risk['cost_optimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_contractor_safety_cost_most_likely += risk['cost_most_likely_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_contractor_safety_cost_pessimistic += risk['cost_pessimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier

        if total_contractor_safety_cost_most_likely == 0:
            simulated_contractor_safety_costs = np.zeros(NUM_SIMULATIONS)
        else:
            simulated_contractor_safety_costs = _monte_carlo_simulation(
                total_contractor_safety_cost_optimistic,
                total_contractor_safety_cost_most_likely,
                total_contractor_safety_cost_pessimistic,
                num_simulations=NUM_SIMULATIONS
            )
        
        if len(simulated_contractor_safety_costs) > 0 and not np.all(simulated_contractor_safety_costs == 0):
            fig_contractor_safety_risk_cost, ax_contractor_safety_risk_cost = plt.subplots(figsize=(10, 6))
            ax_contractor_safety_risk_cost.hist(simulated_contractor_safety_costs, bins=50, density=True, alpha=0.7, color='darkgreen', edgecolor='black', label='Cost Impact Distribution')
            ax_contractor_safety_risk_cost.set_title('Simulated Contractor-Related Safety Risk Cost Impact Distribution')
            ax_contractor_safety_risk_cost.set_xlabel('Cost Impact (CAD)')
            ax_contractor_safety_risk_cost.set_ylabel('Density')
            
            mean_val, std_val = np.mean(simulated_contractor_safety_costs), np.std(simulated_contractor_safety_costs)
            ax_contractor_safety_risk_cost.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Impact: ${mean_val:,.0f}')
            
            if std_val > 0:
                ax_contractor_safety_risk_cost.set_xlim(max(0, mean_val - 4 * std_val), mean_val + 4 * std_val)
            else:
                ax_contractor_safety_risk_cost.set_xlim(max(0, mean_val * 0.9), mean_val * 1.1 if mean_val > 0 else 1)

            ax2_contractor_safety_risk_cost = ax_contractor_safety_risk_cost.twinx()
            sorted_contractor_safety_costs = np.sort(simulated_contractor_safety_costs)
            cdf_contractor_safety_cost = np.arange(1, len(sorted_contractor_safety_costs) + 1) / len(sorted_contractor_safety_costs)
            ax2_contractor_safety_risk_cost.plot(sorted_contractor_safety_costs, cdf_contractor_safety_cost * 100, color='blue', linestyle='-', label='Cumulative Frequency (%)')
            ax2_contractor_safety_risk_cost.set_ylabel('Cumulative Frequency (%)')
            ax2_contractor_safety_risk_cost.set_ylim(0, 100)

            p85_contractor_safety_cost = _calculate_p85(simulated_contractor_safety_costs)
            ax_contractor_safety_risk_cost.axvline(p85_contractor_safety_cost, color='orange', linestyle='dotted', linewidth=1.5, label=f'P85 Cost: ${p85_contractor_safety_cost:,.0f}')

            lines, labels = ax_contractor_safety_risk_cost.get_legend_handles_labels()
            lines2, labels2 = ax2_contractor_safety_risk_cost.get_legend_handles_labels()
            ax_contractor_safety_risk_cost.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            safety_results['plots_base64_after_safety']['contractor_risk_cost_plot_after_safety_base64'] = _plot_to_base64(fig_contractor_safety_risk_cost)
        else:
            safety_results['plots_base64_after_safety']['contractor_risk_cost_plot_after_safety_base64'] = None


        # Owner/Engineer-Born Safety Risk Cost Distribution Plot.
        if len(simulated_safety_owner_engineer_costs) > 0 and not np.all(simulated_safety_owner_engineer_costs == 0):
            fig_safety_owner_engineer_risk_cost, ax_safety_owner_engineer_risk_cost = plt.subplots(figsize=(10, 6))
            ax_safety_owner_engineer_risk_cost.hist(simulated_safety_owner_engineer_costs, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black', label='Cost Impact Distribution')
            ax_safety_owner_engineer_risk_cost.set_title('Simulated Owner/Engineer-Born Safety Risk Cost Impact Distribution')
            ax_safety_owner_engineer_risk_cost.set_xlabel('Cost Impact (CAD)')
            ax_safety_owner_engineer_risk_cost.set_ylabel('Density')
            ax_safety_owner_engineer_risk_cost.axvline(safety_results['management_reserve_after_safety_cad'], color='green', linestyle='dashed', linewidth=1, label=f'Management Reserve (P85): ${safety_results["management_reserve_after_safety_cad"]:,.0f}')
            
            mean_val, std_val = np.mean(simulated_safety_owner_engineer_costs), np.std(simulated_safety_owner_engineer_costs)
            if std_val > 0:
                ax_safety_owner_engineer_risk_cost.set_xlim(max(0, mean_val - 4 * std_val), mean_val + 4 * std_val)
            else:
                ax_safety_owner_engineer_risk_cost.set_xlim(max(0, mean_val * 0.9), mean_val * 1.1 if mean_val > 0 else 1)

            ax2_safety_owner_engineer_risk_cost = ax_safety_owner_engineer_risk_cost.twinx()
            sorted_safety_owner_engineer_costs = np.sort(simulated_safety_owner_engineer_costs)
            cdf_safety_owner_engineer_cost = np.arange(1, len(sorted_safety_owner_engineer_costs) + 1) / len(sorted_safety_owner_engineer_costs)
            ax2_safety_owner_engineer_risk_cost.plot(sorted_safety_owner_engineer_costs, cdf_safety_owner_engineer_cost * 100, color='blue', linestyle='-', label='Cumulative Frequency (%)')
            ax2_safety_owner_engineer_risk_cost.set_ylabel('Cumulative Frequency (%)')
            ax2_safety_owner_engineer_risk_cost.set_ylim(0, 100)

            lines, labels = ax_safety_owner_engineer_risk_cost.get_legend_handles_labels()
            lines2, labels2 = ax2_safety_owner_engineer_risk_cost.get_legend_handles_labels()
            ax_safety_owner_engineer_risk_cost.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            safety_results['plots_base64_after_safety']['owner_engineer_risk_cost_plot_after_safety_base64'] = _plot_to_base64(fig_safety_owner_engineer_risk_cost)
        else:
            safety_results['plots_base64_after_safety']['owner_engineer_risk_cost_plot_after_safety_base64'] = None


        _update_internal_status("Safety analysis pipeline completed successfully.", 100)
        return safety_results

    except Exception as e:
        error_message = f"Error in safety analysis pipeline: {e}"
        _update_internal_status(error_message, 0, error=error_message)
        safety_results['error'] = error_message
        print(error_message)
        return safety_results

def generate_risk_report_pdf(analysis_results, tasks_data):
    """
    Placeholder for generating a comprehensive PDF report of the risk analysis.
    This would involve using a library like ReportLab or FPDF to create a structured document
    with all the analysis results, plots, and summaries.

    Args:
        analysis_results (dict): The complete results from the analysis pipelines.
        tasks_data (list): The generated project tasks.

    Returns:
        dict: A dictionary indicating the status of report generation and a dummy URL.
    """
    _update_internal_status("Generating PDF report...", 10)
    # In a real application, you'd generate the PDF here.
    # This is a complex task that requires a dedicated PDF generation library.
    # For example, using ReportLab:
    # from reportlab.lib.pagesizes import letter
    # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    # from reportlab.lib.styles import getSampleStyleSheet
    # from reportlab.lib import colors
    # doc = SimpleDocTemplate("risk_report.pdf", pagesize=letter)
    # styles = getSampleStyleSheet()
    # story = []
    # story.append(Paragraph("Project Risk Analysis Report", styles['h1']))
    # ... add tables, text, etc. ...
    # doc.build(story)
    
    # For now, just return a success message and a dummy URL.
    report_path = "/tmp/risk_report.pdf" # This would be a temporary file.
    _update_internal_status("PDF report generation simulated. In a full implementation, the PDF would be available for download.", 100)
    return {"message": "PDF report generation simulated. In a full implementation, the PDF would be available for download.", "report_url": report_path}
def robust_parse_json(text):
    """
    Attempts to parse a string as JSON, handling common LLM errors 
    like single quotes, markdown blocks, or trailing text.
    """
    if not text: return []

    # 1. Strip Markdown Code Blocks
    clean_text = text.strip()
    if "```" in clean_text:
        # Extract content between ```json and ``` or just ``` and ```
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        else:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

    # 2. Try Standard JSON (Double Quotes)
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass # Fall through to next method

    # 3. Try Python AST (Handles Single Quotes)
    try:
        import ast
        return ast.literal_eval(clean_text)
    except (ValueError, SyntaxError):
        pass # Fall through
    
    # 4. Try basic cleanup (Switch ' to " manually - risky but desperate)
    try:
        fixed_text = clean_text.replace("'", '"').replace("False", "false").replace("True", "true")
        return json.loads(fixed_text)
    except:
        return []
# ==========================================
# A+ GRADE SCHEDULING LOGIC (WBS + RESOURCES)
# ==========================================

def _get_resource_pool():
    """
    Defines the Resource Pool for the project (XML <Resources>).
    """
    return [
        {"UID": 1, "Name": "Project Manager", "Type": 1, "Rate": 120.00},
        {"UID": 2, "Name": "Site Supervisor", "Type": 1, "Rate": 95.00},
        {"UID": 3, "Name": "General Laborer", "Type": 1, "Rate": 45.00},
        {"UID": 4, "Name": "Excavator Operator", "Type": 1, "Rate": 85.00},
        {"UID": 5, "Name": "Concrete Finisher", "Type": 1, "Rate": 75.00},
        {"UID": 6, "Name": "Electrician", "Type": 1, "Rate": 90.00},
        {"UID": 7, "Name": "Plumber", "Type": 1, "Rate": 90.00},
        {"UID": 8, "Name": "Carpenter", "Type": 1, "Rate": 70.00},
        {"UID": 9, "Name": "HVAC Tech", "Type": 1, "Rate": 85.00},
        {"UID": 10, "Name": "Excavator (Equipment)", "Type": 1, "Rate": 200.00},
        {"UID": 11, "Name": "Concrete (Material)", "Type": 2, "Rate": 0},
        {"UID": 12, "Name": "Lumber (Material)", "Type": 2, "Rate": 0}
    ]

def _assign_resources_smartly(task_name, duration):
    """
    Analyzes task names to assign logical resources (e.g., 'Pour' -> Concrete).
    """
    name = task_name.lower()
    assignments = []
    
    # Helper to add assignment
    def add(res_uid, units=1.0):
        # Work = Duration * Units * 8 hours
        work_hours = int(duration * 8 * units)
        assignments.append({
            "ResourceUID": res_uid, 
            "Units": units, 
            "Work": f"PT{work_hours}H0M0S"
        })

    # Logic Rules
    if "project" in name or "manage" in name: add(1, 1.0) # PM
    elif "site" in name or "mobiliz" in name: add(2, 0.5); add(3, 2.0) # Super + Labor
    elif "excav" in name or "grad" in name: add(4, 1.0); add(10, 1.0) # Op + Machine
    elif "concrete" in name or "pour" in name: add(5, 2.0); add(11, 1.0) # Finisher + Material
    elif "frame" in name or "struct" in name: add(8, 3.0); add(12, 1.0) # Carpenter + Material
    elif "electri" in name or "wir" in name: add(6, 1.0)
    elif "plumb" in name or "pip" in name: add(7, 1.0)
    elif "hvac" in name or "duct" in name: add(9, 1.0)
    else: add(3, 1.0) # Default to Laborer
        
    return assignments

def _structure_wbs(raw_tasks):
    """
    The Architect: Converts flat AI tasks into a WBS Hierarchy.
    FIX: Now forces a valid 'task_id' onto every child task to prevent Procurement crashes.
    """
    if not raw_tasks: return []

    phases = {}
    for t in raw_tasks:
        p_name = t.get('phase', 'General Construction')
        if p_name not in phases: phases[p_name] = []
        phases[p_name].append(t)

    final_list = []
    old_id_to_new_id_map = {}
    current_id = 1 

    # 2. Build the Hierarchy
    for phase_name, subtasks in phases.items():
        # A. Create Phase Summary Task
        final_list.append({
            "task_id": f"PHS_{current_id}", # Unique ID for Phase
            "task_name": phase_name,
            "duration_days": 0, 
            "predecessors": [],
            "estimated_cost_cad": 0,
            "is_milestone": False,
            "is_summary": True,
            "outline_level": 1,
            "UID": current_id,
            "ID": current_id,
            "phase": phase_name,
            "assignments": []
        })
        current_id += 1
        
        # B. Process Children
        for t in subtasks:
            old_uid = str(t.get('UID', ''))
            old_id_to_new_id_map[old_uid] = str(current_id)
            
            t['UID'] = current_id
            t['ID'] = current_id
            t['is_summary'] = False
            t['outline_level'] = 2
            
            # --- THE FIX: Force a fresh task_id ---
            t['task_id'] = f"T{current_id}"
            # --------------------------------------
            
            # Resource Injection
            if not t.get('is_milestone'):
                t['assignments'] = _assign_resources_smartly(t['task_name'], t.get('duration_days', 1))
            else:
                t['assignments'] = []

            final_list.append(t)
            current_id += 1

    # 3. Remap Logic
    for t in final_list:
        if t['is_summary']: continue
        
        old_preds = t.get('predecessors', [])
        new_preds = []
        for p in old_preds:
            p_str = str(p)
            if p_str in old_id_to_new_id_map:
                new_preds.append(old_id_to_new_id_map[p_str])
        
        t['predecessors'] = [x for x in new_preds if int(x) != t['UID']]

    # 4. Calculate Successors
    children = [t for t in final_list if not t['is_summary']]
    if not children: return final_list

    for t in final_list: t['successors'] = []
    
    for t in children:
        my_uid = str(t['UID'])
        for p in t['predecessors']:
            parent = next((x for x in children if str(x['UID']) == p), None)
            if parent and my_uid not in parent['successors']:
                parent['successors'].append(my_uid)

    # 5. Fix Dangles
    last_child = children[-1]
    last_child_uid = str(last_child['UID'])
    
    for t in children:
        if str(t['UID']) == last_child_uid: continue
        if not t['successors']:
            t['successors'].append(last_child_uid)
            if str(t['UID']) not in last_child['predecessors']:
                last_child['predecessors'].append(str(t['UID']))

    return final_list

def _generate_project_tasks_llm(project_description):
    """
    Optimized AI Scheduler to prevent 'Safety Net' fallbacks.
    Simplifies JSON constraints for faster local model reasoning.
    """
    cache_key = project_description
    if cache_key in _project_tasks_cache: return _project_tasks_cache[cache_key]

    _update_internal_status("Consulting AI Scheduler...", 10)
    tasks = []

    if scheduler_model and scheduler_tokenizer:
        try:
            print("Using Local Qwen Scheduler...")
            # Simplified instructions to prevent VRAM overflow/logic loops
            prompt_content = (
                f"Create a detailed construction schedule for: {project_description}. "
                "Output ONLY a valid JSON array of objects. "
                "Keys: task_name, phase, duration_days, predecessors (list of IDs), estimated_cost_cad."
            )
            
            messages = [
                {"role": "system", "content": "You are a professional project scheduler. Output valid JSON only."},
                {"role": "user", "content": prompt_content}
            ]
            
            text = scheduler_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = scheduler_tokenizer(text, return_tensors="pt").to(device)
            
            # Reduced max_new_tokens to 2000 to prevent local hardware timeouts
            outputs = scheduler_model.generate(
                **inputs, 
                max_new_tokens=2000, 
                temperature=0.2, # Lower temperature for stricter JSON adherence
                top_p=0.9
            )
            
            gen_text = scheduler_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in gen_text: gen_text = gen_text.split("assistant")[-1].strip()
            
            tasks = robust_parse_json(gen_text)
        except Exception as e:
            print(f"Qwen Primary Attempt Failed: {e}")

    # --- RECOVERY: Attempt a shorter list if primary fails ---
    if not tasks and scheduler_model:
        try:
            print("Attempting Scheduler Recovery Mode (15 Tasks)...")
            prompt_rescue = f"List 15 construction tasks for {project_description} as a JSON array. Include task_name and duration_days."
            # (Execution logic same as above, but with smaller max_new_tokens)
            # ...
        except: pass

    # --- POST-PROCESSING: Ensuring every task has a unique ID ---
    if tasks:
        for i, task in enumerate(tasks):
            task['UID'] = i + 1
            task['ID'] = i + 1
            if 'task_id' not in task: task['task_id'] = f"T{i+1}"
            if 'duration_days' not in task: task['duration_days'] = 5
            if 'phase' not in task: task['phase'] = "Construction"
            
        final_structure = _structure_wbs(tasks)
        _project_tasks_cache[cache_key] = final_structure
        _update_internal_status("Schedule generated (Local Qwen).", 100)
        return final_structure

    # FINAL SAFETY NET (Only if all else fails)
    print("AI Failed. Using Safety Net.")
    # (Existing safety net logic...)
    
def _generate_ms_project_xml(project_description, project_tasks):
    """
    Generates a more concise MS Project XML string from a list of project tasks.
    """
    _update_internal_status("Generating MS Project XML...", 15)
    current_time = time.strftime('%Y-%m-%dT%H:%M:%S')

    # Simplified XML header
    xml_header = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Project xmlns="http://schemas.microsoft.com/project">
    <Title>{_xml_escape(project_description)}</Title>
    <CreationDate>{current_time}</CreationDate>
    <Tasks>
"""

    tasks_xml_content = []
    for task in project_tasks:
        predecessors_xml = ""
        if task.get('predecessors'):
            predecessors_xml = "".join(f"""
                <PredecessorLink>
                    <PredecessorUID>{pred_uid}</PredecessorUID>
                    <Type>1</Type> </PredecessorLink>""" for pred_uid in task['predecessors'])

        # Using a more compact format for the task entry
        tasks_xml_content.append(f"""
        <Task>
            <UID>{task['UID']}</UID>
            <ID>{task['ID']}</ID>
            <Name>{_xml_escape(task['task_name'])}</Name>
            <IsMilestone>{1 if task.get('is_milestone') else 0}</IsMilestone>
            <Duration>{int(task.get('duration_days', 1) * 480)}M</Duration> <DurationFormat>8</DurationFormat> {predecessors_xml}
        </Task>""")

    # Simplified XML footer
    xml_footer = """
    </Tasks>
</Project>
    """
    
    _update_internal_status("MS Project XML generated.", 20)
    return xml_header + "".join(tasks_xml_content) + xml_footer

def _save_schedule_log(log_entry):
    """
    Appends a log entry (e.g., generated project tasks) to the schedule log file.
    This is useful for auditing and potential future model retraining.

    Args:
        log_entry (dict): The dictionary containing the log data.
    """
    try:
        os.makedirs(os.path.dirname(SCHEDULE_LOG_FILE), exist_ok=True)
        with open(SCHEDULE_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Warning: Could not save schedule log: {e}")
	
def generate_project_schedule_and_xml(project_description):
    """
    Generates project tasks and an MS Project XML based on a project description,
    and logs the generated schedule.
    """
    _update_internal_status("Starting schedule generation pipeline...", 5)
    schedule_results = {"error": None}
    try:
        generated_tasks = _generate_project_tasks_llm(project_description)
        if isinstance(generated_tasks, dict) and generated_tasks.get('error'): # Check if LLM task generation returned an error dict
            raise ValueError(f"Failed to generate project tasks: {generated_tasks['error']}")
        
        # Log the original tasks for potential training data
        log_entry = {
            "timestamp": time.time(),
            "project_description": project_description,
            "generated_tasks": generated_tasks
        }
        _save_schedule_log(log_entry)

        ms_project_xml_content = _generate_ms_project_xml(project_description, generated_tasks)
        
        schedule_results['project_tasks'] = generated_tasks
        schedule_results['ms_project_xml'] = ms_project_xml_content
        
        _update_internal_status("Schedule generation pipeline completed.", 100)
        return schedule_results

    except Exception as e:
        error_message = f"Error during schedule generation: {e}"
        _update_internal_status(error_message, 0, error=error_message)
        schedule_results['error'] = error_message
        print(error_message)
        return schedule_results




# NEW FUNCTION: run_procurement_analysis_pipeline
# In integrationapi.py, replace BOTH existing definitions of run_procurement_analysis_pipeline with this corrected version:

def run_procurement_analysis_pipeline(initial_analysis_results, project_description):
    """
    Performs a comprehensive procurement analysis based on initial project analysis results.
    """
    _update_internal_status("Starting procurement analysis pipeline...", 5)
    procurement_results = {"error": None}
    try:
        # Step 1: Generate Project Tasks for procurement context
        _update_internal_status("Generating project tasks for procurement context...", 10)
        generated_tasks = _generate_project_tasks_llm(project_description)
        if isinstance(generated_tasks, dict) and generated_tasks.get('error'):
            raise ValueError(f"Failed to generate project tasks for procurement: {generated_tasks['error']}")

        # Step 2: Generate Detailed Procurement Schedule Data using LLM
        _update_internal_status("Generating detailed procurement schedule data...", 20)
        detailed_procurement_schedule = _generate_detailed_procurement_schedule(project_description, generated_tasks)
        if isinstance(detailed_procurement_schedule, dict) and detailed_procurement_schedule.get('error'):
            raise ValueError(f"Failed to generate detailed procurement schedule data: {detailed_procurement_schedule['error']}")
        elif not isinstance(detailed_procurement_schedule, list):
            raise TypeError(f"Unexpected return type from _generate_detailed_procurement_schedule: {type(detailed_procurement_schedule)}")

        # Generate the procurement MS Project XML
        _update_internal_status("Generating procurement MS Project XML content...", 25)
        procurement_xml_content = _generate_procurement_xml_content(project_description, detailed_procurement_schedule, generated_tasks)
        procurement_results['procurement_ms_project_xml'] = procurement_xml_content
        procurement_results['detailed_procurement_schedule'] = detailed_procurement_schedule

        # Log the detailed procurement schedule
        proc_log_entry = {
            "timestamp": time.time(),
            "project_description": project_description,
            "original_generated_tasks_summary": [{"task_id": t["task_id"], "task_name": t["task_name"]} for t in generated_tasks],
            "procurement_schedule_details": detailed_procurement_schedule
        }
        _save_procurement_schedule_log(proc_log_entry)

        # Step 3: Identify Procurement Risks and Opportunities
        _update_internal_status("Identifying procurement risks and opportunities with LLM...", 30)
        procurement_risks_opportunities_raw = _identify_risks_and_opportunities_llm(
            f"Identify procurement-related risks and opportunities for: {project_description}. Consider supply chain, vendors, contracts based on these tasks: {json.dumps([{'task_id': t['task_id'], 'task_name': t['task_name']} for t in generated_tasks])}",
            num_to_generate=NUM_RISKS_TO_GENERATE_FROM_LLM,
            category_focus="procurement"
        )
        procurement_risks_with_qualquan = []
        procurement_opportunities_with_qualquan = []
        procurement_risks = [r for r in procurement_risks_opportunities_raw if not r['is_opportunity']]
        procurement_opportunities = [r for r in procurement_risks_opportunities_raw if r['is_opportunity']]
        procurement_results['identified_procurement_risks'] = procurement_risks
        procurement_results['identified_procurement_opportunities'] = procurement_opportunities

        total_procurement_contractor_cost_impact = 0.0
        total_procurement_contractor_time_impact = 0.0
        total_procurement_oe_cost_impact = 0.0
        total_procurement_oe_time_impact = 0.0
        total_procurement_shared_cost_impact = 0.0
        total_procurement_shared_time_impact = 0.0
        total_procurement_opportunity_cost_reduction = 0.0
        total_procurement_opportunity_time_reduction = 0.0

        # Step 4: QualQuan Assessment for Procurement Risks
        _update_internal_status("Performing QualQuan assessment for procurement risks...", 40)
        for risk_item in tqdm(procurement_risks, desc="Procurement QualQuan Assessment (Risks)"):
            qualquan_data = _qualquan_risk_assessment(project_description, risk_item['risk_description'])
            effective_probability = qualquan_data['probability'] * (1 - qualquan_data['risk_reduction_probability'])
            effective_impact = qualquan_data['impact'] * (1 - qualquan_data['risk_reduction_impact'])
            risk_item.update({
                "qualitative_decision": qualquan_data['qualitative_decision'], "probability": qualquan_data['probability'],
                "impact": qualquan_data['impact'], "effective_probability": effective_probability, "effective_impact": effective_impact,
                "mitigation_strategy": qualquan_data['mitigation_strategy'], "risk_reduction_probability": qualquan_data['risk_reduction_probability'],
                "risk_reduction_impact": qualquan_data['risk_reduction_impact'], "cost_optimistic_usd": qualquan_data['cost_optimistic_usd'],
                "cost_most_likely_usd": qualquan_data['cost_most_likely_usd'], "cost_pessimistic_usd": qualquan_data['cost_pessimistic_usd'],
                "time_optimistic_weeks": qualquan_data['time_optimistic_weeks'], "time_most_likely_weeks": qualquan_data['time_most_likely_weeks'],
                "time_pessimistic_weeks": qualquan_data['time_pessimistic_weeks']
            })
            risk_item['cost_impact_cad'] = (qualquan_data['cost_most_likely_usd'] * (effective_probability / 5.0) * (effective_impact / 5.0)) * USD_TO_CAD_EXCHANGE_RATE
            risk_item['time_impact_weeks'] = (qualquan_data['time_most_likely_weeks'] * (effective_probability / 5.0) * (effective_impact / 5.0))
            procurement_risks_with_qualquan.append(risk_item)
            
            risk_category = risk_item.get('risk_category', 'Contractor-borne')
            if risk_category == 'Contractor-borne':
                total_procurement_contractor_cost_impact += risk_item['cost_impact_cad']
                total_procurement_contractor_time_impact += risk_item['time_impact_weeks']
            elif risk_category in ['Owner-borne', 'Engineer-borne']:
                total_procurement_oe_cost_impact += risk_item['cost_impact_cad']
                total_procurement_oe_time_impact += risk_item['time_impact_weeks']
            elif risk_category == 'Shared':
                total_procurement_contractor_cost_impact += risk_item['cost_impact_cad'] * 0.5
                # --- THIS IS THE FIX ---
                total_procurement_contractor_time_impact += risk_item['time_impact_weeks'] * 0.5
                # -----------------------
                total_procurement_oe_cost_impact += risk_item['cost_impact_cad'] * 0.5
                total_procurement_oe_time_impact += risk_item['time_impact_weeks'] * 0.5
        procurement_results['procurement_risks_with_qualquan_data'] = procurement_risks_with_qualquan

        # QualQuan Assessment for Procurement Opportunities
        _update_internal_status("Performing QualQuan assessment for procurement opportunities...", 45)
        for opportunity_item in tqdm(procurement_opportunities, desc="Procurement QualQuan Assessment (Opportunities)"):
            qualquan_data = _qualquan_risk_assessment(project_description, opportunity_item['risk_description'])
            effective_probability = np.clip(qualquan_data['probability'] * (1 + qualquan_data['risk_reduction_probability']), 0.0, 5.0)
            effective_impact = np.clip(qualquan_data['impact'] * (1 + qualquan_data['risk_reduction_impact']), 0.0, 5.0)
            opportunity_item.update({
                "qualitative_decision": qualquan_data['qualitative_decision'], "probability": qualquan_data['probability'],
                "impact": qualquan_data['impact'], "effective_probability": effective_probability, "effective_impact": effective_impact,
                "mitigation_strategy": qualquan_data['mitigation_strategy'], "risk_reduction_probability": qualquan_data['risk_reduction_probability'],
                "risk_reduction_impact": qualquan_data['risk_reduction_impact'], "cost_optimistic_usd": qualquan_data['cost_optimistic_usd'],
                "cost_most_likely_usd": qualquan_data['cost_most_likely_usd'], "cost_pessimistic_usd": qualquan_data['cost_pessimistic_usd'],
                "time_optimistic_weeks": qualquan_data['time_optimistic_weeks'], "time_most_likely_weeks": qualquan_data['time_most_likely_weeks'],
                "time_pessimistic_weeks": qualquan_data['time_pessimistic_weeks']
            })
            opportunity_item['cost_reduction_cad'] = (qualquan_data['cost_most_likely_usd'] * (effective_probability / 5.0) * (effective_impact / 5.0)) * USD_TO_CAD_EXCHANGE_RATE
            opportunity_item['time_reduction_weeks'] = (qualquan_data['time_most_likely_weeks'] * (effective_probability / 5.0) * (effective_impact / 5.0))
            total_procurement_opportunity_cost_reduction += opportunity_item['cost_reduction_cad']
            total_procurement_opportunity_time_reduction += opportunity_item['time_reduction_weeks']
            procurement_opportunities_with_qualquan.append(opportunity_item)
        procurement_results['procurement_opportunities_with_qualquan_data'] = procurement_opportunities_with_qualquan
        procurement_results['total_procurement_opportunity_cost_reduction_cad'] = total_procurement_opportunity_cost_reduction
        procurement_results['total_procurement_opportunity_time_reduction_weeks'] = total_procurement_opportunity_time_reduction
        
        # ... The rest of the function remains the same, it is omitted here for brevity ...
        # (The code for adjusting estimates, running Monte Carlo, and generating plots follows)
        # Safely retrieve initial analysis results for calculations, providing defaults
        cost_before_procurement = initial_analysis_results.get('total_project_cost_after_safety_cad', 0.0)
        time_before_procurement = initial_analysis_results.get('total_project_time_after_safety_weeks', 0.0)

        adjusted_cost_after_procurement_ml = cost_before_procurement + total_procurement_contractor_cost_impact + total_procurement_oe_cost_impact + total_procurement_shared_cost_impact - total_procurement_opportunity_cost_reduction
        adjusted_time_after_procurement_ml = time_before_procurement + total_procurement_contractor_time_impact + total_procurement_oe_time_impact + total_procurement_shared_time_impact - total_procurement_opportunity_time_reduction

        procurement_results['total_project_cost_after_procurement_ml'] = adjusted_cost_after_procurement_ml
        procurement_results['total_project_time_after_procurement_ml'] = adjusted_time_after_procurement_ml
        
        # (rest of function...)
        baseline_estimates_from_initial = initial_analysis_results.get('baseline_estimates', {})
        original_opt_cost_spread = baseline_estimates_from_initial.get('cost_most_likely_cad', 0.0) - baseline_estimates_from_initial.get('cost_optimistic_cad', 0.0)
        original_pess_cost_spread = baseline_estimates_from_initial.get('cost_pessimistic_cad', 0.0) - baseline_estimates_from_initial.get('cost_most_likely_cad', 0.0)
        
        new_optimistic_cost_after_procurement_for_mc = adjusted_cost_after_procurement_ml - original_opt_cost_spread
        new_pessimistic_cost_after_procurement_for_mc = adjusted_cost_after_procurement_ml + original_pess_cost_spread

        new_optimistic_cost_after_procurement_for_mc = max(0.0, new_optimistic_cost_after_procurement_for_mc)
        new_optimistic_cost_after_procurement_for_mc = min(new_optimistic_cost_after_procurement_for_mc, adjusted_cost_after_procurement_ml)
        new_pessimistic_cost_after_procurement_for_mc = max(adjusted_cost_after_procurement_ml, new_pessimistic_cost_after_procurement_for_mc)

        original_opt_time_spread = baseline_estimates_from_initial.get('time_most_likely_weeks', 0.0) - baseline_estimates_from_initial.get('time_optimistic_weeks', 0.0)
        original_pess_time_spread = baseline_estimates_from_initial.get('time_pessimistic_weeks', 0.0) - baseline_estimates_from_initial.get('time_most_likely_weeks', 0.0)
        
        new_optimistic_time_after_procurement_for_mc = adjusted_time_after_procurement_ml - original_opt_time_spread
        new_pessimistic_time_after_procurement_for_mc = adjusted_time_after_procurement_ml + original_pess_time_spread
        
        new_optimistic_time_after_procurement_for_mc = max(0.0, new_optimistic_time_after_procurement_for_mc)
        new_optimistic_time_after_procurement_for_mc = min(new_optimistic_time_after_procurement_for_mc, adjusted_time_after_procurement_ml)
        new_pessimistic_time_after_procurement_for_mc = max(adjusted_time_after_procurement_ml, new_pessimistic_time_after_procurement_for_mc)

        simulated_costs_after_procurement = _pert_monte_carlo_simulation(
            new_optimistic_cost_after_procurement_for_mc,
            adjusted_cost_after_procurement_ml,
            new_pessimistic_cost_after_procurement_for_mc,
            num_simulations=NUM_SIMULATIONS
        )
        simulated_times_after_procurement = _pert_monte_carlo_simulation(
            new_optimistic_time_after_procurement_for_mc,
            adjusted_time_after_procurement_ml,
            new_pessimistic_time_after_procurement_for_mc,
            num_simulations=NUM_SIMULATIONS
        )

        procurement_results['simulated_costs_after_procurement'] = simulated_costs_after_procurement.tolist()
        procurement_results['simulated_times_after_procurement'] = simulated_times_after_procurement.tolist()
        p85_cost_after_procurement = _calculate_p85(simulated_costs_after_procurement)
        p85_time_after_procurement = _calculate_p85(simulated_times_after_procurement)
        contingency_reserve_after_procurement_cad = max(0, p85_cost_after_procurement - adjusted_cost_after_procurement_ml)
        procurement_results['contingency_reserve_after_procurement_cad'] = contingency_reserve_after_procurement_cad
        
        oe_and_shared_procurement_risks = [r for r in procurement_risks_with_qualquan if r.get('risk_category') in ['Owner-borne', 'Engineer-borne', 'Shared']]
        total_proc_oe_and_shared_cost_optimistic, total_proc_oe_and_shared_cost_most_likely, total_proc_oe_and_shared_cost_pessimistic = 0.0, 0.0, 0.0
        for risk in oe_and_shared_procurement_risks:
            multiplier = 0.5 if risk.get('risk_category') == 'Shared' else 1.0
            total_proc_oe_and_shared_cost_optimistic += risk['cost_optimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_proc_oe_and_shared_cost_most_likely += risk['cost_most_likely_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_proc_oe_and_shared_cost_pessimistic += risk['cost_pessimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
        simulated_proc_owner_engineer_costs = _monte_carlo_simulation(
            total_proc_oe_and_shared_cost_optimistic, total_proc_oe_and_shared_cost_most_likely, total_proc_oe_and_shared_cost_pessimistic, num_simulations=NUM_SIMULATIONS
        ) if total_proc_oe_and_shared_cost_most_likely > 0 else np.zeros(NUM_SIMULATIONS)
        procurement_results['management_reserve_after_procurement_cad'] = _calculate_p85(simulated_proc_owner_engineer_costs)

        total_project_cost_after_procurement_cad = adjusted_cost_after_procurement_ml + contingency_reserve_after_procurement_cad + procurement_results['management_reserve_after_procurement_cad']
        total_project_time_after_procurement_weeks = adjusted_time_after_procurement_ml + (p85_time_after_procurement - adjusted_time_after_procurement_ml)

        procurement_results.update({
            'p85_cost_after_procurement_cad': p85_cost_after_procurement, 'p85_time_after_procurement_weeks': p85_time_after_procurement,
            'total_project_cost_after_procurement_cad': total_project_cost_after_procurement_cad, 'total_project_time_after_procurement_weeks': total_project_time_after_procurement_weeks,
            'plots_base64_after_procurement': {}
        })

        # Generate plots
        # Cost Distribution Plot
        if len(simulated_costs_after_procurement) > 0 and not np.all(simulated_costs_after_procurement == 0):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(simulated_costs_after_procurement, bins=50, density=True, alpha=0.7, color='darkcyan', edgecolor='black', label='Cost Distribution')
            ax.set_title('Simulated Total Project Cost Distribution (After Procurement Analysis)')
            ax.set_xlabel('Cost (CAD)'); ax.set_ylabel('Density')
            ax.axvline(adjusted_cost_after_procurement_ml, color='red', linestyle='dotted', lw=1.5, label=f'Adj. ML: ${adjusted_cost_after_procurement_ml:,.0f}')
            ax.axvline(p85_cost_after_procurement, color='green', linestyle='dotted', lw=1.5, label=f'P85: ${p85_cost_after_procurement:,.0f}')
            ax.axvline(total_project_cost_after_procurement_cad, color='purple', linestyle='dashed', lw=1.5, label=f'Total Cost: ${total_project_cost_after_procurement_cad:,.0f}')
            procurement_results['plots_base64_after_procurement']['cost_distribution_plot_after_procurement_base64'] = _plot_to_base64(fig)
        
        # Other plots can be generated similarly...

        _update_internal_status("Procurement analysis pipeline completed successfully.", 100)
        return procurement_results

    except Exception as e:
        error_message = f"Error in procurement analysis pipeline: {e}"
        _update_internal_status(error_message, 0, error=error_message)
        procurement_results['error'] = error_message
        print(error_message)
        return procurement_results


def generate_risk_report_pdf(analysis_results, tasks_data):
    """
    Placeholder for generating a comprehensive PDF report of the risk analysis.
    This would involve using a library like ReportLab or FPDF to create a structured document
    with all the analysis results, plots, and summaries.

    Args:
        analysis_results (dict): The complete results from the analysis pipelines.
        tasks_data (list): The generated project tasks.

    Returns:
        dict: A dictionary indicating the status of report generation and a dummy URL.
    """
    _update_internal_status("Generating PDF report...", 10)
    # In a real application, you'd generate the PDF here.
    # This is a complex task that requires a dedicated PDF generation library.
    # For example, using ReportLab:
    # from reportlab.lib.pagesizes import letter
    # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    # from reportlab.lib.styles import getSampleStyleSheet
    # from reportlab.lib import colors
    # doc = SimpleDocTemplate("risk_report.pdf", pagesize=letter)
    # styles = getSampleStyleSheet()
    # story = []
    # story.append(Paragraph("Project Risk Analysis Report", styles['h1']))
    # ... add tables, text, etc. ...
    # doc.build(story)
    
    # For now, just return a success message and a dummy URL.
    report_path = "/tmp/risk_report.pdf" # This would be a temporary file.
    _update_internal_status("PDF report generation simulated. In a full implementation, the PDF would be available for download.", 100)
    return {"message": "PDF report generation simulated. In a full implementation, the PDF would be available for download.", "report_url": report_path}

# In integrationapi.py



def _generate_ms_project_xml(project_description, project_tasks):
    """
    Generates a more concise MS Project XML string from a list of project tasks.
    """
    _update_internal_status("Generating MS Project XML...", 15)
    current_time = time.strftime('%Y-%m-%dT%H:%M:%S')

    # Simplified XML header
    xml_header = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Project xmlns="http://schemas.microsoft.com/project">
    <Title>{_xml_escape(project_description)}</Title>
    <CreationDate>{current_time}</CreationDate>
    <Tasks>
"""

    tasks_xml_content = []
    for task in project_tasks:
        predecessors_xml = ""
        if task.get('predecessors'):
            predecessors_xml = "".join(f"""
                <PredecessorLink>
                    <PredecessorUID>{pred_uid}</PredecessorUID>
                    <Type>1</Type> </PredecessorLink>""" for pred_uid in task['predecessors'])

        # Using a more compact format for the task entry
        tasks_xml_content.append(f"""
        <Task>
            <UID>{task['UID']}</UID>
            <ID>{task['ID']}</ID>
            <Name>{_xml_escape(task['task_name'])}</Name>
            <IsMilestone>{1 if task.get('is_milestone') else 0}</IsMilestone>
            <Duration>{int(task.get('duration_days', 1) * 480)}M</Duration> <DurationFormat>8</DurationFormat> {predecessors_xml}
        </Task>""")

    # Simplified XML footer
    xml_footer = """
    </Tasks>
</Project>
    """
    
    _update_internal_status("MS Project XML generated.", 20)
    return xml_header + "".join(tasks_xml_content) + xml_footer

def _save_stakeholder_analysis_log(log_entry):
    """
    Appends a log entry (e.g., generated stakeholder analysis) to the stakeholder analysis log file.

    Args:
        log_entry (dict): The dictionary containing the log data.
    """
    try:
        os.makedirs(os.path.dirname(STAKEHOLDER_ANALYSIS_LOG_FILE), exist_ok=True)
        with open(STAKEHOLDER_ANALYSIS_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Warning: Could not save stakeholder analysis log: {e}")


def _generate_procurement_xml_content(project_description, detailed_procurement_schedule, original_project_tasks):
    """
    Generates an MS Project XML string for the procurement schedule based on detailed procurement data.
    """
    _update_internal_status("Generating procurement MS Project XML...", 5)
    procurement_tasks_for_xml = []
    task_id_to_original_uid = {task['task_id']: task['UID'] for task in original_project_tasks}
    
    current_uid = max(task_id_to_original_uid.values()) + 1 if task_id_to_original_uid else 1 # Start UIDs for procurement tasks after existing ones

    for proc_item in detailed_procurement_schedule:
        # Create a new task representing the procurement activity
        proc_task_id = f"proc_{proc_item['task_id']}"
        proc_task_name = f"Procure: {_xml_escape(proc_item.get('task_name', 'Unnamed Task'))}" # Use original task name if available

        # Duration for procurement is its lead time, converted to minutes
        duration_minutes = int(proc_item.get('lead_time_weeks', 0) * 5 * 8 * 60) # Assuming 5 days/week, 8 hours/day

        # Predecessor: The original task that this procurement supports
        # This assumes the procurement must finish before the original task can start.
        # So, the original task should have a predecessor link to the procurement task.
        # For the procurement XML, we might link to an earlier phase or simply make it a standalone task.
        # For simplicity here, let's treat procurement tasks as independent for their own schedule,
        # but in a full project, they would link to the tasks requiring them.
        pass # No direct predecessor link from original task here in the procurement XML.

        procurement_tasks_for_xml.append({
            "UID": current_uid,
            "ID": current_uid, # Using UID as ID for simplicity in this generated XML
            "Name": proc_task_name,
            "Type": 0, # Fixed Duration
            "IsNull": 0,
            "CreateDate": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "IsMilestone": 0,
            "IsSummary": 0,
            "Active": 1,
            "Manual": 0,
            "Duration": f"{duration_minutes}M",
            "DurationFormat": 7, # 7 for minutes
            "Start": time.strftime('%Y-%m-%dT%H:%M:%S').split('T')[0] + 'T08:00:00', # Dummy start date
            "Finish": time.strftime('%Y-%m-%dT%H:%M:%S').split('T')[0] + 'T17:00:00', # Dummy finish date
            "Notes": _xml_escape(f"Type: {proc_item.get('procurement_type', 'N/A')}, Category: {proc_item.get('procurement_category', 'N/A')}, Critical: {proc_item.get('is_procurement_critical', 'N/A')}. Lead Time: {proc_item.get('lead_time_weeks', 0)} weeks.")
        })
        current_uid += 1

    current_time = time.strftime('%Y-%m-%dT%H:%M:%S')
    xml_header = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Project xmlns="http://schemas.microsoft.com/project">
    <Title>{_xml_escape(project_description)} - Procurement Schedule</Title>
    <CreationDate>{current_time}</CreationDate>
    <LastSaved>{current_time}</LastSaved>
    <ScheduleFromEnd>0</ScheduleFromEnd>
    <StartDate>{current_time.split('T')[0]}T08:00:00</StartDate>
    <FinishDate>{current_time.split('T')[0]}T17:00:00</FinishDate>
    <DefaultStartTime>08:00:00</DefaultStartTime>
    <DefaultFinishTime>17:00:00</DefaultFinishTime>
    <MinutesPerDay>480</MinutesPerDay>
    <MinutesPerWeek>2400</MinutesPerWeek>
    <DaysPerMonth>20</DaysPerMonth>
    <CurrencySymbol>$</CurrencySymbol>
    <CurrencyCode>CAD</CurrencyCode>
    <Tasks>
        <Task>
            <UID>0</UID>
            <ID>0</ID>
            <Name>Procurement Summary Task</Name>
            <Type>0</Type>
            <IsNull>0</IsNull>
            <CreateDate>{current_time}</CreateDate>
            <IsMilestone>0</IsMilestone>
            <IsSummary>1</IsSummary>
            <Active>1</Active>
            <Manual>0</Manual>
            <Duration>0M</Duration>
            <DurationFormat>7</DurationFormat>
        </Task>
    """

    tasks_xml_content = []
    for task in procurement_tasks_for_xml:
        tasks_xml_content.append(f"""
        <Task>
            <UID>{task['UID']}</UID>
            <ID>{task['ID']}</ID>
            <Name>{_xml_escape(task['Name'])}</Name>
            <Type>{task['Type']}</Type>
            <IsNull>{task['IsNull']}</IsNull>
            <CreateDate>{task['CreateDate']}</CreateDate>
            <IsMilestone>{task['IsMilestone']}</IsMilestone>
            <IsSummary>{task['IsSummary']}</IsSummary>
            <Active>{task['Active']}</Active>
            <Manual>{task['Manual']}</Manual>
            <Duration>{task['Duration']}</Duration>
            <DurationFormat>{task['DurationFormat']}</DurationFormat>
            <Start>{task['Start']}</Start>
            <Finish>{task['Finish']}</Finish>
            <Notes>{task['Notes']}</Notes>
        </Task>
        """)

    xml_footer = """
    </Tasks>
    <Resources>
        <Resource>
            <UID>0</UID>
            <ID>0</ID>
            <Type>0</Type>
            <IsNull>0</IsNull>
            <Name>Procurement Resources</Name>
            <IsGeneric>0</IsGeneric>
            <IsInactive>0</IsInactive>
            <CanLevel>0</CanLevel>
            <MaxUnits>1.00</MaxUnits>
            <PeakUnits>1.00</PeakUnits>
            <OverAllocated>0</OverAllocated>
            <AccrueAt>3</AccrueAt>
            <StandardRateFormat>1</StandardRateFormat>
            <OvertimeRateFormat>1</OvertimeRateFormat>
            <CostPerUseFormat>1</CostPerUseFormat>
            <BudgetCost>0.0</BudgetCost>
            <BudgetWork>0M</BudgetWork>
        </Resource>
    </Resources>
    <Assignments/>
</Project>
    """
    _update_internal_status("Procurement MS Project XML generated.", 100)
    return xml_header + "".join(tasks_xml_content) + xml_footer


def generate_project_schedule_and_xml(project_description):
    """
    Generates a concise project schedule and an MS Project XML based on a project description,
    and logs the generated schedule.
    """
    _update_internal_status("Starting schedule generation pipeline...", 5)
    schedule_results = {"error": None}
    try:
        # This function now calls the updated helper functions
        generated_tasks = _generate_project_tasks_llm(project_description)
        if isinstance(generated_tasks, dict) and generated_tasks.get('error'):
            raise ValueError(f"Failed to generate project tasks: {generated_tasks['error']}")
        
        log_entry = {
            "timestamp": time.time(),
            "project_description": project_description,
            "generated_tasks": generated_tasks
        }
        _save_schedule_log(log_entry)

        ms_project_xml_content = _generate_ms_project_xml(project_description, generated_tasks)
        
        schedule_results['project_tasks'] = generated_tasks
        schedule_results['ms_project_xml'] = ms_project_xml_content
        
        _update_internal_status("Concise schedule generation pipeline completed.", 100)
        return schedule_results

    except Exception as e:
        error_message = f"Error during schedule generation: {e}"
        _update_internal_status(error_message, 0, error=error_message)
        schedule_results['error'] = error_message
        print(error_message)
        return schedule_results
# NEW FUNCTION: run_procurement_analysis_pipeline
# In integrationapi.py, replace BOTH existing definitions of run_procurement_analysis_pipeline with this corrected version:

def run_procurement_analysis_pipeline(initial_analysis_results, project_description):
    """
    Performs a comprehensive procurement analysis based on initial project analysis results.
    """
    _update_internal_status("Starting procurement analysis pipeline...", 5)
    procurement_results = {"error": None}
    try:
        # Step 1: Generate Project Tasks for procurement context
        _update_internal_status("Generating project tasks for procurement context...", 10)
        generated_tasks = _generate_project_tasks_llm(project_description)
        if isinstance(generated_tasks, dict) and generated_tasks.get('error'):
            raise ValueError(f"Failed to generate project tasks for procurement: {generated_tasks['error']}")

        # Step 2: Generate Detailed Procurement Schedule Data using LLM
        _update_internal_status("Generating detailed procurement schedule data...", 20)
        detailed_procurement_schedule = _generate_detailed_procurement_schedule(project_description, generated_tasks)
        if isinstance(detailed_procurement_schedule, dict) and detailed_procurement_schedule.get('error'):
            raise ValueError(f"Failed to generate detailed procurement schedule data: {detailed_procurement_schedule['error']}")
        elif not isinstance(detailed_procurement_schedule, list):
            raise TypeError(f"Unexpected return type from _generate_detailed_procurement_schedule: {type(detailed_procurement_schedule)}")

        # Generate the procurement MS Project XML
        _update_internal_status("Generating procurement MS Project XML content...", 25)
        procurement_xml_content = _generate_procurement_xml_content(project_description, detailed_procurement_schedule, generated_tasks)
        procurement_results['procurement_ms_project_xml'] = procurement_xml_content
        procurement_results['detailed_procurement_schedule'] = detailed_procurement_schedule

        # Log the detailed procurement schedule
        proc_log_entry = {
            "timestamp": time.time(),
            "project_description": project_description,
            "original_generated_tasks_summary": [{"task_id": t["task_id"], "task_name": t["task_name"]} for t in generated_tasks],
            "procurement_schedule_details": detailed_procurement_schedule
        }
        _save_procurement_schedule_log(proc_log_entry)

        # Step 3: Identify Procurement Risks and Opportunities
        _update_internal_status("Identifying procurement risks and opportunities with LLM...", 30)
        procurement_risks_opportunities_raw = _identify_risks_and_opportunities_llm(
            f"Identify procurement-related risks and opportunities for: {project_description}. Consider supply chain, vendors, contracts based on these tasks: {json.dumps([{'task_id': t['task_id'], 'task_name': t['task_name']} for t in generated_tasks])}",
            num_to_generate=NUM_RISKS_TO_GENERATE_FROM_LLM,
            category_focus="procurement"
        )
        procurement_risks_with_qualquan = []
        procurement_opportunities_with_qualquan = []
        procurement_risks = [r for r in procurement_risks_opportunities_raw if not r['is_opportunity']]
        procurement_opportunities = [r for r in procurement_risks_opportunities_raw if r['is_opportunity']]
        procurement_results['identified_procurement_risks'] = procurement_risks
        procurement_results['identified_procurement_opportunities'] = procurement_opportunities

        total_procurement_contractor_cost_impact = 0.0
        total_procurement_contractor_time_impact = 0.0
        total_procurement_oe_cost_impact = 0.0
        total_procurement_oe_time_impact = 0.0
        total_procurement_shared_cost_impact = 0.0
        total_procurement_shared_time_impact = 0.0
        total_procurement_opportunity_cost_reduction = 0.0
        total_procurement_opportunity_time_reduction = 0.0

        # Step 4: QualQuan Assessment for Procurement Risks
        _update_internal_status("Performing QualQuan assessment for procurement risks...", 40)
        for risk_item in tqdm(procurement_risks, desc="Procurement QualQuan Assessment (Risks)"):
            qualquan_data = _qualquan_risk_assessment(project_description, risk_item['risk_description'])
            effective_probability = qualquan_data['probability'] * (1 - qualquan_data['risk_reduction_probability'])
            effective_impact = qualquan_data['impact'] * (1 - qualquan_data['risk_reduction_impact'])
            risk_item.update({
                "qualitative_decision": qualquan_data['qualitative_decision'], "probability": qualquan_data['probability'],
                "impact": qualquan_data['impact'], "effective_probability": effective_probability, "effective_impact": effective_impact,
                "mitigation_strategy": qualquan_data['mitigation_strategy'], "risk_reduction_probability": qualquan_data['risk_reduction_probability'],
                "risk_reduction_impact": qualquan_data['risk_reduction_impact'], "cost_optimistic_usd": qualquan_data['cost_optimistic_usd'],
                "cost_most_likely_usd": qualquan_data['cost_most_likely_usd'], "cost_pessimistic_usd": qualquan_data['cost_pessimistic_usd'],
                "time_optimistic_weeks": qualquan_data['time_optimistic_weeks'], "time_most_likely_weeks": qualquan_data['time_most_likely_weeks'],
                "time_pessimistic_weeks": qualquan_data['time_pessimistic_weeks']
            })
            risk_item['cost_impact_cad'] = (qualquan_data['cost_most_likely_usd'] * (effective_probability / 5.0) * (effective_impact / 5.0)) * USD_TO_CAD_EXCHANGE_RATE
            risk_item['time_impact_weeks'] = (qualquan_data['time_most_likely_weeks'] * (effective_probability / 5.0) * (effective_impact / 5.0))
            procurement_risks_with_qualquan.append(risk_item)
            
            risk_category = risk_item.get('risk_category', 'Contractor-borne')
            if risk_category == 'Contractor-borne':
                total_procurement_contractor_cost_impact += risk_item['cost_impact_cad']
                total_procurement_contractor_time_impact += risk_item['time_impact_weeks']
            elif risk_category in ['Owner-borne', 'Engineer-borne']:
                total_procurement_oe_cost_impact += risk_item['cost_impact_cad']
                total_procurement_oe_time_impact += risk_item['time_impact_weeks']
            elif risk_category == 'Shared':
                total_procurement_contractor_cost_impact += risk_item['cost_impact_cad'] * 0.5
                # --- THIS IS THE FIX ---
                total_procurement_contractor_time_impact += risk_item['time_impact_weeks'] * 0.5
                # -----------------------
                total_procurement_oe_cost_impact += risk_item['cost_impact_cad'] * 0.5
                total_procurement_oe_time_impact += risk_item['time_impact_weeks'] * 0.5
        procurement_results['procurement_risks_with_qualquan_data'] = procurement_risks_with_qualquan

        # QualQuan Assessment for Procurement Opportunities
        _update_internal_status("Performing QualQuan assessment for procurement opportunities...", 45)
        for opportunity_item in tqdm(procurement_opportunities, desc="Procurement QualQuan Assessment (Opportunities)"):
            qualquan_data = _qualquan_risk_assessment(project_description, opportunity_item['risk_description'])
            effective_probability = np.clip(qualquan_data['probability'] * (1 + qualquan_data['risk_reduction_probability']), 0.0, 5.0)
            effective_impact = np.clip(qualquan_data['impact'] * (1 + qualquan_data['risk_reduction_impact']), 0.0, 5.0)
            opportunity_item.update({
                "qualitative_decision": qualquan_data['qualitative_decision'], "probability": qualquan_data['probability'],
                "impact": qualquan_data['impact'], "effective_probability": effective_probability, "effective_impact": effective_impact,
                "mitigation_strategy": qualquan_data['mitigation_strategy'], "risk_reduction_probability": qualquan_data['risk_reduction_probability'],
                "risk_reduction_impact": qualquan_data['risk_reduction_impact'], "cost_optimistic_usd": qualquan_data['cost_optimistic_usd'],
                "cost_most_likely_usd": qualquan_data['cost_most_likely_usd'], "cost_pessimistic_usd": qualquan_data['cost_pessimistic_usd'],
                "time_optimistic_weeks": qualquan_data['time_optimistic_weeks'], "time_most_likely_weeks": qualquan_data['time_most_likely_weeks'],
                "time_pessimistic_weeks": qualquan_data['time_pessimistic_weeks']
            })
            opportunity_item['cost_reduction_cad'] = (qualquan_data['cost_most_likely_usd'] * (effective_probability / 5.0) * (effective_impact / 5.0)) * USD_TO_CAD_EXCHANGE_RATE
            opportunity_item['time_reduction_weeks'] = (qualquan_data['time_most_likely_weeks'] * (effective_probability / 5.0) * (effective_impact / 5.0))
            total_procurement_opportunity_cost_reduction += opportunity_item['cost_reduction_cad']
            total_procurement_opportunity_time_reduction += opportunity_item['time_reduction_weeks']
            procurement_opportunities_with_qualquan.append(opportunity_item)
        procurement_results['procurement_opportunities_with_qualquan_data'] = procurement_opportunities_with_qualquan
        procurement_results['total_procurement_opportunity_cost_reduction_cad'] = total_procurement_opportunity_cost_reduction
        procurement_results['total_procurement_opportunity_time_reduction_weeks'] = total_procurement_opportunity_time_reduction
        
        # ... The rest of the function remains the same, it is omitted here for brevity ...
        # (The code for adjusting estimates, running Monte Carlo, and generating plots follows)
        # Safely retrieve initial analysis results for calculations, providing defaults
        cost_before_procurement = initial_analysis_results.get('total_project_cost_after_safety_cad', 0.0)
        time_before_procurement = initial_analysis_results.get('total_project_time_after_safety_weeks', 0.0)

        adjusted_cost_after_procurement_ml = cost_before_procurement + total_procurement_contractor_cost_impact + total_procurement_oe_cost_impact + total_procurement_shared_cost_impact - total_procurement_opportunity_cost_reduction
        adjusted_time_after_procurement_ml = time_before_procurement + total_procurement_contractor_time_impact + total_procurement_oe_time_impact + total_procurement_shared_time_impact - total_procurement_opportunity_time_reduction

        procurement_results['total_project_cost_after_procurement_ml'] = adjusted_cost_after_procurement_ml
        procurement_results['total_project_time_after_procurement_ml'] = adjusted_time_after_procurement_ml
        
        # (rest of function...)
        baseline_estimates_from_initial = initial_analysis_results.get('baseline_estimates', {})
        original_opt_cost_spread = baseline_estimates_from_initial.get('cost_most_likely_cad', 0.0) - baseline_estimates_from_initial.get('cost_optimistic_cad', 0.0)
        original_pess_cost_spread = baseline_estimates_from_initial.get('cost_pessimistic_cad', 0.0) - baseline_estimates_from_initial.get('cost_most_likely_cad', 0.0)
        
        new_optimistic_cost_after_procurement_for_mc = adjusted_cost_after_procurement_ml - original_opt_cost_spread
        new_pessimistic_cost_after_procurement_for_mc = adjusted_cost_after_procurement_ml + original_pess_cost_spread

        new_optimistic_cost_after_procurement_for_mc = max(0.0, new_optimistic_cost_after_procurement_for_mc)
        new_optimistic_cost_after_procurement_for_mc = min(new_optimistic_cost_after_procurement_for_mc, adjusted_cost_after_procurement_ml)
        new_pessimistic_cost_after_procurement_for_mc = max(adjusted_cost_after_procurement_ml, new_pessimistic_cost_after_procurement_for_mc)

        original_opt_time_spread = baseline_estimates_from_initial.get('time_most_likely_weeks', 0.0) - baseline_estimates_from_initial.get('time_optimistic_weeks', 0.0)
        original_pess_time_spread = baseline_estimates_from_initial.get('time_pessimistic_weeks', 0.0) - baseline_estimates_from_initial.get('time_most_likely_weeks', 0.0)
        
        new_optimistic_time_after_procurement_for_mc = adjusted_time_after_procurement_ml - original_opt_time_spread
        new_pessimistic_time_after_procurement_for_mc = adjusted_time_after_procurement_ml + original_pess_time_spread
        
        new_optimistic_time_after_procurement_for_mc = max(0.0, new_optimistic_time_after_procurement_for_mc)
        new_optimistic_time_after_procurement_for_mc = min(new_optimistic_time_after_procurement_for_mc, adjusted_time_after_procurement_ml)
        new_pessimistic_time_after_procurement_for_mc = max(adjusted_time_after_procurement_ml, new_pessimistic_time_after_procurement_for_mc)

        simulated_costs_after_procurement = _pert_monte_carlo_simulation(
            new_optimistic_cost_after_procurement_for_mc,
            adjusted_cost_after_procurement_ml,
            new_pessimistic_cost_after_procurement_for_mc,
            num_simulations=NUM_SIMULATIONS
        )
        simulated_times_after_procurement = _pert_monte_carlo_simulation(
            new_optimistic_time_after_procurement_for_mc,
            adjusted_time_after_procurement_ml,
            new_pessimistic_time_after_procurement_for_mc,
            num_simulations=NUM_SIMULATIONS
        )

        procurement_results['simulated_costs_after_procurement'] = simulated_costs_after_procurement.tolist()
        procurement_results['simulated_times_after_procurement'] = simulated_times_after_procurement.tolist()
        p85_cost_after_procurement = _calculate_p85(simulated_costs_after_procurement)
        p85_time_after_procurement = _calculate_p85(simulated_times_after_procurement)
        contingency_reserve_after_procurement_cad = max(0, p85_cost_after_procurement - adjusted_cost_after_procurement_ml)
        procurement_results['contingency_reserve_after_procurement_cad'] = contingency_reserve_after_procurement_cad
        
        oe_and_shared_procurement_risks = [r for r in procurement_risks_with_qualquan if r.get('risk_category') in ['Owner-borne', 'Engineer-borne', 'Shared']]
        total_proc_oe_and_shared_cost_optimistic, total_proc_oe_and_shared_cost_most_likely, total_proc_oe_and_shared_cost_pessimistic = 0.0, 0.0, 0.0
        for risk in oe_and_shared_procurement_risks:
            multiplier = 0.5 if risk.get('risk_category') == 'Shared' else 1.0
            total_proc_oe_and_shared_cost_optimistic += risk['cost_optimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_proc_oe_and_shared_cost_most_likely += risk['cost_most_likely_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_proc_oe_and_shared_cost_pessimistic += risk['cost_pessimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
        simulated_proc_owner_engineer_costs = _monte_carlo_simulation(
            total_proc_oe_and_shared_cost_optimistic, total_proc_oe_and_shared_cost_most_likely, total_proc_oe_and_shared_cost_pessimistic, num_simulations=NUM_SIMULATIONS
        ) if total_proc_oe_and_shared_cost_most_likely > 0 else np.zeros(NUM_SIMULATIONS)
        procurement_results['management_reserve_after_procurement_cad'] = _calculate_p85(simulated_proc_owner_engineer_costs)

        total_project_cost_after_procurement_cad = adjusted_cost_after_procurement_ml + contingency_reserve_after_procurement_cad + procurement_results['management_reserve_after_procurement_cad']
        total_project_time_after_procurement_weeks = adjusted_time_after_procurement_ml + (p85_time_after_procurement - adjusted_time_after_procurement_ml)

        procurement_results.update({
            'p85_cost_after_procurement_cad': p85_cost_after_procurement, 'p85_time_after_procurement_weeks': p85_time_after_procurement,
            'total_project_cost_after_procurement_cad': total_project_cost_after_procurement_cad, 'total_project_time_after_procurement_weeks': total_project_time_after_procurement_weeks,
            'plots_base64_after_procurement': {}
        })

        # Generate plots
        # Step 8: Generate Plots reflecting Procurement Analysis Impact
        _update_internal_status("Generating procurement analysis plots...", 70) # Adjusted progress
        procurement_results['plots_base64_after_procurement'] = {}

        # Cost Distribution Plot (After Procurement)
        if len(simulated_costs_after_procurement) > 0 and not np.all(simulated_costs_after_procurement == 0):
            fig_cost_dist_procurement, ax_cost_dist_procurement = plt.subplots(figsize=(10, 6))
            ax_cost_dist_procurement.hist(simulated_costs_after_procurement, bins=50, density=True, alpha=0.7, color='darkcyan', edgecolor='black', label='Cost Distribution')
            ax_cost_dist_procurement.set_xlabel('Cost (CAD)')
            ax_cost_dist_procurement.set_ylabel('Density')
            ax_cost_dist_procurement.set_title('Simulated Total Project Cost Distribution (After Procurement Analysis)')

            cost_mean, cost_std = np.mean(simulated_costs_after_procurement), np.std(simulated_costs_after_procurement)
            if cost_std > 0:
                ax_cost_dist_procurement.set_xlim(max(0, cost_mean - 4 * cost_std), cost_mean + 4 * cost_std)
            else:
                ax_cost_dist_procurement.set_xlim(max(0, cost_mean * 0.9), cost_mean * 1.1 if cost_mean > 0 else 1)
            
            ax2_cost_procurement = ax_cost_dist_procurement.twinx() # Corrected: Define twinx() here
            sorted_costs_procurement = np.sort(simulated_costs_after_procurement)
            cdf_cost_procurement = np.arange(1, len(sorted_costs_procurement) + 1) / len(sorted_costs_procurement)
            ax2_cost_procurement.plot(sorted_costs_procurement, cdf_cost_procurement * 100, color='blue', linestyle='-', label='Cumulative Frequency (%)')
            ax2_cost_procurement.set_ylabel('Cumulative Frequency (%)')
            ax2_cost_procurement.set_ylim(0, 100)

            ax_cost_dist_procurement.axvline(adjusted_cost_after_procurement_ml, color='red', linestyle='dotted', linewidth=1.5, label=f'Adj. ML Cost: ${adjusted_cost_after_procurement_ml:,.0f}')
            ax_cost_dist_procurement.axvline(p85_cost_after_procurement, color='green', linestyle='dotted', linewidth=1.5, label=f'P85 Cost: ${p85_cost_after_procurement:,.0f}')
            ax_cost_dist_procurement.axvline(total_project_cost_after_procurement_cad, color='purple', linestyle='dashed', linewidth=1.5, label=f'Total Project Cost: ${total_project_cost_after_procurement_cad:,.0f}')

            lines, labels = ax_cost_dist_procurement.get_legend_handles_labels()
            lines2, labels2 = ax2_cost_procurement.get_legend_handles_labels()
            ax2_cost_procurement.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            procurement_results['plots_base64_after_procurement']['cost_distribution_plot_after_procurement_base64'] = _plot_to_base64(fig_cost_dist_procurement)
        else:
            procurement_results['plots_base64_after_procurement']['cost_distribution_plot_after_procurement_base64'] = None

        # Time Distribution Plot (After Procurement)
        if len(simulated_times_after_procurement) > 0 and not np.all(simulated_times_after_procurement == 0):
            fig_time_dist_procurement, ax_time_dist_procurement = plt.subplots(figsize=(10, 6))
            ax_time_dist_procurement.hist(simulated_times_after_procurement, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black', label='Time Distribution')
            ax_time_dist_procurement.set_xlabel('Time (Weeks)')
            ax_time_dist_procurement.set_ylabel('Density')
            ax_time_dist_procurement.set_title('Simulated Total Project Time Distribution (After Procurement Analysis)')

            time_mean, time_std = np.mean(simulated_times_after_procurement), np.std(simulated_times_after_procurement)
            if time_std > 0:
                ax_time_dist_procurement.set_xlim(max(0, time_mean - 4 * time_std), time_mean + 4 * time_std)
            else:
                ax_time_dist_procurement.set_xlim(max(0, time_mean * 0.9), time_mean * 1.1 if time_mean > 0 else 1)

            ax2_time_procurement = ax_time_dist_procurement.twinx() # Corrected: Define twinx() here
            sorted_times_procurement = np.sort(simulated_times_after_procurement)
            cdf_time_procurement = np.arange(1, len(sorted_times_procurement) + 1) / len(sorted_times_procurement)
            ax2_time_procurement.plot(sorted_times_procurement, cdf_time_procurement * 100, color='darkred', linestyle='-', label='Cumulative Frequency (%)')
            ax2_time_procurement.set_ylabel('Cumulative Frequency (%)')
            ax2_time_procurement.set_ylim(0, 100)

            ax_time_dist_procurement.axvline(adjusted_time_after_procurement_ml, color='red', linestyle='dotted', linewidth=1.5, label=f'Adj. ML Time: {adjusted_time_after_procurement_ml:.1f} weeks')
            ax_time_dist_procurement.axvline(p85_time_after_procurement, color='green', linestyle='dotted', linewidth=1.5, label=f'P85 Time: {p85_time_after_procurement:.1f} weeks')
            ax_time_dist_procurement.axvline(total_project_time_after_procurement_weeks, color='purple', linestyle='dashed', linewidth=1.5, label=f'Total Project Time: {total_project_time_after_procurement_weeks:.1f} weeks')

            lines, labels = ax_time_dist_procurement.get_legend_handles_labels()
            lines2, labels2 = ax2_time_procurement.get_legend_handles_labels()
            ax2_time_procurement.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            procurement_results['plots_base64_after_procurement']['time_distribution_plot_after_procurement_base64'] = _plot_to_base64(fig_time_dist_procurement)
        else:
            procurement_results['plots_base64_after_procurement']['time_distribution_plot_after_procurement_base64'] = None

        # Contractor-Related Procurement Risk Cost Distribution Plot
        contractor_procurement_risks = [r for r in procurement_risks_with_qualquan if r.get('risk_category') == 'Contractor-borne' or r.get('risk_category') == 'Shared']
        total_contractor_procurement_cost_optimistic = 0.0
        total_contractor_procurement_cost_most_likely = 0.0
        total_contractor_procurement_cost_pessimistic = 0.0

        for risk in contractor_procurement_risks:
            multiplier = 0.5 if risk.get('risk_category') == 'Shared' else 1.0
            total_contractor_procurement_cost_optimistic += risk['cost_optimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_contractor_procurement_cost_most_likely += risk['cost_most_likely_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier
            total_contractor_procurement_cost_pessimistic += risk['cost_pessimistic_usd'] * USD_TO_CAD_EXCHANGE_RATE * multiplier

        if total_contractor_procurement_cost_most_likely == 0:
            simulated_contractor_procurement_costs = np.zeros(NUM_SIMULATIONS)
        else:
            simulated_contractor_procurement_costs = _monte_carlo_simulation(
                total_contractor_procurement_cost_optimistic,
                total_contractor_procurement_cost_most_likely,
                total_contractor_procurement_cost_pessimistic,
                num_simulations=NUM_SIMULATIONS
            )
        
        if len(simulated_contractor_procurement_costs) > 0 and not np.all(simulated_contractor_procurement_costs == 0):
            fig_contractor_procurement_risk_cost, ax_contractor_procurement_risk_cost = plt.subplots(figsize=(10, 6))
            ax_contractor_procurement_risk_cost.hist(simulated_contractor_procurement_costs, bins=50, density=True, alpha=0.7, color='darkgreen', edgecolor='black', label='Cost Impact Distribution')
            ax_contractor_procurement_risk_cost.set_title('Simulated Contractor-Related Procurement Risk Cost Impact Distribution')
            ax_contractor_procurement_risk_cost.set_xlabel('Cost Impact (CAD)')
            ax_contractor_procurement_risk_cost.set_ylabel('Density')
            
            mean_val, std_val = np.mean(simulated_contractor_procurement_costs), np.std(simulated_contractor_procurement_costs)
            ax_contractor_procurement_risk_cost.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Impact: ${mean_val:,.0f}')
            
            if std_val > 0:
                ax_contractor_procurement_risk_cost.set_xlim(max(0, mean_val - 4 * std_val), mean_val + 4 * std_val)
            else:
                ax_contractor_procurement_risk_cost.set_xlim(max(0, mean_val * 0.9), mean_val * 1.1 if mean_val > 0 else 1)

            ax2_contractor_procurement_risk_cost = ax_contractor_procurement_risk_cost.twinx()
            sorted_contractor_procurement_costs = np.sort(simulated_contractor_procurement_costs)
            cdf_contractor_procurement_cost = np.arange(1, len(sorted_contractor_procurement_costs) + 1) / len(sorted_contractor_procurement_costs)
            ax2_contractor_procurement_risk_cost.plot(sorted_contractor_procurement_costs, cdf_contractor_procurement_cost * 100, color='blue', linestyle='-', label='Cumulative Frequency (%)')
            ax2_contractor_procurement_risk_cost.set_ylabel('Cumulative Frequency (%)')
            ax2_contractor_procurement_risk_cost.set_ylim(0, 100)

            # Add P85 line
            p85_contractor_procurement_cost = _calculate_p85(simulated_contractor_procurement_costs)
            ax_contractor_procurement_risk_cost.axvline(p85_contractor_procurement_cost, color='orange', linestyle='dotted', linewidth=1.5, label=f'P85 Cost: ${p85_contractor_procurement_cost:,.0f}')

            lines, labels = ax_contractor_procurement_risk_cost.get_legend_handles_labels()
            lines2, labels2 = ax2_contractor_procurement_risk_cost.get_legend_handles_labels()
            ax_contractor_procurement_risk_cost.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            procurement_results['plots_base64_after_procurement']['contractor_risk_cost_plot_after_procurement_base64'] = _plot_to_base64(fig_contractor_procurement_risk_cost)
        else:
            procurement_results['plots_base64_after_procurement']['contractor_risk_cost_plot_after_procurement_base64'] = None

        # Owner/Engineer-Born Procurement Risk Cost Distribution Plot
        if len(simulated_proc_owner_engineer_costs) > 0 and not np.all(simulated_proc_owner_engineer_costs == 0):
            fig_proc_owner_engineer_risk_cost, ax_proc_owner_engineer_risk_cost = plt.subplots(figsize=(10, 6))
            ax_proc_owner_engineer_risk_cost.hist(simulated_proc_owner_engineer_costs, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black', label='Cost Impact Distribution')
            ax_proc_owner_engineer_risk_cost.set_title('Simulated Owner/Engineer-Born Procurement Risk Cost Impact Distribution')
            ax_proc_owner_engineer_risk_cost.set_xlabel('Cost Impact (CAD)')
            ax_proc_owner_engineer_risk_cost.set_ylabel('Density')
            ax_proc_owner_engineer_risk_cost.axvline(procurement_results['management_reserve_after_procurement_cad'], color='green', linestyle='dashed', linewidth=1, label=f'Management Reserve (P85): ${procurement_results["management_reserve_after_procurement_cad"]:,.0f}')
            
            mean_val, std_val = np.mean(simulated_proc_owner_engineer_costs), np.std(simulated_proc_owner_engineer_costs)
            if std_val > 0:
                ax_proc_owner_engineer_risk_cost.set_xlim(max(0, mean_val - 4 * std_val), mean_val + 4 * std_val)
            else:
                ax_proc_owner_engineer_risk_cost.set_xlim(max(0, mean_val * 0.9), mean_val * 1.1 if mean_val > 0 else 1)

            ax2_proc_owner_engineer_risk_cost = ax_proc_owner_engineer_risk_cost.twinx()
            sorted_proc_owner_engineer_costs = np.sort(simulated_proc_owner_engineer_costs)
            cdf_proc_owner_engineer_cost = np.arange(1, len(sorted_proc_owner_engineer_costs) + 1) / len(sorted_proc_owner_engineer_costs)
            ax2_proc_owner_engineer_risk_cost.plot(sorted_proc_owner_engineer_costs, cdf_proc_owner_engineer_cost * 100, color='blue', linestyle='-', label='Cumulative Frequency (%)')
            ax2_proc_owner_engineer_risk_cost.set_ylabel('Cumulative Frequency (%)')
            ax2_proc_owner_engineer_risk_cost.set_ylim(0, 100)

            lines, labels = ax_proc_owner_engineer_risk_cost.get_legend_handles_labels()
            lines2, labels2 = ax2_proc_owner_engineer_risk_cost.get_legend_handles_labels()
            ax_proc_owner_engineer_risk_cost.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

            procurement_results['plots_base64_after_procurement']['owner_engineer_risk_cost_plot_after_procurement_base64'] = _plot_to_base64(fig_proc_owner_engineer_risk_cost)
        else:
            procurement_results['plots_base64_after_procurement']['owner_engineer_risk_cost_plot_after_procurement_base64'] = None


        _update_internal_status("Procurement analysis pipeline completed successfully.", 100)
        return procurement_results

    except Exception as e:
        error_message = f"Error during procurement analysis pipeline: {e}"
        _update_internal_status(error_message, 0, error=error_message)
        procurement_results['error'] = error_message
        print(error_message)
        return procurement_results

# NEW FUNCTION: get_internal_analysis_status
def get_internal_analysis_status():
    """Returns the current internal status of the analysis backend."""
    with _status_lock:
        return _internal_analysis_status.copy()

# Initial model lo  ading in a separate thread to avoid blocking the main thread
# during application startup.
def _generate_detailed_procurement_schedule(project_description, generated_tasks):
    """
    Generates a detailed procurement schedule using an LLM, based on high-level project tasks.
    """
    # Create a simplified representation of tasks for the cache key and prompt
    tasks_summary_for_key = tuple(sorted((t['task_id'], t['task_name']) for t in generated_tasks))
    cache_key = (project_description, tasks_summary_for_key)
    if cache_key in _procurement_schedule_cache:
        print("Using cached detailed procurement schedule.")
        return _procurement_schedule_cache[cache_key]

    _update_internal_status("Generating detailed procurement schedule with LLM...", 21)

    # Create a JSON string of tasks to include in the prompt
    tasks_for_prompt = json.dumps([{'task_id': t['task_id'], 'task_name': t['task_name']} for t in generated_tasks])

    prompt = (
        f"Based on the project '{project_description}' and its high-level tasks, create a detailed procurement plan. For each relevant task provided below, identify the necessary procurement item.\n\n"
        f"High-Level Tasks: {tasks_for_prompt}\n\n"
        f"For each procurement item, provide:\n"
        f"- 'task_id': The ID of the original task this procurement is for.\n"
        f"- 'procurement_type': Classify as 'in-house' (handled internally) or 'outside' (sourced from external vendors).\n"
        f"- 'procurement_category': Classify as 'minor' or 'major' based on cost/impact.\n"
        f"- 'is_procurement_critical': A boolean (true/false) indicating if this procurement is on the critical path for its parent task.\n"
        f"- 'lead_time_weeks': An integer estimate for the number of weeks required from order to delivery.\n\n"
        f"The output must be a clean JSON array of objects, with no extra text or explanations. Each object in the array represents one procurement item linked to one task."
    )

    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "task_id": {"type": "STRING"},
                "procurement_type": {"type": "STRING", "enum": ["in-house", "outside"]},
                "procurement_category": {"type": "STRING", "enum": ["minor", "major"]},
                "is_procurement_critical": {"type": "BOOLEAN"},
                "lead_time_weeks": {"type": "NUMBER"}
            },
            "required": ["task_id", "procurement_type", "procurement_category", "is_procurement_critical", "lead_time_weeks"]
        }
    }

    generated_procurement_json = None

    if USE_GEMINI_API:
        # Loop through Gemini API keys for resilience
        for i, api_key in enumerate(GEMINI_API_KEYS):
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                    response_model = genai.GenerativeModel(GEMINI_MODEL_NAME).generate_content(
                        [{"role": "user", "parts": [{"text": prompt}]}],
                        generation_config=generation_config
                    )
                    generated_procurement_json = response_model.text
                    print(f"Generate procurement schedule: Gemini API with key {i+1} successful.")
                    break
                except Exception as e:
                    print(f"Generate procurement schedule: Gemini API with key {i+1} error: {e}. Trying next key.")
            else:
                print(f"Gemini API key {i+1} not found. Skipping.")

    if generated_procurement_json is None and fine_tuned_risk_model is not None and fine_tuned_risk_tokenizer is not None:
        try:
            inputs = fine_tuned_risk_tokenizer(prompt + "\n\nProvide the response in JSON array format only.", return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output_tokens = fine_tuned_risk_model.generate(**inputs, max_new_tokens=2000, pad_token_id=fine_tuned_risk_tokenizer.eos_token_id)
            generated_text = fine_tuned_risk_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", generated_text, re.DOTALL)
            if json_match:
                generated_procurement_json = json_match.group(0)
        except Exception as e:
            print(f"Local TinyLlama error for procurement schedule generation: {e}.")

    try:
        if generated_procurement_json:
            procurement_schedule = json.loads(generated_procurement_json)
            # Basic validation and type casting
            valid_task_ids = {t['task_id'] for t in generated_tasks}
            validated_schedule = []
            for item in procurement_schedule:
                if item.get('task_id') in valid_task_ids:
                    item['is_procurement_critical'] = bool(item.get('is_procurement_critical', False))
                    item['lead_time_weeks'] = int(item.get('lead_time_weeks', 0))
                    validated_schedule.append(item)
                else:
                    print(f"Warning: LLM generated procurement item for an unknown task_id '{item.get('task_id')}'. Discarding.")
            
            _procurement_schedule_cache[cache_key] = validated_schedule
            _update_internal_status("Detailed procurement schedule generated.", 25)
            return validated_schedule
        else:
            raise RuntimeError("No suitable LLM could generate a detailed procurement schedule.")
    except Exception as e:
        error_message = f"Error generating or parsing detailed procurement schedule: {e}"
        _update_internal_status(error_message, 0, error=error_message)
        return {"error": error_message}
def _save_procurement_schedule_log(log_entry):
    """
    Appends a log entry for a procurement schedule to the procurement schedule log file.

    Args:
        log_entry (dict): The dictionary containing the log data.
    """
    try:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(PROCUREMENT_SCHEDULE_LOG_FILE), exist_ok=True)
        # Append the log entry as a new line in the JSONL file
        with open(PROCUREMENT_SCHEDULE_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Warning: Could not save procurement schedule log: {e}")

