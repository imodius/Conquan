# streamlit_app.py - Streamlit application for Project Risk Analysis and AI Chat
# This frontend communicates with a Flask backend.

import streamlit as st
import requests # For making HTTP requests to the Flask backend
import json
import time # For simulation of loading time if needed
import uuid # For unique keys for chat messages
import matplotlib.pyplot as plt # Import matplotlib for plotting
import numpy as np # For numerical operations in plotting
import pandas as pd # For displaying tasks in a structured table
import base64 # For decoding images from backend

# --- Configuration ---
FLASK_BACKEND_URL = "http://127.0.0.1:5000" # Ensure this matches your Flask app's host and port

# Configure Streamlit page settings
st.set_page_config(
    page_title="Construction Risk Analyzer",
    page_icon="📊",
    layout="wide", # Use wide layout for better display of results
    initial_sidebar_state="expanded"
)
# --- Place this right after st.set_page_config ---
# --- ADD THIS FOR PDF PRINTING ---
# --- Place this right after st.set_page_config ---
# --- Global variables for session state management ---
if 'debug_eigen_matrix' not in st.session_state:
    st.session_state.debug_eigen_matrix = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'load_status_message' not in st.session_state:
    st.session_state.load_status_message = "Models not yet loaded."
if 'load_error' not in st.session_state:
    st.session_state.load_error = None
if 'initial_analysis_complete' not in st.session_state: # Track if initial analysis is done
    st.session_state.initial_analysis_complete = False
if 'project_description_for_safety' not in st.session_state: # Store description for safety analysis
    st.session_state.project_description_for_safety = ""
if 'safety_analysis_results' not in st.session_state: # Store safety analysis results
    st.session_state.safety_analysis_results = None
# ADDED: State variables for generated tasks and XML content
if 'generated_tasks' not in st.session_state:
    st.session_state.generated_tasks = []
if 'generated_xml_content' not in st.session_state:
    st.session_state.generated_xml_content = None
# NEW: State variable for procurement analysis results
if 'procurement_analysis_results' not in st.session_state:
    st.session_state.procurement_analysis_results = None
# NEW: State variable for procurement XML content
if 'procurement_ms_project_xml' not in st.session_state:
    st.session_state.procurement_ms_project_xml = None
# NEW: State variable for backend comparison results
if 'backend_comparison_results' not in st.session_state:
    st.session_state.backend_comparison_results = None


# --- Backend Communication Functions ---

def get_backend_status():
    """Fetches the current status from the Flask backend."""
    try:
        response = requests.get(f"{FLASK_BACKEND_URL}/status")
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to Flask backend. Is it running?"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred while fetching status: {e}"}

def load_models_on_backend():
    """Triggers model loading on the Flask backend and polls for completion."""
    st.session_state.load_status_message = "Initiating model load..."
    # Placeholder for loading animation
    status_text_placeholder = st.empty()
    status_text_placeholder.info(st.session_state.load_status_message)
    progress_bar_placeholder = st.progress(0)

    try:
        # First, send the request to trigger loading
        response = requests.post(f"{FLASK_BACKEND_URL}/load_models")
        response.raise_for_status()
        
        # Now, poll for status updates
        while True:
            status_data = get_backend_status()
            message = status_data.get("message", "Loading...")
            progress = status_data.get("progress", 0)
            models_loaded = status_data.get("models_loaded", False)

            st.session_state.load_status_message = message
            status_text_placeholder.info(f"Model Loading Status: {message}")
            progress_bar_placeholder.progress(progress)

            if models_loaded:
                st.session_state.models_loaded = True
                status_text_placeholder.success("Models loaded successfully on backend!")
                progress_bar_placeholder.empty() # Remove progress bar
                break
            elif "error" in status_data.get("message", "").lower():
                st.session_state.load_error = message
                status_text_placeholder.error(f"Model loading failed: {message}")
                progress_bar_placeholder.empty()
                break
            time.sleep(1) # Wait for 1 second before polling again

    except requests.exceptions.ConnectionError:
        st.session_state.load_error = f"Could not connect to Flask backend at {FLASK_BACKEND_URL}. Please ensure it's running."
        st.session_state.load_status_message = "Connection Error"
        status_text_placeholder.error(st.session_state.load_error)
        progress_bar_placeholder.empty()
    except requests.exceptions.RequestException as e:
        st.session_state.load_error = f"Failed to initiate model loading on backend: {e}"
        st.session_state.load_status_message = f"Loading Error: {e}"
        status_text_placeholder.error(st.session_state.load_error)
        progress_bar_placeholder.empty()
    except Exception as e:
        st.session_state.load_error = f"An unexpected error occurred during model loading: {e}"
        st.session_state.load_status_message = "Unknown Error"
        status_text_placeholder.error(st.session_state.load_error)
        progress_bar_placeholder.empty()

def display_eigenmatrix_diagonal(matrix_data):
    """
    Renders the 16x16 matrix with lambda values along the diagonal.
    """
    if not matrix_data:
        return
    
    st.subheader("🔮 Quantum Eigen-Analysis")
    st.info("The diagonal elements (λ) represent the primary risk state probabilities.")
    
    matrix_np = np.array(matrix_data)
    # Extract just the diagonal (eigenvalues/λ)
    diagonal = np.diag(matrix_np)
    
    # Create a clean display for the diagonal elements
    diag_df = pd.DataFrame({
        "State Index (i)": [f"λ_{i}" for i in range(len(diagonal))],
        "Eigenvalue (Probability)": diagonal.real
    })
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Diagonal Values ($\lambda_i$):**")
        st.dataframe(diag_df, hide_index=True)
        
    with col2:
        st.write("**Full 16x16 Matrix View:**")
        # Displaying the full matrix but focusing on the diagonal for the user
        st.dataframe(pd.DataFrame(matrix_np).style.format("{:.4f}").highlight_between(left=0.0001, color="#f0f2f6", axis=None))
        
def analyze_project_via_backend(project_description):
    """Sends project description to Flask backend for analysis."""
    st.session_state.initial_analysis_complete = False # Reset state before new analysis
    st.session_state.safety_analysis_results = None # Clear safety results
    st.session_state.procurement_analysis_results = None # Clear procurement results
    st.session_state.procurement_ms_project_xml = None # Clear procurement XML
    st.session_state.analysis_results = None # Clear previous initial results
    # ADDED: Clear generated tasks and XML content on new analysis
    st.session_state.generated_tasks = []
    st.session_state.generated_xml_content = None

    try:
        headers = {'Content-Type': 'application/json'}
        data = {'project_description': project_description}
        response = requests.post(f"{FLASK_BACKEND_URL}/analyze_project", headers=headers, json=data)
        response.raise_for_status()
        
        results = response.json()
        st.session_state.analysis_results = results
        st.session_state.project_description_for_safety = project_description # Store description for safety analysis
        st.session_state.initial_analysis_complete = True # Mark initial analysis as complete
        if 'debug_eigen_matrix' in results:
            st.session_state.debug_eigen_matrix = results['debug_eigen_matrix']
        return results # Return results for potential display or further processing
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Flask backend for analysis. Is it running?")
        return {"error": "Could not connect to Flask backend."}
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get('error', e.response.text)
        st.error(f"Analysis HTTP Error: {e.response.status_code} - {error_detail}")
        return {"error": f"Analysis HTTP Error: {e.response.status_code} - {error_detail}"}
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {e}")
        return {"error": f"An unexpected error occurred during analysis: {e}"}

def perform_safety_analysis_via_backend():
    """Sends initial analysis results and project description to Flask backend for safety analysis."""
    if not st.session_state.initial_analysis_complete or st.session_state.analysis_results is None:
        st.warning("Initial project analysis must be completed first.")
        return

    st.session_state.safety_analysis_results = None # Clear previous safety results
    
    try:
        headers = {'Content-Type': 'application/json'}
        data = {
            'initial_analysis_results': st.session_state.analysis_results, # Pass the entire initial results
            'project_description': st.session_state.project_description_for_safety # Use stored description
        }
        # Corrected endpoint name to match backend: /perform_safety_analysis
        response = requests.post(f"{FLASK_BACKEND_URL}/perform_safety_analysis", headers=headers, json=data)
        response.raise_for_status()
        
        results = response.json()
        st.session_state.safety_analysis_results = results
        st.success("Safety Analysis Complete!")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Flask backend for safety analysis. Is it running?")
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get('error', e.response.text)
        st.error(f"Safety Analysis HTTP Error: {e.response.status_code} - {error_detail}")
    except Exception as e:
        st.error(f"An unexpected error occurred during safety analysis: {e}")

# ADDED: Function to generate schedule XML via backend
def generate_schedule_xml_via_backend(project_description):
    """Generates project tasks and MS Project XML via the Flask backend."""
    st.session_state.generated_tasks = []
    st.session_state.generated_xml_content = None
    st.session_state.procurement_analysis_results = None # Clear procurement results
    st.session_state.procurement_ms_project_xml = None # Clear procurement XML
    try:
        headers = {'Content-Type': 'application/json'}
        data = {'project_description': project_description}
        response = requests.post(f"{FLASK_BACKEND_URL}/generate_schedule_xml", headers=headers, json=data)
        response.raise_for_status()
        
        results = response.json()
        st.session_state.generated_tasks = results.get('project_tasks', []) # Corrected key
        st.session_state.generated_xml_content = results.get('ms_project_xml') # Corrected key
        st.success("Schedule XML Generation Complete!")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Flask backend for schedule generation. Is it running?")
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get('error', e.response.text)
        st.error(f"Schedule Generation HTTP Error: {e.response.status_code} - {error_detail}")
    except Exception as e:
        st.error(f"An unexpected error occurred during schedule generation: {e}")

# NEW FUNCTION: perform_procurement_analysis_via_backend
def perform_procurement_analysis_via_backend():
    """Sends initial analysis results and project description to Flask backend for procurement analysis."""
    if not st.session_state.initial_analysis_complete or st.session_state.analysis_results is None:
        st.warning("Initial project analysis must be completed first.")
        return
    if not st.session_state.generated_tasks:
        st.warning("Project schedule tasks must be generated first.")
        return

    st.session_state.procurement_analysis_results = None # Clear previous procurement results
    st.session_state.procurement_ms_project_xml = None # Clear previous procurement XML

    try:
        headers = {'Content-Type': 'application/json'}
        data = {
            'initial_analysis_results': st.session_state.analysis_results,
            'project_description': st.session_state.project_description_for_safety # Reuse stored description
        }
        response = requests.post(f"{FLASK_BACKEND_URL}/procurement_analysis", headers=headers, json=data)
        response.raise_for_status()
        
        results = response.json()
        st.session_state.procurement_analysis_results = results
        st.session_state.procurement_ms_project_xml = results.get('procurement_ms_project_xml') # Store procurement XML
        st.success("Procurement Analysis Complete!")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Flask backend for procurement analysis. Is it running?")
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get('error', e.response.text)
        st.error(f"Procurement Analysis HTTP Error: {e.response.status_code} - {error_detail}")
    except Exception as e:
        st.error(f"An unexpected error occurred during procurement analysis: {e}")
# --- Place this near your procurement schedule results ---

def compare_backends_via_backend(input_data):
    """Sends data to Flask backend to compare real vs. simulated quantum backends."""
    st.session_state.backend_comparison_results = None # Clear previous results
    try:
        headers = {'Content-Type': 'application/json'}
        data = {'data': input_data}
        response = requests.post(f"{FLASK_BACKEND_URL}/compare_backends", headers=headers, json=data)
        
        # We handle both success (200) and potential errors (like 500) that still return a body
        st.session_state.backend_comparison_results = response.json()
        if response.status_code == 200:
            st.success("Backend comparison complete!")
        else:
            st.warning("Backend comparison finished with a non-success status. See results below.")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Flask backend for comparison. Is it running?")
        st.session_state.backend_comparison_results = {"error": "Connection error."}
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get('error', e.response.text)
        st.error(f"Backend Comparison HTTP Error: {e.response.status_code} - {error_detail}")
        st.session_state.backend_comparison_results = {"error": error_detail}
    except Exception as e:
        st.error(f"An unexpected error occurred during backend comparison: {e}")
        st.session_state.backend_comparison_results = {"error": str(e)}

def chat_via_backend(message):
    """Sends a chat message to Flask backend for AI response."""
    try:
        headers = {'Content-Type': 'application/json'}
        data = {'message': message}
        response = requests.post(f"{FLASK_BACKEND_URL}/chat", headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('response', 'No response from AI.')
    except requests.exceptions.ConnectionError:
        return "Could not connect to Flask backend for chat. Is it running?"
    except requests.exceptions.HTTPError as e:
        error_detail = response.json().get('error', e.response.text)
        return f"Chat HTTP Error: {e.response.status_code} - {error_detail}"
    except Exception as e:
        return f"An unexpected error occurred during chat: {e}"


# --- Model Loading and Status Check Logic (Run on app start and reruns) ---
# Check backend status and load models if necessary
if not st.session_state.models_loaded:
    load_models_on_backend() # This function now contains the polling logic

# --- Sidebar for Navigation and Info ---
with st.sidebar:
    st.title("Project Risk AI")
    st.subheader("Your AI-Powered Project Management Assistant")
    
    # --- Image Inclusion ---
    # Replace 'unnamed.jpg' with the actual path to your image file.
    # Make sure 'unnamed.jpg' is in the same directory as this script, or provide the full path.
    try:
        # MODIFIED: Set a fixed width for the image to prevent it from taking up too much space
        st.image("/mnt/d/Capstone/ReplacementForAPI/ConQuan1.7/john.jpg", caption="AI Assistant Logo", width=150) 
    except FileNotFoundError:
        st.warning("Image '/mnt/d/Capstone/ReplacementForAPI/ConQuan1.7/john.jpg' not found. Please ensure it's in the correct directory.")
    except Exception as e:
        st.error(f"Error loading image: {e}")
    # --- End Image Inclusion ---

    st.write(st.session_state.load_status_message)
    if st.session_state.load_error:
        st.error(f"Backend Error: {st.session_state.load_error}")
    
    st.markdown("---")
    # Wrap the "About" content in an expander
    with st.expander("About"):
        st.markdown(
            "This application uses advanced AI models, served by a Flask backend, "
            "to assist with project risk management and general AI chat.\n"
            "Ensure your Flask backend (`flask_backend.py`) is running in a separate terminal."
        )
        st.markdown(
            """
            **Ethical & Technical Notes:**
            
            **Decision-Support Tool:** This is an AI assistant. Its outputs are intended to support, not replace, 
            professional engineering and project management judgment. All results should be critically reviewed.
            
            **Methodology:** The term 'Quantum' refers to a **quantum-inspired algorithm** simulated on classical hardware, 
            used to explore complex risk interdependencies.
            
            **Potential for Bias:** As with all AI systems, the models may reflect biases present in their training data. 
            Users are encouraged to independently verify any surprising or critical findings.
            """
        )
    st.markdown("---")
    # Add a simple reset button for the app's state
    if st.button("Reset Frontend"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()


# --- Main Content Area ---
st.header("Construction Project Risk Analysis V1.7") # Changed version to 1.7

# --- Project Analysis Section ---
st.markdown("## 📊 Project Risk Analysis")
st.markdown("Enter your project description below to get a comprehensive risk analysis.")

# Note: We can't directly import integrationapi.DEFAULT_TEST_PROJECT_DESCRIPTION here
# because integrationapi is meant to be run by Flask, not directly by Streamlit for constants.
# A simple hardcoded default or a new Flask endpoint to fetch defaults could be used.
# For now, using a hardcoded default for user convenience.
default_description_value = "A new highspeed train from Calgary to Edmonton that travels at 350km/hr."

project_description_input = st.text_area(
    "Project Description",
    value=default_description_value,
    height=200,
    help="Describe your project, including its scope, goals, key activities, and any initial known challenges or opportunities."
)

# MODIFIED: Changed to 4 columns for the new button
col_buttons = st.columns(4) 

with col_buttons[0]:
    if st.button("Analyze Project", type="primary", disabled=not st.session_state.models_loaded):
        if not st.session_state.models_loaded:
            st.warning("Please wait for models to load on the backend before analyzing.")
        elif not project_description_input.strip():
            st.warning("Please enter a project description to analyze.")
        else:
            with st.spinner("Sending project to backend for analysis..."):
                analyze_project_via_backend(project_description_input) # Call function to handle API call and state update
                if st.session_state.analysis_results and "error" not in st.session_state.analysis_results:
                    st.success("Initial Project Analysis Complete!")
                    # Optionally, scroll to results after completion
                    st.markdown("<div id='results'></div>", unsafe_allow_html=True) # Anchor for scrolling

with col_buttons[1]:
    # Corrected disabled logic: only enable if initial analysis is complete
    if st.button("Perform Safety Analysis", disabled=not st.session_state.initial_analysis_complete):
        with st.spinner("Performing Safety Analysis..."):
            perform_safety_analysis_via_backend()

with col_buttons[2]: # ADDED: New button for schedule generation
    # Button is enabled if models are loaded AND initial analysis is complete
    if st.button("Generate Schedule (MS Project XML)", disabled=not st.session_state.models_loaded or not st.session_state.initial_analysis_complete):
        if not st.session_state.models_loaded:
            st.warning("Please wait for models to load on the backend before generating schedule.")
        elif not st.session_state.initial_analysis_complete:
            st.warning("Please complete the Initial Analysis first to generate the schedule.")
        elif not project_description_input.strip():
            st.warning("Please enter a project description to generate schedule.")
        else:
            with st.spinner("Generating project schedule and XML..."):
                generate_schedule_xml_via_backend(project_description_input)
                if st.session_state.generated_tasks and st.session_state.generated_xml_content:
                    st.success(f"Generated {len(st.session_state.generated_tasks)} tasks and MS Project XML!")
                else:
                    st.error("Failed to generate schedule. Check backend logs for details.")

with col_buttons[3]: # NEW BUTTON: for procurement analysis
    # Button is enabled if models are loaded AND initial analysis is complete AND schedule tasks are generated
    if st.button("Perform Procurement Analysis", disabled=not st.session_state.models_loaded or not st.session_state.initial_analysis_complete or not st.session_state.generated_tasks):
        if not st.session_state.models_loaded:
            st.warning("Please wait for models to load on the backend before performing procurement analysis.")
        elif not st.session_state.initial_analysis_complete:
            st.warning("Please complete the Initial Analysis first.")
        elif not st.session_state.generated_tasks:
            st.warning("Please generate the Project Schedule (MS Project XML) first.")
        else:
            with st.spinner("Performing Procurement Analysis..."):
                perform_procurement_analysis_via_backend()


# --- Function to plot a single risk on a matrix ---
def plot_risk_matrix(probability, impact, is_opportunity, pre_mitigation_probability=None, pre_mitigation_impact=None):
    """
    Generates a 5x5 risk matrix plot with the given risk's probability and impact marked.
    Includes color-coded zones for risk levels (Low, Medium, High).
    Can also plot a 'before mitigation' point if provided.
    Assumes all input probability/impact values are already scaled 1-5 for plotting.
    """
    fig, ax = plt.subplots(figsize=(4, 4)) # Smaller figure size for individual matrix

    # Define risk matrix zones (Impact vs Probability)
    risk_zones = {
        'Low Risk':    {'color': '#d4edda', 'range': (0, 6)},   # Green (P*I <= 6)
        'Medium Risk': {'color': '#ffeeba', 'range': (6, 12)},  # Yellow (6 < P*I <= 12)
        'High Risk':   {'color': '#f8d7da', 'range': (12, 26)}  # Red (P*I > 12, max is 5*5=25)
    }

    # Plot background zones
    for p_val in range(1, 6):
        for i_val in range(1, 6):
            product = p_val * i_val
            color = 'lightgray' # Default background
            for zone_name, zone_info in risk_zones.items():
                min_prod, max_prod = zone_info['range']
                if min_prod < product <= max_prod:
                    color = zone_info['color']
                    break
            
            rect = plt.Rectangle((i_val - 0.5, p_val - 0.5), 1, 1, facecolor=color, edgecolor='none', alpha=0.8)
            ax.add_patch(rect)

    # Ensure current probability and impact are at least 1 for display on 1-5 matrix
    current_plot_prob = max(1.0, probability)
    current_plot_impact = max(1.0, impact)

    # Plot the specific risk point (post-mitigation / current state)
    marker_color = 'blue'
    marker_label = 'Risk (Current)'
    marker_style = 'o' # Circle for risk

    if is_opportunity:
        marker_color = 'green'
        marker_label = 'Opportunity (Current)'
        marker_style = 'D' # Diamond for opportunity
    
    ax.plot(current_plot_impact, current_plot_prob, marker_style, color=marker_color, markersize=10, label=marker_label, markeredgecolor='black', zorder=5)


    # Plot the pre-mitigation point if provided and it's a risk (not an opportunity)
    if pre_mitigation_probability is not None and pre_mitigation_impact is not None and not is_opportunity:
        # Ensure pre-mitigation values are also at least 1 for display
        pre_mitigation_plot_prob = max(1.0, pre_mitigation_probability)
        pre_mitigation_plot_impact = max(1.0, pre_mitigation_impact)
        
        ax.plot(pre_mitigation_plot_impact, pre_mitigation_plot_prob, 'x', color='red', markersize=10, label='Risk (Before Mitigation)', markeredgecolor='black', zorder=4)
        
        # Draw an arrow from pre-mitigation to current (post-mitigation)
        ax.annotate(
            '', xy=(current_plot_impact, current_plot_prob), xytext=(pre_mitigation_plot_impact, pre_mitigation_plot_prob),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6, headlength=6),
            zorder=3
        )


    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlabel('Impact (1-5)')
    ax.set_ylabel('Probability (1-5)')
    ax.set_title('Risk Matrix Position')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.invert_yaxis() # Often probability is high at the top
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal', adjustable='box') # Ensure cells are square
    ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0)) # Add legend outside the plot area

    plt.tight_layout()
    return fig

# --- Function to plot the budget breakdown ---
def plot_budget_breakdown(adjusted_cost, contingency_reserve, management_reserve, total_budget_for_display, baseline_most_likely_cad, plot_title='Project Budget Breakdown (CAD)'):
    """
    Generates a bar graph showing the breakdown of the total project budget.
    """
    labels = ['Baseline ML', 'Contingency Reserve', 'Management Reserve', 'Adjusted Cost']
    values = [baseline_most_likely_cad, contingency_reserve, management_reserve, adjusted_cost]
    colors = ['#FFD700', '#FFC107', '#2196F3', '#4CAF50'] # Gold (Baseline), Yellow (Contingency), Blue (Management), Green (Adjusted Total)

    # Ensure all values are present before plotting
    if not all(val is not None for val in values):
        st.warning(f"Cannot plot {plot_title}: some reserve values are missing.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylabel('Amount (CAD)')
    ax.set_title(plot_title)
    ax.set_xticks(labels)
    ax.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'${yval:,.0f}', va='bottom', ha='center', fontsize=9) # format as currency

    plt.tight_layout()
    return fig


# --- Display Initial Analysis Results ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results

    st.markdown("---")
    st.markdown("### 📈 Initial Project Analysis Results")

    if results.get('error'):
        st.error(f"Initial Analysis failed: {results['error']}")
    else:
        # Create two columns for Baseline Estimates and Budget Breakdown Graph
        col_estimates, col_budget_graph = st.columns([1, 1])

        with col_estimates:
            st.subheader("1. Baseline Estimates (CAD & Weeks)")
            baseline = results.get('baseline_estimates', {})
            if baseline:
                st.write(f"**Cost (Optimistic):** ${baseline.get('cost_optimistic_cad', 0):,.2f} CAD")
                st.write(f"**Cost (Most Likely):** ${baseline.get('cost_most_likely_cad', 0):,.2f} CAD")
                st.write(f"**Cost (Pessimistic):** ${baseline.get('cost_pessimistic_cad', 0):,.2f} CAD")
                st.write(f"**Time (Optimistic):** {baseline.get('time_optimistic_weeks', 0):.1f} weeks")
                st.write(f"**Time (Most Likely):** {baseline.get('time_most_likely_weeks', 0):.1f} weeks")
                st.write(f"**Time (Pessimistic):** {baseline.get('time_pessimistic_weeks', 0):.1f} weeks")
            else:
                st.warning("Baseline estimates not available.")

            st.subheader("2. Adjusted Project Estimates (Monte Carlo)")
            adjusted_cost = results.get('total_project_cost_cad')
            adjusted_time = results.get('total_project_time_weeks')

            if adjusted_cost is not None:
                st.metric(label="Adjusted Total Cost", value=f"${adjusted_cost:,.2f} CAD")
            if adjusted_time is not None:
                st.metric(label="Adjusted Total Time", value=f"{adjusted_time:.1f} weeks")
            if adjusted_cost is None and adjusted_time is None:
                st.info("Adjusted total cost and time not available from initial analysis.")

        with col_budget_graph:
            st.subheader("3. Initial Project Budget Breakdown")
            contingency_reserve = results.get('contingency_reserve_cad')
            management_reserve = results.get('management_reserve_cad')
            total_project_cost_cad_from_backend = results.get('total_project_cost_cad') # Corrected: Use actual total cost from backend
            
            baseline_most_likely_cad = results.get('baseline_estimates', {}).get('cost_most_likely_cad')

            if all(val is not None for val in [adjusted_cost, contingency_reserve, management_reserve, baseline_most_likely_cad, total_project_cost_cad_from_backend]):
                st.write(f"**Baseline Most Likely Cost:** ${baseline_most_likely_cad:,.2f} CAD")
                st.write(f"**Contingency Reserve (P85 Contractor - ML Baseline):** ${contingency_reserve:,.2f} CAD")
                st.write(f"**Management Reserve (P85 Owner/Engineer):** ${management_reserve:,.2f} CAD")
                st.write(f"**Total Project Budget:** ${total_project_cost_cad_from_backend:,.2f} CAD")
                
                # Display the bar graph
                fig_budget = plot_budget_breakdown(
                    adjusted_cost,
                    contingency_reserve,
                    management_reserve,
                    total_project_cost_cad_from_backend, # Pass the correct total budget for display
                    baseline_most_likely_cad,
                    'Initial Project Budget Breakdown (CAD)'
                )
                if fig_budget:
                    st.pyplot(fig_budget)
                    plt.close(fig_budget)
            else:
                st.info("Initial budget breakdown details not available.")


        # Identified Risks and Opportunities
        st.subheader("4. Identified Risks and Opportunities (Detailed Breakdown - Initial Analysis)")
        
        # Combine risks and opportunities for display
        all_initial_items = []
        all_initial_items.extend(results.get('identified_risks_with_qualquan_data', []))
        all_initial_items.extend(results.get('identified_opportunities_with_qualquan_data', []))

        if all_initial_items:
            # Sort items so opportunities appear first, then risks, then by ID
            all_initial_items.sort(key=lambda x: (not x.get('is_opportunity', False), x.get('risk_id', '')))

            for i, item in enumerate(all_initial_items):
                expander_title = f"{'✅ Opportunity' if item.get('is_opportunity') else '⚠️ Risk'} {i+1}: {item.get('risk_description', 'N/A')[:70]}..."
                with st.expander(expander_title):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write(f"**ID:** {item.get('risk_id', 'N/A')}")
                        st.write(f"**Description:** {item.get('risk_description', 'N/A')}")
                        st.write(f"**Strategy:** {item.get('mitigation_strategy', 'N/A')}")
                        st.write(f"**Type:** {'Opportunity' if item.get('is_opportunity') else 'Risk'}")
                        st.write(f"**Qualitative Decision:** {item.get('qualitative_decision', 'N/A')}")
                        st.write(f"**Category:** {item.get('risk_category', 'Unknown')}")
                        
                        st.markdown("#### QualQuan Numerical Predictions")
                        
                        # --- FIX STARTS HERE (Safety Checks for Formatting) ---
                        prob = item.get('probability')
                        imp = item.get('impact')
                        st.write(f"**Probability (Raw Score):** {f'{prob:.2f}' if isinstance(prob, (int, float)) else 'N/A'}")
                        st.write(f"**Impact (Raw Score):** {f'{imp:.2f}' if isinstance(imp, (int, float)) else 'N/A'}")
                        
                        red_prob = item.get('risk_reduction_probability', 0)
                        red_imp = item.get('risk_reduction_impact', 0)
                        # Ensure these are numbers before multiplying
                        if not isinstance(red_prob, (int, float)): red_prob = 0
                        if not isinstance(red_imp, (int, float)): red_imp = 0

                        if item.get('is_opportunity'):
                            st.write(f"**Enhancement Probability (%):** {red_prob*100:.2f}%")
                            st.write(f"**Enhancement Impact (%):** {red_imp*100:.2f}%")
                        else:
                            st.write(f"**Risk Reduction Probability (%):** {red_prob*100:.2f}%")
                            st.write(f"**Risk Reduction Impact (%):** {red_imp*100:.2f}%")

                        # Safe formatting for Costs
                        c_opt = item.get('cost_optimistic_usd')
                        c_ml = item.get('cost_most_likely_usd')
                        c_pess = item.get('cost_pessimistic_usd')
                        c_fmt = lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else "N/A"
                        st.write(f"**Cost (Opt, ML, Pess) USD:** {c_fmt(c_opt)}, {c_fmt(c_ml)}, {c_fmt(c_pess)}")

                        # Safe formatting for Time
                        t_opt = item.get('time_optimistic_weeks')
                        t_ml = item.get('time_most_likely_weeks')
                        t_pess = item.get('time_pessimistic_weeks')
                        t_fmt = lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else "N/A"
                        st.write(f"**Time (Opt, ML, Pess) Weeks:** {t_fmt(t_opt)}, {t_fmt(t_ml)}, {t_fmt(t_pess)}")
                        
                        st.markdown("#### Derived Impacts")
                        if item.get('is_opportunity'):
                            cost_red = item.get('cost_reduction_cad')
                            time_red = item.get('time_reduction_weeks')
                            st.write(f"**Estimated Cost Reduction (CAD):** {c_fmt(cost_red)} CAD")
                            st.write(f"**Estimated Time Reduction (Weeks):** {t_fmt(time_red)} weeks")
                        else:
                            cost_imp = item.get('cost_impact_cad')
                            time_imp = item.get('time_impact_weeks')
                            st.write(f"**Estimated Cost Impact (CAD):** {c_fmt(cost_imp)} CAD")
                            st.write(f"**Estimated Time Impact (Weeks):** {t_fmt(time_imp)} weeks")
                        # --- FIX ENDS HERE ---

                    with col2:
                        # ... (Matrix plotting code remains the same, assuming it handles None internally or uses defaults) ...
                        st.markdown("#### Position on Matrix")
                        # Ensure values are floats for plotting logic
                        prob_val = float(prob) if isinstance(prob, (int, float)) else 0.0
                        imp_val = float(imp) if isinstance(imp, (int, float)) else 0.0
                        
                        # Use these safe values for the rest of the plotting logic
                        is_opp = item.get('is_opportunity', False)
                        RAW_SCORE_MAX_FOR_DISPLAY = 5.0 

                        normalized_prob_0_1 = np.clip(prob_val, 0, RAW_SCORE_MAX_FOR_DISPLAY) / RAW_SCORE_MAX_FOR_DISPLAY
                        normalized_impact_0_1 = np.clip(imp_val, 0, RAW_SCORE_MAX_FOR_DISPLAY) / RAW_SCORE_MAX_FOR_DISPLAY

                        current_prob_for_plot = max(1.0, normalized_prob_0_1 * 5.0)
                        current_impact_for_plot = max(1.0, normalized_impact_0_1 * 5.0)
                        
                        # ... (rest of matrix logic) ...
                        pre_mitigation_prob_for_plot = None
                        pre_mitigation_impact_for_plot = None

                        if not is_opp:
                             # Ensure reduction values are floats
                            reduction_prob_0_1 = float(item.get('risk_reduction_probability', 0.0))
                            reduction_impact_0_1 = float(item.get('risk_reduction_impact', 0.0))

                            pre_mitigation_prob_0_1 = min(1.0, normalized_prob_0_1 + reduction_prob_0_1)
                            pre_mitigation_impact_0_1 = min(1.0, normalized_impact_0_1 + reduction_impact_0_1)
                            
                            pre_mitigation_prob_for_plot = max(1.0, pre_mitigation_prob_0_1 * 5.0)
                            pre_mitigation_impact_for_plot = max(1.0, pre_mitigation_impact_0_1 * 5.0)
                        
                        fig = plot_risk_matrix(
                            current_prob_for_plot, 
                            current_impact_for_plot, 
                            is_opp, 
                            pre_mitigation_probability=pre_mitigation_prob_for_plot, 
                            pre_mitigation_impact=pre_mitigation_impact_for_plot
                        )
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)

        else:
            st.info("No risks or opportunities identified for this initial analysis.")

        # --- THIS IS THE CORRECTED SECTION ---
        # Plots - Now displaying base64 encoded images
        st.subheader("5. Initial Distribution Plots")
        plots_dict = results.get('plots_base64', {})
        cost_plot_base64 = plots_dict.get('cost_distribution_plot_base64')
        time_plot_base64 = plots_dict.get('time_distribution_plot_base64')
        contractor_risk_cost_plot_base64 = plots_dict.get('contractor_risk_cost_plot_base64')
        owner_engineer_risk_cost_plot_base64 = plots_dict.get('owner_engineer_risk_cost_plot_base64')

        if cost_plot_base64:
            st.image(cost_plot_base64, caption='Simulated Total Project Cost Distribution (Initial)', use_container_width=True)
        if time_plot_base64:
            st.image(time_plot_base64, caption='Simulated Total Project Time Distribution (Initial)', use_container_width=True)
        if contractor_risk_cost_plot_base64:
            st.image(contractor_risk_cost_plot_base64, caption='Simulated Contractor-Born Risk Cost Distribution (Initial)', use_container_width=True)
        if owner_engineer_risk_cost_plot_base64:
            st.image(owner_engineer_risk_cost_plot_base64, caption='Simulated Owner/Engineer-Born Risk Cost Distribution (Initial)', use_container_width=True)
        
        if not any([cost_plot_base64, time_plot_base64, contractor_risk_cost_plot_base64, owner_engineer_risk_cost_plot_base64]):
            st.info("Distribution plots were not generated or received from the backend.")

    # --- Display Safety Analysis Results ---
    if st.session_state.safety_analysis_results:
        safety_results = st.session_state.safety_analysis_results

        st.markdown("---")
        st.markdown("### ⛑️ Safety Analysis Results")

        if safety_results.get('error'):
            st.error(f"Safety Analysis failed: {safety_results['error']}")
        else:
            col_safety_estimates, col_safety_budget_graph = st.columns([1, 1])

            with col_safety_estimates:
                st.subheader("1. Safety Adjusted Project Estimates (Monte Carlo)")
                total_before_procurement_cad = safety_results.get('total_project_cost_after_safety_cad') # Corrected: Use actual total cost from backend
                total_project_time_after_safety_weeks = safety_results.get('total_project_time_after_safety_weeks')

                if total_before_procurement_cad is not None:
                    st.metric(label="Total Cost After Safety Risks", value=f"${total_before_procurement_cad:,.2f} CAD")
                if total_project_time_after_safety_weeks is not None:
                    st.metric(label="Total Project Time After Safety Risks", value=f"{total_project_time_after_safety_weeks:.1f} weeks")
                if total_before_procurement_cad is None and total_project_time_after_safety_weeks is None:
                    st.info("Safety-adjusted total cost and time not available.")

            with col_safety_budget_graph:
                st.subheader("2. Safety Budget & Reserves")
                safety_contingency_reserve_cad = safety_results.get('contingency_reserve_after_safety_cad')
                safety_management_reserve_cad = safety_results.get('management_reserve_after_safety_cad')
                safety_total_project_cost_cad = safety_results.get('total_project_cost_after_safety_cad') # Corrected: Use actual total cost from backend
                
                # Use initial baseline from original analysis_results for comparison in budget breakdown
                safety_baseline_most_likely_cad = st.session_state.analysis_results.get('total_project_cost_cad', 0.0)

                if all(val is not None for val in [total_before_procurement_cad, safety_contingency_reserve_cad, safety_management_reserve_cad, safety_baseline_most_likely_cad, safety_total_project_cost_cad]):
                    st.write(f"**Contingency Reserve (Combined Contractor):** ${safety_contingency_reserve_cad:,.2f} CAD")
                    st.write(f"**Management Reserve (Combined Owner/Engineer):** ${safety_management_reserve_cad:,.2f} CAD")
                    st.write(f"**Total Project Budget (After Safety):** ${safety_total_project_cost_cad:,.2f} CAD")
                    
                    # Display the bar graph
                    fig_safety_budget = plot_budget_breakdown(
                        total_before_procurement_cad,
                        safety_contingency_reserve_cad,
                        safety_management_reserve_cad,
                        safety_total_project_cost_cad, # Pass the correct total budget for display
                        safety_baseline_most_likely_cad,
                        'Safety Analysis Budget Breakdown (CAD)'
                    )
                    if fig_safety_budget:
                        st.pyplot(fig_safety_budget)
                        plt.close(fig_safety_budget)
                else:
                    st.info("Safety budget breakdown details not available.")


            st.subheader("3. Identified Safety Risks and Opportunities (Detailed Breakdown)")
            all_safety_items = []
            all_safety_items.extend(safety_results.get('identified_safety_risks_with_qualquan_data', []))
            all_safety_items.extend(safety_results.get('identified_safety_opportunities_with_qualquan_data', [])) # ADDED: Include safety opportunities

            if all_safety_items:
                all_safety_items.sort(key=lambda x: (not x.get('is_opportunity', False), x.get('risk_id', '')))

                for i, item in enumerate(all_safety_items): # Loop through combined list
                    expander_title = f"{'✅ Opportunity' if item.get('is_opportunity') else '⚠️ Risk'} {i+1} (Safety): {item.get('risk_description', 'N/A')[:70]}..."
                    with st.expander(expander_title):
                        col1_s, col2_s = st.columns([2, 1])

                        with col1_s:
                            st.write(f"**ID:** {item.get('risk_id', 'N/A')}")
                            st.write(f"**Description:** {item.get('risk_description', 'N/A')}")
                            st.write(f"**Strategy:** {item.get('mitigation_strategy', 'N/A')}")
                            st.write(f"**Type:** {'Opportunity' if item.get('is_opportunity') else 'Risk'}")
                            st.write(f"**Qualitative Decision:** {item.get('qualitative_decision', 'N/A')}")
                            st.write(f"**Category:** {item.get('risk_category', 'Unknown')}")
                            
                            st.markdown("#### QualQuan Numerical Predictions")
                            st.write(f"**Probability (Raw Score):** {item.get('probability', 'N/A'):.2f}")
                            st.write(f"**Impact (Raw Score):** {item.get('impact', 'N/A'):.2f}")
                            
                            if item.get('is_opportunity'):
                                st.write(f"**Enhancement Probability (%):** {item.get('risk_reduction_probability', 0)*100:.2f}%")
                                st.write(f"**Enhancement Impact (%):** {item.get('risk_reduction_impact', 0)*100:.2f}%")
                            else:
                                st.write(f"**Risk Reduction Probability (%):** {item.get('risk_reduction_probability', 0)*100:.2f}%")
                                st.write(f"**Risk Reduction Impact (%):** {item.get('risk_reduction_impact', 0)*100:.2f}%")

                            st.write(f"**Cost (Opt, ML, Pess) USD:** ${item.get('cost_optimistic_usd', 0):,.2f}, ${item.get('cost_most_likely_usd', 0):,.2f}, ${item.get('cost_pessimistic_usd', 0):,.2f}")
                            st.write(f"**Time (Opt, ML, Pess) Weeks):** {item.get('time_optimistic_weeks', 0):.1f}, {item.get('time_most_likely_weeks', 0):.1f}, {item.get('time_pessimistic_weeks', 0):.1f}")
                            
                            st.markdown("#### Derived Impacts")
                            if item.get('is_opportunity'):
                                st.write(f"**Estimated Cost Reduction (CAD):** ${item.get('cost_reduction_cad', 0):,.2f} CAD")
                                st.write(f"**Estimated Time Reduction (Weeks):** {item.get('time_reduction_weeks', 0):.1f} weeks")
                            else:
                                st.write(f"**Estimated Cost Impact (CAD):** ${item.get('cost_impact_cad', 0):,.2f} CAD")
                                st.write(f"**Estimated Time Impact (Weeks):** {item.get('time_impact_weeks', 0):.1f} weeks")

                        with col2_s:
                            st.markdown("#### Position on Matrix")
                            is_opp = item.get('is_opportunity', False)
                            
                            RAW_SCORE_MAX_FOR_DISPLAY = 5.0 

                            normalized_prob_0_1 = np.clip(item.get('probability', 0.0), 0, RAW_SCORE_MAX_FOR_DISPLAY) / RAW_SCORE_MAX_FOR_DISPLAY
                            normalized_impact_0_1 = np.clip(item.get('impact', 0.0), 0, RAW_SCORE_MAX_FOR_DISPLAY) / RAW_SCORE_MAX_FOR_DISPLAY

                            current_prob_for_plot = max(1.0, normalized_prob_0_1 * 5.0)
                            current_impact_for_plot = max(1.0, normalized_impact_0_1 * 5.0)
                            
                            pre_mitigation_prob_for_plot = None
                            pre_mitigation_impact_for_plot = None

                            if not is_opp:
                                reduction_prob_0_1 = item.get('risk_reduction_probability', 0.0)
                                reduction_impact_0_1 = item.get('risk_reduction_impact', 0.0)

                                pre_mitigation_prob_0_1 = min(1.0, normalized_prob_0_1 + reduction_prob_0_1)
                                pre_mitigation_impact_0_1 = min(1.0, normalized_impact_0_1 + reduction_impact_0_1)
                                
                                pre_mitigation_prob_for_plot = max(1.0, pre_mitigation_prob_0_1 * 5.0)
                                pre_mitigation_impact_for_plot = max(1.0, pre_mitigation_impact_0_1 * 5.0)
                            
                            fig_s = plot_risk_matrix(
                                current_prob_for_plot, 
                                current_impact_for_plot, 
                                is_opp, 
                                pre_mitigation_probability=pre_mitigation_prob_for_plot, 
                                pre_mitigation_impact=pre_mitigation_impact_for_plot
                            )
                            if fig_s:
                                st.pyplot(fig_s)
                                plt.close(fig_s)
            else:
                st.info("No safety risks or opportunities identified for this analysis.")

            st.subheader("4. Safety Analysis Distribution Plots")
            cost_plot_safety_base64 = safety_results.get('plots_base64_after_safety', {}).get('cost_distribution_plot_after_safety_base64')
            time_plot_safety_base64 = safety_results.get('plots_base64_after_safety', {}).get('time_distribution_plot_after_safety_base64')
            contractor_risk_cost_plot_safety_base64 = safety_results.get('plots_base64_after_safety', {}).get('contractor_risk_cost_plot_after_safety_base64')
            owner_engineer_risk_cost_plot_safety_base64 = safety_results.get('plots_base64_after_safety', {}).get('owner_engineer_risk_cost_plot_after_safety_base64')

            if cost_plot_safety_base64:
                st.image(cost_plot_safety_base64, caption='Simulated Total Project Cost Distribution (After Safety)', use_container_width=True)
            if time_plot_safety_base64:
                st.image(time_plot_safety_base64, caption='Simulated Total Project Time Distribution (After Safety)', use_container_width=True)
            if contractor_risk_cost_plot_safety_base64:
                st.image(contractor_risk_cost_plot_safety_base64, caption='Simulated Combined Contractor-Born Risk Cost (After Safety)', use_container_width=True)
            if owner_engineer_risk_cost_plot_safety_base64:
                st.image(owner_engineer_risk_cost_plot_safety_base64, caption='Simulated Owner/Engineer-Born Risk Cost (After Safety)', use_container_width=True)
            
            if not any([cost_plot_safety_base64, time_plot_safety_base64, contractor_risk_cost_plot_safety_base64, owner_engineer_risk_cost_plot_safety_base64]):
                st.info("Some safety analysis distribution plots were not generated or received.")

# ADDED: Display Generated Schedule Tasks and XML (reintegrated)
if st.session_state.generated_tasks:
    st.markdown("---")
    st.markdown("### 🗓️ Generated Project Schedule Tasks")
    st.info("Note: The 'Critical Path' indication is an AI-suggested likelihood. For definitive Critical Path Method (CPM) calculations and detailed scheduling, please import the generated XML into dedicated project management software like Microsoft Project.")

    # Convert tasks to a Pandas DataFrame for better display
    df_tasks = pd.DataFrame(st.session_state.generated_tasks)
    
    # Define the desired column order, handle missing columns gracefully
    display_columns = [
        'task_id', 'task_name', 'UID', 'ID', 'duration_days', 
        'predecessors', 'resources_needed', 'estimated_cost_usd', 
        'is_milestone', 'is_critical', 'is_summary' 
    ]
    
    # Filter columns to only include those present in the DataFrame
    df_columns_present = [col for col in display_columns if col in df_tasks.columns]
    # Use .copy() to ensure df_tasks_display is a new DataFrame, preventing SettingWithCopyWarning
    df_tasks_display = df_tasks[df_columns_present].copy()

    # Safely format columns if they exist in the DataFrame
    if 'estimated_cost_usd' in df_tasks_display.columns:
        df_tasks_display['estimated_cost_usd'] = df_tasks_display['estimated_cost_usd'].apply(lambda x: f"${x:,.2f}")
    
    if 'predecessors' in df_tasks_display.columns:
        # Make sure predecessors (which may be numbers) are strings before joining them
        df_tasks_display['predecessors'] = df_tasks_display['predecessors'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)


    # Style the DataFrame to highlight critical tasks, but only if the 'is_critical' column exists.
    if 'is_critical' in df_tasks_display.columns:
        def highlight_critical(row):
            return ['background-color: #fff3cd' if row['is_critical'] else '' for _ in row]
        
        st.dataframe(df_tasks_display.style.apply(highlight_critical, axis=1))
    else:
        # If the 'is_critical' column is not in the data from the backend, display the table without the problematic styling.
        st.dataframe(df_tasks_display)
if st.session_state.generated_xml_content:
    st.markdown("#### Download Main Project Schedule XML") # Renamed for clarity
    st.download_button(
        label="Download Main Schedule as XML",
        data=st.session_state.generated_xml_content,
        file_name="main_project_schedule.xml",
        mime="application/xml",
        help="Download this file and open it with Microsoft Project or compatible software."
    )

# NEW: Display Procurement Analysis Results
if st.session_state.procurement_analysis_results:
    proc_results = st.session_state.procurement_analysis_results

    st.markdown("---")
    st.markdown("### 📦 Procurement Analysis Results")

    if proc_results.get('error'):
        st.error(f"Procurement Analysis failed: {proc_results['error']}")
    else:
        st.subheader("1. Procurement Adjusted Project Estimates (Monte Carlo)")
        adjusted_cost_after_procurement = proc_results.get('total_project_cost_after_procurement_cad')
        adjusted_time_after_procurement = proc_results.get('total_project_time_after_procurement_weeks')

        if adjusted_cost_after_procurement is not None:
            st.metric(label="Total Cost After Procurement Risks", value=f"${adjusted_cost_after_procurement:,.2f} CAD")
        if adjusted_time_after_procurement is not None:
            st.metric(label="Total Project Time After Procurement Risks", value=f"{adjusted_time_after_procurement:.1f} weeks")
        if adjusted_cost_after_procurement is None and adjusted_time_after_procurement is None:
            st.info("Procurement-adjusted total cost and time not available.")

        col_proc_budget, col_proc_graphs = st.columns([1,1])
        with col_proc_budget:
            st.subheader("2. Procurement Budget & Reserves")
            proc_contingency_reserve_cad = proc_results.get('contingency_reserve_after_procurement_cad')
            proc_management_reserve_cad = proc_results.get('management_reserve_after_procurement_cad')
            proc_total_project_cost_cad = proc_results.get('total_project_cost_after_procurement_cad') # Corrected: Use actual total cost from backend
            
            if st.session_state.initial_analysis_complete and st.session_state.analysis_results:
                initial_baseline_ml_cost = st.session_state.analysis_results['baseline_estimates']['cost_most_likely_cad']
            else:
                initial_baseline_ml_cost = 0.0 # Default if initial analysis not complete

            if all(val is not None for val in [adjusted_cost_after_procurement, proc_contingency_reserve_cad, proc_management_reserve_cad, proc_total_project_cost_cad]):
                st.write(f"**Contingency Reserve (Combined Contractor - Proc):** ${proc_contingency_reserve_cad:,.2f} CAD")
                st.write(f"**Management Reserve (Combined Owner/Engineer - Proc):** ${proc_management_reserve_cad:,.2f} CAD")
                st.write(f"**Total Project Budget (After Procurement):** ${proc_total_project_cost_cad:,.2f} CAD")
                
                # Display the bar graph
                fig_proc_budget = plot_budget_breakdown(
                    adjusted_cost_after_procurement,
                    proc_contingency_reserve_cad,
                    proc_management_reserve_cad,
                    proc_total_project_cost_cad, # Pass the correct total budget for display
                    initial_baseline_ml_cost, # Use initial baseline for consistency in budget breakdown
                    'Procurement Analysis Budget Breakdown (CAD)'
                )
                if fig_proc_budget:
                    st.pyplot(fig_proc_budget)
                    plt.close(fig_proc_budget)
            else:
                st.info("Procurement budget breakdown details not available.")

        with col_proc_graphs:
            st.subheader("3. Procurement Analysis Distribution Plots")
            cost_plot_proc_base64 = proc_results.get('plots_base64_after_procurement', {}).get('cost_distribution_plot_after_procurement_base64')
            time_plot_proc_base64 = proc_results.get('plots_base64_after_procurement', {}).get('time_distribution_plot_after_procurement_base64')
            contractor_procurement_risk_cost_plot_base64 = proc_results.get('plots_base64_after_procurement', {}).get('contractor_risk_cost_plot_after_procurement_base64')
            owner_engineer_procurement_risk_cost_plot_base64 = proc_results.get('plots_base64_after_procurement', {}).get('owner_engineer_risk_cost_plot_after_procurement_base64')


            if cost_plot_proc_base64:
                st.image(cost_plot_proc_base64, caption='Simulated Total Project Cost Distribution (After Procurement)', use_container_width=True)
            if time_plot_proc_base64:
                st.image(time_plot_proc_base64, caption='Simulated Total Project Time Distribution (After Procurement)', use_container_width=True)
            if contractor_procurement_risk_cost_plot_base64:
                st.image(contractor_procurement_risk_cost_plot_base64, caption='Simulated Contractor-Born Procurement Risk Cost (After Procurement)', use_container_width=True)
            if owner_engineer_procurement_risk_cost_plot_base64:
                st.image(
                    owner_engineer_procurement_risk_cost_plot_base64,
                    caption='Simulated Owner/Engineer-Born Procurement Risk Cost (After Procurement)',
                    use_container_width=True
                )
            
            if not any([cost_plot_proc_base64, time_plot_proc_base64, contractor_procurement_risk_cost_plot_base64, owner_engineer_procurement_risk_cost_plot_base64]):
                st.info("Some procurement analysis distribution plots were not generated or received.")


        st.subheader("4. Detailed Procurement Schedule")
        detailed_proc_schedule = proc_results.get('detailed_procurement_schedule', [])
        if detailed_proc_schedule   :
            # Join with original tasks to show task_name
            task_name_map = {task['task_id']: task['task_name'] for task in st.session_state.generated_tasks}
            for item in detailed_proc_schedule:
                item['task_name'] = task_name_map.get(item['task_id'], 'Unknown Task')

            df_proc_schedule = pd.DataFrame(detailed_proc_schedule)
            
            # Reorder columns for better readability
            proc_display_columns = ['task_id', 'task_name', 'procurement_type', 'procurement_category', 'is_procurement_critical', 'lead_time_weeks']
            df_proc_schedule_display = df_proc_schedule[[col for col in proc_display_columns if col in df_proc_schedule.columns]]

            # Sort by criticality, then category, then type
            df_proc_schedule_display['priority_score'] = df_proc_schedule_display.apply(
                lambda row: (
                    (0 if row['procurement_type'] == 'in-house' else 1) * 1000 +
                    (0 if row['procurement_category'] == 'minor' else 1) * 100 +
                    (0 if row['is_procurement_critical'] == False else 1) * 10
                ), axis=1
            )
            df_proc_schedule_display = df_proc_schedule_display.sort_values(by='priority_score', ascending=True).drop(columns='priority_score')
            
            # Format critical column for better display
            df_proc_schedule_display['is_procurement_critical'] = df_proc_schedule_display['is_procurement_critical'].apply(lambda x: 'Critical' if x else 'Non-Critical')


            def highlight_procurement_critical(row):
                # Highlight critical procurements in red, major in yellow, outside in light blue
                styles = [''] * len(row)
                if row['is_procurement_critical'] == 'Critical':
                    styles = ['background-color: #f8d7da'] * len(row) # Light red
                elif row['procurement_category'] == 'major':
                    styles = ['background-color: #ffeeba'] * len(row) # Light yellow
                elif row['procurement_type'] == 'outside':
                    styles = ['background-color: #e2f0fb'] * len(row) # Light blue
                return styles

            st.dataframe(df_proc_schedule_display.style.apply(highlight_procurement_critical, axis=1))

        else:
            st.info("No detailed procurement schedule generated.")
        
        # NEW: Download button for procurement schedule XML
        if st.session_state.procurement_ms_project_xml:
            st.markdown("#### Download Procurement Schedule XML")
            st.download_button(
                label="Download Procurement Schedule as XML",
                data=st.session_state.procurement_ms_project_xml,
                file_name="procurement_schedule.xml",
                mime="application/xml",
                help="Download this file and open it with Microsoft Project or compatible software."
            )
