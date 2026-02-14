Quantum Construction Project Risk Analyzer V1.7
Overview

The Quantum Construction Project Risk Analyzer is an AI-powered project management assistant designed to help with project risk management and general AI chat. This application features a Streamlit frontend that communicates with a Flask backend, where advanced AI models perform comprehensive risk analysis, safety analysis, schedule generation, and procurement analysis.
Ethical & Technical Considerations

This section outlines important ethical and technical notes regarding the use and functionality of the Quantum Construction Project Risk Analyzer:

    Decision-Support Tool: This is an AI assistant. Its outputs are intended to support, not replace, professional engineering and project management judgment. All results should be critically reviewed and validated by qualified professionals before making any critical decisions. The AI provides probabilistic assessments and insights, but human expertise remains paramount for final interpretations and actions.

    Methodology: The term 'Quantum' in the application's name refers to a quantum-inspired algorithm simulated on classical hardware. This methodology is employed to explore complex risk interdependencies and provide more nuanced insights than traditional linear models might offer. It is not indicative of the use of actual quantum computing hardware.

    Potential for Bias: As with all AI systems, the models integrated into this application may reflect biases present in their training data. These biases could potentially influence risk assessments or recommendations. Users are strongly encouraged to independently verify any surprising, unexpected, or critical findings and to consider diverse perspectives to mitigate the impact of potential biases. Continuous monitoring and validation of the AI's outputs are recommended.

Setup Instructions

To run this application, you will need to set up both the Flask backend and the Streamlit frontend, install a pennylane.lightning environment, install a quiskit environment, get a bert model, tiny llama model, train a numerical scaler, train a QML model for the computation, obtain 7 gemini API keys, one open exchange key, and
one key from ibm quantum, label them according to api key calls from integration.api, and call the file key.env. For the scheduler to work you will have to train a qwen model with schedule data.

    Flask Backend:

        Ensure your Flask backend (flask_backend.py) is running in a separate terminal.

        Verify that the FLASK_BACKEND_URL in streamlit_frontend.py matches the host and port where your Flask app is running (e.g., http://192.168.1.84:5000).

    Streamlit Frontend:

        Save the streamlit_frontend.py code.

        Install Streamlit and other necessary Python libraries:

        pip install streamlit requests matplotlib numpy pandas

        Run the Streamlit application from your terminal:

        streamlit run streamlit_frontend.py

Usage

    Model Loading: Upon starting the Streamlit application, it will attempt to connect to the Flask backend and load the necessary AI models. Monitor the status messages in the sidebar.

    Project Risk Analysis: Enter a detailed project description in the provided text area and click "Analyze Project" to get initial risk, cost, and time estimates.

    Specialized Analyses: After the initial analysis, you can perform "Safety Analysis" and "Procurement Analysis" using the respective buttons.

    Schedule Generation: Generate an MS Project XML schedule based on your project description.

    AI Chat Assistant: Use the chat interface to ask general project management questions or seek advice from the AI.

    Reset Frontend: The "Reset Frontend" button in the sidebar can be used to clear the application's session state and start fresh.
