# Medical AI Assistant - Proof of Concept

A voice-enabled medical AI assistant built with LangGraph, Streamlit, and Ollama. This proof-of-concept demonstrates an intelligent agent that can help with medical information retrieval, appointment scheduling, and general health queries.

## Features

- 🎤 Voice input support with Whisper STT
- 🤖 LangGraph-powered agent with tool use
- 🏥 Medical database integration
- 💬 Interactive chat interface
- 🔊 Text-to-speech responses

## Prerequisites

Before installing the project, you need to have:

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running
3. **FFmpeg** (for audio processing)

### Installing Ollama

1. Visit [https://ollama.ai](https://ollama.ai) and download Ollama for your operating system
2. Install Ollama following the instructions for your platform
3. Once installed, pull the required model:

```bash
ollama pull granite4:latest
```

**Note:** The default model is `granite4:latest`. You can use a different model by setting the `MODEL_NAME` environment variable (see Configuration section).

### Installing FFmpeg (macOS)

```bash
brew install ffmpeg
```

For other operating systems, visit [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## Installation

1. **Clone the repository** (or navigate to the project directory):

```bash
cd /path/to/Goal3
```

2. **Create a virtual environment**:

```bash
python3 -m venv venv
```

3. **Activate the virtual environment**:

On macOS/Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

4. **Install required dependencies**:

```bash
pip install -r requirements.txt
```

5. **Set up the database** (optional):

```bash
python src/database/init_db.py
```

## Configuration

Create a `.env` file in the project root directory (optional):

```bash
# Model Configuration (optional - defaults shown)
MODEL_PROVIDER=ollama
MODEL_NAME=granite4:latest

# Other optional configurations
# OPENAI_API_KEY=your_key_here  # If using OpenAI models
```

## Running the Application

1. **Make sure Ollama is running** (it should start automatically after installation, or run `ollama serve`)

2. **Activate your virtual environment** (if not already activated):

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

3. **Launch the Streamlit app**:

```bash
streamlit run app.py
```

4. **Open your browser** to the URL shown in the terminal (typically `http://localhost:8501`)

## Usage

1. Click the microphone button to record your voice input
2. Or type your question in the chat input
3. The AI assistant will process your query and respond with relevant information
4. View the conversation history in the sidebar

## Project Structure

```
Goal3/
├── app.py                  # Main Streamlit application
├── start.py               # Quick start script
├── requirements.txt       # Python dependencies
├── src/
│   ├── agent/            # LangGraph agent implementation
│   ├── database/         # Database models and initialization
│   ├── tools/            # Medical tools for the agent
│   └── utils/            # Utility functions (LLM, STT)
├── data/                 # Data files
└── scripts/              # Helper scripts
```

## Troubleshooting

### Ollama Connection Issues

If you see errors about connecting to Ollama:
- Make sure Ollama is running: `ollama serve`
- Verify the model is downloaded: `ollama list`
- Pull the model if needed: `granite4:latest`

### Audio Recording Issues

If audio recording doesn't work:
- Make sure FFmpeg is installed
- Check browser permissions for microphone access
- Try refreshing the page

### Package Installation Issues

If you encounter installation errors:
- Make sure you're using Python 3.8 or higher: `python --version`
- Upgrade pip: `pip install --upgrade pip`
- Try installing packages individually if bulk installation fails

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## License

This is a proof-of-concept project for demonstration purposes.

A demo application showcasing LLM integration in a medical context with voice input, SQL database queries, and an intelligent multi-agent system inspired by [ClinicalAgent (arXiv:2404.14777)](https://arxiv.org/abs/2404.14777).

## Features

- 🤖 **Multi-Agent Architecture**: Three-stage orchestration (Planning → Specialist → Reasoning) for complex medical queries
- 🏥 **Realistic Medical Data**: ICD-10-CM diagnosis codes, LOINC lab test codes, NDC/RxNorm medication standards
- 🔧 **Dynamic Tool Control**: UI toggles to enable/disable medical tools (medications, labs, appointments, etc.)
- 🎤 **Voice Input**: Whisper-powered speech-to-text for hands-free interaction
- 💬 **Chat Interface**: Clean Streamlit UI with sidebar tool configuration
- � **Flexible LLM Backend**: LiteLLM integration for easy provider switching (local or cloud)
- 📊 **Medical Database**: SQLite database with realistic patient records following healthcare data standards

## Architecture

### Multi-Agent Workflow
Inspired by the ClinicalAgent paper, our system uses a three-stage orchestration:

1. **Planning Agent**: Decomposes complex queries into executable subtasks, checking tool availability
2. **Specialist Agent**: Executes each subtask using appropriate medical tools (database queries)
3. **Reasoning Agent**: Synthesizes specialist outputs into a coherent clinical response

```
User Query → [Planner] → [Specialist + Tools] → [Reasoning] → Final Answer
```

### Technology Stack
- **LangGraph**: Orchestrates the multi-agent workflow with state management
- **LiteLLM**: Unified interface for multiple LLM providers (Ollama, OpenAI, Anthropic)
- **Whisper**: OpenAI's speech recognition for voice input
- **Streamlit**: Interactive web interface with dynamic tool configuration
- **SQLite**: Local database with ICD-10, LOINC, and NDC-compliant medical records

## Prerequisites

1. **Python 3.9+**
2. **Ollama** (for local LLM): 
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull the model:
   ollama pull llama3.2:3b
   ```

## Installation

1. Clone the repository:
   ```bash
   cd Goal3
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Initialize the database:
   ```bash
   python src/database/init_db.py
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Tool Configuration

Use the sidebar to enable/disable medical tools:
- **Search Patients**: Look up patients by name or MRN
- **Get Medications**: Retrieve current medication lists with NDC codes
- **Get Medical History**: Access diagnosis history with ICD-10-CM codes
- **Get Lab Results**: View laboratory results with LOINC codes
- **Get Appointments**: Check scheduled appointments
- **Get Vital Signs**: Retrieve recent vital sign measurements
- **Search by Condition**: Find patients by ICD-10 diagnosis code

The agent will automatically plan queries based on available tools and inform you if a disabled tool is needed.

## Example Queries

Try asking:
- "What medications is Linda Davis taking?" (Returns: Amlodipine besylate 5mg, Albuterol sulfate HFA 90mcg)
- "Show me lab results for William Martinez" (Displays: Cholesterol, HbA1c, TSH with LOINC codes)
- "What is Maria Wilson's medical history?" (Shows: Osteoarthritis [ICD-10: M17.9], Hypertension [I10], GERD [K21.9])
- "Schedule an appointment for patient ID 5"
- "List all patients with diabetes" (Search by ICD-10 code E11.9)

## Switching LLM Providers

Edit `.env` to switch providers:

**Local (Ollama - default)**:
```env
MODEL_PROVIDER=ollama
MODEL_NAME=granite4:latest
```

**OpenAI**:
```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4
```

**Anthropic**:
```env
MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-3-sonnet-20240229
```

## Project Structure

```
Goal3/
├── app.py                      # Main Streamlit application with tool toggles
├── start.py                    # Alternative entry point
├── requirements.txt            # Python dependencies
├── test_agent.py               # Comprehensive test suite
├── ARCHITECTURE_UPDATE.md      # Architecture refactor documentation
├── .env                        # Environment configuration
├── README.md                   # This file
├── data/
│   └── medical_records.db      # SQLite database (generated)
└── src/
    ├── agent/
    │   ├── __init__.py
    │   ├── graph.py            # LangGraph multi-agent orchestration
    │   └── state.py            # Agent state with plan/specialist/reasoning fields
    ├── tools/
    │   ├── __init__.py
    │   └── medical_tools.py    # Medical query tools with ICD-10/LOINC support
    ├── database/
    │   ├── __init__.py
    │   ├── init_db.py          # Database initialization
    │   └── models.py           # Schema with medical coding standards
    └── utils/
        ├── __init__.py
        ├── llm.py              # LiteLLM wrapper
        └── whisper_stt.py      # Whisper speech-to-text
```

## Medical Coding Standards

This project uses industry-standard medical coding systems:
- **ICD-10-CM**: International Classification of Diseases, 10th Revision, Clinical Modification (diagnosis codes)
- **LOINC**: Logical Observation Identifiers Names and Codes (laboratory tests)
- **NDC/RxNorm**: National Drug Code and RxNorm naming conventions (medications)

Sample data includes realistic codes like:
- Diagnoses: I10 (Hypertension), E11.9 (Type 2 Diabetes), M17.9 (Osteoarthritis)
- Lab Tests: 2093-3 (Cholesterol), 4548-4 (HbA1c), 3094-0 (Urea Nitrogen)
- Medications: Lisinopril 10mg, Metformin hydrochloride 1000mg

## Testing

Run the test suite to validate architecture and tool configuration:
```bash
python test_agent.py
```

Tests cover:
- All tools enabled functionality
- Tool disable/enable behavior
- Medical coding (ICD-10/LOINC) display
- Minimal tool configurations

## Security Note

This is a **proof of concept** demo with synthetic data. Do not use with real patient information without proper:
- HIPAA compliance measures
- Data encryption (at rest and in transit)
- Access controls and authentication
- Audit logging
- PHI handling procedures
- Legal review

The medical codes (ICD-10, LOINC, NDC) are real standards, but all patient data is fabricated for demonstration purposes.

## References

- **ClinicalAgent Paper**: [Multi-Agent Systems for Clinical Decision Support](https://arxiv.org/abs/2404.14777) - Inspired our three-stage architecture
- **ICD-10-CM**: [CDC ICD-10-CM](https://www.cdc.gov/nchs/icd/icd-10-cm.htm)
- **LOINC**: [Logical Observation Identifiers Names and Codes](https://loinc.org/)
- **LangGraph**: [LangGraph Documentation](https://python.langchain.com/docs/langgraph)

## License

MIT License - This is a demonstration project.
