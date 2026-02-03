"""
Medical AI Assistant - Streamlit UI
Proof of concept demo with voice input and LangGraph agent
"""

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.agent.graph import create_medical_agent
from src.utils.whisper_stt import get_whisper_instance

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .stButton>button {
        width: 100%;
    }
    .trajectory-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #dee2e6;
    }
    .trajectory-step {
        padding: 10px;
        margin: 8px 0;
        border-left: 4px solid;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .trajectory-step:hover {
        background-color: rgba(255, 255, 255, 1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        with st.spinner("🤖 Initializing AI agent..."):
            st.session_state.agent = create_medical_agent()
    
    if "whisper" not in st.session_state:
        with st.spinner("🎤 Loading speech recognition..."):
            st.session_state.whisper = get_whisper_instance()
    
    # Initialize tool toggles
    if "enabled_tools" not in st.session_state:
        # All tools enabled by default
        st.session_state.enabled_tools = {
            "search_patients": True,
            "get_patient_medications": True,
            "get_medical_history": True,
            "get_lab_results": True,
            "get_appointments": True,
            "get_vital_signs": True,
            "search_by_condition": True,
        }


def display_chat_history():
    """Display chat message history."""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar="🏥"):
                st.markdown(content)


def display_agent_visualization():
    """Display agent internal steps and execution progress in real-time."""
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {
            "execution_trajectory": [],
            "current_step": "Waiting for query...",
            "execution_progress": {},
            "plan": [],
            "detailed_plan": None,
            "specialist_results": {}
        }
    
    state = st.session_state.agent_state
    
    # Check if there's any content to display
    trajectory = state.get("execution_trajectory", [])
    has_plan = state.get("plan") or state.get("detailed_plan")
    has_results = state.get("specialist_results")
    has_progress = state.get("execution_progress", {}).get("total_steps", 0) > 0
    
    # If nothing to show, just return
    if not trajectory and not has_plan and not has_results and not has_progress:
        return
    
    # Container for trajectory
    st.markdown('<div class="trajectory-container">', unsafe_allow_html=True)
    
    # Execution Trajectory Timeline
    if trajectory and len(trajectory) > 0:
        st.markdown("#### 🎯 Live Execution Trajectory")
        
        # Create a container for the trajectory that updates
        for i, step in enumerate(trajectory, 1):
            node = step.get("node", "unknown")
            action = step.get("action", "")
            details = step.get("details", "")
            
            # Node-specific icons and colors
            node_icons = {
                "planner": "🧠",
                "specialist": "🔧",
                "reasoning": "💭"
            }
            
            node_colors = {
                "planner": "#4A90E2",
                "specialist": "#50C878",
                "reasoning": "#9B59B6"
            }
            
            icon = node_icons.get(node, "⚙️")
            color = node_colors.get(node, "#95a5a6")
            
            # Display step with colored badge
            st.markdown(
                f'<div class="trajectory-step" style="border-left-color: {color};">'
                f'<strong>{i}. {icon} <span style="color: {color};">[{node.upper()}]</span></strong> {action}'
                f'</div>',
                unsafe_allow_html=True
            )
            if details:
                st.caption(f"   └─ {details}")
    
    # Current step and progress - Side by side (only if there's trajectory)
    if trajectory:
        st.markdown("---")
    
    # Current step and progress - Side by side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        current_step = state.get("current_step", "Not started")
        st.markdown(f"**📍 Current Step**")
        st.info(current_step)
    
    with col2:
        progress = state.get("execution_progress", {})
        if progress and progress.get("stage"):
            stage = progress.get("stage", "unknown")
            st.markdown(f"**⚙️ Stage**")
            st.info(stage.title())
    
    # Progress bar
    progress = state.get("execution_progress", {})
    if progress and progress.get("total_steps", 0) > 0:
        completed = progress.get("completed_steps", 0)
        total = progress.get("total_steps", 0)
        current_task = progress.get("current_task", "None")
        
        progress_pct = completed / total if total > 0 else 0
        st.progress(progress_pct, text=f"Progress: {completed}/{total} steps completed")
        st.caption(f"⏳ Current task: {current_task}")
    
    # Plan details
    detailed_plan = state.get("detailed_plan")
    simple_plan = state.get("plan", [])
    
    if detailed_plan or simple_plan:
        st.markdown("---")
        st.markdown("#### 📋 Execution Plan")
    
    if detailed_plan:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Type:** Detailed Plan")
        with col2:
            st.markdown(f"**Complexity:** {detailed_plan.get('complexity', 'unknown').title()}")
        st.caption(f"💡 {detailed_plan.get('rationale', 'none')}")
        
        steps = detailed_plan.get("steps", [])
        if steps:
            specialist_results = state.get("specialist_results", {})
            for i, step in enumerate(steps, 1):
                status = "✅" if step["id"] in specialist_results else "⏳"
                st.markdown(f"{status} **{i}. {step['task']}** - {step['description']}")
    elif simple_plan:
        st.markdown(f"**Type:** Simple Plan")
        specialist_results = state.get("specialist_results", {})
        for i, step in enumerate(simple_plan, 1):
            task = step.get("task", "")
            desc = step.get("description", "")
            status = "✅" if task in specialist_results else "⏳"
            st.markdown(f"{status} **{i}. {task}** - {desc}")
    
    # Specialist results (collapsible)
    specialist_results = state.get("specialist_results", {})
    if specialist_results:
        st.markdown("---")
        with st.expander(f"📊 View Specialist Results ({len(specialist_results)} tasks completed)", expanded=False):
            for task, result in specialist_results.items():
                st.markdown(f"**🔧 {task}**")
                st.text_area(f"Result for {task}", result, height=120, disabled=True, key=f"result_{task}", label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)


def process_user_input(user_input: str):
    """Process user input and get agent response with real-time trajectory updates."""
    if not user_input.strip():
        return
    
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Get agent response
    with st.chat_message("assistant", avatar="🏥"):
        try:
            # Recreate agent with current tool settings
            agent = create_medical_agent(enabled_tools=st.session_state.enabled_tools)
            
            # Build conversation history for context (exclude current message)
            conversation_history = []
            for msg in st.session_state.messages[:-1]:  # Exclude the current user message
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", datetime.now().isoformat())
                })
            
            # Use Streamlit's status widget for real-time updates
            with st.status("🤖 Processing query...", expanded=True) as status:
                # Stream state updates
                response_text = None
                final_state = None
                
                for state_update in agent.process_message_stream(user_input, history=conversation_history):
                    # Extract the actual state from the dict
                    for node_name, node_state in state_update.items():
                        final_state = node_state
                        
                        # Get trajectory
                        trajectory = node_state.get("execution_trajectory", [])
                        
                        if trajectory:
                            latest_step = trajectory[-1]
                            node = latest_step.get("node", "unknown")
                            action = latest_step.get("action", "")
                            details = latest_step.get("details", "")
                            
                            # Node-specific icons
                            node_icons = {
                                "planner": "🧠",
                                "specialist": "🔧",
                                "reasoning": "💭"
                            }
                            icon = node_icons.get(node, "⚙️")
                            
                            # Write the step to status
                            st.write(f"{icon} **[{node.upper()}]** {action}")
                            if details:
                                st.caption(f"└─ {details}")
                        
                        # Update session state
                        st.session_state.agent_state = node_state
                
                # Extract final response
                if final_state:
                    for msg in reversed(final_state.get("messages", [])):
                        if isinstance(msg, dict):
                            if msg.get("role") == "assistant":
                                response_text = msg["content"]
                                break
                        else:
                            msg_type = getattr(msg, 'type', None) if hasattr(msg, 'type') else None
                            if msg_type == "ai" or msg_type == "assistant":
                                response_text = msg.content if hasattr(msg, 'content') else str(msg)
                                break
                
                if not response_text:
                    response_text = "I apologize, but I couldn't generate a response. Please try again."
                
                # Update status to complete
                status.update(label="✅ Complete!", state="complete", expanded=False)
            
            # Display final response
            st.markdown("---")
            st.markdown(response_text)
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            })
                
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            import traceback
            traceback.print_exc()


def main():
    """Main application."""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            This is a **proof of concept** medical AI assistant powered by:
            - 🤖 **LangGraph** for agent orchestration
            - 💬 **LiteLLM** for flexible LLM integration
            - 🎤 **Whisper** for voice input
            - 🗄️ **SQLite** database with fabricated patient data
            
            **Disclaimer**: This uses fabricated data for demonstration purposes only.
            """
        )
        
        st.header("⚙️ Configuration")
        model_provider = os.getenv("MODEL_PROVIDER", "ollama")
        model_name = os.getenv("MODEL_NAME", "llama3.2:3b")
        st.text(f"Provider: {model_provider}")
        st.text(f"Model: {model_name}")
        
        st.markdown("---")
        
        st.header("🔧 Medical Tools")
        st.write("Enable/disable tools for agent reasoning:")
        
        # Tool definitions with realistic descriptions
        tool_info = {
            "search_patients": {
                "name": "🔍 Patient Search",
                "description": "Search by name or MRN",
                "standard": "HL7 FHIR Patient"
            },
            "get_patient_medications": {
                "name": "💊 Medications",
                "description": "Active prescriptions & Rx history",
                "standard": "NDC/RxNorm codes"
            },
            "get_medical_history": {
                "name": "📋 Medical History",
                "description": "Conditions & diagnoses",
                "standard": "ICD-10-CM codes"
            },
            "get_lab_results": {
                "name": "🧪 Lab Results",
                "description": "Laboratory test results",
                "standard": "LOINC codes"
            },
            "get_appointments": {
                "name": "📅 Appointments",
                "description": "Past & scheduled visits",
                "standard": "CPT codes"
            },
            "get_vital_signs": {
                "name": "💓 Vital Signs",
                "description": "BP, HR, temp, weight",
                "standard": "LOINC vital signs"
            },
            "search_by_condition": {
                "name": "🔬 Cohort Search",
                "description": "Find patients by condition",
                "standard": "ICD-10 queries"
            }
        }
        
        for tool_key, info in tool_info.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{info['name']}**")
                st.caption(f"{info['description']} · {info['standard']}")
            with col2:
                enabled = st.checkbox(
                    "Enable",
                    value=st.session_state.enabled_tools[tool_key],
                    key=f"toggle_{tool_key}",
                    label_visibility="collapsed"
                )
                st.session_state.enabled_tools[tool_key] = enabled
        
        # Show count of enabled tools
        enabled_count = sum(st.session_state.enabled_tools.values())
        st.info(f"✅ {enabled_count}/{len(tool_info)} tools enabled")
        
        st.markdown("---")
        
        st.header("📋 Example Queries")
        examples = [
            "What medications is patient 1 taking?",
            "Show lab results for patient 2",
            "List all patients with diabetes",
            "Get appointments for patient 3",
            "Show vital signs for Michael Martinez",
            "Find Linda Davis"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                process_user_input(example)
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Agent trajectory visualization (only show if there's data)
    if "agent_state" in st.session_state:
        state = st.session_state.agent_state
        trajectory = state.get("execution_trajectory", [])
        has_plan = state.get("plan") or state.get("detailed_plan")
        has_results = state.get("specialist_results")
        
        if trajectory or has_plan or has_results:
            st.markdown("---")
            st.subheader("🤖 Agent Execution Details")
            
            # Show trajectory
            if trajectory:
                with st.expander("🎯 Execution Trajectory", expanded=True):
                    for i, step in enumerate(trajectory, 1):
                        node = step.get("node", "unknown")
                        action = step.get("action", "")
                        details = step.get("details", "")
                        
                        node_icons = {"planner": "🧠", "specialist": "🔧", "reasoning": "💭"}
                        icon = node_icons.get(node, "⚙️")
                        
                        st.markdown(f"**{i}. {icon} [{node.upper()}]** {action}")
                        if details:
                            st.caption(f"   └─ {details}")
            
            # Show plan
            detailed_plan = state.get("detailed_plan")
            simple_plan = state.get("plan", [])
            
            if detailed_plan or simple_plan:
                with st.expander("📋 Execution Plan", expanded=True):
                    if detailed_plan:
                        st.markdown(f"**Type:** Detailed Plan | **Complexity:** {detailed_plan.get('complexity', 'unknown').title()}")
                        st.caption(f"💡 {detailed_plan.get('rationale', 'none')}")
                        
                        steps = detailed_plan.get("steps", [])
                        specialist_results = state.get("specialist_results", {})
                        for i, step in enumerate(steps, 1):
                            status = "✅" if step["id"] in specialist_results else "⏳"
                            st.markdown(f"{status} **{i}. {step['task']}** - {step['description']}")
                    elif simple_plan:
                        st.markdown(f"**Type:** Simple Plan")
                        specialist_results = state.get("specialist_results", {})
                        for i, step in enumerate(simple_plan, 1):
                            task = step.get("task", "")
                            desc = step.get("description", "")
                            status = "✅" if task in specialist_results else "⏳"
                            st.markdown(f"{status} **{i}. {task}** - {desc}")
            
            # Show specialist results
            specialist_results = state.get("specialist_results", {})
            if specialist_results:
                with st.expander(f"📊 Specialist Results ({len(specialist_results)} tasks)", expanded=False):
                    for task, result in specialist_results.items():
                        st.markdown(f"**🔧 {task}**")
                        st.code(result, language=None)
    
    st.markdown("---")
    
    # Chat section
    st.subheader("💬 Chat")
    
    # Display chat history
    display_chat_history()
    
    # Input area
    st.markdown("---")
    
    # Create tabs for text and voice input
    input_tab1, input_tab2 = st.tabs(["⌨️ Text Input", "🎤 Voice Input"])
    
    with input_tab1:
        # Text input
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_input(
                    "Type your message:",
                    placeholder="Ask about patients, medications, lab results...",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send 📤")
            
            if submit_button and user_input:
                process_user_input(user_input)
                st.rerun()
    
    with input_tab2:
        st.write("Click the microphone to record your question:")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            with st.spinner("🎧 Transcribing audio..."):
                try:
                    # Transcribe audio
                    transcription = st.session_state.whisper.transcribe_audio_bytes(audio_bytes)
                    
                    st.success(f"📝 Transcription: *{transcription}*")
                    
                    # Process the transcription
                    if st.button("✅ Send Transcription"):
                        process_user_input(transcription)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Medical AI Assistant Demo | Built with Streamlit, LangGraph & LiteLLM
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # Check if database exists
    db_path = Path(os.getenv("DATABASE_PATH", "./data/medical_records.db"))
    if not db_path.exists():
        st.error(
            """
            ⚠️ **Database not found!**
            
            Please initialize the database first by running:
            ```
            python src/database/init_db.py
            ```
            """
        )
        st.stop()
    
    main()
