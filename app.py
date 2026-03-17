"""
Medical AI Assistant - Streamlit UI
Proof of concept demo with voice input and LangGraph agent
"""

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import json
import hashlib
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
from src.utils.llm import LLMWrapper

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

    if "visit_llm" not in st.session_state:
        st.session_state.visit_llm = LLMWrapper(temperature=0.1, max_tokens=900)
    
    # Initialize tool toggles
    if "enabled_tools" not in st.session_state:
        # All tools enabled by default
        st.session_state.enabled_tools = {
            "search_persons": True,
            "get_person_encounters": True,
            "get_monitor_messages": True,
            "get_encounter_observations": True,
            "get_encounter_alarms": True,
        }

    if "visit_note" not in st.session_state:
        st.session_state.visit_note = create_empty_visit_note()
    if "visit_transcripts" not in st.session_state:
        st.session_state.visit_transcripts = []
    if "visit_last_audio_fingerprint" not in st.session_state:
        st.session_state.visit_last_audio_fingerprint = ""
        if "visit_checkpoints" not in st.session_state:
            st.session_state.visit_checkpoints = {key: False for key, _ in VISIT_SECTIONS}
        if "visit_note_ready" not in st.session_state:
            st.session_state.visit_note_ready = False


def format_plan_markdown(detailed_plan, simple_plan, specialist_results):
    """Format plan section for display."""
    lines = ["**Plan**"]

    if detailed_plan:
        complexity = detailed_plan.get("complexity", "unknown").title()
        rationale = detailed_plan.get("rationale", "none")
        lines.append(f"- Type: Detailed · Complexity: {complexity}")
        lines.append(f"- Rationale: {rationale}")

        steps = detailed_plan.get("steps", [])
        if steps:
            lines.append("- Steps:")
            for i, step in enumerate(steps, 1):
                status = "✅" if step.get("id") in specialist_results else "⏳"
                lines.append(f"  - {status} {i}. {step.get('task', '')}: {step.get('description', '')}")
        else:
            lines.append("- _No steps available yet._")
    elif simple_plan:
        lines.append("- Type: Simple")
        lines.append("- Steps:")
        for i, step in enumerate(simple_plan, 1):
            task = step.get("task", "")
            desc = step.get("description", "")
            status = "✅" if task in specialist_results else "⏳"
            lines.append(f"  - {status} {i}. {task}: {desc}")
    else:
        lines.append("- _No plan yet._")

    return "\n".join(lines)


def format_results_markdown(specialist_results):
    """Format specialist results for display."""
    lines = ["**Results**"]
    if not specialist_results:
        lines.append("- _No results yet._")
        return "\n".join(lines)

    for task, result in specialist_results.items():
        lines.append(f"- **{task}**")
        lines.append(f"  - {result}")

    return "\n".join(lines)


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
                turn = message.get("turn")
                if turn:
                    st.markdown(f"**Query:** {turn.get('query', '')}")
                    st.markdown(format_plan_markdown(
                        turn.get("detailed_plan"),
                        turn.get("plan", []),
                        turn.get("specialist_results", {})
                    ))
                    st.markdown(format_results_markdown(turn.get("specialist_results", {})))
                    st.markdown("**Answer**")
                    st.markdown(turn.get("answer", content))
                else:
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


VISIT_SECTIONS = [
    ("patient_info", "Patient Info"),
    ("vital_signs", "Vital Signs"),
    ("general_appearance", "General Appearance"),
    ("heent", "Head, Eyes, Ears, Nose, Throat (HEENT)"),
    ("neck", "Neck"),
    ("cardiovascular", "Cardiovascular"),
    ("respiratory", "Respiratory"),
    ("abdomen", "Abdomen"),
    ("extremities", "Extremities"),
    ("neurological_skin", "Neurological/Skin"),
    ("plan_assessment", "Plan & Assessment"),
    ("commonly_examined_areas", "Commonly Examined Areas")
]


def create_empty_visit_note() -> dict:
    """Create an empty structured visit note template."""
    return {key: "" for key, _ in VISIT_SECTIONS}


def _audio_fingerprint(audio_bytes: bytes) -> str:
    """Create a stable fingerprint for audio bytes."""
    digest = hashlib.md5(audio_bytes).hexdigest()[:10]
    return f"{len(audio_bytes)}-{digest}"


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from text."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return {}


def merge_section(existing: str, update: str) -> str:
    """Merge new content into a section, avoiding duplication."""
    if not update or not update.strip():
        return existing
    update = update.strip()
    if not existing:
        return update
    if update.lower() in existing.lower():
        return existing
    return f"{existing}\n{update}"


def extract_visit_update(transcript: str, current_note: dict) -> dict:
    """Use LLM to extract structured visit updates from transcript."""
    system_prompt = (
        "You extract structured doctor visit notes from dictation. "
        "Return JSON only. Include all keys exactly as provided. "
        "Each value must be a concise string; use empty string if not mentioned."
    )

    keys = [key for key, _ in VISIT_SECTIONS]
    schema_hint = {
        key: "" for key in keys
    }

    user_prompt = (
        "CURRENT_NOTE (for context, do not repeat unless updated):\n"
        f"{json.dumps(current_note, ensure_ascii=False)}\n\n"
        "DICTATION TRANSCRIPT:\n"
        f"{transcript}\n\n"
        "Return JSON with keys: " + ", ".join(keys)
    )

    llm = st.session_state.visit_llm
    response_text = llm.generate_text([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    parsed = _extract_json(response_text)
    if not parsed:
        parsed = _extract_json(json.dumps(schema_hint))
    return parsed


def update_visit_checkpoints_from_transcript(transcript: str) -> None:
    """Update checkpoint completion using LLM classification only."""
    keys = [key for key, _ in VISIT_SECTIONS]
    system_prompt = (
        "You classify which visit sections are mentioned in a dictation segment. "
        "Return JSON only with boolean values for the provided keys."
    )
    user_prompt = (
        "DICTATION TRANSCRIPT:\n"
        f"{transcript}\n\n"
        "Return JSON with keys: " + ", ".join(keys)
    )

    llm = st.session_state.visit_llm
    try:
        response_text = llm.generate_text([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        updates = _extract_json(response_text)
    except Exception:
        updates = {}

    if isinstance(updates, dict):
        for key in keys:
            if bool(updates.get(key, False)):
                st.session_state.visit_checkpoints[key] = True


def fill_visit_note_from_segments() -> None:
    """Fill the structured note from all dictation segments using LLM."""
    if not st.session_state.visit_transcripts:
        return
    combined = "\n".join(st.session_state.visit_transcripts)
    current_note = create_empty_visit_note()
    try:
        updates = extract_visit_update(combined, current_note)
    except Exception:
        updates = {}

    for key, _ in VISIT_SECTIONS:
        update_value = updates.get(key, "") if isinstance(updates, dict) else ""
        current_note[key] = merge_section(current_note.get(key, ""), update_value)

    st.session_state.visit_note = current_note


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

            # Placeholders for streaming updates
            query_placeholder = st.empty()
            plan_placeholder = st.empty()
            results_placeholder = st.empty()
            answer_placeholder = st.empty()

            query_placeholder.markdown(f"**Query:** {user_input}")

            response_text = None
            final_state = None

            # Stream state updates
            for state_update in agent.process_message_stream(user_input, history=conversation_history):
                for node_name, node_state in state_update.items():
                    final_state = node_state

                    detailed_plan = node_state.get("detailed_plan")
                    simple_plan = node_state.get("plan", [])
                    specialist_results = node_state.get("specialist_results", {})

                    plan_placeholder.markdown(format_plan_markdown(detailed_plan, simple_plan, specialist_results))
                    results_placeholder.markdown(format_results_markdown(specialist_results))

            # Extract final response
            if final_state:
                for msg in reversed(final_state.get("messages", [])):
                    if isinstance(msg, dict):
                        if msg.get("role") == "assistant":
                            response_text = msg["content"]
                            break
                    else:
                        msg_type = getattr(msg, "type", None) if hasattr(msg, "type") else None
                        if msg_type == "ai" or msg_type == "assistant":
                            response_text = msg.content if hasattr(msg, "content") else str(msg)
                            break

            if not response_text:
                response_text = "I apologize, but I couldn't generate a response. Please try again."

            answer_placeholder.markdown(f"**Answer**\n\n{response_text}")

            # Persist per-turn view data in history
            turn_data = {
                "query": user_input,
                "plan": final_state.get("plan", []) if final_state else [],
                "detailed_plan": final_state.get("detailed_plan") if final_state else None,
                "specialist_results": final_state.get("specialist_results", {}) if final_state else {},
                "answer": response_text
            }

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "turn": turn_data
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
            - 🗄️ **SQLite** database with fabricated monitoring data
            
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
            "search_persons": {
                "name": "🔍 Person Search",
                "description": "Search by name, ID, or ext_ref",
                "standard": "Person registry"
            },
            "get_person_encounters": {
                "name": "🏥 Encounters",
                "description": "Admissions, discharges, outcomes",
                "standard": "Encounter records"
            },
            "get_monitor_messages": {
                "name": "📟 Monitor Messages",
                "description": "Monitor message stream",
                "standard": "Device messaging"
            },
            "get_encounter_observations": {
                "name": "💓 Observations",
                "description": "Vitals and clinical assessments",
                "standard": "Observation events"
            },
            "get_encounter_alarms": {
                "name": "🚨 Alarms",
                "description": "Alarm events and states",
                "standard": "Alarm signaling"
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
            "Find Michael Anderson",
            "Show encounters for Linda Davis",
            "Show observations for Michael Anderson",
            "Get alarms for encounter 1",
            "Show monitor messages for encounter 1",
            "Find person PER12345"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                process_user_input(example)
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("---")
    
    tab_chat, tab_visit = st.tabs(["💬 Chat", "🩺 Doctor Visit"])

    with tab_chat:
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

    with tab_visit:
        st.subheader("🩺 Doctor Visit Dictation")
        st.caption("Speak in short segments. Checkpoints update immediately; the note is filled on submit.")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            if st.session_state.visit_note_ready:
                st.markdown("#### 🧾 Note")
                hide_when_pending = {
                    "heent",
                    "neck",
                    "cardiovascular",
                    "respiratory",
                    "abdomen",
                    "extremities",
                    "neurological_skin"
                }
                for key, label in VISIT_SECTIONS:
                    raw_value = st.session_state.visit_note.get(key, "").strip()
                    if key in hide_when_pending and not raw_value:
                        continue
                    value = raw_value or "_Pending_"
                    st.markdown(f"**{label}**")
                    st.markdown(value)
                    st.markdown("---")
            else:
                st.info("Submit the visit to generate the structured note.")

        with col_right:
            checkpoints = st.session_state.visit_checkpoints
            missing = [label for key, label in VISIT_SECTIONS if not checkpoints.get(key, False)]
            completed = [label for key, label in VISIT_SECTIONS if checkpoints.get(key, False)]

            st.markdown("#### ✅ Checkpoints")
            if completed:
                for label in completed:
                    st.markdown(f"- ✅ {label}")
            if missing:
                st.markdown("**Not yet addressed**")
                for label in missing:
                    st.markdown(f"- ⬜ {label}")

        st.markdown("---")
        st.markdown("#### ✍️ Add Dictation")

        st.write("Click the microphone to record a visit segment:")
        visit_audio = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x",
            key="visit_audio_recorder"
        )

        if visit_audio:
            st.audio(visit_audio, format="audio/wav")
            with st.spinner("🎧 Transcribing visit segment..."):
                try:
                    visit_transcript = st.session_state.whisper.transcribe_audio_bytes(visit_audio)
                    st.success(f"📝 Transcription: *{visit_transcript}*")

                    fingerprint = _audio_fingerprint(visit_audio)
                    should_update = fingerprint != st.session_state.visit_last_audio_fingerprint

                    if should_update:
                        update_visit_checkpoints_from_transcript(visit_transcript)
                        st.session_state.visit_transcripts.append(visit_transcript)
                        st.session_state.visit_last_audio_fingerprint = fingerprint
                        st.toast("Checkpoint update", icon="✅")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")

        st.write("Or paste typed dictation:")
        typed_dictation = st.text_area(
            "Typed dictation",
            placeholder="Paste or type a visit segment...",
            label_visibility="collapsed",
            key="typed_visit_dictation"
        )
        if st.button("➕ Add Dictation", key="add_typed_dictation") and typed_dictation.strip():
            update_visit_checkpoints_from_transcript(typed_dictation.strip())
            st.session_state.visit_transcripts.append(typed_dictation.strip())
            st.rerun()

        submit_disabled = len(st.session_state.visit_transcripts) == 0
        if st.button("✅ Submit Visit", key="submit_visit", disabled=submit_disabled):
            with st.spinner("🧠 Building note from dictation..."):
                fill_visit_note_from_segments()
            st.session_state.visit_note_ready = True
            st.toast("Note filled", icon="✅")
            st.rerun()

        col_reset, col_segments = st.columns([1, 3])
        with col_reset:
            if st.button("🗑️ Reset Visit", key="reset_visit"):
                st.session_state.visit_note = create_empty_visit_note()
                st.session_state.visit_transcripts = []
                st.session_state.visit_last_audio_fingerprint = ""
                st.session_state.visit_checkpoints = {key: False for key, _ in VISIT_SECTIONS}
                st.session_state.visit_note_ready = False
                st.rerun()
        with col_segments:
            if st.session_state.visit_transcripts:
                with st.expander(f"🗣️ Dictation Segments ({len(st.session_state.visit_transcripts)})", expanded=False):
                    for i, segment in enumerate(st.session_state.visit_transcripts, 1):
                        st.markdown(f"**{i}.** {segment}")
    
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
