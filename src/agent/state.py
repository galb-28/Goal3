"""Agent state management for LangGraph."""

from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langgraph.graph import add_messages


class PlanStep(TypedDict):
    """A single step in a detailed plan."""
    id: str
    task: str
    description: str
    tool_name: Optional[str]
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of steps that must complete before this one
    required: bool  # Whether this step is required for plan success


class DetailedPlan(TypedDict):
    """A detailed execution plan."""
    steps: List[PlanStep]
    complexity: str  # "simple", "moderate", "complex"
    rationale: str  # Why this plan was chosen
    estimated_steps: int


class AgentState(TypedDict):
    """State for the medical assistant agent."""
    
    # Messages in the conversation
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # Whether we need more information
    needs_clarification: bool
    
    # Tool results from previous step
    tool_results: List[str]
    
    # Current user query
    current_query: str

    # ---- ClinicalAgent-style orchestration fields ----
    # A structured plan produced by the planning agent, decomposed into sub-tasks.
    plan: List[Dict[str, Any]]

    # Detailed plan for complex tasks
    detailed_plan: Optional[DetailedPlan]

    # Name of the next sub-task to run (used by the executor loop)
    next_task: str

    # Collected outputs from specialist agents keyed by task name.
    specialist_results: Dict[str, str]

    # Optional critique/verifier notes (lightweight evaluation step)
    verifier_notes: str

    # ---- Conversation Memory ----
    # Conversation history for context
    conversation_memory: List[Dict[str, Any]]
    
    # Current execution step for visualization
    current_step: str
    
    # Execution progress for visualization
    execution_progress: Dict[str, Any]
