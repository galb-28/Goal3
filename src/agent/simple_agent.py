"""Simple LangChain agent using ReAct pattern."""

from typing import Dict, Any, List
from datetime import datetime
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatLiteLLM
from langchain.callbacks.base import BaseCallbackHandler

from src.tools.medical_tools import create_medical_tools


class TrajectoryCallbackHandler(BaseCallbackHandler):
    """Callback handler to track agent trajectory."""
    
    def __init__(self):
        self.trajectory = []
        self.current_tool = None
        
    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        self.trajectory.append({
            "node": "agent",
            "action": f"Taking action: {action.tool}",
            "details": f"Input: {str(action.tool_input)[:100]}",
            "timestamp": datetime.now().isoformat()
        })
        self.current_tool = action.tool
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when tool starts."""
        tool_name = serialized.get("name", "unknown")
        self.trajectory.append({
            "node": "specialist",
            "action": f"Executing tool: {tool_name}",
            "details": f"Input: {str(input_str)[:100]}",
            "timestamp": datetime.now().isoformat()
        })
        
    def on_tool_end(self, output, **kwargs):
        """Called when tool ends."""
        self.trajectory.append({
            "node": "specialist",
            "action": f"Tool completed: {self.current_tool or 'unknown'}",
            "details": f"Output length: {len(str(output))} chars",
            "timestamp": datetime.now().isoformat()
        })
        
    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes."""
        self.trajectory.append({
            "node": "reasoning",
            "action": "Agent completed reasoning",
            "details": "Generating final response",
            "timestamp": datetime.now().isoformat()
        })


class SimpleMedicalAgent:
    """Simple medical agent using LangChain's ReAct agent."""
    
    def __init__(self, model_name: str = "ollama/MedAIBase/MedGemma1.5:4b", enabled_tools: Dict[str, bool] = None):
        # Get all tools
        all_tools = create_medical_tools()
        
        # Filter tools based on enabled_tools dict
        if enabled_tools:
            self.tools = [
                tool for tool in all_tools 
                if enabled_tools.get(tool.name, True)
            ]
        else:
            self.tools = all_tools
        
        # Create the LLM
        self.llm = ChatLiteLLM(model=model_name, temperature=0.1)
        
        # Use initialize_agent which is more robust
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            agent_kwargs={
                "prefix": """You are a helpful medical assistant with access to a patient database.

Answer the following questions as best you can. You have access to the following tools:""",
                "suffix": """Important notes:
- When using search_by_condition, pass just the condition name (e.g., "diabetes", not "patients with diabetes")
- Medical conditions can be searched by common names (diabetes, hypertension) or ICD-10 codes (E11, I10)

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
            }
        )
    
    def process_message(self, message: str, history: List[Dict[str, str]] = None) -> tuple[str, Dict[str, Any]]:
        """Process a user message and return a response with state.
        
        Args:
            message: The user's message
            history: Optional conversation history (for compatibility, but not used in basic ReAct)
            
        Returns:
            Tuple of (response text, agent state)
        """
        callback_handler = TrajectoryCallbackHandler()
        
        try:
            # Add initial trajectory step
            callback_handler.trajectory.append({
                "node": "planner",
                "action": "Starting agent execution",
                "details": f"Query: {message[:100]}",
                "timestamp": datetime.now().isoformat()
            })
            
            result = self.agent_executor.invoke(
                {"input": message},
                {"callbacks": [callback_handler]}
            )
            
            response = result.get("output", "I couldn't process that request.")
            
            # Build state object
            state = {
                "execution_trajectory": callback_handler.trajectory,
                "current_step": "Response generated",
                "execution_progress": {
                    "stage": "complete",
                    "completed_steps": len(callback_handler.trajectory),
                    "total_steps": len(callback_handler.trajectory),
                    "current_task": "Complete"
                },
                "plan": [{"task": "react", "description": "Using ReAct agent pattern"}],
                "detailed_plan": None,
                "specialist_results": {},
                "messages": [{"role": "assistant", "content": response}]
            }
            
            return response, state
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error: {str(e)}"
            
            # Build error state
            error_state = {
                "execution_trajectory": callback_handler.trajectory + [{
                    "node": "error",
                    "action": "Error occurred",
                    "details": str(e),
                    "timestamp": datetime.now().isoformat()
                }],
                "current_step": f"Error: {str(e)}",
                "execution_progress": {
                    "stage": "error",
                    "completed_steps": 0,
                    "total_steps": 1,
                    "current_task": "Error"
                },
                "plan": [],
                "detailed_plan": None,
                "specialist_results": {},
                "messages": [{"role": "assistant", "content": error_msg}]
            }
            
            return error_msg, error_state
    
    def process_message_stream(self, message: str, history: List[Dict[str, str]] = None):
        """Process a user message and stream state updates.
        
        Args:
            message: The user's message
            history: Optional conversation history
            
        Yields:
            State updates as the agent processes the message
        """
        callback_handler = TrajectoryCallbackHandler()
        
        # Initial state
        initial_trajectory = [{
            "node": "planner",
            "action": "Starting agent execution",
            "details": f"Query: {message[:100]}",
            "timestamp": datetime.now().isoformat()
        }]
        
        yield {
            "simple_agent": {
                "execution_trajectory": initial_trajectory,
                "current_step": "Initializing",
                "execution_progress": {
                    "stage": "initializing",
                    "completed_steps": 0,
                    "total_steps": 1,
                    "current_task": "Starting"
                },
                "plan": [{"task": "react", "description": "Using ReAct agent pattern"}],
                "detailed_plan": None,
                "specialist_results": {},
                "messages": []
            }
        }
        
        try:
            # Process with streaming trajectory updates
            result = self.agent_executor.invoke(
                {"input": message},
                {"callbacks": [callback_handler]}
            )
            
            response = result.get("output", "I couldn't process that request.")
            
            # Yield final state
            yield {
                "simple_agent": {
                    "execution_trajectory": callback_handler.trajectory,
                    "current_step": "Response generated",
                    "execution_progress": {
                        "stage": "complete",
                        "completed_steps": len(callback_handler.trajectory),
                        "total_steps": len(callback_handler.trajectory),
                        "current_task": "Complete"
                    },
                    "plan": [{"task": "react", "description": "Using ReAct agent pattern"}],
                    "detailed_plan": None,
                    "specialist_results": {},
                    "messages": [{"role": "assistant", "content": response}]
                }
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error: {str(e)}"
            
            yield {
                "simple_agent": {
                    "execution_trajectory": callback_handler.trajectory + [{
                        "node": "error",
                        "action": "Error occurred",
                        "details": str(e),
                        "timestamp": datetime.now().isoformat()
                    }],
                    "current_step": f"Error: {str(e)}",
                    "execution_progress": {
                        "stage": "error",
                        "completed_steps": 0,
                        "total_steps": 1,
                        "current_task": "Error"
                    },
                    "plan": [],
                    "detailed_plan": None,
                    "specialist_results": {},
                    "messages": [{"role": "assistant", "content": error_msg}]
                }
            }


def create_simple_medical_agent(**kwargs) -> SimpleMedicalAgent:
    """Factory function to create a simple medical agent."""
    return SimpleMedicalAgent(**kwargs)
