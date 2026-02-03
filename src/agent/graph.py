"""LangGraph agent implementation for medical assistant."""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
import json
from datetime import datetime

from src.agent.state import AgentState
from src.utils.llm import LLMWrapper
from src.tools.medical_tools import create_medical_tools


class MedicalAgent:
    """LangGraph-based medical assistant agent with LangChain tool integration."""
    
    def __init__(self, llm: LLMWrapper = None, enabled_tools: Dict[str, bool] = None):
        self.llm = llm or LLMWrapper()
        all_tools = create_medical_tools()
        
        # Filter tools based on enabled_tools dict
        if enabled_tools:
            self.tools = [
                tool for tool in all_tools 
                if enabled_tools.get(tool.name, True)
            ]
        else:
            self.tools = all_tools
        
        # Create dict for quick lookup
        self.tools_dict = {tool.name: tool for tool in self.tools}
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.

        Updated to a ClinicalAgent-like architecture (arXiv:2404.14777):
        Planning Agent -> Specialist Agents (with tool calls) -> Reasoning Agent.
        
        For detailed plans, allows specialist to loop until plan completion.
        """

        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("specialist", self._specialist_node)
        workflow.add_node("reason", self._reasoning_node)

        workflow.set_entry_point("planner")

        # planner -> specialist (always)
        workflow.add_edge("planner", "specialist")

        # specialist -> either continue planning or go to reason
        workflow.add_conditional_edges(
            "specialist",
            self._should_continue_planning,
            {
                "continue": "specialist",  # Continue executing detailed plan
                "reason": "reason"         # Plan complete, move to reasoning
            }
        )

        # reason -> end
        workflow.add_edge("reason", END)

        return workflow.compile()
    
    def _should_continue_planning(self, state: AgentState) -> str:
        """Determine if detailed plan execution should continue or move to reasoning."""
        detailed_plan = state.get("detailed_plan")
        specialist_results = state.get("specialist_results", {})
        
        if detailed_plan and detailed_plan.get('steps'):
            # Check if all required steps are completed
            all_step_ids = {step["id"] for step in detailed_plan['steps']}
            completed_steps = set(specialist_results.keys())
            
            if all_step_ids.issubset(completed_steps):
                return "reason"  # All steps done, move to reasoning
            else:
                return "continue"  # More steps to execute
        
        # No detailed plan or simple plan - move to reasoning
        return "reason"

    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Normalize messages to OpenAI-style dicts."""
        llm_messages: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, dict):
                llm_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            else:
                role = getattr(msg, "type", "user") if hasattr(msg, "type") else "user"
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                llm_messages.append({
                    "role": role,
                    "content": msg.content if hasattr(msg, "content") else str(msg),
                })
        return llm_messages

    def _planner_node(self, state: AgentState) -> AgentState:
        """Planning Agent: decompose user query into sub-tasks (Least-to-Most style)."""
        messages = self._normalize_messages(state["messages"])
        user_query = messages[-1]["content"] if messages else state.get("current_query", "")
        conversation_memory = state.get("conversation_memory", [])
        trajectory = state.get("execution_trajectory", [])

        # Assess plan complexity
        complexity = self._assess_plan_complexity(user_query)
        # Special-case: cohort / condition queries (e.g., "list all patients with diabetes")
        # should be handled by the simple planner (search_by_condition) rather than
        # invoking the LLM detailed planner. This avoids unnecessary LLM calls and
        # prevents timeouts when the query is a straightforward cohort search.
        q_lower = user_query.lower()
        cohort_triggers = ["list all", "all patients", "patients with", "who has", "find all"]
        is_cohort_query = any(trigger in q_lower for trigger in cohort_triggers)

        if is_cohort_query:
            complexity = "simple"

        if complexity in ["complex", "moderate"]:
            # Use LLM-powered detailed planning for complex tasks
            detailed_plan = self._create_detailed_plan(user_query, self.tools_dict)
            
            if detailed_plan and detailed_plan.get('steps'):
                # Use detailed plan
                steps = detailed_plan['steps']
                trajectory.append({
                    "node": "planner",
                    "action": "Created detailed plan",
                    "details": f"Complexity: {complexity}, {len(steps)} steps planned",
                    "timestamp": datetime.now().isoformat()
                })
                return {
                    **state,
                    "current_query": user_query,
                    "plan": [],  # Clear simple plan
                    "detailed_plan": detailed_plan,
                    "next_task": steps[0]["id"] if steps else "",
                    "specialist_results": {},
                    "tool_calls": [],
                    "tool_results": [],
                    "verifier_notes": "",
                    "conversation_memory": conversation_memory + [{"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()}],
                    "current_step": "Planning complete (detailed)",
                    "execution_progress": {"stage": "planning", "completed_steps": 0, "total_steps": len(steps) if steps else len(state.get("plan", [])), "current_task": steps[0]["id"] if steps else ""},
                    "execution_trajectory": trajectory
                }
        
        # Fallback to heuristic planning for simple tasks
        plan: List[Dict[str, Any]] = []
        q = user_query.lower()

        # Start with identity resolution when patient-specific info may be asked.
        # Check for common patient-related patterns
        search_patterns = ["search", "find", "look for", "who is", "locate"]
        has_search = any(pattern in q for pattern in search_patterns)
        has_patient_keyword = "patient" in q
        # Check for capitalized names (likely patient names)
        import re
        has_names = bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', user_query))
        
        patient_related = has_patient_keyword or q.strip().isdigit() or (has_search and has_names)
        
        if patient_related and "search_patients" in self.tools_dict:
            plan.append({"task": "identity", "description": "Resolve patient identity (name or ID)."})

        # Add specialist tasks based on query intent (only if tool is enabled).
        if "medication" in q and "get_patient_medications" in self.tools_dict:
            plan.append({"task": "medications", "description": "Retrieve patient medications."})
        if ("lab" in q or "test result" in q) and "get_lab_results" in self.tools_dict:
            plan.append({"task": "labs", "description": "Retrieve patient lab results."})
        if "appointment" in q and "get_appointments" in self.tools_dict:
            plan.append({"task": "appointments", "description": "Retrieve patient appointments."})
        if ("vital" in q or "blood pressure" in q or "heart rate" in q) and "get_vital_signs" in self.tools_dict:
            plan.append({"task": "vitals", "description": "Retrieve patient vital signs."})
        
        # Check for cohort queries (searching patients by condition)
        is_cohort_query = any(w in q for w in ["list all", "all patients", "who has", "patients with", "with ", "find all"])
        has_condition_keywords = ("medical history" in q or "diagnosis" in q or "condition" in q)
        
        if is_cohort_query and "search_by_condition" in self.tools_dict:
            # This is a cohort query - search for patients with a condition
            plan.append({"task": "cohort", "description": "Search patients by condition."})
        elif has_condition_keywords and "get_medical_history" in self.tools_dict:
            # This is asking for a specific patient's medical history
            plan.append({"task": "history", "description": "Retrieve patient medical history."})

        if not plan:
            # Default to answer directly (no tools).
            plan.append({"task": "direct", "description": "Answer from conversation context."})

        trajectory.append({
            "node": "planner",
            "action": "Created simple plan",
            "details": f"{len(plan)} tasks: {', '.join([p['task'] for p in plan])}",
            "timestamp": datetime.now().isoformat()
        })

        return {
            **state,
            "current_query": user_query,
            "plan": plan,
            "detailed_plan": None,  # No detailed plan for simple tasks
            "next_task": plan[0]["task"],
            "specialist_results": {},
            "tool_calls": [],
            "tool_results": [],
            "verifier_notes": "",
            "conversation_memory": conversation_memory + [{"role": "user", "content": user_query, "timestamp": datetime.now().isoformat()}],
            "current_step": "Planning complete (simple)",
            "execution_progress": {"stage": "planning", "completed_steps": 0, "total_steps": len(plan), "current_task": plan[0]["task"] if plan else ""},
            "execution_trajectory": trajectory
        }

    def _specialist_node(self, state: AgentState) -> AgentState:
        """Specialist Agents executor: run the plan sequentially and collect results.

        This avoids node-level recursion by executing tool calls directly in-process.
        """
        query = state.get("current_query", "")
        detailed_plan = state.get("detailed_plan")
        specialist_results: Dict[str, str] = dict(state.get("specialist_results", {}))
        trajectory = state.get("execution_trajectory", [])

        if detailed_plan and detailed_plan.get('steps'):
            # Execute detailed plan with dependencies
            return self._execute_detailed_plan(state, detailed_plan, specialist_results, trajectory)
        else:
            # Execute simple plan (existing logic)
            return self._execute_simple_plan(state, specialist_results, trajectory)

    def _execute_detailed_plan(self, state: AgentState, detailed_plan: Dict[str, Any], specialist_results: Dict[str, str], trajectory: List[Dict[str, Any]]) -> AgentState:
        """Execute a detailed plan with step dependencies."""
        query = state.get("current_query", "")
        
        # Find executable steps (all dependencies satisfied)
        completed_steps = set(specialist_results.keys())
        
        for step in detailed_plan.get('steps', []):
            step_id = step["id"]
            
            # Skip if already completed
            if step_id in completed_steps:
                continue
                
            # Check if all dependencies are satisfied
            dependencies = step.get("dependencies", [])
            if not all(dep in completed_steps for dep in dependencies):
                continue  # Dependencies not met
            
            # Execute this step
            tool_name = step.get("tool_name")
            parameters = step.get("parameters", {})
            
            trajectory.append({
                "node": "specialist",
                "action": f"Executing step: {step['task']}",
                "details": f"Tool: {tool_name}, Dependencies: {dependencies}",
                "timestamp": datetime.now().isoformat()
            })
            
            if tool_name and tool_name in self.tools_dict:
                # Modify parameters to include query context
                enhanced_params = {**parameters, "query": query}
                
                # Execute tool
                tool_calls = [{"tool_name": tool_name, "query": enhanced_params.get("query", query)}]
                tmp_state = {**state, "tool_calls": tool_calls, "tool_results": []}
                tmp_state = self._tool_node(tmp_state)
                result = "\n\n".join(tmp_state.get("tool_results", []))
                specialist_results[step_id] = f"Step '{step['task']}': {result}"
                
                trajectory.append({
                    "node": "specialist",
                    "action": f"Completed step: {step['task']}",
                    "details": f"Result: {result[:100]}..." if len(result) > 100 else f"Result: {result}",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                # No tool needed for this step
                specialist_results[step_id] = f"Step '{step['task']}': Completed (no tool required)"
                trajectory.append({
                    "node": "specialist",
                    "action": f"Completed step: {step['task']}",
                    "details": "No tool required",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Only execute one step per iteration to maintain state
            break

        # Check if plan is complete
        all_step_ids = {step["id"] for step in detailed_plan["steps"]}
        completed_steps = set(specialist_results.keys())
        plan_complete = all_step_ids.issubset(completed_steps)
        
        return {
            **state,
            "specialist_results": specialist_results,
            "tool_calls": [],
            "tool_results": [],
            "next_task": "" if plan_complete else "continue_detailed_plan",  # Signal to continue if not done
            "current_step": f"Executed {len(completed_steps)}/{len(all_step_ids)} detailed plan steps",
            "execution_progress": {
                "stage": "execution",
                "completed_steps": len(completed_steps),
                "total_steps": len(all_step_ids),
                "current_task": list(all_step_ids - completed_steps)[0] if not plan_complete else "Complete"
            },
            "execution_trajectory": trajectory
        }

    def _execute_simple_plan(self, state: AgentState, specialist_results: Dict[str, str], trajectory: List[Dict[str, Any]]) -> AgentState:
        """Execute simple heuristic-based plan using LangChain tools."""
        query = state.get("current_query", "")
        plan = state.get("plan", [])
        
        print(f"DEBUG _execute_simple_plan: query={query}")
        print(f"DEBUG _execute_simple_plan: plan={plan}")

        for step in plan:
            task = step.get("task")
            print(f"DEBUG: Processing task={task}")
            if not task or task in specialist_results:
                continue

            # For direct answers, skip tools
            if task == "direct":
                specialist_results[task] = "No tool calls needed."
                trajectory.append({
                    "node": "specialist",
                    "action": f"Skipping tool execution for task: {task}",
                    "details": "Direct answer from context",
                    "timestamp": datetime.now().isoformat()
                })
                continue

            # Map tasks to tools
            tool_mapping = {
                "identity": "search_patients",
                "medications": "get_patient_medications",
                "labs": "get_lab_results",
                "appointments": "get_appointments",
                "vitals": "get_vital_signs",
                "history": "get_medical_history",
                "cohort": "search_by_condition"
            }
            
            tool_name = tool_mapping.get(task)
            if not tool_name or tool_name not in self.tools_dict:
                specialist_results[task] = f"Tool {tool_name} not available."
                trajectory.append({
                    "node": "specialist",
                    "action": f"Tool not available: {tool_name}",
                    "details": f"Task: {task}",
                    "timestamp": datetime.now().isoformat()
                })
                continue

            # Get the tool
            tool = self.tools_dict[tool_name]
            
            trajectory.append({
                "node": "specialist",
                "action": f"Executing tool: {tool_name}",
                "details": f"Task: {task}, Description: {step.get('description', 'N/A')}",
                "timestamp": datetime.now().isoformat()
            })
            
            # Create a prompt asking LLM to extract the right parameter
            tool_prompt = f"""You are extracting parameters for a medical database tool.

Tool: {tool.name}
Description: {tool.description}
User Query: {query}

Based on the query, what is the exact parameter value needed for this tool?
- For search_patients: provide the patient name or ID
- For get_patient_medications/lab_results/appointments/vitals/history: provide patient name or ID  
- For search_by_condition: provide ONLY the medical condition name (e.g., "diabetes", "hypertension", "asthma") - NOT "patients with diabetes" or "all patients with diabetes"

CRITICAL: For search_by_condition, extract ONLY the condition name, removing phrases like "patients with", "all", "list", etc.

Respond with ONLY the parameter value, no explanation, no quotes."""

            try:
                # Get parameter from LLM
                param_value = self.llm.generate_text([{"role": "user", "content": tool_prompt}]).strip()
                # Clean up quotes
                param_value = param_value.strip('"').strip("'").strip()
                
                print(f"DEBUG: Tool={tool_name}, Extracted param={param_value}")
                
                # Determine the parameter name from tool's args_schema
                if tool_name == "search_patients":
                    param_name = "query"
                elif tool_name == "search_by_condition":
                    param_name = "condition"
                else:
                    param_name = "patient_identifier"
                
                # Invoke the tool with proper parameter name
                result = tool.invoke({param_name: param_value})
                print(f"DEBUG: Result={result[:200]}")
                specialist_results[task] = f"Tool: {tool_name}\nParameter: {param_value}\n\n{result}"
                
                trajectory.append({
                    "node": "specialist",
                    "action": f"Tool execution completed: {tool_name}",
                    "details": f"Parameter: {param_value}, Result length: {len(result)} chars",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                specialist_results[task] = f"Error calling {tool_name}: {str(e)}"
                trajectory.append({
                    "node": "specialist",
                    "action": f"Tool execution failed: {tool_name}",
                    "details": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                import traceback
                traceback.print_exc()

        return {
            **state,
            "specialist_results": specialist_results,
            "tool_calls": [],
            "tool_results": [],
            "next_task": "",  # finished
            "current_step": f"Executed {len(specialist_results)}/{len(plan)} simple plan steps",
            "execution_progress": {
                "stage": "execution",
                "completed_steps": len(specialist_results),
                "total_steps": len(plan),
                "current_task": "Complete"
            },
            "execution_trajectory": trajectory
        }
    
    def _decide_tool_usage(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Decide which tools to use based on the query."""
        
        # Get the user's query
        user_query = messages[-1]["content"].lower() if messages else ""
        
        # Simple keyword-based tool detection
        tool_calls = []
        
        # Pattern matching for common queries
        if any(word in user_query for word in ["search", "find", "look for", "who is"]) and "patient" in user_query:
            # Extract patient name or ID
            query_text = messages[-1]["content"]
            tool_calls.append({"tool_name": "search_patients", "query": query_text})
            
        elif "medication" in user_query:
            # Extract patient identifier
            query_text = messages[-1]["content"]
            tool_calls.append({"tool_name": "get_patient_medications", "query": query_text})
            
        elif "lab result" in user_query or "test result" in user_query:
            query_text = messages[-1]["content"]
            tool_calls.append({"tool_name": "get_lab_results", "query": query_text})
            
        elif "appointment" in user_query:
            query_text = messages[-1]["content"]
            tool_calls.append({"tool_name": "get_appointments", "query": query_text})
            
        elif "vital sign" in user_query or "blood pressure" in user_query or "heart rate" in user_query:
            query_text = messages[-1]["content"]
            tool_calls.append({"tool_name": "get_vital_signs", "query": query_text})
            
        elif "medical history" in user_query or "condition" in user_query or "diagnosis" in user_query:
            query_text = messages[-1]["content"]
            # Check if searching by condition or patient
            if any(word in user_query for word in ["with", "have", "has"]):
                tool_calls.append({"tool_name": "search_by_condition", "query": query_text})
            else:
                tool_calls.append({"tool_name": "get_medical_history", "query": query_text})
        
        # If any patient ID or name is mentioned, try to search
        elif any(word in user_query for word in ["patient", "john", "sarah", "michael", "emily", "david"]) or user_query.strip().isdigit():
            query_text = messages[-1]["content"]
            tool_calls.append({"tool_name": "search_patients", "query": query_text})
        
        return {
            "needs_clarification": False,
            "tool_calls": tool_calls
        }
    
    def _tool_node(self, state: AgentState) -> AgentState:
        """Execute tools."""
        
        tool_calls = state.get("tool_calls", [])
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            query = tool_call.get("query", "")
            
            if tool_name in self.tools_dict:
                try:
                    tool_func = self.tools_dict[tool_name]["function"]
                    
                    # Extract relevant info from query
                    # For patient searches, try to extract name or ID
                    if tool_name == "search_patients":
                        query = self._extract_patient_identifier(query)
                    
                    # For condition searches, extract the condition name
                    elif tool_name == "search_by_condition":
                        query = self._extract_condition_from_query(query)
                    
                    # For other tools, try to extract patient identifier
                    elif tool_name in ["get_patient_medications", "get_medical_history", "get_lab_results", "get_appointments", "get_vital_signs"]:
                        query = self._extract_patient_identifier(query)
                    
                    result = tool_func(query)
                    results.append(f"**{tool_name}** results:\n{result}")
                except Exception as e:
                    results.append(f"Error calling {tool_name}: {str(e)}")
            else:
                results.append(f"Unknown tool: {tool_name}")
        
        return {
            **state,
            "tool_results": results,
            "tool_calls": []
        }

    def _reasoning_node(self, state: AgentState) -> AgentState:
        """Reasoning Agent: synthesize specialist outputs into final answer."""
        messages = self._normalize_messages(state["messages"])
        query = state.get("current_query", "")
        specialist_results = state.get("specialist_results", {})
        conversation_memory = state.get("conversation_memory", [])
        trajectory = state.get("execution_trajectory", [])

        trajectory.append({
            "node": "reasoning",
            "action": "Starting reasoning phase",
            "details": f"Synthesizing {len(specialist_results)} specialist results",
            "timestamp": datetime.now().isoformat()
        })

        # If we had no specialists (direct), just answer normally.
        if not specialist_results and (state.get("plan") and state["plan"][0].get("task") == "direct"):
            # Include recent conversation context for direct answers
            context_messages = []
            if conversation_memory:
                # Add last few exchanges for context (limit to avoid token limits)
                recent_memory = conversation_memory[-6:]  # Last 3 exchanges (6 messages)
                for mem in recent_memory:
                    context_messages.append({
                        "role": mem["role"],
                        "content": f"[Previous] {mem['content']}"
                    })
            
            trajectory.append({
                "node": "reasoning",
                "action": "Direct answer mode",
                "details": "No specialist results, answering from context",
                "timestamp": datetime.now().isoformat()
            })
            
            full_messages = context_messages + messages
            response = self.llm.generate_text(full_messages)
            
            trajectory.append({
                "node": "reasoning",
                "action": "Generated final response",
                "details": f"Response length: {len(response)} chars",
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                **state,
                "messages": [{"role": "assistant", "content": response}],
                "conversation_memory": conversation_memory + [{"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}],
                "current_step": "Response generated",
                "execution_progress": {"stage": "complete", "completed_steps": state.get("execution_progress", {}).get("total_steps", 0), "total_steps": state.get("execution_progress", {}).get("total_steps", 0), "current_task": "Complete"},
                "execution_trajectory": trajectory
            }

        system = {
            "role": "system",
            "content": (
                "You are a medical assistant. You have been given outputs from specialist agents "
                "(database/tool lookups). Synthesize them into a clear, concise, helpful answer.\n\n"
                "IMPORTANT GUIDELINES:\n"
                "- If data was found successfully, present it clearly without asking unnecessary follow-up questions.\n"
                "- If a search returned 'No patients found' or similar, inform the user directly.\n"
                "- Only ask for clarification if there are truly multiple matches or ambiguous results.\n"
                "- If a partial name search succeeds, present the results confidently.\n"
                "- This is demo data for testing purposes."
            ),
        }

        evidence_lines = []
        for k, v in specialist_results.items():
            evidence_lines.append(f"## {k}\n{v}")

        evidence = {
            "role": "system",
            "content": "Specialist agent outputs:\n\n" + "\n\n".join(evidence_lines),
        }

        # Include conversation context
        context_messages = []
        if conversation_memory:
            # Add recent context for better responses
            recent_memory = conversation_memory[-4:]  # Last 2 exchanges
            context_prompt = "Previous conversation context:\n"
            for mem in recent_memory:
                context_prompt += f"{mem['role'].title()}: {mem['content'][:200]}...\n"
            context_messages.append({"role": "system", "content": context_prompt})

        prompt = [system] + context_messages + messages + [
            {"role": "user", "content": f"User question: {query}"},
            evidence,
            {"role": "system", "content": "Write the final answer now. Be direct and confident if data was found."},
        ]

        trajectory.append({
            "node": "reasoning",
            "action": "Calling LLM for synthesis",
            "details": f"Context includes {len(evidence_lines)} specialist results",
            "timestamp": datetime.now().isoformat()
        })

        response = self.llm.generate_text(prompt)
        
        trajectory.append({
            "node": "reasoning",
            "action": "Generated final response",
            "details": f"Response length: {len(response)} chars",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            **state,
            "messages": [{"role": "assistant", "content": response}],
            "conversation_memory": conversation_memory + [{"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()}],
            "current_step": "Response generated",
            "execution_progress": {"stage": "complete", "completed_steps": state.get("execution_progress", {}).get("total_steps", 0), "total_steps": state.get("execution_progress", {}).get("total_steps", 0), "current_task": "Complete"},
            "execution_trajectory": trajectory
        }
    
    def process_message(self, message: str, history: List[Dict[str, str]] = None) -> tuple[str, Dict[str, Any]]:
        """
        Process a user message and return the agent's response.
        
        Args:
            message: User's message
            history: Optional conversation history
            
        Returns:
            Tuple of (Agent's response text, Agent state for visualization)
        """
        # Build messages
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        # Initialize state
        initial_state = {
            "messages": messages,
            "needs_clarification": False,
            "tool_results": [],
            "current_query": message,
            "tool_calls": [],
            "plan": [],
            "next_task": "",
            "specialist_results": {},
            "verifier_notes": "",
            "conversation_memory": history or [],
            "current_step": "Initializing",
            "execution_progress": {"stage": "initializing", "completed_steps": 0, "total_steps": 0},
            "detailed_plan": None,
            "execution_trajectory": []  # Track the trajectory
        }
        
        # Run the graph
        try:
            result = self.graph.invoke(initial_state)
            
            # Extract the last assistant message
            response_text = None
            for msg in reversed(result["messages"]):
                # Handle both dict and LangChain message objects
                if isinstance(msg, dict):
                    if msg.get("role") == "assistant":
                        response_text = msg["content"]
                        break
                else:
                    # Handle LangChain message objects
                    msg_type = getattr(msg, 'type', None) if hasattr(msg, 'type') else None
                    if msg_type == "ai" or msg_type == "assistant":
                        response_text = msg.content if hasattr(msg, 'content') else str(msg)
                        break
            
            if not response_text:
                response_text = "I apologize, but I couldn't generate a response. Please try again."
            
            # Debug: Check trajectory
            print(f"DEBUG process_message: result has {len(result.get('execution_trajectory', []))} trajectory steps")
            print(f"DEBUG process_message: result keys = {result.keys()}")
            
            # Return response and state for visualization
            return response_text, result
            
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"I encountered an error: {str(e)}. Please try rephrasing your question."
            return error_msg, initial_state

    def process_message_stream(self, message: str, history: List[Dict[str, str]] = None):
        """
        Process a user message and yield state updates in real-time.
        
        Args:
            message: User's message
            history: Optional conversation history
            
        Yields:
            State updates as the agent processes the message
        """
        # Build messages
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        # Initialize state
        initial_state = {
            "messages": messages,
            "needs_clarification": False,
            "tool_results": [],
            "current_query": message,
            "tool_calls": [],
            "plan": [],
            "next_task": "",
            "specialist_results": {},
            "verifier_notes": "",
            "conversation_memory": history or [],
            "current_step": "Initializing",
            "execution_progress": {"stage": "initializing", "completed_steps": 0, "total_steps": 0},
            "detailed_plan": None,
            "execution_trajectory": []
        }
        
        # Run the graph and stream intermediate states
        try:
            # Use the stream method to get intermediate states
            for state_update in self.graph.stream(initial_state):
                # Yield each state update for real-time display
                yield state_update
            
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            error_state = {
                **initial_state,
                "current_step": f"Error: {str(e)}",
                "execution_trajectory": initial_state.get("execution_trajectory", []) + [{
                    "node": "error",
                    "action": "Error occurred",
                    "details": str(e),
                    "timestamp": datetime.now().isoformat()
                }]
            }
            yield error_state


    def _extract_patient_identifier(self, query: str) -> str:
        """Extract patient name or ID from a query string.
        
        Args:
            query: The query string to extract from
            
        Returns:
            Extracted patient identifier (name or ID)
        """
        import re
        
        # First, check for "patient X" pattern (ID)
        match = re.search(r'patient\s+(?:id\s+)?(\d+)', query.lower())
        if match:
            return match.group(1)
        
        # Common query prefixes to remove (more aggressive)
        prefixes_to_remove = [
            r'^show\s+(?:me\s+)?(?:the\s+)?vital\s+signs?\s+(?:for\s+)?(?:patient\s+)?',
            r'^get\s+(?:the\s+)?vital\s+signs?\s+(?:for\s+)?(?:patient\s+)?',
            r'^show\s+(?:me\s+)?(?:the\s+)?lab\s+results?\s+(?:for\s+)?(?:patient\s+)?',
            r'^get\s+(?:the\s+)?lab\s+results?\s+(?:for\s+)?(?:patient\s+)?',
            r'^show\s+(?:me\s+)?(?:the\s+)?appointments?\s+(?:for\s+)?(?:patient\s+)?',
            r'^get\s+(?:the\s+)?appointments?\s+(?:for\s+)?(?:patient\s+)?',
            r'^show\s+(?:me\s+)?(?:the\s+)?medical\s+history\s+(?:for\s+)?(?:patient\s+)?',
            r'^get\s+(?:the\s+)?medical\s+history\s+(?:for\s+)?(?:patient\s+)?',
            r'^what\s+medications?\s+is\s+(?:patient\s+)?',
            r'^search\s+(?:for\s+)?(?:patient\s+)?',
            r'^find\s+(?:patient\s+)?',
            r'^list\s+(?:all\s+)?',
        ]
        
        cleaned_query = query
        for prefix in prefixes_to_remove:
            cleaned_query = re.sub(prefix, '', cleaned_query, flags=re.IGNORECASE)
        
        # Remove common suffixes
        suffixes_to_remove = [
            r'\s+taking\??$',
            r'\s+with\s+\w+$',  # "patients with diabetes"
        ]
        
        for suffix in suffixes_to_remove:
            cleaned_query = re.sub(suffix, '', cleaned_query, flags=re.IGNORECASE)
        
        cleaned_query = cleaned_query.strip()
        
        # If we have a clean name, return it
        if cleaned_query and not cleaned_query.lower() in ['search', 'find', 'show', 'get', 'list', 'the', 'a', 'an']:
            return cleaned_query
        
        # Fallback: Extract capitalized names from original query
        # But skip common command words
        command_words = {'Search', 'Find', 'Show', 'Get', 'List', 'What', 'Display', 'Pull', 'Retrieve', 'The'}
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        valid_names = [name for name in names if name not in command_words]
        
        if valid_names:
            return valid_names[0]
        
        # If all else fails, return the original query
        return query

    def _extract_condition_from_query(self, query: str) -> str:
        """Extract medical condition from a cohort query.
        
        Args:
            query: The query string to extract from
            
        Returns:
            Extracted condition name
        """
        import re
        
        # Common patterns for cohort queries
        patterns = [
            r'(?:patients?\s+)?with\s+(.+?)(?:\s*$)',  # "patients with diabetes"
            r'(?:who\s+)?ha(?:s|ve)\s+(.+?)(?:\s*$)',  # "who has diabetes"
            r'list\s+(?:all\s+)?(?:patients?\s+)?with\s+(.+?)(?:\s*$)',  # "list all patients with diabetes"
            r'find\s+(?:all\s+)?(?:patients?\s+)?with\s+(.+?)(?:\s*$)',  # "find patients with diabetes"
            r'show\s+(?:all\s+)?(?:patients?\s+)?with\s+(.+?)(?:\s*$)',  # "show patients with diabetes"
        ]
        
        query_lower = query.lower()
        
        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                condition = match.group(1).strip()
                # Remove trailing punctuation
                condition = re.sub(r'[.!?]+$', '', condition)
                return condition
        
        # If no pattern matches, return the original query
        return query

    def _assess_plan_complexity(self, query: str) -> str:
        """Assess the complexity of a query to determine planning approach."""
        q = query.lower()
        
        # Complex indicators
        complex_indicators = [
            "all patients", "pull up all", "comprehensive", "complete", "full",
            "analyze", "compare", "summarize", "review", "evaluate",
            "multiple", "several", "many", "list all", "show me all",
            "trends", "patterns", "insights", "recommendations"
        ]
        
        # Moderate indicators
        moderate_indicators = [
            "recent", "last", "previous", "current", "active",
            "status", "summary", "overview", "details"
        ]
        
        if any(indicator in q for indicator in complex_indicators):
            return "complex"
        elif any(indicator in q for indicator in moderate_indicators):
            return "moderate"
        else:
            return "simple"

    def _create_detailed_plan(self, query: str, available_tools: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a detailed execution plan using LLM for complex tasks."""
        
        # Convert LangChain tools to descriptions
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in available_tools.items()
        ])
        
        planning_prompt = f"""
You are a medical assistant planning system. Create a detailed execution plan for this query: "{query}"

Available tools:
{tool_descriptions}

For complex queries requiring careful planning, break down the task into sequential steps with dependencies.

IMPORTANT: Respond with ONLY a valid JSON object. No markdown formatting, no explanations, just pure JSON.

Required JSON structure:
{{
    "steps": [
        {{
            "id": "step_1",
            "task": "brief_task_name",
            "description": "detailed description of what this step does",
            "tool_name": "tool_to_use_or_null",
            "parameters": {{"query": "modified_query_if_needed"}},
            "dependencies": ["step_ids_this_depends_on"],
            "required": true
        }}
    ],
    "complexity": "complex",
    "rationale": "why this plan was chosen",
    "estimated_steps": 2
}}

Example for "pull up all patients vital signs":
{{
    "steps": [
        {{
            "id": "identify_patients",
            "task": "get_all_patients",
            "description": "Get list of all patients in the system",
            "tool_name": "search_patients",
            "parameters": {{"query": "all patients"}},
            "dependencies": [],
            "required": true
        }},
        {{
            "id": "collect_vitals",
            "task": "get_vitals_batch",
            "description": "Retrieve vital signs for each patient",
            "tool_name": "get_vital_signs",
            "parameters": {{"query": "batch retrieval"}},
            "dependencies": ["identify_patients"],
            "required": true
        }}
    ],
    "complexity": "complex",
    "rationale": "This requires systematic retrieval of data for multiple patients",
    "estimated_steps": 2
}}
"""
        
        try:
            response = self.llm.generate_text(
                messages=[{"role": "user", "content": planning_prompt}]
            )
            
            # Extract JSON from response
            content = response
            # Find JSON in the response (handle markdown code blocks)
            if "```json" in content:
                # Extract from markdown code block
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end == -1:
                    end = len(content)
                json_str = content[start:end].strip()
            else:
                # Find JSON object
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                else:
                    json_str = content
            
            # Clean up common JSON issues
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            plan_data = json.loads(json_str)
            return plan_data  # Return dict directly since TypedDict is just for typing
        except Exception as e:
            print(f"Error creating detailed plan: {e}")
            return None
def create_medical_agent(enabled_tools: Dict[str, bool] = None) -> MedicalAgent:
    """Create and return a medical agent instance.

    Args:
        enabled_tools: Optional dict mapping tool names to enabled status
        
    Returns:
        Configured MedicalAgent instance
    """
    return MedicalAgent(enabled_tools=enabled_tools)
