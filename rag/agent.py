"""ReAct agent implementation using LangGraph for IDIP."""
import re
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, TypedDict
import httpx
from pydantic import BaseModel

from config import settings
from rag.models import Citation, RAGResponse
from rag.vector_store import SearchResult
from models.ner.service import EntityResult
from models.classifier.service import ClassificationResult

logger = logging.getLogger("idip.rag.agent")

# Define the Graph state shape
class AgentState(TypedDict):
    query: str
    doc_id: Optional[str]
    messages: List[Dict[str, Any]]
    iterations: int
    trace: List[Dict[str, Any]]
    final_response: Optional[str]
    start_time: float

class IDIPAgent:
    """
    ReAct Agent utilizing LangGraph.
    Coordinates vector searches, entity extractions, classifications, calculations,
    and external API lookups to answer complex user queries.
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        ner_service: Any,
        classifier_service: Any,
        llm_service: Any,
        document_store: Optional[Dict[str, Any]] = None
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.ner_service = ner_service
        self.classifier_service = classifier_service
        self.llm_service = llm_service
        
        # Document store cache for classification lookup
        self.document_store = document_store or {}

        # Build LangGraph workflow
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(AgentState)
        
        # Add graph nodes
        workflow.add_node("plan", self.node_plan)
        workflow.add_node("act", self.node_act)
        workflow.add_node("observe", self.node_observe)
        workflow.add_node("respond", self.node_respond)
        
        workflow.set_entry_point("plan")
        
        # Conditional edge: continue to act or end at respond
        workflow.add_conditional_edges(
            "plan",
            self.route_plan,
            {
                "act": "act",
                "respond": "respond"
            }
        )
        
        workflow.add_edge("act", "observe")
        workflow.add_edge("observe", "plan")
        workflow.add_edge("respond", END)
        
        self.app = workflow.compile()

    # --- Tool Definitions ---

    def vector_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Queries vector database backend for semantic context."""
        logger.info(f"Tool Call - vector_search: query='{query}', top_k={top_k}")
        try:
            # Generate query embedding
            query_vector = self.embedding_service.encode_query(query)
            # Fetch candidates from store
            results = self.vector_store.query(query_vector, top_k=top_k)
            return results
        except Exception as e:
            logger.error(f"Error in vector_search tool: {e}")
            return []

    def ner_extract(self, text: str) -> List[EntityResult]:
        """Extracts entities from input text via NER service."""
        logger.info(f"Tool Call - ner_extract: text_len={len(text)}")
        try:
            return self.ner_service.extract_entities(text)
        except Exception as e:
            logger.error(f"Error in ner_extract tool: {e}")
            return []

    def classify_document(self, doc_id: str) -> Optional[ClassificationResult]:
        """Classifies document by doc_id."""
        logger.info(f"Tool Call - classify_document: doc_id='{doc_id}'")
        try:
            doc = self.document_store.get(doc_id)
            if not doc:
                logger.warning(f"Document {doc_id} not found in store registry.")
                return None
            return self.classifier_service.predict(doc)
        except Exception as e:
            logger.error(f"Error in classify_document tool: {e}")
            return None

    def calculate(self, expression: str) -> float:
        """Evaluates basic mathematical arithmetic expressions safely."""
        logger.info(f"Tool Call - calculate: expr='{expression}'")
        # Clean inputs to prevent unsafe code execution
        cleaned = re.sub(r"[^0-9.+\-*/() ]", "", expression)
        if not cleaned.strip():
            raise ValueError("Safe expression is empty after cleaning.")
        try:
            # Evaluate sanitized math operations in blank sandbox
            val = eval(cleaned, {"__builtins__": None}, {})
            return float(val)
        except Exception as e:
            logger.error(f"Arithmetic evaluation failure: {e}")
            raise ValueError(f"Math evaluation failed for '{expression}': {e}") from e

    def fetch_external(self, url: str) -> str:
        """Executes external API lookup request."""
        logger.info(f"Tool Call - fetch_external: url='{url}'")
        try:
            response = httpx.get(url, timeout=5)
            return response.text
        except Exception as e:
            logger.error(f"Failed external API query: {url}, error: {e}")
            return f"Error fetching url: {e}"

    # --- Node Logic ---

    def node_plan(self, state: AgentState) -> AgentState:
        """Plan node: Decides whether to invoke a tool or prepare the final response."""
        state["iterations"] += 1
        
        # Enforce execution timeout
        elapsed = time.time() - state["start_time"]
        if elapsed > 30.0:
            logger.warning("Agent execution exceeded 30-second timeout.")
            state["messages"].append({
                "type": "plan",
                "action": "respond",
                "answer": "Execution timed out. Unable to complete full query path."
            })
            state["trace"].append({
                "node": "plan",
                "action": "timeout reached, forcing response"
            })
            return state

        # Construct trace history for LLM prompt context
        history_lines = []
        for msg in state["messages"]:
            if msg["type"] == "plan":
                if msg["action"] == "tool_call":
                    history_lines.append(f"Thought: I need to call tool '{msg['tool']}' with args {msg['args']}")
                else:
                    history_lines.append(f"Thought: I am ready to respond. Answer: {msg.get('answer')}")
            elif msg["type"] == "observation":
                history_lines.append(f"Observation: {msg['output']}")

        history_str = "\n".join(history_lines)
        
        prompt = f"""You are an IDIP ReAct agent. You have access to the following tools:
- vector_search(query: str, top_k: int) -> List[SearchResult] (finds relevant document contexts)
- ner_extract(text: str) -> List[EntityResult] (extracts entities)
- classify_document(doc_id: str) -> ClassificationResult (identifies document type)
- calculate(expression: str) -> float (solves math and financial equations)
- fetch_external(url: str) -> str (queries external endpoints)

Original Query: {state['query']}
Document ID: {state['doc_id']}

History of execution steps:
{history_str}

Decide the next logical step. You must respond in exactly one of the following formats:

Format 1 (To call a tool):
Action: <tool_name>
Arguments: <json_formatted_arguments>

Format 2 (To return the final answer):
Answer: <synthesized_final_response>
"""
        # Call LLM to make the plan
        llm_output = self.llm_service.generate(prompt).strip()
        logger.info(f"Plan Node - LLM Output: {llm_output}")

        # Parse LLM response
        action_match = re.search(r"Action:\s*(\w+)", llm_output)
        args_match = re.search(r"Arguments:\s*({.*})", llm_output, re.DOTALL)
        answer_match = re.search(r"Answer:\s*(.*)", llm_output, re.DOTALL)

        if action_match and args_match:
            tool_name = action_match.group(1).strip()
            raw_args = args_match.group(1).strip()
            
            # Simple json parser fallback
            import json
            try:
                args = json.loads(raw_args)
            except Exception:
                # heuristic fallback for unquoted values
                args = {}
                for kv in re.findall(r'"(\w+)":\s*"([^"]*)"', raw_args):
                    args[kv[0]] = kv[1]
                for kv in re.findall(r'"(\w+)":\s*(\d+\.?\d*)', raw_args):
                    args[kv[0]] = float(kv[1]) if "." in kv[1] else int(kv[1])

            state["messages"].append({
                "type": "plan",
                "action": "tool_call",
                "tool": tool_name,
                "args": args
            })
            state["trace"].append({
                "node": "plan",
                "action": f"planned tool call: {tool_name} with {args}"
            })
        elif answer_match:
            ans = answer_match.group(1).strip()
            state["messages"].append({
                "type": "plan",
                "action": "respond",
                "answer": ans
            })
            state["trace"].append({
                "node": "plan",
                "action": "resolved final response"
            })
        else:
            # Fallback response parse
            state["messages"].append({
                "type": "plan",
                "action": "respond",
                "answer": llm_output
            })
            state["trace"].append({
                "node": "plan",
                "action": "fallback resolved final response"
            })

        return state

    def node_act(self, state: AgentState) -> AgentState:
        """Act node: Executes the planned tool call and updates execution trace."""
        last_plan = state["messages"][-1]
        tool_name = last_plan.get("tool", "")
        args = last_plan.get("args", {})
        
        output = ""
        try:
            if tool_name == "vector_search":
                q = args.get("query", state["query"])
                k = int(args.get("top_k", 5))
                res = self.vector_search(q, k)
                output = str([{"doc_id": r.doc_id, "text": r.text, "score": r.score} for r in res])
            elif tool_name == "ner_extract":
                t = args.get("text", "")
                res = self.ner_extract(t)
                output = str([{"text": e.text, "label": e.label, "confidence": e.confidence} for e in res])
            elif tool_name == "classify_document":
                d = args.get("doc_id", state["doc_id"])
                res = self.classify_document(d)
                output = str(res.model_dump()) if res else "Document not found"
            elif tool_name == "calculate":
                expr = args.get("expression", "")
                res = self.calculate(expr)
                output = str(res)
            elif tool_name == "fetch_external":
                url = args.get("url", "")
                res = self.fetch_external(url)
                output = res
            else:
                output = f"Unknown tool: {tool_name}"
        except Exception as e:
            output = f"Tool execution failed: {e}"

        state["messages"].append({
            "type": "act",
            "tool": tool_name,
            "args": args,
            "output": output
        })
        state["trace"].append({
            "node": "act",
            "tool": tool_name,
            "output": output
        })
        return state

    def node_observe(self, state: AgentState) -> AgentState:
        """Observe node: Analyzes output from act nodes and stores observation messages."""
        last_act = state["messages"][-1]
        tool_output = last_act.get("output", "")
        
        # Add formatted observation to messages
        state["messages"].append({
            "type": "observation",
            "output": tool_output
        })
        state["trace"].append({
            "node": "observe",
            "observation": tool_output
        })
        return state

    def node_respond(self, state: AgentState) -> AgentState:
        """Respond node: Synthesizes final trace answers to output."""
        last_plan = next((m for m in reversed(state["messages"]) if m.get("type") == "plan" and m.get("action") == "respond"), None)
        
        if last_plan:
            ans = last_plan.get("answer", "")
        else:
            ans = "Max execution steps completed without resolving final answer."

        # If answer needs synthesis from observations, trigger a final LLM run
        has_tool_observations = any(m["type"] == "observation" for m in state["messages"])
        if has_tool_observations and len(ans) < 20:
            prompt = f"""Synthesize a complete final answer based on these observations.
Query: {state['query']}
Observations:
{ans}
"""
            try:
                ans = self.llm_service.generate(prompt).strip()
            except Exception:
                pass

        state["final_response"] = ans
        state["trace"].append({
            "node": "respond",
            "final_response": ans
        })
        return state

    # --- Routing Logic ---

    def route_plan(self, state: AgentState) -> str:
        """Router deciding branch transitions from plan nodes."""
        if state["iterations"] >= 5:
            logger.info("Reached maximum iterations limit (5). Routing to respond.")
            return "respond"
        
        last_msg = state["messages"][-1] if state["messages"] else {}
        if last_msg.get("type") == "plan" and last_msg.get("action") == "tool_call":
            return "act"
        
        return "respond"

    # --- Entry point execution ---

    def run_agent(self, query: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs the ReAct agent flow synchronously."""
        initial_state: AgentState = {
            "query": query,
            "doc_id": doc_id,
            "messages": [],
            "iterations": 0,
            "trace": [],
            "final_response": None,
            "start_time": time.time()
        }

        # Run compiled LangGraph app
        # recursion_limit must accommodate max_iterations * nodes_per_cycle + buffer
        # 5 iterations × 4 nodes (plan→act→observe→respond) = 20 + 5 buffer = 25 but
        # route_plan counts too, so use 40 to comfortably cover 5 full cycles
        final_state = self.app.invoke(initial_state, config={"recursion_limit": 40})
        
        return {
            "final_response": final_state["final_response"],
            "trace": final_state["trace"],
            "iterations": final_state["iterations"]
        }
