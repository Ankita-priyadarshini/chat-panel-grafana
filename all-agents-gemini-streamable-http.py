import asyncio
import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import redis
from zoneinfo import ZoneInfo
import tzlocal
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# MCP + LangChain with HTTP Streamable support
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

def get_session_key(uid: str) -> str:
    return f"session:{uid}"

# FastAPI models
class QueryRequest(BaseModel):
    query: str
    uid: Optional[str] = None

class QueryResponse(BaseModel):
    uid: str
    response: str
    is_new_session: bool
    agent_used: str
    suggested_next_actions: Optional[List[str]] = None

# Agent Classification
def classify_query_intent(query: str) -> str:
    query_lower = query.lower().strip()
    dashboard_keywords = [
        'dashboard', 'panel', 'metric', 'cpu', 'memory', 'disk', 'network',
        'visualization', 'graph', 'chart', 'create dashboard', 'make dashboard',
        'cpu usage', 'memory usage', 'performance', 'system metrics'
    ]
    alert_keywords = [
        'alert', 'notification', 'threshold', 'rule', 'warning', 'alarm',
        'firing', 'trigger', 'escalation', 'alert rule', 'notify'
    ]
    log_keywords = [
        'log', 'error', 'exception', 'debug', 'trace', 'search', 'show logs',
        'error logs', 'list logs', 'find errors', 'show errors', 'log analysis',
        'log search', 'stderr', 'stdout', 'syslog'
    ]
    dashboard_score = sum(1 for keyword in dashboard_keywords if keyword in query_lower)
    alert_score = sum(1 for keyword in alert_keywords if keyword in query_lower)
    log_score = sum(1 for keyword in log_keywords if keyword in query_lower)

    if log_score > dashboard_score and log_score > alert_score:
        return "log_search"
    elif dashboard_score > alert_score:
        return "dashboard"
    elif alert_score > 0:
        return "alert"
    else:
        if any(word in query_lower for word in ['error', 'issue', 'problem', 'fail', 'exception']):
            return "log_search"
        return "coordinator"

# Base Agent Class
class BaseGrafanaAgent:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.environ.get("GOOGLE_API_KEY")
        )
        self.grafana_server_url = os.environ.get("GRAFANA_MCP_SERVER_URL", "http://localhost:8000")

    def get_system_prompt(self) -> str:
        local_tz = ZoneInfo(tzlocal.get_localzone_name())
        current_time = datetime.now(local_tz)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        base_prompt = f"""
You are a specialized AI agent for Grafana observability tasks.
Current time: {formatted_time}
User timezone: IST (Indian Standard Time)

IMPORTANT RULES:
- Always query available datasources first and use their UIDs dynamically
- Convert all timestamps to IST format: YYYY-MM-DD HH:MM:SS IST
- Provide clear, actionable responses
- Be concise but helpful
"""
        return base_prompt + self.get_specialized_prompt()

    def get_specialized_prompt(self) -> str:
        return ""

    async def process_query(self, chat_history: List[Dict[str, str]]) -> str:
        try:
            system_prompt = {"role": "system", "content": self.get_system_prompt()}
            full_history = [system_prompt] + chat_history
            client = MultiServerMCPClient({
                "grafana": {
                    "url": f"{self.grafana_server_url}/mcp",
                    "transport": "streamable_http",
                }
            })
            tools = await client.get_tools()
            agent = create_react_agent(self.model, tools)
            result = await agent.ainvoke({"messages": full_history})

            if result and "messages" in result and len(result["messages"]) > 0:
                content = result["messages"][-1].content
                if isinstance(content, list):
                    if all(isinstance(item, str) for item in content):
                        return "\n".join(content)
                    else:
                        return "\n".join(str(item) for item in content)
                elif isinstance(content, str):
                    return content
                else:
                    return str(content)
            else:
                return "I apologize, but I couldn't process your request. Please try again."
        except Exception as e:
            print(f"Error in {self.agent_type} agent: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

# Specialized Agents
class DashboardAgent(BaseGrafanaAgent):
    def __init__(self):
        super().__init__("dashboard")

    def get_specialized_prompt(self) -> str:
        return """
You are the DASHBOARD SPECIALIST...
... [Prompt continues unchanged]
"""

class AlertAgent(BaseGrafanaAgent):
    def __init__(self):
        super().__init__("alert")

    def get_specialized_prompt(self) -> str:
        return """
You are the ALERT SPECIALIST...
... [Prompt continues unchanged]
"""

class LogSearchAgent(BaseGrafanaAgent):
    def __init__(self):
        super().__init__("log_search")

    def get_specialized_prompt(self) -> str:
        return """
You are the LOG SEARCH SPECIALIST...
... [Prompt continues unchanged]
"""

class CoordinatorAgent(BaseGrafanaAgent):
    def __init__(self):
        super().__init__("coordinator")

    def get_specialized_prompt(self) -> str:
        return """
You are the COORDINATOR agent...
... [Prompt continues unchanged]
"""

# Multi-Agent System
class MultiAgentGrafanaSystem:
    def __init__(self):
        self.dashboard_agent = DashboardAgent()
        self.alert_agent = AlertAgent()
        self.log_agent = LogSearchAgent()
        self.coordinator = CoordinatorAgent()

    async def process_query(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        agent_type = classify_query_intent(query)
        if agent_type == "dashboard":
            agent = self.dashboard_agent
        elif agent_type == "alert":
            agent = self.alert_agent
        elif agent_type == "log_search":
            agent = self.log_agent
        else:
            agent = self.coordinator
        response = await agent.process_query(chat_history)
        suggestions = self.get_suggestions(agent_type, response)
        return {
            "response": response,
            "agent_used": agent_type,
            "suggested_next_actions": suggestions
        }

    def get_suggestions(self, agent_type: str, response: str) -> List[str]:
        suggestion_map = {
            "dashboard": [
                "Add more panels to dashboard",
                "Create alerts for metrics",
                "Configure dashboard variables"
            ],
            "alert": [
                "Review alert thresholds",
                "Configure notifications",
                "Test alert conditions"
            ],
            "log_search": [
                "Create log dashboard",
                "Set up error alerts",
                "Filter by specific service"
            ],
            "coordinator": [
                "Create a dashboard",
                "Search for errors",
                "Set up alerts"
            ]
        }
        return suggestion_map.get(agent_type, [])[:3]

# FastAPI Setup
app = FastAPI(title="Multi-Agent Grafana MCP API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

multi_agent_system = MultiAgentGrafanaSystem()

@app.get("/")
async def root():
    return {
        "message": "Multi-Agent Grafana MCP API is running",
        "agents": {
            "dashboard": "Handles dashboard creation and metrics visualization",
            "alert": "Manages alert rules and notifications",
            "log_search": "Searches and analyzes logs",
            "coordinator": "Handles general queries and guidance"
        }
    }

@app.post("/query")
async def query_grafana(request: QueryRequest):
    if request.uid is None:
        uid = str(uuid.uuid4())
        is_new_session = True
        redis_client.hset(f"meta:{uid}", mapping={
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        })
    else:
        uid = request.uid
        is_new_session = not redis_client.exists(get_session_key(uid))
        if is_new_session:
            redis_client.hset(f"meta:{uid}", mapping={
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat()
            })
        else:
            redis_client.hset(f"meta:{uid}", "last_used", datetime.now().isoformat())

    history_raw = redis_client.lrange(get_session_key(uid), 0, -1)
    chat_history = []
    for item in history_raw:
        entry = json.loads(item)
        chat_history.append({"role": "user", "content": entry["query"]})
        chat_history.append({"role": "assistant", "content": entry["response"]})

    chat_history.append({"role": "user", "content": request.query})

    try:
        result = await multi_agent_system.process_query(request.query, chat_history)
        entry = {
            "query": request.query,
            "response": result["response"],
            "agent_used": result["agent_used"],
            "timestamp": datetime.now().isoformat()
        }
        redis_client.rpush(get_session_key(uid), json.dumps(entry))
        return QueryResponse(
            uid=uid,
            response=result["response"],
            is_new_session=is_new_session,
            agent_used=result["agent_used"],
            suggested_next_actions=result.get("suggested_next_actions", [])
        )
    except Exception as e:
        return QueryResponse(
            uid=uid,
            response=f"I apologize, but I encountered an error processing your request: {str(e)}",
            is_new_session=is_new_session,
            agent_used="error",
            suggested_next_actions=["Try rephrasing your question", "Check system status"]
        )

@app.get("/session/{uid}")
async def get_session_history(uid: str):
    if not redis_client.exists(get_session_key(uid)):
        return {"error": "Session not found"}
    history_raw = redis_client.lrange(get_session_key(uid), 0, -1)
    history = [json.loads(item) for item in history_raw]
    meta = redis_client.hgetall(f"meta:{uid}")
    return {
        "uid": uid,
        "history": history,
        "created_at": meta.get("created_at"),
        "last_used": meta.get("last_used"),
        "total_queries": len(history)
    }

@app.get("/sessions")
async def list_all_sessions():
    session_keys = redis_client.keys("meta:*")
    sessions = []
    for key in session_keys:
        uid = key.split(":")[1]
        meta = redis_client.hgetall(key)
        total_queries = redis_client.llen(get_session_key(uid))
        sessions.append({
            "uid": uid,
            "created_at": meta.get("created_at"),
            "last_used": meta.get("last_used"),
            "total_queries": total_queries
        })
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

@app.get("/health")
async def health_check():
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    return {
        "status": "healthy",
        "redis": redis_status,
        "agents": ["dashboard", "alert", "log_search", "coordinator"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)
