import json
import os
import re
import zipfile
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict
from urllib.parse import urlsplit

import requests
from langgraph.graph import END, StateGraph

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
except Exception:
    AzureChatOpenAI = None
    ChatOpenAI = None


TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".php", ".rb", ".cs",
    ".cpp", ".c", ".h", ".hpp", ".swift", ".scala", ".sql", ".md", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".env", ".xml", ".html", ".css", ".scss", ".sh", ".ps1", ".dockerfile",
}

PERSONA_SDE = "sde"
PERSONA_PM = "pm"
ProgressCallback = Callable[[int, str], None]


class AnalysisState(TypedDict, total=False):
    repo_zip_path: str
    project_name: str
    persona: str
    progress_callback: ProgressCallback
    coordinator_note: str
    files: List[Dict[str, str]]
    summary: str
    technical_analysis: str
    business_analysis: str
    auth_logic: str
    mermaid: str
    web_research: str
    documentation_json: str


class AskState(TypedDict, total=False):
    question: str
    summary: str
    core_logic: str
    auth_logic: str
    mermaid: str
    documentation_json: str
    answer: str


def _emit_progress(state: AnalysisState, percent: int, message: str) -> None:
    callback = state.get("progress_callback")
    if callback:
        try:
            callback(percent, message)
        except Exception:
            pass


def _get_chat_model() -> Optional[Any]:
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if azure_endpoint:
        parsed = urlsplit(azure_endpoint)
        if parsed.scheme and parsed.netloc:
            azure_endpoint = f"{parsed.scheme}://{parsed.netloc}"

    if azure_api_key and azure_endpoint and azure_deployment and AzureChatOpenAI is not None:
        return AzureChatOpenAI(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0,
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and ChatOpenAI is not None:
        return ChatOpenAI(
            api_key=openai_api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    return None


def _safe_read_zip_files(repo_zip_path: str, max_files: int = 120, max_chars: int = 140000) -> List[Dict[str, str]]:
    if not os.path.exists(repo_zip_path):
        raise ValueError("Repository archive was not found")

    files: List[Dict[str, str]] = []
    total_chars = 0

    try:
        with zipfile.ZipFile(repo_zip_path, "r") as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            for member in members:
                if len(files) >= max_files or total_chars >= max_chars:
                    break

                file_path = member.filename
                lowered = file_path.lower()

                if lowered.endswith((".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".jar", ".class", ".exe", ".dll")):
                    continue
                if "/.git/" in lowered or lowered.endswith(".gitignore"):
                    continue

                _, ext = os.path.splitext(lowered)
                if ext and ext not in TEXT_EXTENSIONS:
                    continue

                try:
                    raw = archive.read(member)
                    if b"\x00" in raw:
                        continue
                    text = raw.decode("utf-8", errors="ignore")
                    if not text.strip():
                        continue
                except Exception:
                    continue

                files.append({"path": file_path, "content": text[:3000]})
                total_chars += len(files[-1]["content"])
    except zipfile.BadZipFile:
        raise ValueError("Repository archive is not a valid ZIP file")

    return files


def _llm_refine(model: Any, title: str, instruction: str, content: str, fallback: str) -> str:
    if model is None:
        return fallback
    prompt = (
        f"You are an expert repository analyst.\n"
        f"Task: {title}\n"
        f"Instruction: {instruction}\n"
        f"Use concise and clear output.\n\n"
        f"Repository content:\n{content}"
    )
    try:
        response = model.invoke(prompt)
        text = getattr(response, "content", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return fallback
    except Exception:
        return fallback


def _coordinator_agent(state: AnalysisState) -> AnalysisState:
    persona = state.get("persona", PERSONA_SDE)
    note = (
        "Route analysis for Software Developer: emphasize architecture, code-level flow, risks, and implementation details."
        if persona == PERSONA_SDE
        else "Route analysis for Project Manager: emphasize features, business workflows, dependencies, delivery risks, and roadmap hints."
    )
    _emit_progress(state, 5, "Coordinator initialized analysis plan")
    return {"coordinator_note": note}


def _repo_scanner_agent(state: AnalysisState) -> AnalysisState:
    _emit_progress(state, 12, "Scanning repository archive")
    files = _safe_read_zip_files(state["repo_zip_path"])
    _emit_progress(state, 22, f"Preprocessing complete ({len(files)} readable files)")
    return {"files": files}


def _summary_agent(state: AnalysisState) -> AnalysisState:
    files = state.get("files", [])
    project_name = state.get("project_name", "Project")

    ext_counter = Counter()
    top_dirs = Counter()
    for item in files:
        path = item["path"]
        _, ext = os.path.splitext(path.lower())
        if ext:
            ext_counter[ext] += 1
        root = path.split("/")[0] if "/" in path else "root"
        top_dirs[root] += 1

    fallback = (
        f"Project {project_name} includes {len(files)} readable files. "
        f"Top types: {', '.join([f'{e}({c})' for e, c in ext_counter.most_common(6)]) or 'unknown'}. "
        f"Top folders: {', '.join([f'{d}({c})' for d, c in top_dirs.most_common(6)]) or 'none'}."
    )

    model = _get_chat_model()
    compact = "\n\n".join([f"FILE: {f['path']}\n{f['content'][:700]}" for f in files[:20]])
    summary = _llm_refine(
        model,
        "Repository Summary",
        "Create concise summary of system purpose, modules, and technology stack.",
        compact,
        fallback,
    )

    _emit_progress(state, 35, "Repository summary completed")
    return {"summary": summary}


def _technical_analysis_agent(state: AnalysisState) -> AnalysisState:
    files = state.get("files", [])
    keys = ["main", "app", "router", "service", "controller", "handler", "workflow", "api", "model"]
    matches = []
    for item in files:
        text = f"{item['path']}\n{item['content']}".lower()
        if any(k in text for k in keys):
            matches.append(item["path"])
        if len(matches) >= 15:
            break

    fallback = "Technical hotspots:\n" + "\n".join([f"- {m}" for m in matches]) if matches else "Technical hotspots not confidently identified."
    model = _get_chat_model()
    compact = "\n\n".join([f"FILE: {f['path']}\n{f['content'][:1000]}" for f in files[:25]])
    technical = _llm_refine(
        model,
        "Technical Analysis Agent",
        "Explain architecture, key code paths, data flow, and engineering risks for developers.",
        compact,
        fallback,
    )

    _emit_progress(state, 50, "Technical analysis agent completed")
    return {"technical_analysis": technical}


def _business_analysis_agent(state: AnalysisState) -> AnalysisState:
    files = state.get("files", [])
    model = _get_chat_model()
    compact = "\n\n".join([f"FILE: {f['path']}\n{f['content'][:900]}" for f in files[:25]])
    fallback = (
        "Business analysis fallback: identify probable user-facing capabilities from routes/controllers, "
        "dependencies, and delivery considerations."
    )
    business = _llm_refine(
        model,
        "Business Analysis Agent",
        "Describe business workflows, user value, product capabilities, constraints, and roadmap opportunities for PM audience.",
        compact,
        fallback,
    )

    _emit_progress(state, 62, "Business analysis agent completed")
    return {"business_analysis": business}


def _auth_analysis_agent(state: AnalysisState) -> AnalysisState:
    files = state.get("files", [])
    patterns = ["auth", "jwt", "token", "login", "password", "oauth", "bearer", "session", "role"]
    matches = []
    for item in files:
        text = f"{item['path']}\n{item['content']}".lower()
        if any(p in text for p in patterns):
            matches.append(item["path"])
        if len(matches) >= 12:
            break

    fallback = (
        "Detected authentication/authorization files:\n" + "\n".join([f"- {m}" for m in matches])
        if matches
        else "No explicit authentication files detected in analyzed sample."
    )

    model = _get_chat_model()
    compact = "\n\n".join([f"FILE: {f['path']}\n{f['content'][:1000]}" for f in files[:25]])
    auth_logic = _llm_refine(
        model,
        "Authentication Agent",
        "Explain authentication, authorization, token/session flows, and potential security gaps.",
        compact,
        fallback,
    )

    _emit_progress(state, 72, "Authentication analysis completed")
    return {"auth_logic": auth_logic}


def _diagram_agent(state: AnalysisState) -> AnalysisState:
    files = state.get("files", [])
    folders = []
    seen = set()
    for item in files:
        parts = item["path"].split("/")
        top = parts[0] if len(parts) > 1 else "root"
        if top not in seen:
            seen.add(top)
            folders.append(top)
        if len(folders) >= 8:
            break

    root = re.sub(r"[^a-zA-Z0-9_]", "_", state.get("project_name", "Project")) or "Project"
    fallback = ["graph TD", f"    {root}[{state.get('project_name', 'Project')}]"]
    for idx, folder in enumerate(folders, start=1):
        fallback.append(f"    {root} --> N{idx}[{folder}]")
    fallback_mermaid = "\n".join(fallback)

    model = _get_chat_model()
    compact = "\n".join([f["path"] for f in files[:40]])
    mermaid = _llm_refine(
        model,
        "Diagram Agent",
        "Return only valid Mermaid graph TD code for high-level repository structure and flow.",
        compact,
        fallback_mermaid,
    )
    if "graph" not in mermaid.lower():
        mermaid = fallback_mermaid

    _emit_progress(state, 82, "Mermaid diagram generated")
    return {"mermaid": mermaid}


def _extract_requirements_from_files(files: List[Dict[str, str]]) -> List[str]:
    for item in files:
        if item["path"].lower().endswith("requirements.txt"):
            deps = []
            for line in item["content"].splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                dep = re.split(r"[<>=~! ]", line)[0].strip()
                if dep:
                    deps.append(dep)
            return deps[:8]
    return []


def _web_search_agent(state: AnalysisState) -> AnalysisState:
    files = state.get("files", [])
    deps = _extract_requirements_from_files(files)
    if not deps:
        _emit_progress(state, 88, "Web research agent skipped (no requirements.txt found)")
        return {"web_research": "No dependency metadata found for web-based best-practice checks."}

    lines = []
    for dep in deps:
        try:
            response = requests.get(f"https://pypi.org/pypi/{dep}/json", timeout=4)
            if response.status_code == 200:
                latest = response.json().get("info", {}).get("version", "unknown")
                lines.append(f"- {dep}: latest available version on PyPI is {latest}")
            else:
                lines.append(f"- {dep}: unable to fetch latest version")
        except Exception:
            lines.append(f"- {dep}: unable to fetch latest version")

    lines.append("- Best practice: pin critical runtime dependencies and review security advisories regularly.")
    web_research = "\n".join(lines)
    _emit_progress(state, 92, "Web research agent completed")
    return {"web_research": web_research}


def _documentation_agent(state: AnalysisState) -> AnalysisState:
    persona = state.get("persona", PERSONA_SDE)
    audience = "Software Developer" if persona == PERSONA_SDE else "Project Manager"
    primary_detail = state.get("technical_analysis", "") if persona == PERSONA_SDE else state.get("business_analysis", "")

    doc = {
        "format_version": "1.0",
        "audience": audience,
        "title": f"Repository Analysis - {state.get('project_name', 'Project')}",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sections": [
            {"id": "overview", "title": "Overview", "content": state.get("summary", "")},
            {
                "id": "primary_analysis",
                "title": "Technical Deep Dive" if persona == PERSONA_SDE else "Business Workflow Analysis",
                "content": primary_detail,
            },
            {"id": "authentication", "title": "Authentication and Access", "content": state.get("auth_logic", "")},
            {"id": "web_research", "title": "Web Research and Best Practices", "content": state.get("web_research", "")},
        ],
        "mermaid_diagrams": [
            {"id": "repo_overview", "title": "Repository Structure and Flow", "code": state.get("mermaid", "")}
        ],
        "agent_trace": [
            "Coordinator Agent",
            "Repository Scanner Agent",
            "Technical Analysis Agent",
            "Business Analysis Agent",
            "Authentication Agent",
            "Diagram Agent",
            "Web Research Agent",
            "Documentation Agent",
        ],
    }

    _emit_progress(state, 98, "Documentation agent packaged final output")
    return {"documentation_json": json.dumps(doc)}


def run_analysis_graph(
    repo_zip_path: str,
    project_name: str,
    persona: str = PERSONA_SDE,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    workflow = StateGraph(AnalysisState)
    workflow.add_node("coordinator_step", _coordinator_agent)
    workflow.add_node("repo_scan_step", _repo_scanner_agent)
    workflow.add_node("summary_step", _summary_agent)
    workflow.add_node("technical_step", _technical_analysis_agent)
    workflow.add_node("business_step", _business_analysis_agent)
    workflow.add_node("auth_step", _auth_analysis_agent)
    workflow.add_node("diagram_step", _diagram_agent)
    workflow.add_node("web_research_step", _web_search_agent)
    workflow.add_node("documentation_step", _documentation_agent)

    workflow.set_entry_point("coordinator_step")
    workflow.add_edge("coordinator_step", "repo_scan_step")
    workflow.add_edge("repo_scan_step", "summary_step")
    workflow.add_edge("summary_step", "technical_step")
    workflow.add_edge("technical_step", "business_step")
    workflow.add_edge("business_step", "auth_step")
    workflow.add_edge("auth_step", "diagram_step")
    workflow.add_edge("diagram_step", "web_research_step")
    workflow.add_edge("web_research_step", "documentation_step")
    workflow.add_edge("documentation_step", END)

    graph = workflow.compile()
    result = graph.invoke(
        {
            "repo_zip_path": repo_zip_path,
            "project_name": project_name,
            "persona": persona,
            "progress_callback": progress_callback,
        }
    )

    if progress_callback:
        progress_callback(100, "Analysis completed")

    return {
        "summary": result.get("summary", ""),
        "core_logic": result.get("technical_analysis", ""),
        "business_analysis": result.get("business_analysis", ""),
        "auth_logic": result.get("auth_logic", ""),
        "mermaid": result.get("mermaid", ""),
        "documentation_json": result.get("documentation_json", "{}"),
        "persona": persona,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
    }


def _answer_node(state: AskState) -> AskState:
    question = state.get("question", "").strip()
    question_lower = question.lower()

    if not question:
        return {"answer": "Please provide a question about the repository."}

    if "mermaid" in question_lower or "diagram" in question_lower:
        return {"answer": state.get("mermaid", "Mermaid diagram is not available yet.")}
    if "core" in question_lower and "logic" in question_lower:
        return {"answer": state.get("core_logic", "Core logic is not available yet.")}
    if "auth" in question_lower or "authentication" in question_lower or "authorization" in question_lower:
        return {"answer": state.get("auth_logic", "Authentication logic is not available yet.")}

    model = _get_chat_model()
    context = (
        f"Summary:\n{state.get('summary', '')}\n\n"
        f"Core Logic:\n{state.get('core_logic', '')}\n\n"
        f"Auth Logic:\n{state.get('auth_logic', '')}\n\n"
        f"Mermaid:\n{state.get('mermaid', '')}\n\n"
        f"Documentation JSON:\n{state.get('documentation_json', '{}')}"
    )

    if model is None:
        return {"answer": f"LLM is not configured. Available context:\n\n{context}\n\nQuestion: {question}"}

    prompt = (
        "You are a repository assistant. Answer only from provided context. "
        "If uncertain, say clearly that information is insufficient.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    try:
        response = model.invoke(prompt)
        text = getattr(response, "content", "")
        if isinstance(text, str) and text.strip():
            return {"answer": text.strip()}
    except Exception:
        pass

    return {"answer": "Unable to generate an answer right now. Please try again."}


def run_question_graph(
    question: str,
    summary: str,
    core_logic: str,
    auth_logic: str,
    mermaid: str,
    documentation_json: str,
) -> str:
    workflow = StateGraph(AskState)
    workflow.add_node("answer_step", _answer_node)
    workflow.set_entry_point("answer_step")
    workflow.add_edge("answer_step", END)
    graph = workflow.compile()
    result = graph.invoke(
        {
            "question": question,
            "summary": summary,
            "core_logic": core_logic,
            "auth_logic": auth_logic,
            "mermaid": mermaid,
            "documentation_json": documentation_json,
        }
    )
    return result.get("answer", "Unable to answer at this time.")
