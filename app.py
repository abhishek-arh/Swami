import json
import time
from typing import Any, Dict, Optional

import requests
import streamlit as st


st.set_page_config(page_title="Project Management UI", layout="wide")


def init_state() -> None:
	if "api_base_url" not in st.session_state:
		st.session_state.api_base_url = "http://localhost:8000"
	if "token" not in st.session_state:
		st.session_state.token = ""
	if "username" not in st.session_state:
		st.session_state.username = ""
	if "role" not in st.session_state:
		st.session_state.role = ""
	if "active_analysis_job_id" not in st.session_state:
		st.session_state.active_analysis_job_id = ""
	if "active_analysis_project_id" not in st.session_state:
		st.session_state.active_analysis_project_id = ""


def auth_headers() -> Dict[str, str]:
	if not st.session_state.token:
		return {}
	return {"Authorization": f"Bearer {st.session_state.token}"}


def api_request(
	method: str,
	path: str,
	json_body: Optional[Dict[str, Any]] = None,
	data_body: Optional[Dict[str, Any]] = None,
	files: Optional[Dict[str, Any]] = None,
) -> requests.Response:
	url = f"{st.session_state.api_base_url.rstrip('/')}{path}"
	return requests.request(
		method=method,
		url=url,
		headers=auth_headers(),
		json=json_body,
		data=data_body,
		files=files,
		timeout=60,
	)


def show_response(response: requests.Response, success_message: str) -> None:
	try:
		payload = response.json()
	except Exception:
		payload = {"detail": response.text}

	if 200 <= response.status_code < 300:
		st.success(success_message)
		st.json(payload)
	else:
		message = payload.get("detail", "Request failed") if isinstance(payload, dict) else "Request failed"
		st.error(f"{response.status_code}: {message}")
		st.json(payload)


def render_documentation(documentation_json: str) -> None:
	if not documentation_json:
		st.info("No structured documentation found in analysis output.")
		return

	try:
		doc = json.loads(documentation_json)
	except json.JSONDecodeError:
		st.warning("Documentation payload is not valid JSON.")
		st.code(documentation_json)
		return

	st.markdown(f"### {doc.get('title', 'Repository Documentation')}")
	st.caption(f"Audience: {doc.get('audience', 'Unknown')} | Generated: {doc.get('generated_at', 'N/A')}")

	for section in doc.get("sections", []):
		st.markdown(f"#### {section.get('title', 'Section')}")
		st.markdown(section.get("content", ""))

	for diagram in doc.get("mermaid_diagrams", []):
		st.markdown(f"#### Mermaid: {diagram.get('title', 'Diagram')}")
		st.code(diagram.get("code", ""), language="mermaid")

	agents = doc.get("agent_trace", [])
	if agents:
		st.markdown("#### Agent Orchestration")
		for agent in agents:
			st.markdown(f"- {agent}")


def auth_section() -> None:
	st.subheader("Authentication")
	auth_tabs = st.tabs(["Login", "Register", "Session"])

	with auth_tabs[0]:
		with st.form("login_form"):
			username = st.text_input("Username", key="login_username")
			password = st.text_input("Password", type="password", key="login_password")
			submit = st.form_submit_button("Login")
			if submit:
				try:
					response = api_request("POST", "/login", json_body={"username": username, "password": password})
					if response.status_code == 200:
						token_data = response.json()
						st.session_state.token = token_data.get("access_token", "")
						st.session_state.username = username
						st.success("Login successful")
						st.rerun()
					else:
						show_response(response, "")
				except Exception as exc:
					st.error(f"Login failed: {exc}")

	with auth_tabs[1]:
		with st.form("register_form"):
			username = st.text_input("Username", key="register_username")
			email = st.text_input("Email", key="register_email")
			password = st.text_input("Password", type="password", key="register_password")
			role = st.selectbox("Role", options=["user", "admin"], key="register_role")
			submit = st.form_submit_button("Register")
			if submit:
				body = {
					"username": username,
					"email": email,
					"password": password,
					"role": role,
				}
				try:
					response = api_request("POST", "/register", json_body=body)
					if response.status_code in (200, 201):
						token_data = response.json()
						st.session_state.token = token_data.get("access_token", "")
						st.session_state.username = username
						st.session_state.role = role
						st.success("Registration successful")
						st.rerun()
					else:
						show_response(response, "")
				except Exception as exc:
					st.error(f"Registration failed: {exc}")

	with auth_tabs[2]:
		st.write(f"Logged in user: {st.session_state.username or 'Not logged in'}")
		st.write(f"Token present: {'Yes' if st.session_state.token else 'No'}")
		if st.button("Logout"):
			st.session_state.token = ""
			st.session_state.username = ""
			st.session_state.role = ""
			st.success("Logged out")
			st.rerun()


def projects_section() -> None:
	st.subheader("Projects")
	col1, col2 = st.columns([1, 1])

	with col1:
		st.markdown("### Create Project")
		with st.form("create_project_form"):
			name = st.text_input("Project name")
			description = st.text_area("Description", value="")
			submit = st.form_submit_button("Create")
			if submit:
				if not st.session_state.token:
					st.error("Please login first")
				else:
					response = api_request("POST", "/projects", json_body={"name": name, "description": description or None})
					show_response(response, "Project created")

	with col2:
		st.markdown("### List Projects")
		if st.button("Refresh Projects"):
			if not st.session_state.token:
				st.error("Please login first")
			else:
				response = api_request("GET", "/projects")
				show_response(response, "Projects fetched")

	st.markdown("### Get Single Project")
	project_identifier = st.text_input("Project ID or UUID", key="single_project_identifier")
	if st.button("Get Project"):
		if not st.session_state.token:
			st.error("Please login first")
		elif not project_identifier:
			st.error("Please provide project identifier")
		else:
			response = api_request("GET", f"/projects/{project_identifier}")
			show_response(response, "Project fetched")


def repository_section() -> None:
	st.subheader("Repository")
	repo_tabs = st.tabs(["Upload ZIP", "Link GitHub"])

	with repo_tabs[0]:
		project_identifier = st.text_input("Project ID/UUID", key="zip_project_identifier")
		uploaded = st.file_uploader("Upload .zip file", type=["zip"])
		if st.button("Upload ZIP"):
			if not st.session_state.token:
				st.error("Please login first")
			elif not project_identifier or uploaded is None:
				st.error("Provide project identifier and zip file")
			else:
				files = {"file": (uploaded.name, uploaded.getvalue(), "application/zip")}
				response = api_request("POST", f"/projects/{project_identifier}/upload-zip", files=files)
				show_response(response, "ZIP uploaded")

	with repo_tabs[1]:
		project_identifier = st.text_input("Project ID/UUID", key="github_project_identifier")
		github_url = st.text_input("GitHub URL", key="github_url")
		if st.button("Link GitHub"):
			if not st.session_state.token:
				st.error("Please login first")
			elif not project_identifier or not github_url:
				st.error("Provide project identifier and GitHub URL")
			else:
				response = api_request(
					"POST",
					f"/projects/{project_identifier}/link-github",
					data_body={"github_url": github_url},
				)
				show_response(response, "GitHub repo linked")


def analysis_section() -> None:
	st.subheader("Repository Analysis")
	tabs = st.tabs(["Analyze with Progress", "Get Cached Analysis", "Ask Question"])

	with tabs[0]:
		project_identifier = st.text_input("Project ID/UUID", key="analyze_project_identifier")
		persona_label = st.radio(
			"Evaluation Mode",
			options=["Software Developer", "Project Manager"],
			horizontal=True,
		)
		persona = "sde" if persona_label == "Software Developer" else "pm"
		force_refresh = st.checkbox("Force refresh", value=False)

		if st.button("Start Analysis Job"):
			if not st.session_state.token:
				st.error("Please login first")
			elif not project_identifier:
				st.error("Provide project identifier")
			else:
				response = api_request(
					"POST",
					f"/projects/{project_identifier}/analyze/start",
					json_body={"force_refresh": force_refresh, "persona": persona},
				)
				if 200 <= response.status_code < 300:
					payload = response.json()
					st.session_state.active_analysis_job_id = payload.get("job_id", "")
					st.session_state.active_analysis_project_id = project_identifier
					st.success("Analysis job started")
				else:
					show_response(response, "")

		active_job_id = st.session_state.active_analysis_job_id
		active_project_id = st.session_state.active_analysis_project_id
		if active_job_id and active_project_id:
			status_resp = api_request("GET", f"/projects/{active_project_id}/analysis/jobs/{active_job_id}")
			if 200 <= status_resp.status_code < 300:
				job = status_resp.json()
				progress = int(job.get("progress_percent", 0))
				status = job.get("status", "unknown")
				step = job.get("current_step", "")

				st.markdown(f"**Job ID:** {active_job_id}")
				st.markdown(f"**Status:** {status}")
				st.progress(progress / 100 if progress <= 100 else 1.0, text=f"{progress}% - {step}")

				events = job.get("events", [])
				if events:
					st.markdown("#### Real-time Progress Log")
					for event in events[-15:]:
						st.markdown(f"- {event.get('progress_percent', 0)}% | {event.get('message', '')}")

				if status in ("queued", "running"):
					time.sleep(1.2)
					st.rerun()
				elif status == "completed":
					st.success("Analysis completed. Fetch cached analysis to view formatted documentation.")
				elif status == "failed":
					st.error(job.get("error_message") or "Analysis failed")
			else:
				show_response(status_resp, "")

	with tabs[1]:
		project_identifier = st.text_input("Project ID/UUID", key="analysis_project_identifier")
		if st.button("Fetch Analysis"):
			if not st.session_state.token:
				st.error("Please login first")
			elif not project_identifier:
				st.error("Provide project identifier")
			else:
				response = api_request("GET", f"/projects/{project_identifier}/analysis")
				if 200 <= response.status_code < 300:
					payload = response.json()
					st.success("Analysis fetched")
					st.json(payload)
					render_documentation(payload.get("documentation_json", ""))
				else:
					show_response(response, "")

	with tabs[2]:
		project_identifier = st.text_input("Project ID/UUID", key="ask_project_identifier")
		question = st.text_area(
			"Question",
			value="Show me the core logic flow",
			help="Examples: 'show mermaid diagram', 'explain authentication logic', 'what does this repository do?'",
		)
		if st.button("Ask"):
			if not st.session_state.token:
				st.error("Please login first")
			elif not project_identifier:
				st.error("Provide project identifier")
			elif not question.strip():
				st.error("Question cannot be empty")
			else:
				response = api_request(
					"POST",
					f"/projects/{project_identifier}/ask",
					json_body={"question": question},
				)
				show_response(response, "Answer generated")


def admin_section() -> None:
	st.subheader("Admin Controls")
	st.caption("Use an admin account that owns the project for grant/revoke operations")

	tabs = st.tabs(["Grant Access", "Revoke Access"])

	with tabs[0]:
		project_identifier = st.text_input("Project ID/UUID", key="grant_project_identifier")
		user_id = st.number_input("User ID to grant", min_value=1, step=1, value=1)
		if st.button("Grant"):
			if not st.session_state.token:
				st.error("Please login first")
			elif not project_identifier:
				st.error("Provide project identifier")
			else:
				response = api_request(
					"POST",
					f"/projects/{project_identifier}/grant-access",
					json_body={"user_id": int(user_id)},
				)
				show_response(response, "Access granted")

	with tabs[1]:
		project_identifier = st.text_input("Project ID/UUID", key="revoke_project_identifier")
		user_id = st.number_input("User ID to revoke", min_value=1, step=1, value=1, key="revoke_user_id")
		if st.button("Revoke"):
			if not st.session_state.token:
				st.error("Please login first")
			elif not project_identifier:
				st.error("Provide project identifier")
			else:
				response = api_request(
					"DELETE",
					f"/projects/{project_identifier}/revoke-access/{int(user_id)}",
				)
				show_response(response, "Access revoked")


def raw_request_tool() -> None:
	st.subheader("Raw API Tester")
	st.caption("Use this for any endpoint not covered above")

	method = st.selectbox("Method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
	path = st.text_input("Path", value="/projects")
	body = st.text_area("JSON body", value="{}")

	if st.button("Send Request"):
		try:
			parsed = json.loads(body) if body.strip() else None
		except json.JSONDecodeError:
			st.error("Invalid JSON body")
			return

		response = api_request(method, path, json_body=parsed)
		show_response(response, "Request successful")


def main() -> None:
	init_state()

	st.title("Project Management + Repository Analysis UI")
	st.caption("FastAPI backend companion UI")

	with st.sidebar:
		st.header("Settings")
		st.session_state.api_base_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
		st.write(f"Current user: {st.session_state.username or 'Not logged in'}")
		st.write(f"Token: {'Present' if st.session_state.token else 'Missing'}")

	auth_section()

	if not st.session_state.token:
		st.info("Login or register to access project, repository, analysis, and admin actions.")
		return

	sections = st.tabs([
		"Projects",
		"Repository",
		"Analysis",
		"Admin",
		"Raw API",
	])

	with sections[0]:
		projects_section()
	with sections[1]:
		repository_section()
	with sections[2]:
		analysis_section()
	with sections[3]:
		admin_section()
	with sections[4]:
		raw_request_tool()


if __name__ == "__main__":
	main()
