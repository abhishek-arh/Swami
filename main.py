from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table, Enum as SQLEnum, Text, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, validator
from enum import Enum
import uuid
import json
import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

from repo_agent import run_analysis_graph, run_question_graph

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-env")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_DIR = "uploads"
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Database setup
DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Enums
class RoleEnum(str, Enum):
    admin = "admin"
    user = "user"


class PersonaEnum(str, Enum):
    sde = "sde"
    pm = "pm"


class AnalysisJobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"

# Database Models
project_access = Table(
    'project_access',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('project_id', Integer, ForeignKey('projects.id'))
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(RoleEnum), nullable=False)
    
    owned_projects = relationship("Project", back_populates="owner")
    accessible_projects = relationship("Project", secondary=project_access, back_populates="users_with_access")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    repo_type = Column(String)  # "zip" or "github"
    repo_location = Column(String)  # file path or github URL
    
    owner = relationship("User", back_populates="owned_projects")
    users_with_access = relationship("User", secondary=project_access, back_populates="accessible_projects")
    analysis = relationship("RepoAnalysis", back_populates="project", uselist=False)


class RepoAnalysis(Base):
    __tablename__ = "repo_analysis"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), unique=True, nullable=False)
    summary = Column(Text)
    core_logic = Column(Text)
    business_analysis = Column(Text)
    auth_logic = Column(Text)
    mermaid = Column(Text)
    persona = Column(String)
    documentation_json = Column(Text)
    analyzed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="analysis")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    persona = Column(String, nullable=False, default="sde")
    status = Column(SQLEnum(AnalysisJobStatus), nullable=False, default=AnalysisJobStatus.queued)
    progress_percent = Column(Integer, nullable=False, default=0)
    current_step = Column(String)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AnalysisJobEvent(Base):
    __tablename__ = "analysis_job_events"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("analysis_jobs.id"), nullable=False, index=True)
    progress_percent = Column(Integer, nullable=False)
    message = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# Create tables
Base.metadata.create_all(bind=engine)


def ensure_repo_analysis_schema() -> None:
    with engine.begin() as connection:
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(repo_analysis)")).fetchall()
        }
        if "business_analysis" not in columns:
            connection.execute(text("ALTER TABLE repo_analysis ADD COLUMN business_analysis TEXT"))
        if "persona" not in columns:
            connection.execute(text("ALTER TABLE repo_analysis ADD COLUMN persona VARCHAR"))
        if "documentation_json" not in columns:
            connection.execute(text("ALTER TABLE repo_analysis ADD COLUMN documentation_json TEXT"))


ensure_repo_analysis_schema()

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: RoleEnum
    
    @validator('password')
    def validate_password(cls, v):
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password cannot be longer than 72 bytes')
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: int
    unique_id: str
    name: str
    description: Optional[str]
    owner_id: int
    repo_type: Optional[str]
    repo_location: Optional[str]
    
    class Config:
        from_attributes = True

class GrantAccessRequest(BaseModel):
    user_id: int


class RepoAnalysisResponse(BaseModel):
    project_id: int
    summary: Optional[str]
    core_logic: Optional[str]
    business_analysis: Optional[str]
    auth_logic: Optional[str]
    mermaid: Optional[str]
    persona: Optional[str]
    documentation_json: Optional[str]
    analyzed_at: datetime

    class Config:
        from_attributes = True


class AnalyzeRequest(BaseModel):
    force_refresh: bool = False
    persona: PersonaEnum = PersonaEnum.sde


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class AnalysisStartResponse(BaseModel):
    job_id: str
    status: AnalysisJobStatus
    progress_percent: int
    current_step: str


class AnalysisJobEventResponse(BaseModel):
    progress_percent: int
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisJobResponse(BaseModel):
    job_id: str
    status: AnalysisJobStatus
    progress_percent: int
    current_step: Optional[str]
    error_message: Optional[str]
    persona: str
    events: List[AnalysisJobEventResponse]

# Helper Functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    # Truncate password to 72 bytes for bcrypt compatibility
    password_bytes = password.encode('utf-8')[:72]
    return pwd_context.hash(password_bytes.decode('utf-8', errors='ignore'))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Truncate password to 72 bytes for bcrypt compatibility
    password_bytes = plain_password.encode('utf-8')[:72]
    return pwd_context.verify(password_bytes.decode('utf-8', errors='ignore'), hashed_password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    if "sub" in to_encode and to_encode["sub"] is not None:
        to_encode["sub"] = str(to_encode["sub"])
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

        try:
            user_id_int = int(user_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired. Please login again")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token. Please login again")
    
    user = db.query(User).filter(User.id == user_id_int).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def validate_github_url(url: str) -> bool:
    """Validate if GitHub URL is correct and public"""
    github_pattern = r'^https?://github\.com/[\w-]+/[\w.-]+/?$'
    if not re.match(github_pattern, url):
        return False
    
    try:
        # Check if repository exists and is public
        url = url.rstrip('/')
        api_url = url.replace('github.com', 'api.github.com/repos')
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            repo_data = response.json()
            return not repo_data.get('private', True)
        return False
    except Exception:
        return False

def save_upload_file_with_size_limit(upload_file: UploadFile, destination: str) -> None:
    total_size = 0
    try:
        with open(destination, "wb") as f:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail="File size must be less than 10MB"
                    )
                f.write(chunk)
    except HTTPException:
        if os.path.exists(destination):
            os.remove(destination)
        raise
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def download_github_zip(github_url: str, unique_id: str) -> str:
    url = github_url.rstrip('/')
    api_url = url.replace('github.com', 'api.github.com/repos')
    response = requests.get(api_url, timeout=5)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="GitHub repository not found")
    repo_data = response.json()
    default_branch = repo_data.get('default_branch', 'main')

    owner_repo = url.split('github.com/')[1]
    repo_name = owner_repo.split('/')[-1]
    archive_url = f"https://github.com/{owner_repo}/archive/refs/heads/{default_branch}.zip"
    file_location = os.path.join(UPLOAD_DIR, f"{unique_id}_{repo_name}.zip")

    try:
        with requests.get(archive_url, stream=True, timeout=10) as r:
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download repository zip")

            content_length = r.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(status_code=400, detail="Repository zip exceeds 10MB limit")

            total_size = 0
            with open(file_location, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total_size += len(chunk)
                    if total_size > MAX_UPLOAD_SIZE_BYTES:
                        raise HTTPException(status_code=400, detail="Repository zip exceeds 10MB limit")
                    f.write(chunk)
    except HTTPException:
        if os.path.exists(file_location):
            os.remove(file_location)
        raise
    except Exception:
        if os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail="Failed to download repository zip")

    return file_location


def get_project_by_identifier(db: Session, project_identifier: str) -> Project:
    project = None
    if project_identifier.isdigit():
        project = db.query(Project).filter(Project.id == int(project_identifier)).first()
    if project is None:
        project = db.query(Project).filter(Project.unique_id == project_identifier).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def check_project_access(project: Project, current_user: User) -> None:
    if project.owner_id != current_user.id and current_user not in project.users_with_access:
        raise HTTPException(status_code=403, detail="You don't have access to this project")


def run_and_store_analysis(db: Session, project: Project, persona: str = PersonaEnum.sde.value, progress_callback=None) -> RepoAnalysis:
    if not project.repo_location or not os.path.exists(project.repo_location):
        raise HTTPException(status_code=400, detail="Repository file is not available for analysis")

    try:
        result = run_analysis_graph(project.repo_location, project.name, persona=persona, progress_callback=progress_callback)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to analyze repository. Please try again")

    analysis = db.query(RepoAnalysis).filter(RepoAnalysis.project_id == project.id).first()
    if analysis is None:
        analysis = RepoAnalysis(project_id=project.id)
        db.add(analysis)

    analysis.summary = result.get("summary")
    analysis.core_logic = result.get("core_logic")
    analysis.business_analysis = result.get("business_analysis")
    analysis.auth_logic = result.get("auth_logic")
    analysis.mermaid = result.get("mermaid")
    analysis.persona = result.get("persona")
    analysis.documentation_json = result.get("documentation_json")
    analysis.analyzed_at = datetime.utcnow()
    db.commit()
    db.refresh(analysis)
    return analysis


def precompute_project_analysis(db: Session, project: Project, persona: str = PersonaEnum.sde.value) -> Optional[str]:
    try:
        run_and_store_analysis(db, project, persona=persona)
        return None
    except HTTPException as exc:
        return exc.detail
    except Exception:
        return "Repository linked, but automatic analysis failed"

# FastAPI App
app = FastAPI(title="Project Management API", version="1.0")


def append_job_event(db: Session, job: AnalysisJob, percent: int, message: str) -> None:
    job.progress_percent = max(0, min(100, percent))
    job.current_step = message
    job.updated_at = datetime.utcnow()
    db.add(
        AnalysisJobEvent(
            job_id=job.id,
            progress_percent=job.progress_percent,
            message=message,
            created_at=datetime.utcnow(),
        )
    )
    db.commit()


def run_analysis_job(job_id: str, project_id: int, persona: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        project = db.query(Project).filter(Project.id == project_id).first()
        if job is None or project is None:
            return

        job.status = AnalysisJobStatus.running
        job.updated_at = datetime.utcnow()
        db.commit()
        append_job_event(db, job, 2, "Analysis job started")

        def progress(percent: int, message: str) -> None:
            local_job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if local_job is None:
                return
            append_job_event(db, local_job, percent, message)

        run_and_store_analysis(db, project, persona=persona, progress_callback=progress)

        final_job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if final_job:
            final_job.status = AnalysisJobStatus.completed
            final_job.progress_percent = 100
            final_job.current_step = "Completed"
            final_job.updated_at = datetime.utcnow()
            db.commit()
            append_job_event(db, final_job, 100, "Repository analysis completed")
    except Exception as exc:
        failed_job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if failed_job:
            failed_job.status = AnalysisJobStatus.failed
            failed_job.error_message = str(exc)
            failed_job.updated_at = datetime.utcnow()
            db.commit()
            append_job_event(db, failed_job, failed_job.progress_percent or 0, f"Failed: {str(exc)}")
    finally:
        db.close()

# Authentication Endpoints
@app.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if username exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password),
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": str(db_user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    db_user = db.query(User).filter(User.username == user.username).first()
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": str(db_user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

# Project Endpoints
@app.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    project: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new project"""
    # Generate unique identifier
    unique_id = str(uuid.uuid4())
    
    # Create project
    db_project = Project(
        unique_id=unique_id,
        name=project.name,
        description=project.description,
        owner_id=current_user.id
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    
    return db_project

@app.post("/projects/{project_identifier}/upload-zip")
def upload_zip_to_project(
    project_identifier: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a zip file to a project"""
    project = get_project_by_identifier(db, project_identifier)
    
    # Check ownership
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only project owner can upload files")

    # Prevent overwriting an existing repo upload/link
    if project.repo_type == "zip" and project.repo_location and os.path.exists(project.repo_location):
        raise HTTPException(
            status_code=400,
            detail="Project file already exists. Create a new project to upload another file."
        )
    
    # Validate file extension
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")
    
    # Save file
    file_location = os.path.join(UPLOAD_DIR, f"{project.unique_id}_{file.filename}")
    save_upload_file_with_size_limit(file, file_location)
    
    # Update project
    project.repo_type = "zip"
    project.repo_location = file_location
    db.commit()

    analysis_error = precompute_project_analysis(db, project, persona=PersonaEnum.sde.value)
    response = {"message": "File uploaded successfully", "location": file_location}
    if analysis_error:
        response["analysis_warning"] = analysis_error
    else:
        response["analysis_status"] = "completed"

    return response

@app.post("/projects/{project_identifier}/link-github")
def link_github_to_project(
    project_identifier: str,
    github_url: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Link a GitHub repository to a project"""
    project = get_project_by_identifier(db, project_identifier)
    
    # Check ownership
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only project owner can link GitHub repository")
    
    # Validate GitHub URL
    if not validate_github_url(github_url):
        raise HTTPException(
            status_code=400, 
            detail="Invalid GitHub URL or repository is private. Please provide a valid public GitHub repository URL"
        )

    # Download zip from GitHub
    file_location = download_github_zip(github_url, project.unique_id)
    
    # Update project
    project.repo_type = "github"
    project.repo_location = file_location
    db.commit()

    analysis_error = precompute_project_analysis(db, project, persona=PersonaEnum.sde.value)
    response = {"message": "GitHub repository linked successfully", "url": github_url, "location": file_location}
    if analysis_error:
        response["analysis_warning"] = analysis_error
    else:
        response["analysis_status"] = "completed"

    return response

@app.get("/projects", response_model=List[ProjectResponse])
def get_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all projects user has created or has access to"""
    # Get owned projects
    owned_projects = db.query(Project).filter(Project.owner_id == current_user.id).all()
    
    # Get projects with access
    accessible_projects = current_user.accessible_projects
    
    # Combine and remove duplicates
    all_projects = list({p.id: p for p in owned_projects + accessible_projects}.values())
    
    return all_projects

@app.get("/projects/{project_identifier}", response_model=ProjectResponse)
def get_project(
    project_identifier: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific project"""
    project = get_project_by_identifier(db, project_identifier)
    check_project_access(project, current_user)
    
    return project


@app.post("/projects/{project_identifier}/analyze", response_model=RepoAnalysisResponse)
def analyze_project_repo(
    project_identifier: str,
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze repository and cache summary, core/auth logic, and Mermaid diagram"""
    project = get_project_by_identifier(db, project_identifier)
    check_project_access(project, current_user)

    if not project.repo_location:
        raise HTTPException(status_code=400, detail="Please upload or link a repository first")

    existing = db.query(RepoAnalysis).filter(RepoAnalysis.project_id == project.id).first()
    if existing is not None and not request.force_refresh:
        return existing

    return run_and_store_analysis(db, project, persona=request.persona.value)


@app.post("/projects/{project_identifier}/analyze/start", response_model=AnalysisStartResponse)
def start_analysis_job(
    project_identifier: str,
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start async analysis job with real-time progress events"""
    project = get_project_by_identifier(db, project_identifier)
    check_project_access(project, current_user)

    if not project.repo_location:
        raise HTTPException(status_code=400, detail="Please upload or link a repository first")

    active = (
        db.query(AnalysisJob)
        .filter(AnalysisJob.project_id == project.id, AnalysisJob.status.in_([AnalysisJobStatus.queued, AnalysisJobStatus.running]))
        .first()
    )
    if active:
        return {
            "job_id": active.id,
            "status": active.status,
            "progress_percent": active.progress_percent,
            "current_step": active.current_step or "In progress",
        }

    job_id = str(uuid.uuid4())
    job = AnalysisJob(
        id=job_id,
        project_id=project.id,
        persona=request.persona.value,
        status=AnalysisJobStatus.queued,
        progress_percent=0,
        current_step="Queued",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    db.add(AnalysisJobEvent(job_id=job.id, progress_percent=0, message="Queued", created_at=datetime.utcnow()))
    db.commit()

    background_tasks.add_task(run_analysis_job, job.id, project.id, request.persona.value)

    return {
        "job_id": job.id,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "current_step": job.current_step,
    }


@app.get("/projects/{project_identifier}/analysis/jobs/{job_id}", response_model=AnalysisJobResponse)
def get_analysis_job_status(
    project_identifier: str,
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = get_project_by_identifier(db, project_identifier)
    check_project_access(project, current_user)

    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id, AnalysisJob.project_id == project.id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    events = (
        db.query(AnalysisJobEvent)
        .filter(AnalysisJobEvent.job_id == job.id)
        .order_by(AnalysisJobEvent.id.asc())
        .all()
    )

    return {
        "job_id": job.id,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "current_step": job.current_step,
        "error_message": job.error_message,
        "persona": job.persona,
        "events": events,
    }


@app.get("/projects/{project_identifier}/analysis", response_model=RepoAnalysisResponse)
def get_project_analysis(
    project_identifier: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get cached repository analysis"""
    project = get_project_by_identifier(db, project_identifier)
    check_project_access(project, current_user)

    analysis = db.query(RepoAnalysis).filter(RepoAnalysis.project_id == project.id).first()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found. Run /analyze first")
    return analysis


@app.post("/projects/{project_identifier}/ask", response_model=AskResponse)
def ask_project_repo(
    project_identifier: str,
    request: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ask questions about repository analysis, core logic, auth logic, or Mermaid diagram"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    project = get_project_by_identifier(db, project_identifier)
    check_project_access(project, current_user)

    analysis = db.query(RepoAnalysis).filter(RepoAnalysis.project_id == project.id).first()
    if analysis is None:
        analysis = run_and_store_analysis(db, project)

    answer = run_question_graph(
        question=request.question,
        summary=analysis.summary or "",
        core_logic=analysis.core_logic or "",
        auth_logic=analysis.auth_logic or "",
        mermaid=analysis.mermaid or "",
        documentation_json=analysis.documentation_json or "{}",
    )
    return {"answer": answer}

@app.post("/projects/{project_identifier}/grant-access")
def grant_access(
    project_identifier: str,
    request: GrantAccessRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Grant access to a user for a project (Admin only)"""
    # Check if current user is admin
    if current_user.role != RoleEnum.admin:
        raise HTTPException(status_code=403, detail="Only admins can grant access to projects")
    
    project = get_project_by_identifier(db, project_identifier)
    
    # Check if current user owns the project
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only project owner can grant access")
    
    # Get user to grant access
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Grant access
    if user not in project.users_with_access:
        project.users_with_access.append(user)
        db.commit()
        return {"message": f"Access granted to user {user.username}"}
    else:
        return {"message": "User already has access to this project"}

@app.delete("/projects/{project_identifier}/revoke-access/{user_id}")
def revoke_access(
    project_identifier: str,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke access from a user for a project (Admin only)"""
    # Check if current user is admin
    if current_user.role != RoleEnum.admin:
        raise HTTPException(status_code=403, detail="Only admins can revoke access to projects")
    
    project = get_project_by_identifier(db, project_identifier)
    
    # Check if current user owns the project
    if project.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only project owner can revoke access")
    
    # Get user to revoke access
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Revoke access
    if user in project.users_with_access:
        project.users_with_access.remove(user)
        db.commit()
        return {"message": f"Access revoked from user {user.username}"}
    else:
        raise HTTPException(status_code=400, detail="User doesn't have access to this project")

@app.get("/")
def root():
    return {"message": "Project Management API", "version": "1.0"}
