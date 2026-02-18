
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table, Enum as SQLEnum
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
import os
import requests
import re
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "changeme")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024))

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
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

# Create tables
Base.metadata.create_all(bind=engine)

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

# FastAPI App
app = FastAPI(title="Project Management API", version="1.0")

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
    
    return {"message": "File uploaded successfully", "location": file_location}

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
    
    return {"message": "GitHub repository linked successfully", "url": github_url, "location": file_location}

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
    
    # Check if user has access
    if project.owner_id != current_user.id and current_user not in project.users_with_access:
        raise HTTPException(status_code=403, detail="You don't have access to this project")
    
    return project

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
