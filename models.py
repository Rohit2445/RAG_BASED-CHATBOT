# models.py
from sqlmodel import SQLModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy import Column
from sqlalchemy import JSON as SA_JSON

class UserTable(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, nullable=False)
    email: str
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentChunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_document: str = Field(index=True)
    content: str
    mock_embedding: List[float] = Field(default_factory=list, sa_column=Column(SA_JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    role: str
    message: str
    context_chunks: List[str] = Field(default_factory=list, sa_column=Column(SA_JSON))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
