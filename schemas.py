# schemas.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    session_id: str = Field(description="Unique ID for the conversation session.")
    query: str = Field(description="The user's question.")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    context_used: List[str] = Field(description="The chunks of policy text used to generate the answer.")

class ChatHistoryOut(BaseModel):
    session_id: str
    role: str
    message: str
    context_chunks: List[str]
    timestamp: datetime
    class Config:
        from_attributes = True
