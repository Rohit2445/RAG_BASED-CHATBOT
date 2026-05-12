# database.py
from sqlmodel import SQLModel, create_engine, Session
import os
from typing import Generator

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_chatbot_policy.db")
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

def create_db_and_tables():
    # Import models at runtime so SQLModel sees them
    from models import UserTable, DocumentChunk, ChatHistory
    print(f"INFO: Creating database and tables at {DATABASE_URL}")
    SQLModel.metadata.create_all(engine)
