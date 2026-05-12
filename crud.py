# crud.py
from sqlmodel import Session, select
from typing import List, Optional
from models import DocumentChunk, ChatHistory, UserTable
from auth import get_password_hash, verify_password

def save_document_chunk(session: Session, chunk: DocumentChunk) -> DocumentChunk:
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    return chunk

def get_all_document_chunks(session: Session) -> List[DocumentChunk]:
    statement = select(DocumentChunk)
    return session.exec(statement).all()

def save_chat_message(session: Session, session_id: str, role: str, message: str, context_chunks: List[str]) -> ChatHistory:
    chat_entry = ChatHistory(
        session_id=session_id,
        role=role,
        message=message,
        context_chunks=context_chunks
    )
    session.add(chat_entry)
    session.commit()
    session.refresh(chat_entry)
    return chat_entry

def get_chat_history(session: Session, session_id: str) -> List[ChatHistory]:
    statement = select(ChatHistory).where(ChatHistory.session_id == session_id).order_by(ChatHistory.timestamp)
    return session.exec(statement).all()

def get_user_by_username(session: Session, username: str) -> Optional[UserTable]:
    statement = select(UserTable).where(UserTable.username == username)
    return session.exec(statement).first()

def create_user_db(session: Session, username: str, email: str, password: str) -> UserTable:
    hashed_password = get_password_hash(password)
    db_user = UserTable(username=username, email=email, hashed_password=hashed_password)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

def authenticate_user_db(session: Session, username: str, password: str) -> Optional[UserTable]:
    user = get_user_by_username(session, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
