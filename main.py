# main.py
import os
import traceback, sys
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Session
from datetime import timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from database import create_db_and_tables, get_session, engine
from rag_service import init_rag_db, RagService
from models import UserTable
from schemas import UserCreate, UserOut, Token, ChatRequest, ChatResponse, ChatHistoryOut
from crud import create_user_db, authenticate_user_db, get_user_by_username, save_chat_message, get_chat_history
from auth import create_access_token, SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from dotenv import load_dotenv
load_dotenv()   # loads .env file into environment variables


PDF_FOLDER = os.getenv("PDFS_FOLDER", "data/pdfs")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("INFO: startup...")
    create_db_and_tables()
    # init rag db
    try:
        with next(iter([get_session()])) as _:
            pass
    except Exception:
        pass

    # Use engine session directly to init RAG
    from sqlmodel import Session as SQLSession
    from database import engine as _engine
    try:
        with SQLSession(_engine) as session:
            init_rag_db(session, folder_path=PDF_FOLDER, reinit=False)
    except Exception:
        traceback.print_exc(file=sys.stdout)
    yield
    print("INFO: shutdown...")

app = FastAPI(title="FastAPI RAG Policy Chatbot", lifespan=lifespan)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(session: Session = Depends(get_session), token: str = Depends(oauth2_scheme)) -> UserTable:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate":"Bearer"}
    )
    import jwt
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user_by_username(session, username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/users/", response_model=UserOut, status_code=status.HTTP_201_CREATED, tags=["Auth"])
def create_user(user_in: UserCreate, session: Session = Depends(get_session)):
    db_user = get_user_by_username(session, username=user_in.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user = create_user_db(session, user_in.username, user_in.email, user_in.password)
    return UserOut.from_orm(new_user)

@app.post("/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(session: Session = Depends(get_session), form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user_db(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate":"Bearer"})
    access_token_expires = timedelta(minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")))
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
def chat_with_policy(request: ChatRequest, session: Session = Depends(get_session), current_user: UserTable = Depends(get_current_user)):
    rag = RagService(session)
    # Save user message
    save_chart = save_chat_message(session, request.session_id, "user", request.query, [])
    # Generate
    answer, ctx = rag.generate_response(request.query)
    # Save assistant message
    save_chat_message(session, request.session_id, "assistant", answer, ctx)
    return ChatResponse(session_id=request.session_id, answer=answer, context_used=ctx)

@app.get("/history/{session_id}", response_model=List[ChatHistoryOut], tags=["Chatbot"])
def get_session_history(session_id: str, session: Session = Depends(get_session), current_user: UserTable = Depends(get_current_user)):
    history = get_chat_history(session, session_id)
    return [ChatHistoryOut.from_orm(h) for h in history]
