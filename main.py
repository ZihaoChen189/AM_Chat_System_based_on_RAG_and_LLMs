import os
import io
import uuid
import time
import json
import logging
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel 
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse

# import previously created MultiModelRAGSystem
import sys
sys.path.append('.')
from rag_system import MultiModelRAGSystem

# default global variables
rag_system: Optional[MultiModelRAGSystem] = None
force_rebuild_index = False
model_type = "gpt-4o"

def save_conversation_history(conversation_id: str, user_message: str, assistant_response: str, folder_name: Optional[str] = None):
    """save the conversation history to JSON files"""
    try:
        # create conversation history directories if needed
        conversations_dir = Path("./conversations")
        conversations_dir.mkdir(exist_ok=True)

        # unique identifier
        conversation_file = conversations_dir / f"{conversation_id}.json"
        
        # read existing conversation history if possible
        if conversation_file.exists():
            with open(conversation_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {
                "conversation_id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }  # ready to store the conversation

        history["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "folder_name": folder_name
        })  # update
        
        history["updated_at"] = datetime.now().isoformat()

        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)  # save
        logger.info(f"Conversation History Saved with ID: {conversation_id}.")
        
    except Exception as e:  # Exception Handling
        logger.error(f"Failed to Save Conversation History: {e}.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# manage application lifespan events
# define a lifecycle management function for FastAPI
# FastAPI will call this function automatically at startup and shutdown
# "app" parameter is the FastAPI application instance
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system, force_rebuild_index, model_type

    try:
        # get the corresponding LLM API key
        if model_type in ["deepseek-chat", "deepseek-reasoner"]:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("DEEPSEEK_API_KEY Not Found. Please Set it As an Environment Variable.")
                yield
                return  # do not execute following code
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY Not Found. Please Set it As an Environment Variable.")
                yield
                return  # do not execute following code

        # create necessary directories if needed
        Path("./documents").mkdir(exist_ok=True)
        Path("./storage").mkdir(exist_ok=True)
        
        # initialize RAG
        rag_system = MultiModelRAGSystem(
            api_key=api_key,
            model_type=model_type,
            documents_dir="./documents",
            storage_dir="./storage"
        )

        if force_rebuild_index:
            logger.info("🚀 Start Rebuild Index...")
            t_start = time.time()  ##########
            documents = rag_system.load_documents()  # load document pages
            t_end = time.time()  ##########
            load_time = t_end - t_start  ##########
            logger.info(f"CHECK------Global Documents Loaded. Time used: {load_time:.2f}s")  # emergency fix
            if documents:
                logger.info("🚀 Start Rebuild Global Chat Engine Index...")
                rag_system.build_index(documents, force_rebuild=True)  # split, convert and store
                rag_system.create_chat_engine()  # global engine
                logger.info(f"Global Chat Engine Index Built Successfully with {len(documents)} Document Pages.")
            else:
                logger.info(f"No Document Found for Global Chat Engine Index.")
            
            try:
                logger.info("🚀 Start Rebuild Folder Index for Expert Chat Engine...")
                rag_system.build_folder_indexes(force_rebuild=force_rebuild_index)  # split, convert and store for expert engines
                folders = rag_system.get_available_folders()
                if folders:
                    logger.info(f"Folder Index Built Successfully for Expert Chat Engine from Folders: {folders}.")
                else:
                    logger.info(f"No Folder Found for Expert Chat Engine Index.")

            except Exception as e:  # Exception Handling
                logger.error(f"Failed to Rebuild Folder Index for Expert Chat Engine: {e}.")
        
        else:
            logger.info("This Run Will Attempt to Load the Existing (Folder) Index to Create Global Chat Engine and Expert Chat Engine.")

            try:
                rag_system.build_index([], force_rebuild=False)  # load existing vectors with index
                if rag_system.index is not None:
                    rag_system.create_chat_engine()  # create the global chat engine if vectors with index are loaded
                    logger.info("Global Chat Engine Created Successfully.")
                else:
                    logger.info("No Existing Index Found for Global Chat Engine.")

            except Exception as e:  # Exception Handling
                logger.error(f"Failed to Load Existing Index for Global Chat Engine: {e}.")
            
            try:
                rag_system.build_folder_indexes(force_rebuild=force_rebuild_index)  # load existing expert engines
                folders = rag_system.get_available_folders()
                if folders:
                    logger.info(f"Folder Index Loaded Successfully from Folders: {folders}.")
                else:
                    logger.info("No Folder Found for Expert Chat Engine.")

            except Exception as e:  # Exception Handling
                logger.error(f"Failed to Load Existing Folder Index for Expert Chat Engine: {e}.")
            
    except Exception as e:  # Exception Handling
        logger.error(f"System Startup Failed: {e}.")
    
    yield
    # the code before yield is executed when FastAPI starts
    # the code after yield is executed when FastAPI shuts down
    
    # shutdown
    logger.info("Shutting Down System...")
    logger.info("Bye, See You Next Time!")
    if rag_system:
        # closing database connections...
        pass

app = FastAPI(
    title="Chat System", 
    description="Conversational System for AM Based on RAG and LLMs",
    lifespan=lifespan
)

# access management
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow requests from all origins
    #  allow_origins=["http://localhost:3000", "https://yourfrontend.com"],  # only allow the frontend access from these two domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    folder_name: Optional[str] = None

class ChatResponse(BaseModel):  
    response: str
    conversation_id: str
    timestamp: str

class DocumentInfo(BaseModel):
    filename: str
    file_size: int
    upload_time: str
    file_type: str

class FolderInfo(BaseModel):
    folder_name: str
    documents_count: int
    has_index: bool
    index_status: str  # index_status = "Ready" if has_index else "Not built"


@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents():
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=404, detail="RAG System Not Found.")
    
    try:
        documents_dir = Path("./documents")
        document_list = []
        
        # iterate all files and folders in "documents_dir"
        for file_path in documents_dir.rglob("*"):  
            if file_path.is_file():  # if it is a file
                stat = file_path.stat()  # get metadata
                relative_path = file_path.relative_to(documents_dir)  # get the relative path to the "documents_dir"
                document_list.append(DocumentInfo(
                    filename=str(relative_path),  # "folder1/file1.pdf"
                    file_size=stat.st_size,
                    # st_mtime: the last modified time of the file
                    # datetime.fromtimestamp(stat.st_mtime): convert to a datetime object
                    # .isoformat(): convert datetime to ISO string format
                    upload_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    file_type=file_path.suffix  # ".pdf"
                ))
        return sorted(document_list, key=lambda x: x.upload_time, reverse=True)  # latest files appear firstly
        
    except Exception as e:  # Exception Handling
        raise HTTPException(status_code=500, detail=f"Failed to Get Document List: {e}.")

@app.get("/api/folders", response_model=List[FolderInfo])
async def get_folders():
    global rag_system

    if not rag_system:
        raise HTTPException(status_code=404, detail="RAG System Not Found.")
    
    try:
        folders_info = []
        documents_dir = Path("./documents")

        # get all direct subfolders in the "documents_dir" (not recursive)
        for item in documents_dir.iterdir():
            if item.is_dir():  # if it is a folder
                folder_name = item.name  # get the folder name without path information
                documents_count = rag_system.get_folder_documents_count(folder_name)  # get the number of documents in this folder
                has_index = folder_name in rag_system.folder_indexes  # check if the expert engine exist for this folder

                folders_info.append(FolderInfo(
                    folder_name=folder_name,
                    documents_count=documents_count,
                    has_index=has_index,
                    index_status="Ready" if has_index else "Not built"
                ))

        return sorted(folders_info, key=lambda x: x.folder_name)

    except Exception as e:  # Exception Handling
        raise HTTPException(status_code=500, detail=f"Failed to Get Folder List: {e}.")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_documents(chat_message: ChatMessage):
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=404, detail="RAG System Not Found.")
    
    try:
        # generate a new conversation ID if not provided, otherwise use the existing one
        conversation_id = chat_message.conversation_id or str(uuid.uuid4())
        
        # if the specific expert engine is selected
        if chat_message.folder_name:
            if chat_message.folder_name not in rag_system.folder_indexes:
                raise HTTPException(status_code=404, detail=f"Folder Index for Folder '{chat_message.folder_name}' Not Found.")
            start_folder_time = time.time()  ##########
            response = rag_system.chat_with_folder(chat_message.folder_name, chat_message.message)  # get the response from the expert engine
            end_folder_time = time.time() - start_folder_time  ##########
            logger.info(f"Generating Response from {chat_message.folder_name} Folder Index Time: {end_folder_time:.3f} Seconds.")  # 输出日志
            response_prefix = f"[Based on Folder: {chat_message.folder_name}] "  # prefix
        else:
            if not rag_system.chat_engine:
                raise HTTPException(status_code=404, detail="Global RAG system not ready, please ensure there are available documents in the document directory")
            start_general_time = time.time()  ##########
            response = rag_system.chat(chat_message.message)  # get the response from the global engine
            end_general_time = time.time() - start_general_time  ##########
            logger.info(f"Generating Response from Global Chat Engine Time: {end_general_time:.3f} Seconds.")
            response_prefix = "[Global Search] "  # prefix
        
        # save conversation history of this time
        save_conversation_history(conversation_id, chat_message.message, response, chat_message.folder_name)
        
        return ChatResponse(
            response=response_prefix + response,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )  # follow the designed requirement
        
    except Exception as e:  # Exception Handling
        raise HTTPException(status_code=500, detail=f"Failed to Chat: {e}.")

@app.get("/api/download/all")
async def download_all_documents():
    """Download ZIP archive of all documents"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=404, detail="RAG System Not Found.")
    
    try:
        documents_dir = Path("./documents")
        if not documents_dir.exists():
            raise HTTPException(status_code=404, detail="Document Directory Not Found.")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()  # 在内存里创建一个二进制缓冲区，等会儿把 ZIP 内容写进来（不落磁盘）
        
        # 目标文件就是刚刚的 zip_buffer 也就是在内存里创建这个zip包
        # 新建一个 ZIP 文件对象 把它的输出指向 zip_buffer 写入模式 使用 DEFLATED 压缩
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Recursively add all files to ZIP
            for file_path in documents_dir.rglob("*"):
                if file_path.is_file():  # 递归遍历 documents 下的全部条目；只处理文件，跳过目录
                    # Get path relative to documents directory
                    relative_path = file_path.relative_to(documents_dir)  # 文件结构
                    zip_file.write(file_path, relative_path)   #写进去
        
        zip_buffer.seek(0)  # 写完后把内存指针回到起点，准备从头读取整个 ZIP 内容
        # 如果不拉回开头 此时指针在“文件末尾” 你用zip_buffer.read()去读内容时 会什么都读不到 因为从当前位置往后已经没有内容了
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"documents_backup_{timestamp}.zip"
        
        # Return ZIP file stream
        # Content-Disposition: attachment 告诉浏览器“我是个附件（需要下载）” 不要直接在浏览器窗口里打开 而是让用户“保存到本地”
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),  # 只要拿到zip文件的 二进制内容 就可以把这些内容包到 io.BytesIO(...) 里 构造成一个“内存文件”对象
            media_type="application/zip",  # 告诉浏览器“这是一个zip文件” 浏览器会自动弹出下载窗口。
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )   # Content-Disposition: attachment; filename=... 告诉浏览器下载为附件并建议文件名
        
    except Exception as e:  # Exception Handling
        raise HTTPException(status_code=500, detail=f"Failed to Download Resource File Zip: {e}.")

@app.get("/")
async def root():
    # redirect the root path to frontend interface
    return RedirectResponse(url="/static/index.html")

# serve static files from the "static" directory at the "/static" URL path
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # set the private LLM API key
    
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Conversational System for AM Based on RAG and LLM.")
    parser.add_argument("--model", choices=["deepseek-chat", "deepseek-reasoner", "gpt4o", "gpt4.1"], default="gpt4o")
    # action="store_true" -> Boolean flag
    parser.add_argument("--rebuild-index", action="store_true", help="Force Rebuild Index on Startup (Even If Index Files Exist).")
    parser.add_argument("--host", default="localhost", help="Server Host Address (Default: localhost).")
    parser.add_argument("--port", type=int, default=8000, help="Server Port (Default: 8000).")
    args = parser.parse_args()

    # assign command line arguments to global variables
    model_type = args.model
    force_rebuild_index = args.rebuild_index
    
    if args.rebuild_index:
        print("Notice: The Index Will Be Rebuilt During This Run (Due To --rebuild-index Flag).")

    print(f"FastAPI Server Root URL:       http://{args.host}:{args.port}")
    print(f"FastAPI API Documentation:     http://{args.host}:{args.port}/docs")
    print(f"Frontend Interface Entry:      http://{args.host}:{args.port}/static/index.html")

    
    import uvicorn  # start the FastAPI application with Uvicorn web server
    uvicorn.run(
        app,
        host=args.host, 
        port=args.port, 
    )  # expose the FastAPI app as an HTTP API service on the specified host and port
