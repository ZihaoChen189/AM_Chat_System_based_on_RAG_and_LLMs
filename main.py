from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel 
from typing import List, Optional
from contextlib import asynccontextmanager  # 定义FastAPI的生命周期
import os
import uuid
import json
from pathlib import Path
import logging
from datetime import datetime
import argparse 
import zipfile 
import io
import time

# Import the previously created RAG system
import sys
sys.path.append('.')
from rag_system import MultiModelRAGSystem

# Global variables SOS
rag_system: Optional[MultiModelRAGSystem] = None
force_rebuild_index = False  # Whether to force rebuild index
model_type = "gpt-4o"  # Default model type


# Conversation history saving function
def save_conversation_history(conversation_id: str, user_message: str, assistant_response: str, folder_name: Optional[str] = None):
    """Save conversation history to JSON file"""
    try:
        conversations_dir = Path("./conversations")
        conversations_dir.mkdir(exist_ok=True)
        
        conversation_file = conversations_dir / f"{conversation_id}.json"
        
        # Read existing conversation history
        if conversation_file.exists():
            with open(conversation_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {
                "conversation_id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        # Add new conversation record
        history["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "folder_name": folder_name
        })
        
        history["updated_at"] = datetime.now().isoformat()
        
        # Save to file
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Conversation History Saved with ID: {conversation_id}.")
        
    except Exception as e:
        logger.error(f"Failed to Save Conversation History: {e}.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):  # 定义生命周期管理函数 FastAPI会自动在启动时进入 关闭时退出
    """Application lifecycle management"""
    global rag_system, force_rebuild_index, model_type
    
    # Initialize RAG system on startup
    try:
        # Get corresponding API key based on model type
        if model_type in ["deepseek-chat", "deepseek-reasoner"]:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("DEEPSEEK_API_KEY Not Found. Please Set it As an Environment Variable.")
                yield  # 每次执行到 yield，就“暂停”函数运行，把 yield 后的值“抛”给外部调用者
                return
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY Not Found. Please Set it As an Environment Variable.")
                yield 
                return
        
        # Create necessary directories
        Path("./documents").mkdir(exist_ok=True)
        Path("./storage").mkdir(exist_ok=True)
        
        # Initialize RAG system
        rag_system = MultiModelRAGSystem(
            api_key=api_key,
            model_type=model_type,
            documents_dir="./documents",
            storage_dir="./storage"
        )
        
        # Track total rebuild time if force rebuilding
        if force_rebuild_index:  # 如果命令行参数或环境要求**强制**重建索引 记录开始时间 并打印日志提醒
            logger.info("🚀 Start Rebuild Index...")
            documents = rag_system.load_documents()
            if documents:
                logger.info("🚀 Start Rebuild Global Chat Engine Index...")
                rag_system.build_index(documents, force_rebuild=True)
                rag_system.create_chat_engine()
                logger.info(f"Global Chat Engine Index Built Successfully with {len(documents)} Document Pages.")
            else:
                logger.info(f"No Document Found for Global Chat Engine Index.")
            
            try:
                logger.info("🚀 Start Rebuild Folder Index for Expert Chat Engine...")
                rag_system.build_folder_indexes(force_rebuild=force_rebuild_index)
                folders = rag_system.get_available_folders()
                if folders:
                    logger.info(f"Folder Index Built Successfully for Expert Chat Engine from Folders: {folders}.")
                else:
                    logger.info(f"No Folder Found for Expert Chat Engine Index.")
            except Exception as e:
                logger.error(f"Failed to Rebuild Folder Index for Expert Chat Engine: {e}.")
        
        else:   # 通常在RAG系统里 会有两种模式
                # 重建索引 (传文档 + force_rebuild=True): 把所有文档重新读一遍 生成新的索引 覆盖原有
                # 加载已有索引 (传空文档 + force_rebuild=False): 只检查本地有没有已经保存的索引文件 直接加载 不做重建
            logger.info("This Run Will Attempt to Load the Existing (Folder) Index to Create Global Chat Engine and Expert Chat Engine.")

            # Try to load existing index without loading documents
            try:
                rag_system.build_index([], force_rebuild=False)  ### SOS []
                if rag_system.index is not None:
                    rag_system.create_chat_engine()  # 已经根据创建的index构建了chat对话引擎
                    logger.info("Global Chat Engine Created Successfully.")
                else:
                    logger.info("No Existing Index Found for Global Chat Engine.")
            except Exception as e:
                logger.error(f"Failed to Load Existing Index for Global Chat Engine: {e}.")
            
            try:
                rag_system.build_folder_indexes(force_rebuild=force_rebuild_index)  # 加载folder index
                folders = rag_system.get_available_folders()
                if folders:
                    logger.info(f"Folder Index Loaded Successfully from Folders: {folders}.")
                else:
                    logger.info("No Folder Found for Expert Chat Engine.")
            except Exception as e:
                logger.error(f"Failed to Load Existing Folder Index for Expert Chat Engine: {e}.")
            
    except Exception as e:
        logger.error(f"System Startup Failed: {e}.")
    
    yield
    
    # Clean up resources on shutdown
    logger.info("Shutting Down System...")
    if rag_system:
        # Add cleanup logic here, such as closing database connections
        pass
 

app = FastAPI(
    title="Chat System", 
    description="Conversational System for AM Based on RAG and LLMs",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be restricted to specific domains in production
    #  allow_origins=["http://localhost:3000", "https://yourfrontend.com"],  # 只允许这两个域名的前端访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
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
    index_status: str  # Ready, Not built


### 后端主页路由  ###
@app.get("/api/model")
async def get_model_info():
    """Get current model information"""
    global model_type
    
    model_descriptions = {
        "deepseek": "DeepSeek Chat Model",
        "deepseek-reasoner": "DeepSeek Reasoner Model",
        "gpt4.1": "GPT-4.1",
        "gpt4o": "GPT-4o"
    }
    
    return {
        "model_type": model_type,
        "model_name": model_descriptions.get(model_type, "Unknown Model"),
        "api_base": {
            "deepseek-chat": "https://api.deepseek.com/v1",
            "deepseek-reasoner": "https://api.deepseek.com/v1",
            "gpt4.1": "https://api.openai.com/v1", ###############
            "gpt4o": "https://api.openai.com/v1"   ############### SOS
        }.get(model_type, "Unknown")
    }


@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get document list"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        documents_dir = Path("./documents")
        document_list = []
        
        # Recursively scan files in all subdirectories
        for file_path in documents_dir.rglob("*"):  
            if file_path.is_file():  # 只处理文件
                stat = file_path.stat()
                # Get path relative to documents directory
                relative_path = file_path.relative_to(documents_dir)
                document_list.append(DocumentInfo(
                    filename=str(relative_path),
                    file_size=stat.st_size,
                    upload_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    file_type=file_path.suffix
                ))
        
        return sorted(document_list, key=lambda x: x.upload_time, reverse=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document list: {e}")

@app.get("/api/folders", response_model=List[FolderInfo])
async def get_folders():
    """Get folder list and their status"""
    global rag_system

    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    try:
        folders_info = []

        # Get all subfolders
        documents_dir = Path("./documents")
        for item in documents_dir.iterdir():
            if item.is_dir():
                folder_name = item.name
                documents_count = rag_system.get_folder_documents_count(folder_name)
                has_index = folder_name in rag_system.folder_indexes

                folders_info.append(FolderInfo(
                    folder_name=folder_name,
                    documents_count=documents_count,
                    has_index=has_index,
                    index_status="Ready" if has_index else "Not built"
                ))

        return sorted(folders_info, key=lambda x: x.folder_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get folder list: {e}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_documents(chat_message: ChatMessage):
    """Chat with documents (supports specifying folder)"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Generate or use existing conversation ID
        conversation_id = chat_message.conversation_id or str(uuid.uuid4())
        
        # Choose different processing methods based on whether folder is specified
        if chat_message.folder_name:
            # Query by folder
            if chat_message.folder_name not in rag_system.folder_indexes:
                raise HTTPException(status_code=404, detail=f"Index for folder '{chat_message.folder_name}' does not exist")
            start_folder_time = time.time()  ### 记录开始时间
            response = rag_system.chat_with_folder(chat_message.folder_name, chat_message.message)
            end_folder_time = time.time() - start_folder_time  ### 计算耗时
            logger.info(f"Generating Response from {chat_message.folder_name} Folder Index time: {end_folder_time:.3f} seconds")  # 输出日志
            response_prefix = f"[Based on folder: {chat_message.folder_name}] "
        else:  # 如果没指定文件夹（全局检索）：
            # Global query
            if not rag_system.chat_engine:  # 检查全局chat引擎是否可用
                raise HTTPException(status_code=503, detail="Global RAG system not ready, please ensure there are available documents in the document directory")
            start_general_time = time.time()
            response = rag_system.chat(chat_message.message)  # 用全局问答返回结果
            end_general_time = time.time() - start_general_time
            logger.info(f"Generating Response from General Index time: {end_general_time:.3f} seconds")

            response_prefix = "[Global search] "  # 设置全局前缀。
        
        # Save conversation history to file
        save_conversation_history(conversation_id, chat_message.message, response, chat_message.folder_name)
        
        return ChatResponse(
            response=response_prefix + response,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.post("/api/query")
async def query_documents(chat_message: ChatMessage):
    """Single query (no conversation history, supports specifying folder)"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Choose different processing methods based on whether folder is specified
        if chat_message.folder_name:
            # Query by folder
            if chat_message.folder_name not in rag_system.folder_indexes:
                raise HTTPException(status_code=404, detail=f"Index for folder '{chat_message.folder_name}' does not exist")
            
            response = rag_system.query_folder(chat_message.folder_name, chat_message.message)
            response_prefix = f"[Based on folder: {chat_message.folder_name}] "
        else:
            # Global query
            if not rag_system.index:
                raise HTTPException(status_code=503, detail="Global RAG system not ready, please ensure there are available documents in the document directory")
            
            # Need to add query method to rag_system
            query_engine = rag_system.index.as_query_engine(
                similarity_top_k=3,
                response_mode="tree_summarize"
            )
            response = str(query_engine.query(chat_message.message))
            response_prefix = "[Global search] "
        
        return {
            "response": response_prefix + response,
            "folder_name": chat_message.folder_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    
    
@app.get("/api/download/all")
async def download_all_documents():
    """Download ZIP archive of all documents"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        documents_dir = Path("./documents")
        if not documents_dir.exists():
            raise HTTPException(status_code=404, detail="Documents directory does not exist")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()  # 内存中临时创建一个ZIP压缩文件的 缓冲区 避免频繁读写硬盘 更快也更方便传输
        
        # 目标文件就是刚刚的 zip_buffer 也就是在内存里创建这个zip包
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Recursively add all files to ZIP
            for file_path in documents_dir.rglob("*"):
                if file_path.is_file():
                    # Get path relative to documents directory
                    relative_path = file_path.relative_to(documents_dir)  # 文件结构
                    zip_file.write(file_path, relative_path)
        
        zip_buffer.seek(0) 
        # 如果不拉回开头 此时指针在“文件末尾” 你用zip_buffer.read()去读内容时 会什么都读不到 因为从当前位置往后已经没有内容了
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"documents_backup_{timestamp}.zip"
        
        # Return ZIP file stream
        # Content-Disposition: attachment 告诉浏览器“我是个附件（需要下载）” 不要直接在浏览器窗口里打开 而是让用户“保存到本地”
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),  # 只要拿到zip文件的 二进制内容 就可以把这些内容包到 io.BytesIO(...) 里 构造成一个“内存文件”对象
            media_type="application/zip",  # 告诉浏览器“这是一个zip文件” 浏览器会自动弹出下载窗口。
            headers={"Content-Disposition": f"attachment; filename={filename}"}  # 下载时用这个文件名保存 而不是直接打开
        )   # 返回zip文件内容作为“流式响应”
        
    except Exception as e:
        logger.error(f"Failed to create ZIP file: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

# Root path redirect to frontend
@app.get("/")
async def root():
    """Redirect root path to frontend interface"""
    return RedirectResponse(url="/static/index.html")

# Static file service (for frontend)
# "/static" 这是URL前缀（挂载点）
# StaticFiles(directory="static") 就是把 static 文件夹里的内容变成可以通过 /static/ 访问的静态资源
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Conversational System for AM Based on RAG and LLMs")
    parser.add_argument("--model", choices=["deepseek-chat", "deepseek-reasoner", "gpt4o", "gpt4.1"], default="gpt4o", help="Select the LLM API to Use (Default: gpt4o)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force Rebuild Index on Startup (Even If Index Files Exist)")  # 把命令行参数变成一个布尔值 True/False 开关 命令行没出现这个 就是False
    parser.add_argument("--host", default="localhost", help="Server Host Address (Default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server Port (Default: 8000)")
    args = parser.parse_args()

    # Set global variables
    model_type = args.model
    force_rebuild_index = args.rebuild_index
    
    if args.rebuild_index:
        print("Notice: The Index Will Be Rebuilt During This Run (Due To --rebuild-index Flag).")

    print(f"FastAPI Server Root URL:       http://{args.host}:{args.port}")
    print(f"FastAPI API Documentation:     http://{args.host}:{args.port}/docs")
    print(f"Frontend Interface Entry:      http://{args.host}:{args.port}/static/index.html")

    
    import uvicorn  # 用Uvicorn启动FastAPI项目 绑定刚才设置的主机IP和端口
    uvicorn.run(
        app,
        host=args.host, 
        port=args.port, 
    )  # 让你的 FastAPI 项目变成一个对外开放的 HTTP API 服务 监听指定的 host 和 port 是后端启动 让前端 比如Vue可以通过 HTTP 请求访问这个后端
