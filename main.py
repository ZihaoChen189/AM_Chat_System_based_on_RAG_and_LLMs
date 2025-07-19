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
            
        logger.info(f"Conversation history saved: {conversation_id}")
        
    except Exception as e:
        logger.error(f"Failed to save conversation history: {e}")

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
        if model_type in ["deepseek", "deepseek-reasoner"]:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("DEEPSEEK_API_KEY environment variable not set")
                yield  # 每次执行到 yield，就“暂停”函数运行，把 yield 后的值“抛”给外部调用者
                return
        else:  # gpt4.1 or gpt4o
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
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
            rebuild_start_time = time.time()
            logger.info("=" * 80)
            logger.info("🚀 STARTING COMPLETE INDEX REBUILD")
            logger.info("=" * 80)
            
            # Load and build main index
            logger.info("📚 Phase 1: Loading documents and building main index...")
            main_index_start = time.time()
            documents = rag_system.load_documents()
            if documents:
                rag_system.build_index(documents, force_rebuild=True)
                rag_system.create_chat_engine()
                main_index_time = time.time() - main_index_start
                logger.info(f"✅ Main index built successfully with {len(documents)} documents (time: {main_index_time:.2f}s)")
            else:
                main_index_time = time.time() - main_index_start
                logger.info(f"⚠️  No documents found for main index (time: {main_index_time:.2f}s)")
            
            # Build folder indexes
            logger.info("\n📁 Phase 2: Building folder indexes...")
            folder_index_start = time.time()
            try:
                rag_system.build_folder_indexes(force_rebuild=force_rebuild_index)
                folders = rag_system.get_available_folders()
                folder_index_time = time.time() - folder_index_start
                if folders:
                    logger.info(f"✅ Folder indexes built successfully for {len(folders)} folders (time: {folder_index_time:.2f}s)")
                else:
                    logger.info(f"⚠️  No subfolders found for folder indexing (time: {folder_index_time:.2f}s)")
            except Exception as e:
                folder_index_time = time.time() - folder_index_start
                logger.warning(f"❌ Warning during folder index building (time: {folder_index_time:.2f}s): {e}")
            
            # Total rebuild summary
            total_rebuild_time = time.time() - rebuild_start_time
            logger.info("\n" + "=" * 80)
            logger.info("🎯 COMPLETE INDEX REBUILD SUMMARY")
            logger.info("=" * 80)
            logger.info(f"🕒 Total rebuild time: {total_rebuild_time:.2f} seconds")
            logger.info(f"📚 Main index time: {main_index_time:.2f} seconds ({main_index_time/total_rebuild_time*100:.1f}%)")  # 主索引耗时占总重建用时的百分比”，保留一位小数
            logger.info(f"📁 Folder index time: {folder_index_time:.2f} seconds ({folder_index_time/total_rebuild_time*100:.1f}%)")  # 代表文件夹索引阶段占整个重建流程的百分比
            if documents:
                logger.info(f"📄 Total documents processed: {len(documents)}")
                avg_time_per_doc = total_rebuild_time / len(documents)  # 平均每份文档花了多少
                logger.info(f"⚡ Average time per document: {avg_time_per_doc:.3f} seconds")
            logger.info("=" * 80)
            
        else:   # 通常在RAG系统里 会有两种模式
                # 重建索引 (传文档 + force_rebuild=True): 把所有文档重新读一遍 生成新的索引 覆盖原有
                # 加载已有索引 (传空文档 + force_rebuild=False): 只检查本地有没有已经保存的索引文件 直接加载 不做重建
            logger.info("Normal startup, skipping document loading for faster startup")
            # Try to load existing index without loading documents
            try:
                rag_system.build_index([], force_rebuild=False)  ### SOS []
                if rag_system.index is not None:
                    rag_system.create_chat_engine()  # 已经根据创建的index构建了chat对话引擎
                    logger.info("System started successfully, using existing index")
                else:
                    logger.info("System started successfully, no existing index found")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                logger.info("System started successfully, index will be built when documents are uploaded")  ### SOS
            
            # Build folder indexes (normal startup)
            try:
                logger.info("Starting to build folder indexes...")
                rag_system.build_folder_indexes(force_rebuild=force_rebuild_index)  # 加载folder index
                folders = rag_system.get_available_folders()
                if folders:
                    logger.info(f"Folder indexes built successfully, available folders: {folders}")
                else:
                    logger.info("No subfolders found, skipping folder index building")
            except Exception as e:
                logger.warning(f"Warning during folder index building: {e}")
            
    except Exception as e:
        logger.error(f"System startup failed: {e}")
    
    yield
    
    # Clean up resources on shutdown
    logger.info("Shutting down RAG system...")
    if rag_system:
        # Add cleanup logic here, such as closing database connections
        pass

app = FastAPI(
    title="Multi-Model RAG Chat System", 
    description="Multi-model RAG chat system based on LlamaIndex, supporting DeepSeek, DeepSeek Reasoner, GPT-4.1, GPT-4o",
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

class SystemStatus(BaseModel):
    is_ready: bool
    documents_count: int
    index_status: str  # Error, Ready, Not initialized, Not built
    error_message: Optional[str] = None  # "RAG system not initialized" or other errors

class FolderInfo(BaseModel):
    folder_name: str
    documents_count: int 
    has_index: bool
    index_status: str  # Ready, Not built

# Global variables SOS
rag_system: Optional[MultiModelRAGSystem] = None
force_rebuild_index = False  # Whether to force rebuild index
model_type = "deepseek"  # Default model type

class ConnectionManager:  # 管理所有活跃的WebSocket连接
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        # 遇到 I/O 操作或“慢任务”时，把控制权让出去，让别的代码先跑，等到结果准备好了再回来继续执行。
        await websocket.accept()  # await只能在async修饰的函数里用 让我在这里等这个异步操作完 完成前我可以把CPU让给其他请求/任务
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)  # 向指定的某一个WebSocket连接发消息

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)  # 向所有当前在线的WebSocket连接广播同一条消息

manager = ConnectionManager()

### 后端主页路由  ###
@app.get("/api/model")
async def get_model_info():
    """Get current model information"""
    global model_type
    
    model_descriptions = {
        "deepseek": "DeepSeek Chat Model",
        "deepseek-reasoner": "DeepSeek Reasoner Model",
        "gpt4.1": "GPT-4 Turbo Preview",
        "gpt4o": "GPT-4o"
    }
    
    return {
        "model_type": model_type,
        "model_name": model_descriptions.get(model_type, "Unknown Model"),
        "api_base": {
            "deepseek": "https://api.deepseek.com/beta",
            "deepseek-reasoner": "https://api.deepseek.com/v1",
            "gpt4.1": "https://api.openai.com/v1", ###############
            "gpt4o": "https://api.openai.com/v1"   ############### SOS
        }.get(model_type, "Unknown")
    }

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status"""
    global rag_system
    
    if not rag_system:
        return SystemStatus(
            is_ready=False,
            documents_count=0,
            index_status="Not initialized",
            error_message="RAG system not initialized"
        )
    
    try:
        documents = rag_system.list_documents()
        return SystemStatus(
            is_ready=rag_system.index is not None,
            documents_count=len(documents),
            index_status="Ready" if rag_system.index else "Not built",
        )
    except Exception as e:
        return SystemStatus(
            is_ready=False,  
            documents_count=0,
            index_status="Error",
            error_message=str(e)
        )

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

@app.websocket("/ws")  # 负责和前端建立WebSocket连接 **只**处理WebSocket协议 一旦前端发起WebSocket连接 就会走到这个协程
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data.get("type") == "ping":

                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            elif message_data.get("type") == "chat":
                # Handle real-time chat
                if rag_system and rag_system.chat_engine:
                    try:
                        user_message = message_data.get("message", "")
                        response = rag_system.chat(user_message)

                        # Generate conversation ID if not provided
                        conversation_id = message_data.get("conversation_id") or str(uuid.uuid4())

                        # Save conversation history to file
                        save_conversation_history(conversation_id, user_message, response, None)

                        await manager.send_personal_message(
                            json.dumps({
                                "type": "chat_response",
                                "response": response,
                                "conversation_id": conversation_id,
                                "timestamp": datetime.now().isoformat()
                            }),
                            websocket
                        )
                    except Exception as e:
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "error",
                                "message": f"Failed to process message: {e}"
                            }),
                            websocket
                        )
                else:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "message": "RAG system not ready"
                        }),
                        websocket
                    )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-model RAG chat system")
    parser.add_argument("--model", choices=["deepseek", "gpt4.1", "gpt4o","deepseek-reasoner"], default="deepseek", help="Select model type to use (default: deepseek)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild index on startup (even if index files exist)")
    parser.add_argument("--host", default="localhost", help="Server host address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    # 'http://localhost:8000/api'
    args = parser.parse_args()
    
    # Set global variables
    model_type = args.model
    force_rebuild_index = args.rebuild_index
    
    # Display startup information
    print(f"🤖 Using model: {model_type.upper()}")
    if model_type == "deepseek":
        print("📋 Please ensure DEEPSEEK_API_KEY environment variable is set")
    else:
        print("📋 Please ensure OPENAI_API_KEY environment variable is set")
    
    if args.rebuild_index:
        print("🔄 Startup parameter: Force rebuild index")  # 如果加了 --rebuild-index 会有提示
    
    print(f"🚀 Starting server: http://{args.host}:{args.port}")  # 服务端
    print(f"📖 API documentation: http://{args.host}:{args.port}/docs")  # API 文档
    print(f"🌐 Frontend interface: http://{args.host}:{args.port}/static/index.html")  # 前端页面各自的入口地址
    
    import uvicorn  # 用Uvicorn启动FastAPI项目 绑定刚才设置的主机IP和端口
    uvicorn.run(
        app,
        host=args.host, 
        port=args.port, 
    )
