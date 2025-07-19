import logging
import random
import shutil
import time
from typing import List, Optional, Dict
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

from llama_index.core import (
    VectorStoreIndex,  # 创建向量索引对象
    SimpleDirectoryReader,
    Document,
    StorageContext,  # 索引数据要存在哪里 怎么存 用什么方式存
    load_index_from_storage,  # 用于从本地加载已有的索引 要传入一个StorageContext对象
    Settings
)

from llama_index.core.node_parser import SentenceSplitter  # node_parser用到了
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGConfig:
    # Document processing configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md", ".html"]
    
    # LLM configuration
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048  # 限制LLM生成的最大token数
    
    # Retrieval configuration
    SIMILARITY_TOP_K = 3
    TOKEN_LIMIT = 3000  # 多轮对话时 保存多少token的历史
    
    # Random seed
    RANDOM_SEED = 42


class SystemPrompts:
    DEFAULT = (
        "You are an intelligent assistant that answers user questions based on provided document content."
        "Please answer based on the retrieved relevant document content. If there is no relevant information in the documents, please clearly state so."
        "When answering, please be accurate, detailed, and helpful. If possible, please cite specific document sources."
    )
# Answer ONLY with the facts listed in the sources below.
# If there isn’t enough information below, say you don’t know. DO NOT generate answers that don’t use the sources below. 
# If asking a clarifying question to the user would help, ask the question.
# If the question is not in English, translate the question to English before generating the search query.

    @staticmethod
    def folder_specific(folder_name: str) -> str:
        return (
            f"You are an intelligent assistant that specializes in answering user questions based on document content from the '{folder_name}' folder."
            "Please answer based on the retrieved relevant document content. If there is no relevant information in the documents, please clearly state so."
            "When answering, please be accurate, detailed, and helpful. If possible, please cite specific document sources."
        )


class MultiModelRAGSystem:
    def __init__(self, api_key: str, model_type: str = "deepseek", documents_dir: str = "./documents", storage_dir: str = "./storage", embedding_model: str = "BAAI/bge-small-zh-v1.5", random_seed: Optional[int] = None):
        # Set random seed for reproducibility
        self.random_seed = random_seed or RAGConfig.RANDOM_SEED  # 如果init没传 有默认的
        self._set_random_seeds()  # defined function below
        
        self.api_key = api_key
        self.model_type = model_type.lower()
        self.documents_dir = Path(documents_dir)
        self.storage_dir = Path(storage_dir)
        
        # Create directories
        self.documents_dir.mkdir(exist_ok=True)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Configure LLM based on model type
        self.llm = self._configure_llm()  # 初始化LLM对象 # defined function below
        
        logger.info(f"Using model: {self.model_type}")
        logger.info(f"Random seed: {self.random_seed}")
        
        # Configure embedding model (using local model with CUDA acceleration)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Embedding model using device: {device}")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            cache_folder="./embedding_cache",
            device=device
        )
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = RAGConfig.CHUNK_SIZE
        Settings.chunk_overlap = RAGConfig.CHUNK_OVERLAP
        
        # Configure node parser
        self.node_parser = SentenceSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
        )
        
        self.index = None
        self.chat_engine = None
        
        # Folder-based indexing functionality
        self.folder_indexes: Dict[str, VectorStoreIndex] = {}
        self.folder_chat_engines: Dict[str, any] = {}
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)  # PyTorch的CPU随机数生成器
        if torch.cuda.is_available():  # GPU
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        # Set deterministic behavior
        # 这两行用于彻底保证操作的确定性 有的底层实现有“非确定性加速优化” 这样写能关掉所有不确定性操作 牺牲一点速度换来“完全可复现”
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set: {self.random_seed}")
    
    def _get_base_llm_config(self) -> dict:
        """Get base LLM configuration"""
        # 专门负责基础公共配置 api_key temperature max_tokens 这些参数都通用
        return {
            'api_key': self.api_key,
            'temperature': RAGConfig.DEFAULT_TEMPERATURE,
            'max_tokens': RAGConfig.DEFAULT_MAX_TOKENS,
            'is_chat_model': True,  # 声明这是“对话大模型”
        }
    
    def _configure_llm(self):  # 根据选的模型类型 动态合成全量配置 并创建模型对象
        """
        Configure LLM based on model type
        
        Returns:
            Configured LLM instance
        """
        base_config = self._get_base_llm_config()
        
        model_configs = {
            "deepseek": {
                "model": "deepseek-chat",  # DeepSeek-V3-0324 Jul-16
                "api_base": "https://api.deepseek.com/v1",  # DeepSeek-R1-0528 JUl-16
            },
            "deepseek-reasoner": {
                "model": "deepseek-reasoner",
                "api_base": "https://api.deepseek.com/v1",
            },
            "gpt4.1": {
                "model": "gpt-4-turbo-preview",
                "api_base": "https://api.openai.com/v1",
            },
            "gpt4o": {
                "model": "gpt-4o",
                "api_base": "https://api.openai.com/v1",
            }
        }
        
        if self.model_type not in model_configs:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {', '.join(model_configs.keys())}")
        # base_config 是通用的LLM配置信息;
        # model_configs 是具体LLM品牌的配置信息
        config = {**base_config, **model_configs[self.model_type]}
        return OpenAILike(**config)  # 参数解包 Keyword Arguments unpacking
        
    def _add_document_metadata(self, doc: Document, folder_name: Optional[str] = None) -> None:
        """Add standard metadata to document"""
        # hasattr 是否有某个属性
        if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:
            file_path = Path(doc.metadata['file_path'])
            doc.metadata.update({
                'filename': file_path.name,
                'file_type': file_path.suffix,
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            })
            if folder_name:
                doc.metadata['folder'] = folder_name
                
    def load_documents(self, file_extensions: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from specified directory
        
        Args:
            file_extensions: List of supported file extensions
            
        Returns:
            List of documents
        """
        if file_extensions is None:
            file_extensions = RAGConfig.SUPPORTED_EXTENSIONS
        
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(self.documents_dir),
                recursive=True,
                required_exts=file_extensions,
            )
            
            documents = reader.load_data(show_progress=True,num_workers=4)
            logger.info(f"Successfully loaded {len(documents)} document pages")  # 多少page
            
            # Add metadata to each document
            for doc in documents:
                self._add_document_metadata(doc)  # 给每一个doc添加元数据
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> None:
        """
        Build or load vector index
        
        Args:
            documents: List of documents
            force_rebuild: Whether to force rebuild index
        """
        index_store_file = self.storage_dir / "index_store.json"
        docstore_file = self.storage_dir / "docstore.json"
        
        try:
            # Check if saved index exists
            if index_store_file.exists() and docstore_file.exists() and not force_rebuild:
                start_time = time.time()
                logger.info("Loading existing index...")

                # StorageContext,  # 索引数据要存在哪里 怎么存 用什么方式存
                # load_index_from_storage,  # 用于从本地加载已有的索引 要传入一个StorageContext对象

                with tqdm(total=1, desc="Loading index", unit="step") as pbar:
                    storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))  # 创建一个索引存储的上下文对象 用于后续索引的加载 这是llama_index读取本地向量索引时的标准写法
                    self.index = load_index_from_storage(storage_context)  # 用llama_index的接口 把本地存储里的向量索引数据加载到内存 并赋值给self.index 后续就可以检索了
                    pbar.update(1)  # 手动把进度条推进1步 表示“加载索引”这一步已经完成
                load_time = time.time() - start_time
                logger.info(f"Index loading completed in {load_time:.2f} seconds")
            else:
                # If no documents provided and no existing index, skip building
                if not documents:  # 如果documents这个列表本身是空的（没有可用文档）那连索引都没法建
                    logger.info("No documents provided and no existing index found, skipping index building")
                    self.index = None
                    return
                
                start_time = time.time()  ####################
                logger.info("Building new vector index...")
                
                # Parse document nodes 切块操作
                parse_start = time.time()  ####################
                with tqdm(total=len(documents), desc="Parsing documents", unit="doc") as pbar:  # 总数=文档数 单位是doc
                    nodes = []
                    for doc in documents:  # 遍历所有文档  所有切的块
                        doc_nodes = self.node_parser.get_nodes_from_documents([doc])  # 每份文档用node_parser（通常是分句/分块器）切分成若干“节点”（chunk）
                        nodes.extend(doc_nodes)  # 把这些节点合并到nodes总列表
                        pbar.update(1)  # bar+=1
                parse_time = time.time() - parse_start  #################### 切块时间
                
                logger.info(f"Documents split into {len(nodes)} nodes in {parse_time:.2f} seconds")  # 这次所有文档总共切成多少个节点 每个节点后续都会做embedding
                
                # Create vector index 做embedding 生成向量并建立索引
                index_start = time.time()  ####################
                with tqdm(total=1, desc="Building vector index", unit="step") as pbar:
                    self.index = VectorStoreIndex(nodes, show_progress=True)  # 调用VectorStoreIndex(nodes, ...) 对所有节点做embedding并建立向量检索索引 索引对象赋值给self.index 供后续检索用
                    pbar.update(1)  # 手动+1
                index_time = time.time() - index_start  ####################生成向量并建立索引用了多长时间
                
                # Save index 保存到本地磁盘
                save_start = time.time()  ####################
                with tqdm(total=1, desc="Saving index", unit="step") as pbar:
                    self.index.storage_context.persist(persist_dir=str(self.storage_dir))  # 把当前索引完整保存到本地指定目录
                    pbar.update(1)
                save_time = time.time() - save_start  ####################保存到本地磁盘用了多长时间
                
                total_time = time.time() - start_time  #################### 一共花了多长时间
                logger.info(f"Vector index built and saved successfully in {total_time:.2f} seconds")
                logger.info(f"  - Parsing: {parse_time:.2f}s, Building: {index_time:.2f}s, Saving: {save_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def build_folder_indexes(self, force_rebuild: bool = False) -> None:  # 分文件夹批量构建索引的核心函数
        """
        Build indexes grouped by folder 批量为每个文件夹下的文档构建向量索引
        
        Args:
            force_rebuild: Whether to force rebuild indexes
        """
        try:
            overall_start_time = time.time()  # 整个过程的起始时间
            # Get all subfolders
            folders = []  # 所有自文件夹的名字
            for item in self.documents_dir.iterdir():  # 获取文档主目录下的所有子文件夹名称 iterdir(): 列举当前目录下所有的文件和文件夹
                if item.is_dir():
                    folders.append(item.name)
            
            logger.info(f"Found {len(folders)} folders: {folders}")
            logger.info("=" * 80)
            logger.info("📊 FOLDER INDEX BUILD TIMING REPORT")
            logger.info("=" * 80)
            
            # Statistics tracking
            folder_stats = []
            total_documents = 0
            total_nodes = 0
            successful_builds = 0
            
            # Iterate through folders with progress bar
            for folder_name in tqdm(folders, desc="Processing folders", unit="folder"):
                folder_start_time = time.time()  # 记录每个文件夹的起始时间folder_start_time
                folder_path = self.documents_dir / folder_name
                storage_path = self.storage_dir / f"index_{folder_name}"
                
                logger.info(f"\n📁 Processing folder: '{folder_name}'")
                
                # Check if saved index exists
                if storage_path.exists() and not force_rebuild:
                    load_start = time.time()  ### 记录加载现有索引的开始时间
                    logger.info(f"  ⏳ Loading existing index for folder '{folder_name}'...")
                    try:  # 尝试加载指定文件夹已有的向量索引（如果已经构建过）并把结果保存到系统内存的索引字典里
                        storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))  # 创建一个索引存储的上下文对象 用于后续索引的加载 这是llama_index读取本地向量索引时的标准写法
                        self.folder_indexes[folder_name] = load_index_from_storage(storage_context)  # 把磁盘上的向量索引数据加载到内存 并放进 self.folder_indexes 这个字典里（以文件夹名为key）
                        load_time = time.time() - load_start  ### 加载现有索引用了多少时间
                        folder_total_time = time.time() - folder_start_time  ### 整个这个文件夹的用时
                        
                        logger.info(f"  ✅ Index for folder '{folder_name}' loaded successfully")
                        logger.info(f"  ⏱️  Load time: {load_time:.2f}s, Total time: {folder_total_time:.2f}s")
                        
                        folder_stats.append({
                            'name': folder_name,
                            'action': 'loaded',
                            'total_time': folder_total_time,
                            'load_time': load_time,
                            'documents': 0,
                            'nodes': 0
                        })
                        successful_builds += 1
                        continue  # 跳过本轮循环 已经加载完成 不需要继续处理 就是后面的代码
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to load index for folder '{folder_name}', will rebuild: {e}")
                
                # Load documents from this folder
                try:
                    doc_load_start = time.time()  ### 记录文档加载的起始时间
                    reader = SimpleDirectoryReader(
                        input_dir=str(folder_path),
                        recursive=True,
                        required_exts=RAGConfig.SUPPORTED_EXTENSIONS,
                    )
                    
                    documents = reader.load_data()
                    doc_load_time = time.time() - doc_load_start  ### 记录文档加载花了多长时间
                    
                    if not documents:  # 如果没读到文档 列表为空 输出警告
                        logger.warning(f"  ⚠️  No valid documents found in folder '{folder_name}'")
                        folder_total_time = time.time() - folder_start_time  ###
                        folder_stats.append({
                            'name': folder_name,
                            'action': 'skipped',
                            'total_time': folder_total_time,
                            'documents': 0,
                            'nodes': 0
                        })
                        continue  # 跳过这次循环
                    
                    # Add folder identification and metadata to documents
                    for doc in documents:
                        self._add_document_metadata(doc, folder_name)  # 给这些子文件夹的文件添加metadata 定义好的函数
                    
                    logger.info(f"  📄 Found {len(documents)} documents (loaded in {doc_load_time:.2f}s)")
                    logger.info(f"  🔨 Building index for folder '{folder_name}'...")
                    
                    # Parse document nodes
                    parse_start = time.time()  ### 记录“拆分节点”开始时间
                    with tqdm(total=len(documents), desc=f"Parsing {folder_name}", unit="page", leave=False) as pbar:  # False 进度条用完后要不要留在终端（或日志）里。
                        nodes = []
                        for doc in documents:
                            doc_nodes = self.node_parser.get_nodes_from_documents([doc])
                            nodes.extend(doc_nodes)
                            pbar.update(1)  # 进度条前进1步
                    parse_time = time.time() - parse_start  ### 拆分这些节点所用时间
                    
                    logger.info(f"  📝 Split into {len(nodes)} nodes (parsing: {parse_time:.2f}s)")
                    
                    # Create vector index 创建embedding向量索引
                    index_start = time.time()
                    with tqdm(total=1, desc=f"Building {folder_name} index", unit="step", leave=False) as pbar:
                        folder_index = VectorStoreIndex(nodes, show_progress=True)
                        pbar.update(1)
                    index_time = time.time() - index_start  # 记录索引构建所用的时间
                    
                    # Save index
                    save_start = time.time()
                    with tqdm(total=1, desc=f"Saving {folder_name} index", unit="step", leave=False) as pbar:
                        storage_path.mkdir(exist_ok=True)
                        folder_index.storage_context.persist(persist_dir=str(storage_path))  # 把刚刚建立的索引结构 向量 元数据等 保存到磁盘上 方便下次直接加载 无需重建
                        pbar.update(1)
                    save_time = time.time() - save_start  # 保存索引耗时
                    
                    # Store in memory
                    self.folder_indexes[folder_name] = folder_index  # 在那个字典结构里保存一下
                    
                    folder_total_time = time.time() - folder_start_time  # 统计本次总用时（从这个子文件夹开始处理到现在）
                    
                    logger.info(f"  ✅ Index for folder '{folder_name}' built successfully")
                    logger.info(f"  ⏱️  Total: {folder_total_time:.2f}s | Doc Load: {doc_load_time:.2f}s | Parse: {parse_time:.2f}s | Build: {index_time:.2f}s | Save: {save_time:.2f}s")
                    
                    # Update statistics
                    total_documents += len(documents)
                    total_nodes += len(nodes)
                    successful_builds += 1
                    
                    folder_stats.append({
                        'name': folder_name,
                        'action': 'built',
                        'total_time': folder_total_time,
                        'doc_load_time': doc_load_time,
                        'parse_time': parse_time,
                        'index_time': index_time,
                        'save_time': save_time,
                        'documents': len(documents),
                        'nodes': len(nodes)
                    })
                    
                except Exception as e:
                    folder_total_time = time.time() - folder_start_time
                    logger.error(f"  ❌ Error building index for folder '{folder_name}' (time: {folder_total_time:.2f}s): {e}")
                    folder_stats.append({
                        'name': folder_name,
                        'action': 'failed',
                        'total_time': folder_total_time,
                        'error': str(e)
                    })
                    continue  # continue 跳过本轮 进入下一个文件夹
            
            overall_time = time.time() - overall_start_time
            
            # Print detailed timing report
            logger.info("\n" + "=" * 80)
            logger.info("📈 DETAILED TIMING REPORT")
            logger.info("=" * 80)
            
            # Sort by total time for better analysis
            folder_stats.sort(key=lambda x: x.get('total_time', 0), reverse=True)
            
            for stat in folder_stats:
                if stat['action'] == 'built':
                    logger.info(f"📁 {stat['name']:20} | 🔨 BUILT   | {stat['total_time']:6.2f}s | Docs: {stat['documents']:3d} | Nodes: {stat['nodes']:5d}")
                    logger.info(f"   {'':20} |          | Load: {stat['doc_load_time']:4.2f}s | Parse: {stat['parse_time']:4.2f}s | Build: {stat['index_time']:4.2f}s | Save: {stat['save_time']:4.2f}s")
                elif stat['action'] == 'loaded':  # 仅加载了已有索引 只需要显示加载用时
                    logger.info(f"📁 {stat['name']:20} | 📂 LOADED | {stat['total_time']:6.2f}s | Load: {stat['load_time']:4.2f}s")
                elif stat['action'] == 'failed':  # 构建失败 打印错误内容
                    logger.info(f"📁 {stat['name']:20} | ❌ FAILED  | {stat['total_time']:6.2f}s | Error: {stat.get('error', 'Unknown')}")
                elif stat['action'] == 'skipped':  # 文件夹没有文档 直接跳过 输出提示
                    logger.info(f"📁 {stat['name']:20} | ⏭️  SKIPPED | {stat['total_time']:6.2f}s | No documents found")
            
            # 总用时 文件夹数 成功构建/加载数 文档总数 节点总数 内存里最终的索引数量
            logger.info("\n" + "=" * 80)
            logger.info("📊 SUMMARY STATISTICS")
            logger.info("=" * 80)
            logger.info(f"🕒 Total processing time: {overall_time:.2f} seconds")
            logger.info(f"📁 Total folders processed: {len(folders)}")
            logger.info(f"✅ Successful builds/loads: {successful_builds}")
            logger.info(f"📄 Total documents processed: {total_documents}")
            logger.info(f"📝 Total nodes created: {total_nodes}")
            logger.info(f"🏗️  Final folder indexes: {len(self.folder_indexes)}")
            
            if total_documents > 0:
                avg_time_per_doc = overall_time / total_documents
                logger.info(f"⚡ Average time per document: {avg_time_per_doc:.3f} seconds")
            
            if total_nodes > 0:
                avg_time_per_node = overall_time / total_nodes
                logger.info(f"⚡ Average time per node: {avg_time_per_node:.4f} seconds")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error building folder indexes: {e}")
            raise
    
    def create_chat_engine(self, similarity_top_k: Optional[int] = None):
        """
        Create chat engine
        
        Args:
            similarity_top_k: Number of similar documents to retrieve
        """
        if self.index is None:
            raise ValueError("Please build index first")
        
        if similarity_top_k is None:
            similarity_top_k = RAGConfig.SIMILARITY_TOP_K
        
        try:
            # Create chat memory
            memory = ChatMemoryBuffer.from_defaults(token_limit=RAGConfig.TOKEN_LIMIT)
            
            # Create chat engine
            self.chat_engine = self.index.as_chat_engine(
                chat_mode="context",  # 用上下文模式 每次对话都结合历史和检索内容
                memory=memory,
                similarity_top_k=similarity_top_k,
                system_prompt=SystemPrompts.DEFAULT,  # 标准提示词 不是模版的那个
                verbose=True #  详细输出内部日志 便于调试
            )
            
            logger.info("Chat engine created successfully")
            
        except Exception as e:
            logger.error(f"Error creating chat engine: {e}")
            raise
    
    def create_folder_chat_engine(self, folder_name: str, similarity_top_k: Optional[int] = None):  # 为指定文件夹创建一个“专属聊天引擎”
        """
        Create chat engine for specified folder
        
        Args:
            folder_name: Folder name
            similarity_top_k: Number of similar documents to retrieve
        """
        if folder_name not in self.folder_indexes:
            raise ValueError(f"Index for folder '{folder_name}' does not exist")
        
        if similarity_top_k is None:
            similarity_top_k = RAGConfig.SIMILARITY_TOP_K
        
        try:
            # Create chat memory
            memory = ChatMemoryBuffer.from_defaults(token_limit=RAGConfig.TOKEN_LIMIT)
            
            # Create chat engine
            chat_engine = self.folder_indexes[folder_name].as_chat_engine(
                chat_mode="context",
                memory=memory,
                similarity_top_k=similarity_top_k,
                system_prompt=SystemPrompts.folder_specific(folder_name),
                verbose=True
            )
            
            self.folder_chat_engines[folder_name] = chat_engine
            logger.info(f"Chat engine for folder '{folder_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Error creating chat engine for folder '{folder_name}': {e}")
            raise
    
    def chat_with_folder(self, folder_name: str, message: str) -> str:  # 实际和某个文件夹的知识库对话
        """
        Chat with documents in specified folder
        
        Args:
            folder_name: Folder name
            message: User message
            
        Returns:
            AI response
        """
        if folder_name not in self.folder_chat_engines:
            # If chat engine doesn't exist, try to create it
            if folder_name in self.folder_indexes:
                self.create_folder_chat_engine(folder_name)
            else:
                raise ValueError(f"Index for folder '{folder_name}' does not exist")
        
        try:
            response = self.folder_chat_engines[folder_name].chat(message)
            return str(response)
        except Exception as e:
            logger.error(f"Error during folder '{folder_name}' chat: {e}")
            return f"Sorry, an error occurred while processing your question: {e}"
    
    def query_folder(self, folder_name: str, question: str, similarity_top_k: Optional[int] = None) -> str:  # 对指定文件夹的知识库做一次“单轮检索型问答”
        """
        Query documents in specified folder (single query without conversation history)
        
        Args:定义方法 输入是文件夹名 用户提问 检索top_k数量 返回字符串类型答案
            folder_name: Folder name
            question: Query question
            similarity_top_k: Number of similar documents to retrieve
            
        Returns:
            Query result
        """
        if folder_name not in self.folder_indexes:  # 如果这个文件夹还没有构建向量索引 直接报错
            raise ValueError(f"Index for folder '{folder_name}' does not exist")
        
        if similarity_top_k is None:
            similarity_top_k = RAGConfig.SIMILARITY_TOP_K
        
        try:
            query_engine = self.folder_indexes[folder_name].as_query_engine(  # 把本文件夹的向量索引对象转成一个纯检索的query_engine
                similarity_top_k=similarity_top_k,
                response_mode="tree_summarize"  # 用树状摘要法 自动整合多段检索结果 输出精炼答案
            )
            
            response = query_engine.query(question)
            return str(response)
            
        except Exception as e:
            logger.error(f"Error querying folder '{folder_name}': {e}")
            return f"Sorry, an error occurred while processing your query: {e}"
    
    def get_available_folders(self) -> List[str]:
        """
        Get list of available folders
        
        Returns:
            List of folder names
        """
        return list(self.folder_indexes.keys())
    
    def get_folder_documents_count(self, folder_name: str) -> int:
        """
        Get document count for specified folder
        
        Args:
            folder_name: Folder name
            
        Returns:
            Document count
        """
        try:
            folder_path = self.documents_dir / folder_name
            if not folder_path.exists() or not folder_path.is_dir():
                return 0  # 如果该路径不存在 或者不是文件夹 直接返回0（没有任何文件）
            
            count = 0
            for file_path in folder_path.rglob("*"):  # 用rglob("*")递归遍历该文件夹下所有层级的文件和子文件夹
                if file_path.is_file():
                    count += 1  # 如果遍历到的是“文件”，就计数器加1
            return count
        except Exception as e:
            logger.error(f"Error getting document count for folder '{folder_name}': {e}")
            return 0
    
    def chat(self, message: str) -> str:
        """
        Chat with the system
        
        Args:
            message: User message
            
        Returns:
            System response
        """
        if self.chat_engine is None:
            raise ValueError("Please create chat engine first")
        
        try:
            response = self.chat_engine.chat(message)
            return str(response)
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            return f"Sorry, an error occurred while processing your question: {e}"
    
    def add_document(self, file_path: str) -> bool:
        """
        Add new document to system
        
        Args:
            file_path: File path
            
        Returns:
            Whether successfully added
        """
        try:
            # Copy file to documents directory
            source_path = Path(file_path)  # 把输入的文件路径（字符串）转为 Path 对象。  
            dest_path = self.documents_dir / source_path.name  # 构造目标路径（目标目录是系统的documents目录，文件名保持不变）。
            
            if not dest_path.exists():  # 检查目标文件是否已存在，防止重复添加。
                shutil.copy2(source_path, dest_path)  # 如果不存在，复制源文件到系统文档目录。
                logger.info(f"Document {source_path.name} added to system")
                
                # Reload documents and rebuild index
                documents = self.load_documents()  # 重新扫描并加载所有文档，获取新的文档对象列表。
                if documents:  # 如果文档列表不为空，强制重建向量索引，保证新文件被纳入知识库。
                    self.build_index(documents, force_rebuild=True)
                    if self.chat_engine:
                        self.create_chat_engine()  # 如果聊天引擎已经存在，则重新创建，确保和新知识库同步
                    return True
            else:
                logger.warning(f"Document {source_path.name} already exists")
                return False
                
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def list_documents(self) -> List[str]:  # 完全支持有多层子文件夹的目录结构
        """
        List all documents in system (recursively scan subdirectories)
        
        Returns:
            List of document filenames (including relative paths)
        """
        try:
            files = []
            # Recursively scan files in all subdirectories
            for file_path in self.documents_dir.rglob("*"):  # 会递归遍历documents_dir下的所有内容，包括所有子文件夹里的文件，不管有几层。
                if file_path.is_file():
                    # Get path relative to documents directory
                    relative_path = file_path.relative_to(self.documents_dir)  # 会把每个文件的全路径转换成相对于根目录的路径，比如：
                    files.append(str(relative_path))
            return sorted(files)  # 最终所有文件（无论在哪一层）都会被包含进返回结果，而且是清楚的“带层次”的相对路径。
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
        