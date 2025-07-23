import time
import torch
import random
import logging
import hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict

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
    def __init__(self, api_key: str, model_type: str = "deepseek-chat", documents_dir: str = "./documents", storage_dir: str = "./storage", embedding_model: str = "BAAI/bge-small-zh-v1.5", random_seed: Optional[int] = None):
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
        
        logger.info(f"Using the LLM API: {self.model_type}.")
        
        # Configure embedding model (using local model with CUDA acceleration)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Detected Device in Current Environment: {device}.")

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
        logger.info(f"Manually Set Random Seed: {self.random_seed} in All Libraries.")
    
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
            "deepseek-chat": {
                "model": "deepseek-chat",  # DeepSeek-V3-0324 Jul-22
                "api_base": "https://api.deepseek.com/v1",  
            },
            "deepseek-reasoner": {
                "model": "deepseek-reasoner",  # DeepSeek-R1-0528 JUl-22
                "api_base": "https://api.deepseek.com/v1",
            },
            "gpt4.1": {
                "model": "gpt-4.1-2025-04-14",
                "api_base": "https://api.openai.com/v1",
            },
            "gpt4o": {
                "model": "gpt-4o-2024-08-06",
                "api_base": "https://api.openai.com/v1",
            }
        }
        
        if self.model_type not in model_configs:
            raise ValueError(f"Unsupported LLM API: {self.model_type}. Supported Ones: {', '.join(model_configs.keys())}")
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
            logger.info(f"Successfully Loaded {len(documents)} Document Pages.")  # 多少page
            
            # Add metadata to each document
            for doc in documents:
                self._add_document_metadata(doc)  # 给每一个doc添加元数据
            ########################################################################################
            unique_docs = []
            seen_hashes = set()
            for doc in documents:
                content_hash = hashlib.md5(doc.text.encode('utf-8')).hexdigest()  # 对每个页面内容（doc.text）做md5哈希运算，得到一个唯一的“指纹”。
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)
            logger.info(f"There Were {len(documents)} Document Pages Before.")
            logger.info(f"After Hash Deduplication, {len(unique_docs)} Document Pages Remain.")
            return unique_docs
            #########################################################################################
            # return documents
            
        except Exception as e:
            logger.error(f"Error on Loading Document Page: {e}.")
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
                logger.info("Loading Existing Index for Global Chat Engine...")

                # StorageContext,  # 索引数据要存在哪里 怎么存 用什么方式存
                # load_index_from_storage,  # 用于从本地加载已有的索引 要传入一个StorageContext对象

                with tqdm(total=1, desc="Loading Existing Index for Global Chat Engine.", unit="Step") as pbar:
                    storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))  # 创建一个索引存储的上下文对象 用于后续索引的加载 这是llama_index读取本地向量索引时的标准写法
                    self.index = load_index_from_storage(storage_context)  # 用llama_index的接口 把本地存储里的向量索引数据加载到内存 并赋值给self.index 后续就可以检索了
                    pbar.update(1)  # 手动把进度条推进1步 表示“加载索引”这一步已经完成
                load_time = time.time() - start_time
                logger.info(f"Loaded Existing Index for Global Chat Engine in {load_time:.2f} Seconds.")
            else:
                # If no documents provided and no existing index, skip building
                if not documents:  # 如果documents这个列表本身是空的（没有可用文档）那连索引都没法建
                    logger.info("No Document Found and No Existing Index Found.")
                    self.index = None
                    return
                
                start_time = time.time()  ####################
                logger.info("Building New Index...")

                logger.info("Firstly, Parse Document Page.")
                # Parse document nodes 切块操作
                parse_start = time.time()  ####################
                with tqdm(total=len(documents), desc="Parsing Document Pages.", unit="Document") as pbar:  # 总数=文档数 单位是doc
                    nodes = []
                    for doc in documents:  # 遍历所有文档  所有切的块
                        doc_nodes = self.node_parser.get_nodes_from_documents([doc])  # 每份文档用node_parser（通常是分句/分块器）切分成若干“节点”（chunk）
                        nodes.extend(doc_nodes)  # 把这些节点合并到nodes总列表
                        pbar.update(1)  # bar+=1
                parse_time = time.time() - parse_start  #################### 切块时间
                
                logger.info(f"Document Pages Split Into {len(nodes)} Nodes in {parse_time:.2f} Seconds.")  # 这次所有文档总共切成多少个节点 每个节点后续都会做embedding

                # Create vector index 做embedding 生成向量并建立索引
                logger.info("Secondly, Create Index.")
                index_start = time.time()  ####################
                with tqdm(total=1, desc="Creating Vector Index.", unit="Step") as pbar:
                    self.index = VectorStoreIndex(nodes, show_progress=True)  # 调用VectorStoreIndex(nodes, ...) 对所有节点做embedding并建立向量检索索引 索引对象赋值给self.index 供后续检索用
                    pbar.update(1)  # 手动+1
                index_time = time.time() - index_start  ####################生成向量并建立索引用了多长时间
                
                # Save index 保存到本地磁盘
                logger.info("Thirdly, Save Index.")
                save_start = time.time()  ####################
                with tqdm(total=1, desc="Saving Index.", unit="Step") as pbar:
                    self.index.storage_context.persist(persist_dir=str(self.storage_dir))  # 把当前索引完整保存到本地指定目录
                    pbar.update(1)
                save_time = time.time() - save_start  ####################保存到本地磁盘用了多长时间
                
                total_time = time.time() - start_time  #################### 一共花了多长时间
                logger.info(f"Build Index Successfully.")
                logger.info(f"Document Page Parsed, Vector Index Built, and Saved Successfully in {total_time:.2f} Seconds Totally.")
                logger.info(f"Parsing in {parse_time:.2f} Seconds,   Building in {index_time:.2f} Seconds,   Saving in {save_time:.2f} Seconds.")
                
        except Exception as e:
            logger.error(f"Error on Building Index: {e}.")
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
            folders = []
            for item in self.documents_dir.iterdir():  # 获取文档主目录下的所有子文件夹名称 iterdir(): 列举当前目录下所有的文件和文件夹
                if item.is_dir():
                    folders.append(item.name)
            
            logger.info(f"Found {len(folders)} Folders: {folders}.")
            logger.info("📊 START RECORD TIME ABOUT FOLDER INDEX.")
            
            # Statistics tracking
            folder_stats = []
            total_documents = 0
            total_nodes = 0
            successful_builds = 0
            
            # Iterate through folders with progress bar
            for folder_name in tqdm(folders, desc="Processing Folders.", unit="Folder"):
                folder_start_time = time.time()  # 记录每个文件夹的起始时间folder_start_time
                folder_path = self.documents_dir / folder_name
                storage_path = self.storage_dir / f"index_{folder_name}"
                
                logger.info(f"📁 Processing Folder: '{folder_name}'.")
                
                # Check if saved index exists
                if storage_path.exists() and not force_rebuild:
                    load_start = time.time()  ### 记录加载现有索引的开始时间
                    logger.info(f"Loading Existing Folder Index for Folder '{folder_name}'...")
                    try:  # 尝试加载指定文件夹已有的向量索引（如果已经构建过）并把结果保存到系统内存的索引字典里
                        storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))  # 创建一个索引存储的上下文对象 用于后续索引的加载 这是llama_index读取本地向量索引时的标准写法
                        self.folder_indexes[folder_name] = load_index_from_storage(storage_context)  # 把磁盘上的向量索引数据加载到内存 并放进 self.folder_indexes 这个字典里（以文件夹名为key）
                        load_time = time.time() - load_start  ### 加载现有索引用了多少时间
                        folder_total_time = time.time() - folder_start_time  ### 整个这个文件夹的用时
                        
                        logger.info(f"Loaded Existing Folder Index for Folder '{folder_name}' Successfully.")
                        logger.info(f"Load Time: {load_time:.2f} Seconds,   Total Time: {folder_total_time:.2f} Seconds.")
                        
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
                        logger.error(f"Failed to Load Existing Folder Index '{folder_name}', {e}.")
                
                # Load documents from this folder
                try:
                    logger.info(f"🚀 Building Folder Index for Folder '{folder_name}'...")
                    doc_load_start = time.time()  ### 记录文档加载的起始时间
                    reader = SimpleDirectoryReader(
                        input_dir=str(folder_path),
                        recursive=True,
                        required_exts=RAGConfig.SUPPORTED_EXTENSIONS,
                    )
                    
                    documents = reader.load_data()
                    doc_load_time = time.time() - doc_load_start  ### 记录文档加载花了多长时间
                    
                    if not documents:  # 如果没读到文档 列表为空 输出警告
                        logger.error(f"⚠️ No Document Found in Folder '{folder_name}'.")
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
                    
                    logger.info(f"Read {len(documents)} Document Pages in {doc_load_time:.2f} Seconds under Current Folder '{folder_name}'.")

                    # Parse document nodes
                    logger.info("Firstly, Parse Document Pages under Current Folder.")
                    parse_start = time.time()  ### 记录“拆分节点”开始时间
                    with tqdm(total=len(documents), desc=f"Parsing Document Page under Folder: {folder_name}.", unit="Page", leave=False) as pbar:  # False 进度条用完后要不要留在终端（或日志）里。
                        nodes = []
                        for doc in documents:
                            doc_nodes = self.node_parser.get_nodes_from_documents([doc])
                            nodes.extend(doc_nodes)
                            pbar.update(1)  # 进度条前进1步
                    parse_time = time.time() - parse_start  ### 拆分这些节点所用时间
                    
                    logger.info(f"Document Pages Split Into {len(nodes)} Nodes in {parse_time:.2f} Seconds under Current Folder '{folder_name}'.")
                    
                    # Create vector index 创建embedding向量索引
                    logger.info("Secondly, Create Folder Index under Current Folder.")
                    index_start = time.time()
                    with tqdm(total=1, desc=f"Creating Folder Index for Folder {folder_name}.", unit="Step", leave=False) as pbar:
                        folder_index = VectorStoreIndex(nodes, show_progress=True)
                        pbar.update(1)
                    index_time = time.time() - index_start  # 记录索引构建所用的时间
                    
                    # Save index
                    logger.info("Thirdly, Save Folder Index under Current Folder.")
                    save_start = time.time()
                    with tqdm(total=1, desc=f"Saving Folder Index for Folder {folder_name}.", unit="Step", leave=False) as pbar:
                        storage_path.mkdir(exist_ok=True)
                        folder_index.storage_context.persist(persist_dir=str(storage_path))  # 把刚刚建立的索引结构 向量 元数据等 保存到磁盘上 方便下次直接加载 无需重建
                        pbar.update(1)
                    save_time = time.time() - save_start  # 保存索引耗时
                    
                    # Store in memory
                    self.folder_indexes[folder_name] = folder_index  # 在那个字典结构里保存一下
                    
                    folder_total_time = time.time() - folder_start_time  # 统计本次总用时（从这个子文件夹开始处理到现在）

                    logger.info(f"Already Build Folder Index for Folder '{folder_name}' Successfully.")
                    logger.info(f"Total Time: {folder_total_time:.2f} Seconds,   Read Time: {doc_load_time:.2f} Seconds,   Parsed Time: {parse_time:.2f} Seconds,   Built Time: {index_time:.2f} Seconds,   Saved Time: {save_time:.2f} Seconds.")
                    
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
                    logger.error(f"Error on Building Folder Index for Folder '{folder_name}': {e} (Time: {folder_total_time:.2f} Seconds).")
                    folder_stats.append({
                        'name': folder_name,
                        'action': 'failed',
                        'total_time': folder_total_time,
                        'error': str(e)
                    })
                    continue  # continue 跳过本轮 进入下一个文件夹
            
            overall_time = time.time() - overall_start_time
            
            # Print detailed timing report
            logger.info("DETAILED TIMING REPORT FOR FOLDER INDEX.")
            # Sort by total time for better analysis
            folder_stats.sort(key=lambda x: x.get("total_time", 0), reverse=True)
            for stat in folder_stats:
                if stat['action'] == 'built':
                    logger.info(f"📁 Folder {stat['name']:20}, TOTAL BUILT TIME: {stat['total_time']:6.2f}s, PAGE COUNT: {stat['documents']:3d}, NODE COUNT: {stat['nodes']:5d}, READ TIME: {stat['doc_load_time']:4.2f}s, PARSE TIME: {stat['parse_time']:4.2f}s, CREATE INDEX TIME: {stat['index_time']:4.2f}s, SAVE TIME: {stat['save_time']:4.2f}s.")

                elif stat['action'] == 'failed':  # 构建失败 打印错误内容
                    logger.info(f"❌ FAILED on Folder {stat['name']:20}, Running Time: {stat['total_time']:6.2f} Seconds, With Error: {stat.get('error', 'Unknown Error')}.")
                elif stat['action'] == 'skipped':  # 文件夹没有文档 直接跳过 输出提示
                    logger.info(f"⏭️ SKIPPED   Folder {stat['name']:20}, Running Time: {stat['total_time']:6.2f} Seconds, No Document Found.")
            
            # 总用时 文件夹数 成功构建/加载数 文档总数 节点总数 内存里最终的索引数量
            logger.info("📊 STATISTICS SUMMARY FOR FOLDER INDEX.")
            logger.info(f"✅ Total Time Taken:                {overall_time:.2f} Seconds.")
            logger.info(f"✅ Total Folder Processed:          {len(folders)}.")
            logger.info(f"✅ Total Folder Index Built/Loaded: {successful_builds}.")
            logger.info(f"✅ Total Folder Index:              {len(self.folder_indexes)}.")
            logger.info(f"✅ Total Document Page Processed:   {total_documents}.")
            logger.info(f"✅ Total Node Created:              {total_nodes}.")
            
            
        except Exception as e:
            logger.error(f"Error on Building Folder Index: {e}.")
            raise
    
    def create_chat_engine(self, similarity_top_k: Optional[int] = None):
        """
        Create chat engine
        
        Args:
            similarity_top_k: Number of similar documents to retrieve
        """
        if self.index is None:
            raise ValueError("Please Build Global Chat Engine Index First.")
        
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
            logger.info(f"Global Chat Engine Created Successfully with Maximal {RAGConfig.TOKEN_LIMIT} MEMORY Tokens.")
            
        except Exception as e:
            logger.error(f"Error on Creating Global Chat Engine: {e}.")
            raise
    
    def create_folder_chat_engine(self, folder_name: str, similarity_top_k: Optional[int] = None):  # 为指定文件夹创建一个“专属聊天引擎”
        """
        Create chat engine for specified folder
        
        Args:
            folder_name: Folder name
            similarity_top_k: Number of similar documents to retrieve
        """
        if folder_name not in self.folder_indexes:
            raise ValueError(f"Folder Index for Folder '{folder_name}' Not Found.")
        
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
            logger.info(f"🚀 Expert Chat Engine for Folder '{folder_name}' Created Successfully.")
            
        except Exception as e:
            logger.error(f"Error on Creating Expert Chat Engine for Folder '{folder_name}'; {e}.")
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
                raise ValueError(f"Folder Index for Folder '{folder_name}' Not Found.")
        
        try:
            response = self.folder_chat_engines[folder_name].chat(message)
            return str(response)
        except Exception as e:
            logger.error(f"Error During Chat under Folder '{folder_name}'; {e}.")
            return f"Sorry, Error Occurred When Processing the Question; {e}."
    
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
            logger.error(f"Error on Getting Document File Count in Folder '{folder_name}'; {e}.")
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
            raise ValueError("Please Create Global Chat Engine First.")
        
        try:
            response = self.chat_engine.chat(message)
            return str(response)
        except Exception as e:
            logger.error(f"Error During Chat: {e}.")
            return f"Sorry, Error Occurred When Processing the Question; {e}."
    
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
            logger.error(f"Error on Listing Document: {e}.")
            return []
        