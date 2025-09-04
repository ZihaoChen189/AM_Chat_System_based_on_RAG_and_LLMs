import time
import torch
import random
import hashlib
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Dict

from llama_index.core import (
    VectorStoreIndex,         # create embedding vectors for chunks with index mappings
    SimpleDirectoryReader,    # load files (documents) from a folder
    Document,                 # Document Objective
    StorageContext,           # container for embedding vectors and index mappings
    load_index_from_storage,  # load existing embedding vectors with index mappings
    Settings                  # global configurations
)

from llama_index.core.node_parser import SentenceSplitter            # node_parser
from llama_index.llms.openai_like import OpenAILike                  # integrate LLMs
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # integrate the embedding model
from llama_index.core.memory import ChatMemoryBuffer                 # chat history buffer for storing conversation context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGConfig:
    RANDOM_SEED = 42

    # document processing configurations for RAG
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md", ".html"]

    # LLM configurations
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 1024

    # retrieval configurations
    SIMILARITY_TOP_K = 3
    TOKEN_LIMIT = 4000


class SystemPrompts:
    DEFAULT = (
        "You are an intelligent conversational assistant for additive manufacturing based on Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs)."
        "Answer user questions strictly based on the retrieved document content and cite the specific document source in your answers."
        "If there is no relevant information in the retrieved document content, explicitly state: I do not know."
        "Do not generate answers that are not supported by the retrieved document content."
        "If asking the user a clarifying question would be helpful, please do so."
    )

    @staticmethod
    def folder_specific(folder_name: str) -> str:
        return (  
            "You are an intelligent conversational assistant for additive manufacturing based on Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs)."  
            f"You specialize in answering user questions strictly based on the content of documents retrieved from the '{folder_name}' folder."
            "In your answers, always cite the specific document source."
            "If there is no relevant information in the retrieved document content, explicitly state: I do not know. In this case, do not cite any document source."
            "Do not generate any answers that are not supported by the retrieved document content."
            "If asking the user a clarifying question would be helpful, please do so."
        )


class MultiModelRAGSystem:
    def __init__(self, api_key: str, model_type: str = "deepseek-chat", documents_dir: str = "./documents", storage_dir: str = "./storage", embedding_model: str = "BAAI/bge-large-en-v1.5", random_seed: Optional[int] = None):
        # set random seeds for reproducibility
        self.random_seed = random_seed or RAGConfig.RANDOM_SEED
        self._set_random_seeds()  # defined function below
        
        self.api_key = api_key
        self.model_type = model_type.lower()
        self.documents_dir = Path(documents_dir)
        self.storage_dir = Path(storage_dir)
        
        # create necessary directories if needed
        self.documents_dir.mkdir(exist_ok=True)
        self.storage_dir.mkdir(exist_ok=True)
        
        # configure the specific LLM based on "model_type"
        self.llm = self._configure_llm()  # defined function below
        logger.info(f"Using the LLM API: {self.model_type}.")
        
        # GPU or MPS acceleration
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Detected Device in Current Environment: {device}.")

        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            cache_folder="./embedding_cache",
            device=device
        )  # the embedding model
        
        # configure global LLM settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = RAGConfig.CHUNK_SIZE
        Settings.chunk_overlap = RAGConfig.CHUNK_OVERLAP
        
        # configure node parser to split documents into chunks
        self.node_parser = SentenceSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
        )

        # for global engine
        self.index = None
        self.chat_engine = None
        
        # for expert engine
        self.folder_indexes: Dict[str, VectorStoreIndex] = {}
        self.folder_chat_engines: Dict[str, any] = {}
    
    def _set_random_seeds(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        # set deterministic behaviors in pytorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Manually Set Random Seed: {self.random_seed} in All Libraries.")
    
    def _get_base_llm_config(self) -> dict:
        # provide universal parameters: "api_key", "temperature" and "max_tokens"
        return {
            'api_key': self.api_key,
            'temperature': RAGConfig.DEFAULT_TEMPERATURE,
            'max_tokens': RAGConfig.DEFAULT_MAX_TOKENS,
            'is_chat_model': True,  # specify this is a chat model
        }
    
    def _configure_llm(self):  # build full configurations for the selected LLM
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
            raise ValueError(f"Unsupported LLM API: {self.model_type}. Supported Ones: {', '.join(model_configs.keys())}.")
        # base_config:   universal LLM parameters
        # model_configs: brand-specific LLM parameters
        config = {**base_config, **model_configs[self.model_type]}
        return OpenAILike(**config)  # keyword arguments unpacking
        
    def _add_document_metadata(self, doc: Document, folder_name: Optional[str] = None) -> None:  # SimpleDirectoryReader for file_path
        # hasattr(): check if the "doc" has a "metadata" attribute
        if hasattr(doc, 'metadata') and 'file_path' in doc.metadata:  # file_path default attribute
            file_path = Path(doc.metadata['file_path'])  # convert string path to a Path object for easy file operations
            doc.metadata.update({
                'filename': file_path.name,
                'file_type': file_path.suffix,
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            })
            if folder_name:
                doc.metadata['folder'] = folder_name  # for expert engine
                
    def load_documents(self, file_extensions: Optional[List[str]] = None) -> List[Document]:
        if file_extensions is None:
            file_extensions = RAGConfig.SUPPORTED_EXTENSIONS
        
        try:
            # create a reader to load files from a directory
            reader = SimpleDirectoryReader(  # file_path
                input_dir=str(self.documents_dir),  # specific directory path
                recursive=True,                     # load files in subfolders as well
                required_exts=file_extensions,      # supported file extensions
            )
            
            documents = reader.load_data(show_progress=True,num_workers=4)
            logger.info(f"Successfully Loaded {len(documents)} Document Pages.")  # the number of document pages
            
            # metadata attachment
            for doc in documents:
                self._add_document_metadata(doc)

            unique_docs = []
            seen_hashes = set()
            for doc in documents:  # for each document page
                content_hash = hashlib.md5(doc.text.encode('utf-8')).hexdigest()  # create a unique identifier based on the content of this document page
                if content_hash not in seen_hashes:  # if this document page is a unique one and not seen before
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)
            logger.info(f"There Were {len(documents)} Document Pages Before.")
            logger.info(f"After Hash Deduplication, {len(unique_docs)} Document Pages Remain.")
            return unique_docs
            
        except Exception as e:  # Exception Handling
            logger.error(f"Error on Loading Document Page: {e}.")
            return []
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> None:
        # define file paths for vectors and index
        index_store_file = self.storage_dir / "index_store.json"
        docstore_file = self.storage_dir / "docstore.json"
        
        try:
            # check if vectors with index exist
            if index_store_file.exists() and docstore_file.exists() and not force_rebuild:
                start_time = time.time()
                logger.info("Loading Existing Index for Global Chat Engine...")

                with tqdm(total=1, desc="Loading Existing Index for Global Chat Engine.", unit="Step", leave=False, disable=True) as pbar:
                    # create a container for embedding vectors and index mappings, specifying where vector and index data are stored
                    storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
                    # load existing embedding vectors with index mappings
                    self.index = load_index_from_storage(storage_context)
                    pbar.update(1)
                load_time = time.time() - start_time
                logger.info(f"Loaded Existing Index for Global Chat Engine in {load_time:.2f} Seconds.")
            else:
                # if no document and no existing vector with index, skip building
                if not documents:
                    logger.info("No Document Found and No Existing Index Found.")
                    self.index = None
                    return
                
                start_time = time.time()  ##########
                logger.info("Building New Index...")


                logger.info("Firstly, Parse Document Page.")
                parse_start = time.time()  ##########
                with tqdm(total=len(documents), desc="Parsing Document Pages.", unit="Document", leave=False, disable=True) as pbar:
                    nodes = []
                    for doc in documents:
                        doc_nodes = self.node_parser.get_nodes_from_documents([doc])  # split document pages into chunks
                        nodes.extend(doc_nodes)  # all chunks
                        pbar.update(1)
                parse_time = time.time() - parse_start  ##########
                logger.info(f"Document Pages Split Into {len(nodes)} Nodes in {parse_time:.2f} Seconds.")


                logger.info("Secondly, Create Index.")
                index_start = time.time()  ##########
                with tqdm(total=1, desc="Creating Vector Index.", unit="Step", leave=False, disable=True) as pbar:
                    # convert chunks into embedding vectors with index mappings and assign to self.index
                    self.index = VectorStoreIndex(nodes, show_progress=True)  # not real engine, just vectors
                    pbar.update(1)
                index_time = time.time() - index_start  ##########


                logger.info("Thirdly, Save Index.")
                save_start = time.time()  ##########
                with tqdm(total=1, desc="Saving Index.", unit="Step", leave=False) as pbar:
                    # save self.index to self.storage_dir
                    self.index.storage_context.persist(persist_dir=str(self.storage_dir))
                    pbar.update(1)
                save_time = time.time() - save_start  ##########
                
                total_time = time.time() - start_time  ##########
                logger.info(f"Build Global Index Successfully.")
                logger.info(f"GLobal Index: Document Page Parsed, Vector Index Built, and Saved Successfully in {total_time:.2f} Seconds Totally.")
                logger.info(f"Parsing in {parse_time:.2f} Seconds,   Building in {index_time:.2f} Seconds,   Saving in {save_time:.2f} Seconds.")
                
        except Exception as e:  # Exception Handling
            logger.error(f"Error on Building Index: {e}.")
            raise
    
    def build_folder_indexes(self, force_rebuild: bool = False) -> None:
        try:
            overall_start_time = time.time()  ##########
            # get all subfolders
            folders = []
            for item in self.documents_dir.iterdir():  # iterate all files and folders in "self.documents_dir"
                if item.is_dir():
                    folders.append(item.name)  # catch the subfolder
            
            logger.info(f"Found {len(folders)} Folders: {folders}.")
            logger.info("📊 START RECORD TIME ABOUT FOLDER INDEX.")
            
            # statistics
            folder_stats = []
            total_documents = 0
            total_nodes = 0
            successful_builds = 0
            
            # for each subfolder
            for folder_name in tqdm(folders, desc="Processing Folders.", unit="Folder", leave=False, disable=True):
                folder_start_time = time.time()  ##########


                #################### similar to the global engine  ####################
                folder_path = self.documents_dir / folder_name  # the specific path
                storage_path = self.storage_dir / f"index_{folder_name}"  # the specific path
                logger.info(f"📁 Processing Folder: '{folder_name}'.")

                if storage_path.exists() and not force_rebuild:
                    load_start = time.time()  ##########
                    logger.info(f"Loading Existing Folder Index for Folder '{folder_name}'...")
                    try:
                        # create a container for embedding vectors and index mappings, specifying where vector and index data are stored
                        storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
                        # load existing embedding vectors with index mappings
                        self.folder_indexes[folder_name] = load_index_from_storage(storage_context)
                        load_time = time.time() - load_start  ##########
                        folder_total_time = time.time() - folder_start_time  ##########
                        
                        logger.info(f"Loaded Existing Folder Index for Folder '{folder_name}' Successfully.")
                        logger.info(f"Load Time: {load_time:.2f} Seconds,   Total Time: {folder_total_time:.2f} Seconds.")
                        
                        folder_stats.append({
                            'name': folder_name,
                            'action': 'loaded',
                            'total_time': folder_total_time,
                            'load_time': load_time,
                            'documents': 0,
                            'nodes': 0
                        })  # update

                        successful_builds += 1
                        continue  # next folder

                    except Exception as e:  # Exception Handling
                        logger.error(f"Failed to Load Existing Folder Index '{folder_name}', {e}.")


                try:
                    logger.info(f"🚀 Building Folder Index for Folder '{folder_name}'...")
                    doc_load_start = time.time()  ##########
                    # create a reader to load files from a directory
                    reader = SimpleDirectoryReader(
                        input_dir=str(folder_path),  # specific directory path
                        recursive=True,  # load files in subfolders as well
                        required_exts=RAGConfig.SUPPORTED_EXTENSIONS,  # supported file extensions
                    )
                    documents = reader.load_data()   ##########
                    doc_load_time = time.time() - doc_load_start   ##########
                    
                    if not documents:
                        logger.error(f"⚠️ No Document Found in Folder '{folder_name}'.")
                        folder_total_time = time.time() - folder_start_time  ##########
                        folder_stats.append({
                            'name': folder_name,
                            'action': 'skipped',
                            'total_time': folder_total_time,
                            'documents': 0,
                            'nodes': 0
                        })  # update
                        continue
                    
                    # metadata attachment
                    for doc in documents:
                        self._add_document_metadata(doc, folder_name)

                    logger.info(f"Read {len(documents)} Document Pages in {doc_load_time:.2f} Seconds under Current Folder '{folder_name}'.")


                    logger.info("Firstly, Parse Document Pages under Current Folder.")
                    parse_start = time.time()  ##########
                    with tqdm(total=len(documents), desc=f"Parsing Document Page under Folder: {folder_name}.", unit="Page", leave=False, disable=True) as pbar:
                        nodes = []
                        for doc in documents:
                            doc_nodes = self.node_parser.get_nodes_from_documents([doc])  # split document pages into chunks
                            nodes.extend(doc_nodes)  # all chunks
                            pbar.update(1)
                    parse_time = time.time() - parse_start  ##########
                    logger.info(f"Document Pages Split Into {len(nodes)} Nodes in {parse_time:.2f} Seconds under Current Folder '{folder_name}'.")
                    

                    logger.info("Secondly, Create Folder Index under Current Folder.")
                    index_start = time.time()
                    with tqdm(total=1, desc=f"Creating Folder Index for Folder {folder_name}.", unit="Step", leave=False, disable=True) as pbar:
                        # convert chunks into embedding vectors with index mappings and assign to folder_index
                        folder_index = VectorStoreIndex(nodes, show_progress=True)  # not real engine, just vectors
                        pbar.update(1)
                    index_time = time.time() - index_start  ##########
                    

                    logger.info("Thirdly, Save Folder Index under Current Folder.")
                    save_start = time.time()  ##########
                    with tqdm(total=1, desc=f"Saving Folder Index for Folder {folder_name}.", unit="Step", leave=False, disable=True) as pbar:
                        storage_path.mkdir(exist_ok=True)
                        folder_index.storage_context.persist(persist_dir=str(storage_path))  # save "folder_index" to "storage_path"
                        pbar.update(1)
                    save_time = time.time() - save_start  ##########

                    self.folder_indexes[folder_name] = folder_index  # store "folder_index" in the dictionary
                    
                    folder_total_time = time.time() - folder_start_time  ##########

                    logger.info(f"Already Build Folder Index for Folder '{folder_name}' Successfully.")
                    logger.info(f"Total Time: {folder_total_time:.2f} Seconds,   Read Time: {doc_load_time:.2f} Seconds,   Parsed Time: {parse_time:.2f} Seconds,   Built Time: {index_time:.2f} Seconds,   Saved Time: {save_time:.2f} Seconds.")

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
                    })  # update
                    
                except Exception as e:  # Exception Handling
                    folder_total_time = time.time() - folder_start_time
                    logger.error(f"Error on Building Folder Index for Folder '{folder_name}': {e} (Time: {folder_total_time:.2f} Seconds).")
                    folder_stats.append({
                        'name': folder_name,
                        'action': 'failed',
                        'total_time': folder_total_time,
                        'error': str(e)
                    })  # update
                    continue
            
            overall_time = time.time() - overall_start_time  ##########

            logger.info("DETAILED TIMING REPORT FOR FOLDER INDEX.")
            # folders with the longest processing time will appear firstly
            folder_stats.sort(key=lambda x: x.get("total_time", 0), reverse=True)

            # {stat['documents']:3d}:       display at least 3 characters wide
            # {stat['doc_load_time']:4.2f}: display at least 4 characters wide, with 2 decimal places

            for stat in folder_stats:
                if stat['action'] == 'built':
                    logger.info(f"📁 Folder {stat['name']:20}, TOTAL BUILT TIME: {stat['total_time']:6.2f}s, PAGE COUNT: {stat['documents']:3d}, NODE COUNT: {stat['nodes']:5d}, READ TIME: {stat['doc_load_time']:4.2f}s, PARSE TIME: {stat['parse_time']:4.2f}s, CREATE INDEX TIME: {stat['index_time']:4.2f}s, SAVE TIME: {stat['save_time']:4.2f}s.")

                elif stat['action'] == 'failed':
                    logger.info(f"❌ FAILED on Folder {stat['name']:20}, Running Time: {stat['total_time']:6.2f} Seconds, With Error: {stat.get('error', 'Unknown Error')}.")
                    
                elif stat['action'] == 'skipped':
                    logger.info(f"⏭️ SKIPPED   Folder {stat['name']:20}, Running Time: {stat['total_time']:6.2f} Seconds, No Document Found.")

            logger.info("📊 STATISTICS SUMMARY FOR FOLDER INDEX.")
            logger.info(f"✅ Total Time Taken:                {overall_time:.2f} Seconds.")
            logger.info(f"✅ Total Folder Processed:          {len(folders)}.")
            logger.info(f"✅ Total Folder Index Built/Loaded: {successful_builds}.")
            logger.info(f"✅ Total Folder Index:              {len(self.folder_indexes)}.")
            logger.info(f"✅ Total Document Page Processed:   {total_documents}.")
            logger.info(f"✅ Total Node Created:              {total_nodes}.")

        except Exception as e:  # Exception Handling
            logger.error(f"Error on Building Folder Index: {e}.")
            raise
    
    def create_chat_engine(self, similarity_top_k: Optional[int] = None):
        if self.index is None:
            raise ValueError("Please Build Global Chat Engine Index First.")
        if similarity_top_k is None:
            similarity_top_k = RAGConfig.SIMILARITY_TOP_K
        
        try:
            # chat history buffer for storing conversation context
            memory = ChatMemoryBuffer.from_defaults(token_limit=RAGConfig.TOKEN_LIMIT)
            
            # create the global engine
            self.chat_engine = self.index.as_chat_engine(
                chat_mode="context",  # context conversation mode
                memory=memory,
                similarity_top_k=similarity_top_k,
                system_prompt=SystemPrompts.DEFAULT,  # prompt engineering
                verbose=True  # show detailed logs
            )
            logger.info(f"Global Chat Engine Created Successfully with Maximal {RAGConfig.TOKEN_LIMIT} MEMORY Tokens.")
            
        except Exception as e:  # Exception Handling
            logger.error(f"Error on Creating Global Chat Engine: {e}.")
            raise
    
    def create_folder_chat_engine(self, folder_name: str, similarity_top_k: Optional[int] = None):
        if folder_name not in self.folder_indexes:
            raise ValueError(f"Folder Index for Folder '{folder_name}' Not Found.")
        if similarity_top_k is None:
            similarity_top_k = RAGConfig.SIMILARITY_TOP_K

        try:
            # chat history buffer for storing conversation context
            memory = ChatMemoryBuffer.from_defaults(token_limit=RAGConfig.TOKEN_LIMIT)
            
            # create the expert engine based on the specific folder database
            chat_engine = self.folder_indexes[folder_name].as_chat_engine(
                chat_mode="context",
                memory=memory,
                similarity_top_k=similarity_top_k,
                system_prompt=SystemPrompts.folder_specific(folder_name),  # prompt engineering
                verbose=True
            )
            self.folder_chat_engines[folder_name] = chat_engine
            logger.info(f"🚀 Expert Chat Engine for Folder '{folder_name}' Created Successfully.")
            
        except Exception as e:  # Exception Handling
            logger.error(f"Error on Creating Expert Chat Engine for Folder '{folder_name}'; {e}.")
            raise

    def chat(self, message: str) -> str:
        if self.chat_engine is None:
            raise ValueError("Please Create Global Chat Engine First.")
        
        try:
            response = self.chat_engine.chat(message)  # get the response from the global engine
            return str(response)

        except Exception as e:  # Exception Handling
            logger.error(f"Error During Chat: {e}.")
            return f"Sorry, Error Occurred When Processing the Question; {e}."

    def chat_with_folder(self, folder_name: str, message: str) -> str:
        if folder_name not in self.folder_chat_engines:
            if folder_name in self.folder_indexes:
                self.create_folder_chat_engine(folder_name)  # important code - emergency fix
            else:
                raise ValueError(f"Folder Index for Folder '{folder_name}' Not Found.")
        
        try:
            response = self.folder_chat_engines[folder_name].chat(message)  # get the response from the expert engine
            return str(response)

        except Exception as e:  # Exception Handling
            logger.error(f"Error During Chat under Folder '{folder_name}'; {e}.")
            return f"Sorry, Error Occurred When Processing the Question; {e}."

    def get_available_folders(self) -> List[str]:
        return list(self.folder_indexes.keys())

    def get_folder_documents_count(self, folder_name: str) -> int:
        try:
            folder_path = self.documents_dir / folder_name
            if not folder_path.exists() or not folder_path.is_dir():
                return 0
            
            count = 0
            for file_path in folder_path.rglob("*"):  # iterate all files and folders in "folder_path"
                if file_path.is_file():
                    count += 1  # if it is a file, +=1
            return count

        except Exception as e:  # Exception Handling
            logger.error(f"Error on Getting Document File Count in Folder '{folder_name}'; {e}.")
            return 0
