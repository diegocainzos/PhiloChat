"""
 Leer PDFs (PyPDFLoader).

    Limpiar texto y Chunking (RecursiveCharacterTextSplitter).

    Embeddings (GoogleGenerativeAIEmbeddings).

    Carga a Postgres (PGVector).
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from rich import print
import re

# FUNCTIONS
def book_processing(docs, skip_n=5, min_length=100):
    pattern = re.compile(r'[\xad\n\d+]')
    
    def clean(doc):
        doc.page_content = ' '.join(pattern.sub(' ', doc.page_content).split())
        return doc

    return [clean(d) for d in docs[skip_n:] if len(d.page_content) > min_length]
    
# CONSTANTS
FOLDER_PATH = Path('etl/sources/')

# CODE
file_routes = [str(file) for file in FOLDER_PATH.iterdir() if file.is_file()][:1]

loaders = [PyPDFLoader(doc).load() for doc in file_routes]
processed_doc_list = [book_processing(doc) for doc in loaders]

# CHUNKING

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,)

all_splits = [text_splitter.split_documents(doc) for doc in processed_doc_list]

# sisifo = file_routes[0]
# loader = PyPDFLoader(sisifo)
# docs_sisifo  = loader.load()



# PRINTS
print(f"Split blog post into {len(processed_doc_list[0])} pages.")
print(f"Split blog post into {len(all_splits[0])} sub-documents.")
print(all_splits[0])
#print(all_splits)
# print(loaders)
# print('el tipo es:' ,type(docs_sisifo))
# print(docs_sisifo[0].metadata, f'numero de chunks {len(docs_sisifo)}') 
# print(book_processing(docs_sisifo))


