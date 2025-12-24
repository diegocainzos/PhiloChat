"""
 Leer PDFs (PyPDFLoader).

    Limpiar texto y Chunking (RecursiveCharacterTextSplitter).

    Embeddings (GoogleGenerativeAIEmbeddings).

    Carga a Postgres (PGVector).

    ETL FULL TEXT
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Importante para crear el doc unificado
from pathlib import Path
import re

# ... (Tus funciones e imports previos se mantienen igual) ...

# CODE
# Asumo que loaders es una lista de listas (una lista de páginas por cada libro)
# Para este ejemplo procesamos el primer libro:
# FUNCTIONS
def book_processing(docs, skip_n=5, min_length=100):
    page_num = docs[0].metadata.get('page','Unknown')
    def clean(doc):
        # Nota la 'r' antes de las comillas del patrón
        doc.page_content = ' '.join(re.sub(r'[-\d]', '', doc.page_content).split())
        return doc

    return [clean(d) for d in docs[skip_n:] if len(d.page_content) > min_length]
    
# CONSTANTS
FOLDER_PATH = Path('etl/sources/')

# CODE
file_routes = [str(file) for file in FOLDER_PATH.iterdir() if file.is_file()][:1]

docs = [PyPDFLoader(doc).load() for doc in file_routes]
processed_doc_list = [book_processing(doc) for doc in docs]

book_pages = processed_doc_list[0] 

# --- PASO CRÍTICO: UNIFICACIÓN ---
# 1. Unimos el texto de todas las páginas en un solo string gigante.
# Usamos " " como separador para no pegar la última palabra de una pág con la primera de la otra.
full_text = " ".join([page.page_content for page in book_pages])

# 2. (Opcional) Recuperamos los metadatos generales del libro (del primer doc)
# Nota: Perderás el número de página específico ('page': 9), pero ganarás continuidad.
book_metadata = book_pages[0].metadata
# Limpiamos metadatos específicos de página si existen para no confundir
if 'page' in book_metadata:
    del book_metadata['page']
if 'page_label' in book_metadata:
    del book_metadata['page_label']

# 3. Creamos un único Documento gigante
full_doc = Document(page_content=full_text, metadata=book_metadata)

# --- CHUNKING ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""] # Definir prioridades ayuda
)

# Ahora split_documents recibe una lista con UN solo documento gigante y lo trocea
splits = text_splitter.split_documents([full_doc])

# VERIFICACIÓN
def show_chunk(splits):
    import random
    i = random.randint(0,len(splits))
    print(f"Total chunks generados: {len(splits)}")
    print(f"Longitud del {i} chunk: {len(splits[i].page_content)}")
    print(f"--- Muestra del {i} chunk ---")
    print(splits[i].page_content)
    print("--- Metadatos ---")
    print(splits[i].metadata)


from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

