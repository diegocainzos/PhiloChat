"""
 Leer PDFs (PyPDFLoader).

    Limpiar texto y Chunking (RecursiveCharacterTextSplitter).

    Embeddings (GoogleGenerativeAIEmbeddings).

    Carga a Postgres (PGVector).

    ETL FULL TEXT
"""
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from pathlib import Path
import re
from rich import print
from dotenv import load_dotenv

load_dotenv()

# ... (Tus funciones e imports previos se mantienen igual) ...

# CODE
# Asumo que loaders es una lista de listas (una lista de páginas por cada libro)
# Para este ejemplo procesamos el primer libro:
# FUNCTIONS
def book_processing(docs : list[Document], skip_n=5, min_length=100):
    def clean(doc):
        doc.metadata["id"] = str(uuid.uuid4())
        # Nota la 'r' antes de las comillas del patrón
        doc.page_content = ' '.join(re.sub(r'[-\d]', '', doc.page_content).split())
        return doc

    return [clean(d) for d in docs[skip_n:] if len(d.page_content) > min_length]
    
# CONSTANTS
FOLDER_PATH = Path('etl/sources/')

# CODE
file_routes = [str(file) for file in FOLDER_PATH.iterdir() if file.is_file()]

docs = [PyPDFLoader(doc).load() for doc in file_routes]
processed_doc_list = [book_processing(doc) for doc in docs]

# --- PASO CRÍTICO: UNIFICACIÓN ---
# 1. Unimos el texto de todas las páginas en un solo string gigante.
# Usamos " " como separador para no pegar la última palabra de una pág con la primera de la otra.
def unificacion(book_pages: list[Document]):
    # Join all page contents with proper spacing
    full_text = " ".join(page.page_content for page in book_pages)
    
    full_text = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', full_text)  # Remove control characters
    full_text = re.sub(r'\s+', ' ', full_text)  # Normalize multiple spaces/newlines
    full_text = full_text.strip()  # Remove leading/trailing whitespace
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
    return full_doc

full_document_list = [unificacion(doc) for doc in processed_doc_list]
print(type(full_document_list), len(full_document_list))

# --- CHUNKING ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""] # Definir prioridades ayuda
)

# Ahora split_documents recibe una lista con UN solo documento gigante y lo trocea
fully_procesed_books = [text_splitter.split_documents([doc]) for doc in full_document_list]


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

#show_chunk(splits)

#EMBEDDINGS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# PGVECTOR DB
from langchain_postgres import PGVectorStore, PGEngine

def create_engine_and_table(db_connection_url: str,table_name: str, vector_size: int):
    engine = PGEngine.from_connection_string(url=db_connection_url)
    engine.drop_table(table_name)

    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=VECTOR_SIZE,
        metadata_columns=['author']
    )
    return engine

CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
VECTOR_SIZE = 3072
TABLE_NAME = "books"

engine = create_engine_and_table(CONNECTION_STRING, TABLE_NAME, VECTOR_SIZE)


store = PGVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embeddings_model,
)

for book in fully_procesed_books:
    store.add_documents(book)

# query = "Quien era Sisifo?"
# docs = store.similarity_search(query)
# print(docs)


