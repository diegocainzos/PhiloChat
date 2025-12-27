import re
import uuid
import hashlib
from pathlib import Path
from rich import print
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVectorStore, PGEngine

load_dotenv()

# --- CONFIG ---
FOLDER_PATH = Path('etl/sources/')
CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
TABLE_NAME = "books"
VECTOR_SIZE = 3072 

# --- 1. LIMPIEZA DE PÁGINAS ---
def generate_uuid_from_content(content: str, source: str) -> str:
    # 1. Crear un hash único del contenido
    unique_string = content + source
    hash_object = hashlib.md5(unique_string.encode())
    
    # 2. Convertir el hash (32 chars hex) directamente a un UUID válido
    # Esto crea un UUID determinista (siempre el mismo para el mismo texto)
    return str(uuid.UUID(hex=hash_object.hexdigest()))

def clean_page_content(text: str) -> str:
    # Ojo: Tu regex original [-\d] elimina TODOS los números (años, cantidades). 
    # Si eso es lo que quieres, déjalo, si no, usa r'\s+' para limpiar espacios.
    text = re.sub(r'[-\d]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_and_unify_book(file_path: str, skip_n=5) -> Document:
    """
    Carga, limpia y unifica un PDF en un solo Documento gigante.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Saltamos las primeras N páginas (índices, portadas)
    pages = pages[skip_n:]
    
    # Extraemos el texto limpio de todas las páginas
    cleaned_texts = [clean_page_content(p.page_content) for p in pages]
    
    # Unimos con espacio para mantener continuidad entre salto de página
    full_text = " ".join(cleaned_texts)
    
    # Gestión de Metadatos: Tomamos la fuente del primer doc y limpiamos basura
    if not pages:
        return None
        
    meta = pages[0].metadata.copy()
    meta.pop('page', None)       # Borrar número de página
    meta.pop('page_label', None) # Borrar etiquetas de página
    
    # Retornamos UN solo documento por libro
    return Document(page_content=full_text, metadata=meta)


# --- 2. PIPELINE DE EJECUCIÓN ---

# A. Carga y Unificación
file_routes = [str(file) for file in FOLDER_PATH.iterdir() if file.is_file()]
print(f"Procesando {len(file_routes)} libros...")

# Creamos la lista de LIBROS (Documentos gigantes)
raw_books = []
for route in file_routes:
    book_doc = process_and_unify_book(route)
    if book_doc and len(book_doc.page_content) > 100: # Filtro mínimo
        raw_books.append(book_doc)

print(f"Libros unificados: {len(raw_books)}")

# B. Chunking (Splitter)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

# AQUI LA MEJORA: Pasamos la lista de libros directamente.
# El splitter devolverá una lista PLANA de todos los chunks de todos los libros.
all_chunks = text_splitter.split_documents(raw_books)

ids_para_insertar = []
docs_para_insertar = []

for chunk in all_chunks:
    # 1. Generar ID determinista
    doc_id = generate_uuid_from_content(chunk.page_content, chunk.metadata.get('source', ''))
    
    # 2. (Opcional) Asignarlo al documento si usas versiones nuevas de LangChain
    chunk.id = doc_id 
    
    docs_para_insertar.append(chunk)
    ids_para_insertar.append(doc_id)

print(f"Total chunks generados listos para insertar: {len(all_chunks)}")

# --- 3. INGESTIÓN A POSTGRES ---

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Inicialización de DB
engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
engine.init_vectorstore_table(
    table_name=TABLE_NAME,
    vector_size=VECTOR_SIZE,
    metadata_columns=['source', 'chunk_id'] # Definir columnas ayuda a filtrar luego
)

store = PGVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embeddings_model,
)

BATCH_SIZE = 100

total_docs = len(docs_para_insertar)

for i in range(0, total_docs, BATCH_SIZE):
    batch_docs = docs_para_insertar[i : i + BATCH_SIZE] 
    batch_ids = ids_para_insertar[i : i + BATCH_SIZE]
    store.add_documents(documents=batch_docs, ids=batch_ids)
    
    print(f"Insertado lote {i} a {min(i + BATCH_SIZE, total_docs)}")
print("¡Proceso ETL finalizado!")