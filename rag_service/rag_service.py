from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

CONNECTION_STRING=os.environ['CONNECTION_STRING']
TABLE_NAME=os.environ['TABLE_NAME']


class RAGService:
    def __init__(self):
        #VECTORSTORE
        self.engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        #EMBEDDINGS
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store = PGVectorStore.create_sync(
            self.engine,
            embedding_service=self.embedding,
            table_name=TABLE_NAME
        )
        #RETRIEVER
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        #LLM
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview",
                                          temperature=0)
        
        self.prompt = ChatPromptTemplate.from_template("""
            Eres un asistente experto en literatura y filosofía.
            Usa el siguiente contexto recuperado para responder a la pregunta.
            Si no sabes la respuesta basándote en el contexto, di "No encuentro información en los libros procesados".
            
            Contexto:
            {context}
            
            Pregunta: {question}
            
            Respuesta detallada:
        """)
        
        #CHAIN
        self.chain = ({ 'context' : self.retriever | self.format_docs, 'question' : RunnablePassthrough()} 
                     | self.prompt
                     | self.llm
                     | StrOutputParser())
    def format_docs(self, docs):
        """Une el contenido de los documentos recuperados en un solo string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_answer(self, question: str) -> str:
        """Método público para obtener respuesta."""
        print(f"🧠 Pensando respuesta para: {question}...")
        try:
            # invoke ejecuta la cadena
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"Error generando respuesta: {str(e)}"
        
    def get_sources(self, question: str):
            """Método extra para depurar: ver qué documentos está encontrando."""
            docs = self.retriever.invoke(question)
            return [{"source": d.metadata.get("source"), "content_preview": d.page_content[:100]} for d in docs]

# --- USO DEL CÓDIGO ---
if __name__ == "__main__":
    rag = RAGService()

    # Prueba 1: Ver fuentes
    print("🔍 Buscando fuentes...")
    fuentes = rag.get_sources("¿Quién es Sísifo?")
    print(fuentes)

    # Prueba 2: Generar respuesta
    respuesta = rag.get_answer("Hecho de menos a mi exnovia.")
    print("\n🤖 RESPUESTA DE GEMINI:\n")
    print(respuesta)


