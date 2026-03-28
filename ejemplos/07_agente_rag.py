import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from rag_helper.build_vector_db import crear_vector_store
from rag_helper.query_rag import QueryRAG


def main(args=None):
    #Configurar logging a archivo (sobreescribe el archivo en cada ejecución)
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    load_dotenv(override=True)

    project_name = os.getenv("LANGCHAIN_PROJECT")
    print("Proyecto LangSmith:", project_name)
    COLLECTION_NAME = "futbol-peruano-hybrid-markdown-chunks"
    
    #Vector stores para RAG de Agente IA
    vs_custom_chunks = crear_vector_store(COLLECTION_NAME)
    #Iniciamos el modelo de lenguaje para el agente RAG, se puede usar el mismo modelo para el agente RAG y para el reranker por LLM, 
    #pero en esta evaluación usaremos un modelo más pequeño para el reranker y así reducir costos en esta etapa de evaluación del agente RAG.
    llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0.0)

    #inicializamos la clase QueryRAG con el modelo de lenguaje, el vector store y sin reranker para la primera evaluación del agente RAG
    query_rag_hybrid = QueryRAG(model=llm, 
                                vector_store=vs_custom_chunks,
                                reranker=None,
                                )

    query = "Identifica al segundo máximo goleador histórico de la primera división peruana, muéstrame su nombre, y en una tabla la temporada en la que fue máximo goleador, el club en el que jugaba y la cantidad de goles anotados."
    
    print("Prueba de consulta con Agente RAG sin Reranker y respuesta resumen (sin datos de los documentos recuperados)")
    query_rag_hybrid.consultar(query, max_docs_recuperados=3, max_busquedas=3, limite_recursion=10, imprimir_respuesta="resumen")
    print("\n","*"*100, "\n")
    
    # #Asignamos reranker a la clase para evaluar el mismo agente RAG pero con reranker por modelo reranker ejecutado localmente
    query_rag_hybrid.reranker = "local"
    print("Prueba de consulta con Agente RAG usando Reranker con un modelo reranker local, sin filtrado por umbral de relevancia y respuesta resumen")
    query_rag_hybrid.consultar(query, max_docs_recuperados=3, max_busquedas=3, limite_recursion=10, imprimir_respuesta="resumen")
    print("\n","*"*100, "\n")
    
    print("Prueba de consulta con Agente RAG usando Reranker con un modelo reranker local, con umbral de relevancia y respuesta resumen")
    query_rag_hybrid.consultar(query, max_docs_recuperados=3, max_busquedas=3, limite_recursion=10, imprimir_respuesta="resumen", umbral=0.5)
    print("\n","*"*100, "\n")
    
    # #Asignamos reranker a la clase para evaluar el mismo agente RAG pero con reranker por LLM
    query_rag_hybrid.reranker = "llm"
    #Usaremos un llm más pequeño para el reranker y así reducir costos en esta etapa de evaluación del agente RAG, ya que el reranker se ejecuta varias veces durante la consulta.
    query_rag_hybrid.llm_reranker = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)
    print("Prueba de consulta con Agente RAG usando Reranker por LLM, y con umbral de relevancia = 0.5 y respuesta resumen")
    query_rag_hybrid.consultar(query, max_docs_recuperados=3, max_busquedas=3, limite_recursion=10, imprimir_respuesta="resumen", umbral=0.5)

    # respuestas_ok = [
    #     """Oswaldo Ramírez
    #     | Temporada | Club                | Goles |
    #     |-----------|---------------------|-------|
    #     | 1968      | Sport Boys          | 26 |
    #     | 1980      | Sporting Cristal    | 19 |
    #     """,
    # ]
    
if __name__ == "__main__":
    main()

