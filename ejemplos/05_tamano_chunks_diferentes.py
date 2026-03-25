import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from rag_helper.build_vector_db import (crear_dataset_evaluacion,
                                        crear_vector_store,
                                        evaluar_documentos_recuperados)


def main(args=None):
    #Configurar logging a archivo (sobreescribe el archivo en cada ejecución)
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    load_dotenv(override=True)

    project_name = os.getenv("LANGCHAIN_PROJECT")
    print("Proyecto LangSmith:", project_name)
    COLLECTION_NAME = "futbol-peruano-hybrid-doc-chunks"
    COLLECTION_NAME_SMALL = "futbol-peruano-denso-default"
    
    #Vector stores para probar diferentes tamaños de chunks
    vs_doc_chunks_denso_small = crear_vector_store(COLLECTION_NAME_SMALL, 
                                      modo_recuperacion="dense",
                                      )

    vs_doc_chunks_denso = crear_vector_store(COLLECTION_NAME, 
                                      modo_recuperacion="dense",
                                      )
    
    querys = [
        "¿Cuáles torneos internacionales ha ganado Cienciano y en que año?",
        "¿Cuáles clubes peruanos han sido campeones o subcampeones de torneos internacionales?, indicar los torneos y en que año.",
        "¿Cuál es el club con más titulos de campeón en el futbol peruano?, indicar la cantidad de títulos ganados."
    ]

    respuestas_ok = [
        """Torneos internacionales oficiales ganados por Cienciano:

| Torneo | Año |
|--------|-----|
| Copa Sudamericana | 2003 |
| Recopa Sudamericana | 2004 |
""",
        """Clubes peruanos campeones o subcampeones de torneos internacionales:

| Club | Torneo | Año | Puesto |
|------|--------|-----|--------|
| Cienciano | Copa Sudamericana | 2003 | Campeón |
| Cienciano | Recopa Sudamericana | 2004 | Campeón |
| Universitario de Deportes | Copa Libertadores | 1972 | Subcampeón |
| Sporting Cristal | Copa Libertadores | 1997 | Subcampeón |
""",
        "El club **Universitario de Deportes** con 29 títulos.",
    ]

    llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)

    #Preparar datasets para probar diferentes tamaños de chunks
    dataset_doc_chunks_denso_small = crear_dataset_evaluacion(vs_doc_chunks_denso_small, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=3,
                            )
    dataset_doc_chunks_denso = crear_dataset_evaluacion(vs_doc_chunks_denso, 
                            llm, 
                            querys,
                            respuestas_ok,
                            max_docs_recuperados=3,
                            )

    print("Probando busqueda semantica con diferentes tamaños de chunks:")
    print(f"Colección {COLLECTION_NAME_SMALL} de máximo 256 tokens")
    evaluacion_token_size = evaluar_documentos_recuperados(dataset_doc_chunks_denso_small, llm, imprimir_resultados=True)
    print("="*50)
    print(f"Colección {COLLECTION_NAME} de máximo 1024 tokens")
    evaluacion_token_size = evaluar_documentos_recuperados(dataset_doc_chunks_denso, llm, imprimir_resultados=True)


if __name__ == "__main__":
    main()
