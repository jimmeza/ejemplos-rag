max_busquedas = 2
max_docs = 5
umbral = 0.5
pregunta = "¿Cuál es el primer club ganador de la Copa Bicentenario y en que año?"


prompt = ("Puedes buscar los documentos relevantas usando las herramientas disponibles,"
    f"hasta en {max_busquedas} ocasiones pero con criterios de búsqueda diferentes (ASEGURATE DE NO SUPERAR ESTE LIMITE)."
    f"Con un máximo de {max_docs} documentos recuperados por cada búsqueda, y un umbral de relevancia de {umbral},"
    f"responder a las siguiente pregunta:\n{pregunta}"
)

print(prompt)
