querys = [
    #02_recuperar_rag_embeddings
    "¿Cuando fue el primer encuentro de fútbol en el Perú?",
    "¿Cuál fue el club ganador de copa bicentenario 2019?",
    "primer título internacional de selección nacional de Perú",
    #04_recuperar_rag_hibrido
    "¿Cuando fue el primer encuentro de fútbol en el Perú?",
    "¿Qué nos explica Jorge Basadre sobre el fútbol peruano?",
    "primer título internacional de selección nacional de Perú",
    #05_tamano_chunks_diferentes
    "¿Cuáles torneos internacionales ha ganado Cienciano y en que año?",
    "¿Cuáles clubes peruanos han sido campeones o subcampeones de torneos internacionales?, indicar los torneos y en que año.",
    "¿Cuál es el club con más titulos de campeón en el futbol peruano?, indicar la cantidad de títulos ganados.",
    #06_estrategia_chunking_diferentes
    "¿Cuál fue el primer club peruano en ser fundado y en que año?",
    "¿Quien ganó la Supercopa 'Copa Federación' del 2012?, indicar el subcampeon.",
    #Se repitio esta pregunta del priemr ejemplo:    "¿Cuál fue el club ganador de copa bicentenario 2019?",
    #Otras preguntas
    "¿En qué temporada ganó Juan Aurich su primer campeonato?",
    "¿Cómo le fue a Perú en los juegos olimpicos de 1936?",
    "¿Cómo le fue a Perú en los juegos olimpicos de 1960?",
]

respuestas_ok = [
    #02_recuperar_rag_embeddings
    "el primer registro de un partido de fútbol en el Perú, corresponde al domingo 7 de agosto de 1892",
    "El club Atlético Grau gano la copa Bicentenario del 2019",
    "El primer título internacional de la selección de Perú fue en los Juegos Bolivarianos de 1938",
    #04_recuperar_rag_hibrido
   "el primer registro de un partido de fútbol en el Perú, corresponde al domingo 7 de agosto de 1892",
    "Jorge Basadre explica que el primer registro de un partido de fútbol en el Perú ocurrió el domingo 7 de agosto 1892, cuando ingleses y peruanos jugaron representando al Callao y a Lima. El club Lima Cricket and Lawn Tennis(el primer club del país) organizó encuentros en el campo Santa Sofía de su propiedad. La guerra del Pacífico provocó la destrucción de varias ciudades costeras, incluida Lima, lo que detuvo temporalmente la difusión del fútbol y otros deportes en el país.",
    "El primer título internacional de la selección de Perú fue en los Juegos Bolivarianos de 1938",
    #05_tamano_chunks_diferentes
    " Cienciano ha ganado los torneos internacionales oficiales Copa Sudamericana el 2003 y Recopa Sudamericana el 2004.",
    """**Clubes peruanos que han sido campeones o subcampeones de torneos internacionales**

| Club | Torneo | Año | Puesto |
|------|--------|-----|--------|
| Cienciano | Copa Sudamericana | 2003 | Campeón |
| Cienciano | Recopa Sudamericana | 2004 | Campeón |
| Universitario de Deportes | Copa Libertadores | 1972 | Subcampeón |
| Sporting Cristal | Copa Libertadores | 1997 | Subcampeón |
""",
    "El club **Universitario de Deportes** con 29 títulos es el más ganador.",
    #06_estrategia_chunking_diferentes
    "El primer club peruano fue el Lima Cricket and Football Club y fue fundado en 1859.",
    "José Gálvez ganó la Supercopa “Copa Federación” del 2012, Juan Aurich fue el subcampeón.",
    #Se repitio esta pregunta del priemr ejemplo:   "El club Atlético Grau gano la copa Bicentenario del 2019",
    #Otras preguntas
    "Juan Aurich gano su primer campeonato en la temporada del año 2011.",
    "En los Juegos Olímpicos de Berlín 1936, Perú empezó derrotando a Finlandia por 7 goles contra 3. En cuartos de final, enfrentó a Austria en un partido muy controvertido, ganando Perú en tiempo extra por 4 a 2 y se retiró del torneo porque los austriacos exigieron una revancha con el argumento de que los aficionados peruanos habían invadido el campo y maltratado a los jugadores austriacos. La defensa peruana nunca fue escuchada, y el Comité Olímpico y la FIFA favorecieron a los austriacos.",
    "En los Juegos Olímpicos de Roma 1960, el equipo de Perú logró clasificarse y venció a la Selección de fútbol de la India, luego terminó perdiendo ante Francia y Hungría.",
 ]