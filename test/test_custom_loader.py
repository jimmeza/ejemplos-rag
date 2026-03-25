# import sys
# import os
# # Esto sube un nivel ('..') para encontrar la raíz del proyecto
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_helper.custom_docling_loader import md_table_to_json, reemplazar_tablas_md_a_json, extraer_tablas_md, separar_texto_y_tablas, markdown_to_plain_text, cadena_json

md_text = """
| Edición | Campeón | Resultado | Subcampeón | Final | Categorías |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copa de la Liga del Perú | | | | | |
| 1928 | Bandera de Lima Alianza Lima | 3 - 0 | Bandera de Lima Liga Balnearios del Sur | Lima | Primera División Segunda División |
| Copa Presidente de la República  | | | | | |
| 1970 | Bandera de Lima Universitario | 4 - 0 | Bandera de Arequipa FBC Melgar | Lima | Primera División Tercera División |
| Torneo Intermedio 1993   | | | | | |
| 1993 | Bandera de Lima Deportivo Municipal | 2 - 2 (4-3 pen .) | Bandera del departamento de Junín Deportivo Sipesa | Lima | Primera División |
| Copa del Inca / Torneo del Inca      | | | | | |
| 2011 | José Gálvez | 7 - 3 ( glo. ) | Sport Ancash | Huaraz Chimbote y | Primera División Segunda División Tercera División |
| 2012 | Federación Peruana de Fútbol Cancelada por decisión de la argumentando problemas contractuales televisivos.  | | | | |
| 2014 | Bandera de Lima Alianza Lima | 3 - 3 ( pró. ) (5-3 pen. ) | Bandera de Lima Universidad San Martín | Callao | Primera División |
| Copa Bicentenario  | | | | | |
| 2019 | Bandera de Piura Atlético Grau | 0 - 0 ( pró. ) (4-3 pen. ) | Bandera del departamento de Junín Sport Huancayo | Callao | Primera División Segunda División |
| 2020 | pandemia de COVID-19 Suspendida por la  | | | | |
| 2021 | Bandera de Lima Sporting Cristal | 2 - 1 | Bandera del departamento de La Libertad (Perú) Carlos Mannucci | Lima | Primera División Segunda División |
| 2022 | Federación Peruana de Fútbol Cancelada por decisión de la .  | | | | |
| Copa de la Liga de Fútbol Profesional | | | | | |
| 2025 | Federación Peruana de Fútbol Cancelada por decisión de la argumentando problemas contractuales televisivos.  | | | | |
"""

texto_tabla = """# Posible titulo
Esto es un texto inicial...
| Año | Campeonato | Puesto |
|-------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| 1927 | Campeonato Sudamericano de 1927 (Copa América) | Tercer lugar |
| 1929 | Campeonato Sudamericano de 1929 (Copa América) | Cuarto lugar |
| 1935 | Campeonato Sudamericano de 1935 (Copa América) | Tercer lugar |
| 1938 | Juegos Bolivarianos de 1938 | Medalla de oro |
| 1939 | Campeonato Sudamericano de 1939 (Copa América) | Campeón |
| 1941 | Campeonato Sudamericano de 1941 (Copa América) | Cuarto lugar |
| 1947 | Juegos Bolivarianos de 1947 | Medalla de oro |
| 1949 | Campeonato Sudamericano de 1949 (Copa América) | Tercer lugar |
| 1951 | Juegos Bolivarianos de 1951 | Medalla de bronce |
| 1955 | Campeonato Sudamericano de 1955 (Copa América) | Tercer lugar |
| 1957 | Campeonato Sudamericano de 1957 (Copa América) | Cuarto lugar |
| 1959 | Campeonato Sudamericano de 1959 (Copa América) | Cuarto lugar |
| 1975 | Copa América 1975 | Campeón |
| 1961 | Juegos Bolivarianos de 1961 | Medalla de oro |
| 1973 | Juegos Bolivarianos de 1973 | Medalla de oro |
| 1977 | Juegos Bolivarianos de 1977 | Medalla de bronce |
| 1979 | Copa América 1979 | Tercer lugar (compartido con Brasil)  |
| 1981 | Juegos Bolivarianos de 1981 | Medalla de oro |
| 1983 | Copa América 1983 | Tercer lugar (compartido con Paraguay)  |
| 1985 | Juegos Bolivarianos de 1985 | Medalla de bronce |
| 1997 | Copa América 1997 | Cuarto lugar |
| 2011 | Copa América 2011 | Tercer lugar |
| 2015 | Copa América 2015 | Tercer lugar |
| 2019 | Copa América 2019 | Subcampeón |
| 2021 | Copa América 2021 | Cuarto lugar |

| Año | Campeonato | Puesto |
|-------|-------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| 1960 | Torneo Preolímpico Sudamericano Sub-23 de 1960 | Subcampeón  |
| 1964 | Torneo Preolímpico Sudamericano Sub-23 de 1964 | Tercer lugar |
| 1972 | Torneo Preolímpico Sudamericano Sub-23 de 1972 | Cuarto lugar  |
| 1980 | Torneo Preolímpico Sudamericano Sub-23 de 1980 | Tercer lugar |
| 1982 | Juegos Suramericanos de 1982 | Medalla de bronce |
| 1990 | Juegos Suramericanos de 1990 | Medalla de oro |
| 1994 | Juegos Suramericanos de 1994 | Medalla de bronce.  |
| 1997 | Juegos Bolivarianos de 1997 | Medalla de plata |
| 2001 | Juegos Bolivarianos de 2001 | Medalla de oro |
| 2013 | Juegos Bolivarianos de 2013 | Medalla de bronce  |
| 2013 | Campeonato Sudamericano Sub-15 de 2013 | Campeón  |
| 2014 | Juegos Olímpicos de la Juventud 2014 | Medalla de oro  |
## Subtitulo
### Otro posible subtitulo
Esto es un texto final
"""
texto_tabla_2 = """
| Primera División del Perú Liga1 de Fútbol Profesional | | |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------|
| Temporada o torneo actual Liga1 2026 | | |
| | | |
| Datos generales | Datos generales | Datos generales |
| Deporte | Fútbol | |
| Sede | Perú Perú | |
| Nivel de competencia | Nacional (desde 1966) | |
| Federación | Federación Peruana de Fútbol | |
| Confederación | Conmebol | |
| Nombre oficial | Liga1 de Fútbol Profesional | Liga1 de Fútbol Profesional |
| Nombre comercial | Liga1 Te Apuesto | Liga1 Te Apuesto |
| Lema | «El ADN del hincha» | |
| Organizador | FPF LFP | |
| Director ejecutivo | Jesús Gonzales Hurtado | Jesús Gonzales Hurtado |
| Presidente | Bandera de Perú Agustín Lozano | |
| Equipos participantes | 18 | 18 |
| Datos históricos | Datos históricos | Datos históricos |
| Fundación | 1912 (114 años) | 1912 (114 años) |
| Equipos fundacionales | Ver lista Bandera de Lima Association F. C. Bandera de Lima Escuela Militar de Chorrillos Bandera de Lima Jorge Chávez N.º 1 Bandera de Lima Lima Cricket Bandera de Lima Miraflores S. C. Bandera de Lima Sport Alianza Bandera de Lima Sport Inca Bandera de Lima Sport Vitarte | |
| Primera temporada | 1912 | |
| Primer campeón | Bandera de Lima Lima Cricket | |
| Goleador histórico | Bandera de Argentina Bandera de Perú Sergio Ibarra (274) | |
| Datos estadísticos | Datos estadísticos | Datos estadísticos |
| Campeón actual | Bandera de Lima Universitario | |
| Subcampeón actual | Departamento del Cusco Cusco FC | |
| Más campeonatos | Bandera de Lima Universitario (29) | |
| Datos de competencia | Datos de competencia | Datos de competencia |
| Categoría | 1.ª División | |
| Descenso a | Segunda División del Perú | |
| Clasificación a | Copa Libertadores Copa Sudamericana | |
| Copa nacional | Copa LFP - FPF | |
| Otros datos | Otros datos | Otros datos |
| Patrocinador | Ver lista Bandera de Perú Te Apuesto Bandera de Alemania Adidas Bandera de Perú Cerveza Cristal Bandera de Chile LATAM Airlines Bandera de Perú Sporade Bandera de Perú Caja Huancayo Bandera de Australia Vantage Bandera de Brasil Smart Fit Bandera de Estados Unidos inDrive Bandera de Perú Big Cola | |
| Socio de TV | | |
| Sitio web oficial | https://liga1.pe/ | |
| Cronología | Cronología | Cronología |
| | Campeonato Descentralizado 1966 - 2018 | Liga1 de Fútbol Profesional 2019 - actualidad | texto | |---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|----| | | |
|  [editar datos en Wikidata ] | | |
"""

texto_tabla_3 = """
| Año | Campeón | Subcampeón |
|-------|---------------------------|---------------------------|
| 1951 | Sport Boys | Deportivo Municipal |
| 1952 | Alianza Lima | Sport Boys |
| 1953 | Mariscal Sucre | Alianza Lima |
| 1954 | Alianza Lima | Sporting Tabaco |
| 1955 | Alianza Lima | Universitario de Deportes |
| 1956 | Sporting Cristal | Alianza Lima |
| 1957 | Centro Iqueño | Atlético Chalaco |
| 1958 | Sport Boys | Atlético Chalaco |
| 1959 | Universitario de Deportes | Sport Boys |
| 1960 | Universitario de Deportes | Sport Boys |
| 1961 | Sporting Cristal | Alianza Lima |
"""

texto_md_2 = """
El campeonato peruano fue calificado por la [Federación Internacional de Historia y Estadística de Fútbol](/wiki/IFFHS) como la [10.° mejor liga del mundo en 2004](/wiki/Anexo:Clasificaci%C3%B3n_mundial_de_ligas_nacionales_de_la_IFFHS) . En 2010, ocupó el puesto 16 en la misma lista, superando a la liga de [Chile](/wiki/Primera_Divisi%C3%B3n_de_Chile) y [Paraguay](/wiki/Primera_Divisi%C3%B3n_de_Paraguay) . [[ 1 ]](#cite_note-1) En la primera década del siglo XXI , fue considerada como la quinta liga más fuerte de [Sudamérica](/wiki/Am%C3%A9rica_del_Sur) y la sexta de [América](/wiki/Am%C3%A9rica) , mientras que a nivel mundial se ubicó en el puesto 20. [[ 2 ]](#cite_note-2)

## Historia

[ [editar](/w/index.php?title=Primera_Divisi%C3%B3n_del_Per%C3%BA&action=edit&section=1) ]

### Introducción del fútbol en el Perú y los primeros clubes

[ [editar](/w/index.php?title=Primera_Divisi%C3%B3n_del_Per%C3%BA&action=edit&section=2) ]

Vista del puerto del Callao , frecuentado por marinos ingleses a finales del siglo XIX .
"""

texto = texto_tabla_2
ultimo_titulo = "Titulo de la tabla"

#Guardar resultado en archivo
# with open("temp/resultado_markdown_to_plain_text.txt", "w", encoding="utf-8") as f:
#     f.write(markdown_to_plain_text(texto))

# print("md_table_to_json(texto): ", cadena_json(md_table_to_json(texto_tabla_2, ultimo_titulo)))
# print("\n\n")


# print("longitud del texto: ", len(texto))
# tablas = extraer_tablas_md(texto)
# print("extraer_tablas_md(texto): ", tablas)
# for tabla, inicio, fin in tablas:
#     print("tabla: ", tabla)
#     print("longitud de la tabla: ", len(tabla))
#     extracto = texto[inicio:fin]
#     print("extracto: ", extracto)
#     print("longitud del extracto: ", len(extracto))
#     print("Tabla es igual a extracto? ", tabla == extracto)
#     print("\n\n")

# segmentos = separar_texto_y_tablas(texto)
# print("segmentos: ", len(segmentos))
# print("separar_texto_y_tablas(texto): ", segmentos)
# for tipo, segmento, titulo in reversed(segmentos):
#     if titulo != "":
#         ultimo_titulo = titulo
#         break
# print("\n\n")

print("reemplazar_tablas_md_a_json(texto): \n", reemplazar_tablas_md_a_json(texto, ultimo_titulo))
