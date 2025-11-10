from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup
import polars as pl
import time as tm
import random as rnd

# -----------------------------------------------
# Configuración De Paginas y bla bla bla
# Primero, se define el rango de inicio, y luego el rango de término
PAGINA_INICIO = 0
PAGINA_FIN = 1
# Como hay alrededor de 478 páginas (actualización: 30/10/2025), la última es el índice 477 (0 a 477 son 478 páginas)

#Este es el link que se usará, el base, el completo
BASE_LINK = "https://chilepropiedades.cl/propiedades/arriendo-mensual/departamento/region-metropolitana-de-santiago-rm/"
#Este es el link que se usará para el salto de páginas, debido a que en esta pág solo el valor final cambia para pasar a otra pag
base_url = "https://www.chilepropiedades.cl" # La URL base para construir enlaces de detalle

# Tabla Para guardar valores en polars más tarde (Se inicia antes del bloque entero de inicialización del modelo sincronico como variable global)
tabla = []
# -----------------------------------------------

# Se inicia con playwright SINCRONICO (asincronico no pa evitar problemas ya que sin orden no contaria que datos saco/duplicó, etc)
with sync_playwright() as puente:

    # Navegador a usar (Chrome, Safari, entre otros)
    navUsar = puente.chromium.launch(headless=True) #<- El headless es para abrir o cerrar el nav en tiempo real, True para cerrar y False para abrirlo, es como para ver que clickea y que no
    pagc = navUsar.new_page()
    
    # Se hace presente el uso de headers(user-agents) para tener una estadía más "humana" y asi disminuir el chance de baneo
    headers = {
    "User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
             "AppleWebKit/537.36 (KHTML, like Gecko) "
                 "Chrome/118.0.5993.90 Safari/537.36",
            "Accept-Language": "es-CL,es;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://www.houm.com/"
         }
    pagc.set_extra_http_headers(headers)

    # Se crea la primera función que es para poder bloquear iamgenes, videos, etc para ahorrar recursos y que la pag cargue + rapido
    def bloquearImgyVideos(ruta):
         tipoDatos = ruta.request.resource_type
         if tipoDatos in ["image", "media"]:
             ruta.abort()
         else:
             ruta.continue_()
    
    #Un equivalente de la función sería este -> pagc.route(**/*, lambda ruta: ruta.abort() if ruta.request.resource_type in ["image", "media"] else ruta.continue_())
    #Define la función altiro, como usar ()=> en node.js

    # -------------------------------------------------------------------
    # Bucle de Paginas y bla bla bla

    # El codigo se repite para cada numero de página,es necesario tenerlo dentro de un bucle para completar la cantidad de páginas totales globables del sitio de arriendos
    for num_pagina in range(PAGINA_INICIO, PAGINA_FIN + 1):
        
        #Contador de Pags
        link_paginado = BASE_LINK + str(num_pagina)
        print(f"\n==========================================================")
        print(f"📢 SCRAPEANDO PÁGINA: {num_pagina + 1} de {PAGINA_FIN + 1} ({link_paginado})")
        print(f"==========================================================")

        pagc.goto(link_paginado) # Se accede al link de la página actual

        # Esperar a que carguen los anuncios de la pag (anuncios de arriendo, se agarra el total de todos, no solo uno)
        pagc.wait_for_selector("div.clp-publication-list") 

        # Se parsea el contenido de la pag, tipo navega por toda la "insepccion" de contenido y agarra lo necesario (lo que le pedimos), a diferencia de bs4 aqui se usa .content() en vez de .text
        pagEntera = pagc.content()
        soup = BeautifulSoup(pagEntera, "html.parser")

        # Se cargan los anuncios 
        anunciosPag = soup.find_all("div", class_="clp-publication-element clp-highlighted-container")
        
        # -------------------------------------------------------------------
        # En esta sección de acá es neto iimportante, ya que aquí se genera la extracción de datos específicos a buscar
        # Primero, se realiza un contador por página (reinicio) para los anuncios que vaya visitando
        for i, anun in enumerate(anunciosPag,1): 

            # Se obtiene el link del anuncio en la clase que se ubica el anuncio
            linkInfo= anun.find("a", class_="clp-listing-image-link")
            if not linkInfo:
                continue

            hrefRel = linkInfo["href"]
            hrefSol = base_url + hrefRel # Aquí obtienes la URL completa

            # Se va al detalle del arriendo encontrado (tipo "clickear")
            pagc.goto(hrefSol)
            # Se espera a que cargue el titulo
            pagc.wait_for_selector("h1")

            #Ahora se descarga el contenido del detalle del anuncio
            detalleHtml = pagc.content()
            soupDetalle = BeautifulSoup(detalleHtml, "html.parser")

            #------------------------------------------------------Desde ahora se empieza a extraer la info de detalles del Arriendo----------------------------------------------

            #Titulo del Arriendo
            tituloPrincipal = soupDetalle.select_one("h1", class_="ui-pdp-title")

            #-----------------------------------------------------------------
            #Precio
            precio = "N/A"
            precioEspecifico = soupDetalle.find("div", class_="clp-description-label col-6", string="Valor:")

            if precioEspecifico:
                valorPrecio = precioEspecifico.find_next_sibling("div", class_="clp-description-value col-6")
                if valorPrecio:
                    precio = valorPrecio.get_text(strip=True)
            #------------------------------------------------------------------
            #Ubicacion Precisa del Arriendo
            ubicacion = "N/A"
            ubicacionEspecifica = soupDetalle.find("div", class_="col-6 clp-description-label", string="Dirección:")

            if ubicacionEspecifica:
                ubicacionBosai = ubicacionEspecifica.find_next_sibling("div", class_="col-6 clp-description-value")
                if ubicacionBosai:
                    ubicacion = ubicacionBosai.get_text(strip=True)

            #------------------------------------------------------------------
            #Gastos Comunes (Si es que aplica)
            gastosComunes = "No Informado"
            gastosEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Gastos Comunes:") 
            
            if gastosEspecificos:
                gastosBosaii = gastosEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                if gastosBosaii:
                    gastosComunes = gastosBosaii.get_text(strip=True)

            #------------------------------------------------------------------
            #Metros
            metros = "N/A"
            metrosEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Superficie Total:")
            if metrosEspecificos:
                metrosBosaii = metrosEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                if metrosBosaii:
                    metros = metrosBosaii.get_text(strip=True)
            #------------------------------------------------------------------
            #Dormitorios
            habitaciones = "N/A"
            habEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Habitaciones:")
            if habEspecificos:
                habBosaii = habEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                if habBosaii:
                    habitaciones = habBosaii.get_text(strip=True)
            #------------------------------------------------------------------
            #Baños
            baños = "N/A"
            bañosEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Baño:")
            if bañosEspecificos:
                bañosbosaii = bañosEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                if bañosbosaii:
                    baños = bañosbosaii.get_text(strip=True)
            
            #Luego De obtener los datos, se agregan a la tabla
            tabla.append({
                "Nombre": tituloPrincipal.get_text(strip=True) if tituloPrincipal else "N/A",
                "Precio": precio,
                "Gastos_Comunes": gastosComunes,
                "Ubicacion": ubicacion,
                "Metros": metros,
                "Habitaciones": habitaciones,
                "Baños": baños
            })
            
            #Se imprime por cada arriendo procesado
            print(f"   -> Anuncio {i} de la Pág {num_pagina+1} Procesado Correctamente!")
            
            #Tiempo de espera randomizado entre extracciones de anuncios
            tm.sleep(rnd.uniform(5, 8.3))

        print(f"✅ PÁGINA {num_pagina + 1} ({len(anunciosPag)} anuncios) PROCESADA.")


    # -------------------------------------------------------------------
    # Se cierra el navegador total
    # Despues de completar todas las páginas, deja un breve tiempo de espera para os procesos remanentes, y despues se cierra todo el navegador
    pagc.close()
    tm.sleep(0.6)
    navUsar.close()
    
#Se crea el DataFrame en polars
data = pl.DataFrame(tabla)
print("\n============== DATOS FINALES (Primeras 5 Filas) ==============")
print(data.head(5))

#Exportacion de Datos a Csv
data.write_csv("ArriendosChilePropiedadesDatos.csv") # Descomenta para exportar