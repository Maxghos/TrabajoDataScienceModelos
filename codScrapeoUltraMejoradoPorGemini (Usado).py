from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup
import polars as pl
import time as tm
import random as rnd

# -------------------------------------------------------------------
## ⚙️ Configuración de Paginación y Enlaces
# -------------------------------------------------------------------
PAGINA_INICIO = 361
PAGINA_FIN = 400 # ¡458 páginas! (Índices 0 a 457)
BASE_LINK = "https://chilepropiedades.cl/propiedades/arriendo-mensual/departamento/region-metropolitana-de-santiago-rm/"
base_url = "https://www.chilepropiedades.cl" 

# Tabla Para guardar valores en polars más tarde
tabla = [] 

# Functión para bloquear recursos (imagenes/media)
def bloquearImgyVideos(ruta):
    tipoDatos = ruta.request.resource_type
    if tipoDatos in ["image", "media"]:
        ruta.abort()
    else:
        ruta.continue_()

# Headers HTTP (se definen una sola vez)
headers = {
"User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0.5993.90 Safari/537.36",
              "Accept-Language": "es-CL,es;q=0.9",
              "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
              "Referer": "https://www.houm.com/"
            }

#Se inicia con playwright SINCRICO
with sync_playwright() as puente:

    # -------------------------------------------------------------------
    # 1. INICIALIZACIÓN DEL NAVEGADOR (Primera Apertura)
    # -------------------------------------------------------------------
    navUsar = puente.chromium.launch(headless=True) 
    pagc = navUsar.new_page()
    pagc.set_extra_http_headers(headers)
    pagc.route("**/*", bloquearImgyVideos)
    
    # -------------------------------------------------------------------
    ## 🔄 Bucle Principal de Paginación
    # -------------------------------------------------------------------
    for num_pagina in range(PAGINA_INICIO, PAGINA_FIN + 1):
        
        link_paginado = BASE_LINK + str(num_pagina)
        print(f"\n==========================================================")
        print(f"📢 SCRAPEANDO PÁGINA: {num_pagina + 1} de {PAGINA_FIN + 1} ({link_paginado})")
        print(f"==========================================================")

        try:
            pagc.goto(link_paginado, timeout=60000) 
            pagc.wait_for_selector("div.clp-publication-list", timeout=30000) 
        except Exception as e:
            print(f"❌ Error CRÍTICO al cargar la página {num_pagina + 1}. Mensaje: {e}. SALTANDO.")
            continue

        pagEntera = pagc.content()
        soup = BeautifulSoup(pagEntera, "html.parser")
        anunciosPag = soup.find_all("div", class_="clp-publication-element clp-highlighted-container")
        
        # -------------------------------------------------------------------
        ## 📝 Bucle Interno de Extracción de Detalles
        # -------------------------------------------------------------------
        for i, anun in enumerate(anunciosPag,1): 
            hrefSol = "N/A"
            try: 
                linkInfo= anun.find("a", class_="clp-listing-image-link")
                if not linkInfo: continue

                hrefRel = linkInfo["href"]
                hrefSol = base_url + hrefRel 

                pagc.goto(hrefSol, timeout=60000)
                pagc.wait_for_selector("h1", timeout=30000)

                detalleHtml = pagc.content()
                soupDetalle = BeautifulSoup(detalleHtml, "html.parser")

                # --- EXTRACCIÓN DE DETALLES (Tu lógica) ---
                tituloPrincipal = soupDetalle.select_one("h1", class_="ui-pdp-title")
                
                # Precio
                precio = "N/A"
                precioEspecifico = soupDetalle.find("div", class_="clp-description-label col-6", string="Valor:")
                if precioEspecifico:
                    valorPrecio = precioEspecifico.find_next_sibling("div", class_="clp-description-value col-6")
                    if valorPrecio:
                        precio = valorPrecio.get_text(strip=True)
                
                # Ubicacion
                ubicacion = "N/A"
                ubicacionEspecifica = soupDetalle.find("div", class_="col-6 clp-description-label", string="Dirección:")
                if ubicacionEspecifica:
                    ubicacionBosai = ubicacionEspecifica.find_next_sibling("div", class_="col-6 clp-description-value")
                    if ubicacionBosai:
                        ubicacion = ubicacionBosai.get_text(strip=True)

                # Gastos Comunes
                gastosComunes = "No Informado"
                gastosEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Gastos Comunes:") 
                if gastosEspecificos:
                    gastosBosaii = gastosEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                    if gastosBosaii:
                        gastosComunes = gastosBosaii.get_text(strip=True)

                # Metros
                metros = "N/A"
                metrosEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Superficie Total:")
                if metrosEspecificos:
                    metrosBosaii = metrosEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                    if metrosBosaii:
                        metros = metrosBosaii.get_text(strip=True)
                
                # Dormitorios
                habitaciones = "N/A"
                habEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Habitaciones:")
                if habEspecificos:
                    habBosaii = habEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                    if habBosaii:
                        habitaciones = habBosaii.get_text(strip=True)
                
                # Baños
                baños = "N/A"
                bañosEspecificos = soupDetalle.find("div", class_="clp-description-label col-6", string="Baño:")
                if bañosEspecificos:
                    bañosbosaii = bañosEspecificos.find_next_sibling("div", class_="clp-description-value col-6")
                    if bañosbosaii:
                        baños = bañosbosaii.get_text(strip=True)
                
                # Agregar a la tabla
                tabla.append({
                    "Nombre": tituloPrincipal.get_text(strip=True) if tituloPrincipal else "N/A",
                    "Precio": precio,
                    "Gastos_Comunes": gastosComunes,
                    "Ubicacion": ubicacion,
                    "Metros": metros,
                    "Habitaciones": habitaciones,
                    "Baños": baños
                })
                
                print(f"   -> Anuncio {i} de la Pág {num_pagina+1} Procesado Correctamente!")
                
                tm.sleep(rnd.uniform(7, 10.5)) 

            except Exception as e:
                print(f"❌ Error al procesar el ANUNCIO {i} de la Pág {num_pagina+1}. URL: {hrefSol}. Mensaje: {e}. Continuado.")
                tm.sleep(5) 
                continue
        
        # ------------------------------------------------
        # ✨ CHECKPOINT DE GUARDADO Y RECICLAJE
        # ------------------------------------------------
        if (num_pagina + 1) % 20 == 0:
            
            # 1. GUARDADO (Checkpoint)
            temp_data = pl.DataFrame(tabla)
            nombre_archivo = f"Arriendos_Checkpoint_Pag_{num_pagina + 1}.csv"
            temp_data.write_csv(nombre_archivo)
            print("----------------------------------------------------------")
            print(f"💾 CHECKPOINT: Guardado en {nombre_archivo}")
            
            # 2. CIERRE (Libera la RAM)
            pagc.close()
            navUsar.close()
            tm.sleep(5) 
            
            # 3. REINICIO (Nueva instancia con memoria limpia)
            navUsar = puente.chromium.launch(headless=True) 
            pagc = navUsar.new_page()
            pagc.set_extra_http_headers(headers)
            pagc.route("**/*", bloquearImgyVideos) # Se re-activa el bloqueo
            
            print(f"🔄 Navegador Reiniciado. Recursos liberados. Continuamos...")
            print("----------------------------------------------------------")
            
        print(f"✅ PÁGINA {num_pagina + 1} ({len(anunciosPag)} anuncios) PROCESADA.")


    # -------------------------------------------------------------------
    ## 🛑 Cierre Final del Navegador
    # -------------------------------------------------------------------
    pagc.close()
    tm.sleep(0.6)
    navUsar.close()
    
#Se crea el DataFrame en polars
data = pl.DataFrame(tabla)
print("\n============== DATOS FINALES (Primeras 5 Filas) ==============")
print(data.head(5))

#Exportacion de Datos a Csv (Guarda todos los datos acumulados al finalizar)
data.write_csv("ArriendosChilePropiedades_FINAL.csv") 
print(f"Archivo FINAL exportado: ArriendosChilePropiedades_FINAL.csv con {len(tabla)} registros.")