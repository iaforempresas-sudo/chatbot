- No inventes información.
- No agregues detalles que no estén en el contexto.
- Responde de forma clara y directa.
- Si la respuesta no está en el contexto, dilo expresamente."""
    respuesta = gemini.models.generate_content(
        model=CHAT_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3)
    )
    return respuesta.text or "Gemini no devolvió texto."


# Estado temporal de cada sesión del navegador: namespace aislado,
# historial del chat y contador de fragmentos indexados.
# No equivale a autenticación ni limpia automáticamente datos en Pinecone.
if "namespace" not in st.session_state:
    st.session_state.namespace = "alumno-" + uuid.uuid4().hex[:20]
if "historial" not in st.session_state:
    st.session_state.historial = []
if "fragmentos_total" not in st.session_state:
    st.session_state.fragmentos_total = 0

st.title("🧠 Laboratorio RAG con IA")
st.caption("Práctica educativa · Gemini + Pinecone · Documentos PDF, Word y TXT")

try:
    # Las claves se almacenan en Secrets, nunca dentro del código público.
    clave_gemini = st.secrets["GEMINI_API_KEY"]
    clave_pinecone = st.secrets["PINECONE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Faltan GEMINI_API_KEY y PINECONE_API_KEY en los secretos de Streamlit Cloud.")
    st.stop()

# Panel de contexto: estado de la sesión y parámetros de los modelos.
with st.sidebar:
    st.subheader("📚 Tu espacio de trabajo")
    st.info("Los documentos de esta sesión se guardan en un espacio aislado de Pinecone. Evita subir datos confidenciales.")
    st.metric("Fragmentos indexados en la sesión", st.session_state.fragmentos_total)
    st.caption("El aislamiento es por sesión del navegador, no una cuenta de usuario. Si se recarga o cierra la sesión, puede perderse el acceso a ese espacio.")
    if st.button("Comenzar nueva sesión", use_container_width=True):
        st.session_state.namespace = "alumno-" + uuid.uuid4().hex[:20]
        st.session_state.historial = []
        st.session_state.fragmentos_total = 0
        st.rerun()
    st.divider()
    st.caption(f"Embeddings: {EMBED_MODEL} · 3072 dimensiones")
    st.caption(f"Respuestas: {CHAT_MODEL} · Temperatura 0.3")

try:
    gemini, pc = clientes(clave_gemini, clave_pinecone)
except Exception as e:
    st.error(f"No fue posible inicializar los clientes: {e}")
    st.stop()

# Interfaz dividida en dos etapas: alimentar conocimiento y consultarlo.
subir, conversar = st.tabs(["📄 1. Cargar documentos", "💬 2. Consultar con IA"])
# ETAPA 1: carga e indexación de documentos.
with subir:
    st.subheader("Carga e indexación")
    st.write("Selecciona tus documentos y pulsa **Procesar documentos** para convertirlos en vectores y almacenarlos en Pinecone.")
    archivos = st.file_uploader("Selecciona archivos", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Procesar documentos", type="primary", disabled=not archivos):
        try:
            with st.spinner("Conectando con Pinecone y creando embeddings..."):
                indice = obtener_index(pc)
                progreso = st.progress(0)
                total, fallos, detalle = indexar_archivos(archivos, gemini, indice, st.session_state.namespace, progreso)
            st.session_state.fragmentos_total += total
            if total:
                st.success(f"Indexación terminada: {total} fragmentos almacenados.")
            if fallos:
                st.warning(f"No se pudieron procesar {fallos} archivo(s).")
            for linea in detalle:
                st.write(linea)
        except Exception as e:
            st.error(f"Error durante la indexación: {e}")
    with st.expander("¿Qué estamos aprendiendo?"):
        st.write("1. Extraer texto de los archivos. 2. Dividirlo en fragmentos. 3. Crear embeddings con Gemini. 4. Guardar vectores en Pinecone. 5. Buscar fragmentos similares cuando el alumno pregunta.")

# ETAPA 2: interacción con los documentos mediante chat RAG.
with conversar:
    st.subheader("Consulta tus documentos")
    if not st.session_state.fragmentos_total:
        st.info("Primero indexa al menos un documento en la pestaña anterior.")
    for mensaje in st.session_state.historial:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["texto"])
            if mensaje.get("fuentes"):
                with st.expander("🔎 Ver fragmentos recuperados"):
                    for i, fuente in enumerate(mensaje["fuentes"], 1):
                        st.markdown(f"**{i}. {fuente['documento']}** · similitud {fuente['similitud']:.3f}" if fuente["similitud"] is not None else f"**{i}. {fuente['documento']}**")
                        st.write(fuente["fragmento"])
    pregunta = st.chat_input("¿De qué trata el documento?", disabled=st.session_state.fragmentos_total == 0)
    if pregunta:
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        with st.chat_message("user"):
            st.write(pregunta)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando fragmentos y generando respuesta..."):
                    indice = obtener_index(pc)
                    # Recuperación antes de generación: base del enfoque RAG.
                    fuentes = buscar(indice, gemini, pregunta, st.session_state.namespace)
                    respuesta = responder(gemini, pregunta, fuentes)
                st.markdown(respuesta)
                if fuentes:
                    with st.expander("🔎 Ver fragmentos recuperados"):
                        for i, fuente in enumerate(fuentes, 1):
                            st.markdown(f"**{i}. {fuente['documento']}** · similitud {fuente['similitud']:.3f}" if fuente["similitud"] is not None else f"**{i}. {fuente['documento']}**")
                            st.write(fuente["fragmento"])
                st.session_state.historial.append({"rol": "assistant", "texto": respuesta, "fuentes": fuentes})
            except Exception as e:
                st.error(f"No se pudo completar la consulta: {e}")
