"""Laboratorio RAG basado en Practicapdf2026 (1).ipynb."""
import io
import re
import uuid
import unicodedata

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

st.set_page_config(page_title="Laboratorio RAG | Gemini + Pinecone", page_icon="🧠", layout="wide")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
INDEX_NAME = "documentos-tae2026-01"
DIMENSIONS = 3072


def extraer_texto(archivo):
    datos = archivo.getvalue()
    nombre = archivo.name.lower()
    if nombre.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(datos)).pages)
    if nombre.endswith(".docx"):
        return "\n".join(p.text for p in Document(io.BytesIO(datos)).paragraphs)
    if nombre.endswith(".txt"):
        try:
            return datos.decode("utf-8-sig")
        except UnicodeDecodeError:
            return datos.decode("latin-1")
    raise ValueError("Formato no admitido")


def crear_chunks(texto, max_chars=350):
    """Reproduce el chunker aproximado por puntos del notebook original."""
    partes = texto.split(".")
    resultado, actual = [], ""
    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue
        frase = parte + ". "
        if len(actual) + len(frase) < max_chars:
            actual += frase
        else:
            if actual.strip():
                resultado.append(actual.strip())
            actual = frase
    if actual.strip():
        resultado.append(actual.strip())
    return resultado


def limpiar_id(texto):
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_-]", "_", texto)).strip("_") or "documento"


@st.cache_resource(show_spinner=False)
def clientes(api_gemini, api_pinecone):
    return genai.Client(api_key=api_gemini), Pinecone(api_key=api_pinecone)


def obtener_index(pc):
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, dimension=DIMENSIONS, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    indice = pc.Index(INDEX_NAME)
    return indice


def indexar_archivos(archivos, gemini, indice, namespace, barra):
    total, fallos, detalle = 0, 0, []
    total_archivos = len(archivos)
    for numero, archivo in enumerate(archivos, 1):
        try:
            texto = extraer_texto(archivo)
            if not texto.strip():
                detalle.append(f"⚠️ {archivo.name}: no se extrajo texto (PDF escaneado requiere OCR).")
                continue
            chunks = crear_chunks(texto)
            if not chunks:
                detalle.append(f"⚠️ {archivo.name}: no hay fragmentos que indexar.")
                continue
            lote = []
            doc_id = limpiar_id(archivo.name) + "-" + uuid.uuid4().hex[:8]
            for i, chunk in enumerate(chunks):
                resultado = gemini.models.embed_content(model=EMBED_MODEL, contents=chunk)
                lote.append({
                    "id": f"{doc_id}-{i}",
                    "values": resultado.embeddings[0].values,
                    "metadata": {"texto": chunk, "doc": archivo.name, "chunk": i},
                })
                if len(lote) == 25:
                    indice.upsert(vectors=lote, namespace=namespace)
                    total += len(lote)
                    lote = []
            if lote:
                indice.upsert(vectors=lote, namespace=namespace)
                total += len(lote)
            detalle.append(f"✅ {archivo.name}: {len(chunks)} fragmentos procesados.")
        except Exception as e:
            fallos += 1
            detalle.append(f"❌ {archivo.name}: {str(e)}")
        barra.progress(numero / total_archivos)
    return total, fallos, detalle


def buscar(indice, gemini, pregunta, namespace, k=4):
    embedding = gemini.models.embed_content(model=EMBED_MODEL, contents=pregunta)
    consulta = indice.query(
        vector=embedding.embeddings[0].values,
        top_k=k, include_metadata=True, namespace=namespace
    )
    encontrados = []
    for item in consulta.matches:
        meta = item.metadata or {}
        if meta.get("texto"):
            encontrados.append({
                "documento": meta.get("doc", "Sin nombre"),
                "fragmento": meta["texto"],
                "similitud": float(item.score) if item.score is not None else None,
            })
    return encontrados


def responder(gemini, pregunta, fragmentos):
    contexto = "\n\n".join(x["fragmento"] for x in fragmentos)
    if not contexto.strip():
        return "No encontré información en los documentos cargados para responder esa pregunta."
    prompt = f"""Usa EXCLUSIVAMENTE el siguiente contexto para responder.
CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}\n\nINSTRUCCIONES:
- No inventes información.
- No agregues detalles que no estén en el contexto.
- Responde de forma clara y directa.
- Si la respuesta no está en el contexto, dilo expresamente."""
    respuesta = gemini.models.generate_content(
        model=CHAT_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3)
    )
    return respuesta.text or "Gemini no devolvió texto."


if "namespace" not in st.session_state:
    st.session_state.namespace = "alumno-" + uuid.uuid4().hex[:20]
if "historial" not in st.session_state:
    st.session_state.historial = []
if "fragmentos_total" not in st.session_state:
    st.session_state.fragmentos_total = 0

st.title("🧠 Laboratorio RAG con IA")
st.caption("Práctica educativa · Gemini + Pinecone · Documentos PDF, Word y TXT")

try:
    clave_gemini = st.secrets["GEMINI_API_KEY"]
    clave_pinecone = st.secrets["PINECONE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("Faltan GEMINI_API_KEY y PINECONE_API_KEY en los secretos de Streamlit Cloud.")
    st.stop()

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

subir, conversar = st.tabs(["📄 1. Cargar documentos", "💬 2. Consultar con IA"])
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
