import streamlit as st
from dotenv import load_dotenv
from rag_ejemplo import ask, ingest

load_dotenv()

st.set_page_config(
    page_title="EcoMarket RAG",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Asistente Inteligente EcoMarket")
st.write("Sistema RAG para atención al cliente basado en documentos internos.")

with st.sidebar:
    st.header("Configuración")
    if st.button("Indexar base de conocimiento"):
        with st.spinner("Creando índice vectorial..."):
            try:
                ingest()
                st.success("Índice creado correctamente.")
            except Exception as e:
                st.error(f"Error al indexar: {e}")

st.subheader("Consulta del cliente")

pregunta = st.text_area(
    "Escribe una pregunta:",
    placeholder="Ejemplo: ¿Cuál es el estado del pedido ORD-00001?"
)

if st.button("Enviar consulta"):
    if not pregunta.strip():
        st.warning("Por favor escribe una pregunta.")
    else:
        with st.spinner("Consultando la base de conocimiento..."):
            try:
                respuesta = ask(pregunta)
                st.success("Respuesta generada:")
                st.write(respuesta)
            except Exception as e:
                st.error(f"Error: {e}")