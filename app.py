import os
import datetime
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()); #para leer y cargar la api key del archivo .env

current_date = datetime.datetime.now().date()

llm_model = "gpt-3.5-turbo-0301" 

from langchain_classic.chains import create_retrieval_chain; # crea la estructura principal. une el buscador de documentos con el modelo de IA
from langchain_classic.chains.combine_documents import create_stuff_documents_chain;#prepara los documentos que el sistema encontró para ponerlos en el prompt que se le va a enviar al modelo
from langchain_community.document_loaders import PyPDFLoader; # para cargar el archivo PDF
from langchain_community.vectorstores import DocArrayInMemorySearch;#guarda los textos convertidos a vectores directamente en la memoria RAM de la compu 
from langchain_core.prompts import ChatPromptTemplate; #permite armar la estructura fija del mensaje que va a recibir la IA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings;# el encargado de leer la info y redactar la respuesta final (chatopenai) y el modelo que convierte los vectores en datos (OpenAIEmbedding)

#cargo el aarchivo PDF
file_path = "data.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

#Creo los embeddings y los guardo en la base de datos de la memoria
embeddings= OpenAIEmbeddings()
vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings= embeddings)

