import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
INDEX_NAME = os.getenv('INDEX_NAME')

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    
def check_credentials(username, password):
    correct_password = os.getenv('USER_PASSWORD')
    return username == "luccid" and password == correct_password

def display_login_form():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        if login_button:
            if check_credentials(username, password):
                st.session_state['logged_in'] = True
                st.success("Logged in successfully.")
                # Using st.experimental_rerun() to force the app to rerun might help, but use it judiciously.
                st.rerun()
            else:
                st.error("Incorrect username or password.")

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = INDEX_NAME

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{index_name}' does not exist. Please run the indexing script first.")

index = pc.Index(index_name)

vector_store = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY,
                                   index_name=INDEX_NAME,
                                   embedding=embeddings_model)


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

prompt_template = """
Koristi trenutni kontekst da odgovoriš na pitanje.
Ako ne znaš odgovor, reci da ne znaš. U tom slučaju odgovor mora da bude NE ZNAM.
Nemoj da izmišljaš odgovor i za odgovaranje koristi samo kontekst koji ti je proslijeđen.
Svi odgovori moraju biti na srpskom jeziku.
Kontekst:
{context}

Pitanje: {question}
Answer:"""

def fetch_all_chunks(article_id, title):
    metadata_filter = {
        "chapter": {"$eq": article_id},
        "title": {"$eq": title}
    }
    
    all_chunks = []
    next_page_token = None

    index_stats = index.describe_index_stats()
    vector_dim = index_stats.dimension

    dummy_vector = [0.0] * vector_dim

    while True:
        query_response = index.query(
            vector=dummy_vector,  
            filter=metadata_filter,
            top_k=10,  
            include_metadata=True,
            namespace="",  
        )

        all_chunks.extend(query_response['matches'])
        
        if 'next_page_token' not in query_response or not query_response['next_page_token']:
            break

        next_page_token = query_response['next_page_token']

    return all_chunks





def display_main_app():
    st.title('LUCCID AI Bot')
    query = st.text_input("Unesite pitanje:")

    if st.button("Pošalji upit"):
        with st.spinner("Obrada..."):
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            try:
                result = qa_chain({"query": query})
                st.success("Upit je uspješno izvršen.")
                st.subheader("Odgovor:")
                st.write(result['result'])
                st.subheader("Izvor odgovora:")
                st.write(result['source_documents'])
            except Exception as e:
                st.error(f"Desila se greška: {str(e)}")
                st.error(f"Detalji greške: {e.__class__.__name__}")
                return

            if 'result' in locals():
                result_embedding = embeddings_model.embed_query(result['result'])

                best_doc = None
                max_similarity = -1

                for doc in result['source_documents']:
                    doc_embedding = embeddings_model.embed_query(doc.page_content)
                    similarity = cosine_similarity([result_embedding], [doc_embedding])[0][0]
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_doc = doc

                if best_doc:
                    metadata_info = {
                        "Naziv": best_doc.metadata.get("title", "N/A"),
                        "Član": best_doc.metadata.get("chapter", "N/A")
                    }
                else:
                    metadata_info = {"title": "N/A", "chapter": "N/A"}

                final_response = {
                    "answer": result['result'],
                    "metadata": metadata_info
                }

                chunks = fetch_all_chunks(final_response['metadata']['Član'], final_response['metadata']['Naziv'])
                chunks.sort(key=lambda x: x['id'])
                whole_article = "\n".join(map(lambda x: x['metadata']['text'], chunks))

                st.subheader("Konačan odgovor:")
                st.json(final_response)
                st.subheader("Cijeli članak:")
                st.write(whole_article)
            else:
                st.warning("Nema rezultata za prikaz.")



if not st.session_state['logged_in']:
    display_login_form()
else:
    display_main_app()