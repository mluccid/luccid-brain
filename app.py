import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import requests
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
INDEX_NAME = os.getenv('INDEX_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False


def run_jina_reranker(
    documents: List[str],
    query: str,
    url:str = 'https://api.jina.ai/v1/rerank',
    model:str = "jina-reranker-v2-base-multilingual",
    top_n: int = 5
) -> dict:
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        raise ValueError("JINA_API_KEY is not set")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {jina_api_key}'
    }
    data = {
        "model": model,
        "query": query,
        "top_n": top_n,
        "documents": documents
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()
    
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

model = 'models/embedding-001'
embeddings = GoogleGenerativeAIEmbeddings(model=model)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = INDEX_NAME

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{index_name}' does not exist. Please run the indexing script first.")

index = pc.Index(index_name)

vector_store = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY,
                                   index_name=INDEX_NAME,
                                   embedding=embeddings)


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.01)

prompt_template = """
Odgovaraj na pitanja vezana za zakone i tehničke standarde u oblasti planiranja, projektovanja i izgradnje u Republici Srbiji. Sve informacije koje pružaš moraju biti zasnovane isključivo na vektorizovanoj bazi podataka članaka iz zakona i pravilnika, bez ikakvih pretpostavki ili dodavanja informacija van baze. Ako član nije dostupan, jasno to saopšti korisniku.

1. Referenciranje izvora i citiranje:
Svaki odgovor mora biti precizno potkrepljen: 'Prema članu X zakona Y' ili 'Pravilnik o [naziv pravilnika], član X'. Navedeni član i naziv moraju odgovarati tačnoj formulaciji u bazi.
U slučaju da se pronađe više članaka sa sličnim informacijama, uključite sve relevantne članke kako bi korisnik imao sveobuhvatan uvid.
Ako pitanje uključuje upit za određeni datum (npr. “Koji su važeći propisi za 2022. godinu?”), odgovori koristeći članke važeće na taj datum i naglasi ako su se propisi promenili.
2. Preciznost i jasnost odgovora:
Koristi tačne dimenzije, klasifikacione brojeve, kategorije objekata ili druge numeričke vrednosti iz članaka. Uvek naglasi iz kog članka i pravilnika su ove vrednosti preuzete, kako bi korisnik mogao da proveri autentičnost informacije.
Ako se zakon ili član izričito ne odnosi na postavljeno pitanje, reci: 'Za ovo pitanje nema direktno definisanih odredbi u članu X zakona Y'.
Ukoliko se pronađe delimično relevantan član, objasni korisniku u kojoj meri se član odnosi na njegovo pitanje i koji delovi su primenljivi.
3. Jezik i ton:
Jezik odgovora je srpski, prilagođen korisnicima bez pravnog znanja. Definiši složene pravne pojmove jednostavno, ali zadrži tačnost formulacija.
Koristi konkretne primere kada je to moguće, npr. ako se zakon odnosi na "izgrađene objekte u prvom stepenu zaštite", dodaj objašnjenje šta to podrazumeva u praksi.
4. Izbegavanje halucinacija i tačnost izvora:
Podaci i reference moraju dolaziti isključivo iz članaka u bazi: Nemoj dodavati zakone, članke ili pravne reference koje nisu stvarno prisutne u bazi. Svaki deo odgovora mora biti zasnovan na konkretnoj pravnoj osnovi iz baze.
Prilikom generisanja odgovora, proveri da li svaki citirani član i zakon stvarno postoje u bazi podataka. Ako član ne postoji, umesto generisanja, odgovori: 'Za ovo pitanje nema relevantnih podataka u dostupnoj bazi članaka.'
5. Specifični upiti i scenariji:
Klasifikacija objekata: Kada korisnik postavi pitanje o klasifikacionim brojevima i kategorijama objekata, odgovori moraju uključivati tačan klasifikacioni broj i kategoriju, sa objašnjenjem šta oni podrazumevaju.
Dimenzije i tehničke karakteristike: Za pitanja koja se odnose na dimenzije, kao što su minimalne dimenzije stepeništa, liftova ili rampi, uvek citiraj izvor dimenzija sa tačnim vrednostima i članom iz kog je informacija preuzeta. Ako postoje različiti standardi za različite tipove objekata, navedi sve relevantne standarde i njihove izvore.
Zabrana legalizacije objekata: Kada korisnik postavi pitanje o uslovima za legalizaciju ili objekatima koji se ne mogu legalizovati, pruži celokupnu listu uslova i tačaka iz relevantnih članaka, čak i ako nisu sve direktno pomenute u pitanju.
Nejasni ili nepotpuni upiti: Ako korisnički upit nije dovoljno specifičan, npr. 'Kolika je visina zgrade?', traži dodatne informacije pre nego što pružiš odgovor: 'Da li vas interesuje minimalna ili maksimalna dozvoljena visina prema zakonu o planiranju?'.
6. Fallback mehanizam:
Upozorenje o nedostatku informacija: Ako baza nema član koji odgovara na korisničko pitanje, odgovori sa: 'Informacije za postavljeno pitanje nisu dostupne u trenutnoj bazi članaka. Preporučujemo da se obratite pravnom stručnjaku za dodatne informacije.'
Jasno označi nesigurnost: Ako se informacije koje baza daje mogu tumačiti na više načina, dodaj upozorenje: 'Tumačenje ove informacije može zavisiti od specifičnog slučaja, preporučujemo dodatnu konsultaciju sa stručnjakom.'

Kontekst:
{context}

Pitanje: {question}
Answer:"""

prompt_gpt = """
Bazirano na poslatim dokumentima, izvuci odgovor na pitanje: {question}
Odgovaraj na pitanja vezana za zakone i tehničke standarde u oblasti planiranja, projektovanja i izgradnje u Republici Srbiji. Sve informacije koje pružaš moraju biti zasnovane isključivo na vektorizovanoj bazi podataka članaka iz zakona i pravilnika, bez ikakvih pretpostavki ili dodavanja informacija van baze. Ako član nije dostupan, jasno to saopšti korisniku.

1. Referenciranje izvora i citiranje:
Svaki odgovor mora biti precizno potkrepljen: 'Prema članu X zakona Y' ili 'Pravilnik o [naziv pravilnika], član X'. Navedeni član i naziv moraju odgovarati tačnoj formulaciji u bazi.
U slučaju da se pronađe više članaka sa sličnim informacijama, uključite sve relevantne članke kako bi korisnik imao sveobuhvatan uvid.
Ako pitanje uključuje upit za određeni datum (npr. “Koji su važeći propisi za 2022. godinu?”), odgovori koristeći članke važeće na taj datum i naglasi ako su se propisi promenili.
2. Preciznost i jasnost odgovora:
Koristi tačne dimenzije, klasifikacione brojeve, kategorije objekata ili druge numeričke vrednosti iz članaka. Uvek naglasi iz kog članka i pravilnika su ove vrednosti preuzete, kako bi korisnik mogao da proveri autentičnost informacije.
Ako se zakon ili član izričito ne odnosi na postavljeno pitanje, reci: 'Za ovo pitanje nema direktno definisanih odredbi u članu X zakona Y'.
Ukoliko se pronađe delimično relevantan član, objasni korisniku u kojoj meri se član odnosi na njegovo pitanje i koji delovi su primenljivi.
3. Jezik i ton:
Jezik odgovora je srpski, prilagođen korisnicima bez pravnog znanja. Definiši složene pravne pojmove jednostavno, ali zadrži tačnost formulacija.
Koristi konkretne primere kada je to moguće, npr. ako se zakon odnosi na "izgrađene objekte u prvom stepenu zaštite", dodaj objašnjenje šta to podrazumeva u praksi.
4. Izbegavanje halucinacija i tačnost izvora:
Podaci i reference moraju dolaziti isključivo iz članaka u bazi: Nemoj dodavati zakone, članke ili pravne reference koje nisu stvarno prisutne u bazi. Svaki deo odgovora mora biti zasnovan na konkretnoj pravnoj osnovi iz baze.
Prilikom generisanja odgovora, proveri da li svaki citirani član i zakon stvarno postoje u bazi podataka. Ako član ne postoji, umesto generisanja, odgovori: 'Za ovo pitanje nema relevantnih podataka u dostupnoj bazi članaka.'
5. Specifični upiti i scenariji:
Klasifikacija objekata: Kada korisnik postavi pitanje o klasifikacionim brojevima i kategorijama objekata, odgovori moraju uključivati tačan klasifikacioni broj i kategoriju, sa objašnjenjem šta oni podrazumevaju.
Dimenzije i tehničke karakteristike: Za pitanja koja se odnose na dimenzije, kao što su minimalne dimenzije stepeništa, liftova ili rampi, uvek citiraj izvor dimenzija sa tačnim vrednostima i članom iz kog je informacija preuzeta. Ako postoje različiti standardi za različite tipove objekata, navedi sve relevantne standarde i njihove izvore.
Zabrana legalizacije objekata: Kada korisnik postavi pitanje o uslovima za legalizaciju ili objekatima koji se ne mogu legalizovati, pruži celokupnu listu uslova i tačaka iz relevantnih članaka, čak i ako nisu sve direktno pomenute u pitanju.
Nejasni ili nepotpuni upiti: Ako korisnički upit nije dovoljno specifičan, npr. 'Kolika je visina zgrade?', traži dodatne informacije pre nego što pružiš odgovor: 'Da li vas interesuje minimalna ili maksimalna dozvoljena visina prema zakonu o planiranju?'.
6. Fallback mehanizam:
Upozorenje o nedostatku informacija: Ako baza nema član koji odgovara na korisničko pitanje, odgovori sa: 'Informacije za postavljeno pitanje nisu dostupne u trenutnoj bazi članaka. Preporučujemo da se obratite pravnom stručnjaku za dodatne informacije.'
Jasno označi nesigurnost: Ako se informacije koje baza daje mogu tumačiti na više načina, dodaj upozorenje: 'Tumačenje ove informacije može zavisiti od specifičnog slučaja, preporučujemo dodatnu konsultaciju sa stručnjakom.'
Dokumenti:
{context}
"""

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
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            try:
                result = qa_chain({"query": query})
                st.success("Upit je uspješno izvršen.")
                st.subheader("Odgovor:")
                if (result['result'].strip()=='NE ZNAM'):
                    PROMPTGPT = PromptTemplate(
                        template=prompt_gpt, input_variables=["context", "question"]
                    )
                    llm2 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm2,
                        chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PROMPTGPT}
                    )
                    
                    result = qa_chain({"query": query})

                if (result['result'].strip()=='NE ZNAM'):
                    top_30_docs, reranker_docs = [], []
                    retriever = vector_store.as_retriever(search_kwargs={"k": 30})
                    retrieved_docs = retriever.get_relevant_documents(query) 
                    for doc in retrieved_docs:
                        top_30_docs.append(doc.page_content)

                    reranker_results = run_jina_reranker(top_30_docs, query)
                    for reranker_result in reranker_results['results']:
                        reranker_docs.append(reranker_result['document']['text'])

                    context = "\n\n".join([doc for doc in reranker_docs])

                    # Create the prompt using the context and the user query
                    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                    # Create an LLMChain to send the query and context to the LLM
                    qa_chain = LLMChain(llm=llm, prompt=prompt)

                    # Run the query through the LLM with the reranked context
                    result['result'] = qa_chain.run({"context": context, "question": query})
                
                st.write(result['result'])
                st.subheader("Izvor odgovora:")
                st.write(result['source_documents'])
            except Exception as e:
                st.error(f"Desila se greška: {str(e)}")
                st.error(f"Detalji greške: {e.__class__.__name__}")
                return

            if 'result' in locals():
                result_embedding = embeddings.embed_query(result['result'])

                best_doc = None
                max_similarity = -1

                for doc in result['source_documents']:
                    doc_embedding = embeddings.embed_query(doc.page_content)
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
