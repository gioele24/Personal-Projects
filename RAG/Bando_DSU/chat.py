import os
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# Directory dove Chroma salva il database vettoriale
DB_DIR = "db"


# ============================================================
# 1. LLM + EMBEDDINGS + DATABASE VETTORIALE
# ============================================================

# Modello LLM usato per riscrivere la query (query expansion)
llm_rewrite = ChatOllama(
    model="gemma3:4b",
    temperature=0.4,  # un po' di creatività per generare varianti utili
)

# LLM finale per generare la risposta
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.0,  # massima precisione
)

# Modello di embedding locale (Ollama) per convertire testi in vettori
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

# Carica il database Chroma persistente
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

# Retriever semantico (vector search)
retriever_semantic = vectordb.as_retriever(search_kwargs={"k": 10})



# ============================================================
# 2. RECUPERO DI TUTTI I DOCUMENTI (PER BM25)
# ============================================================

def load_all_docs():
    """Recupera tutti i documenti da Chroma e li ricostruisce come Document."""
    results = vectordb.get(include=["documents", "metadatas"])
    docs = results["documents"]
    metas = results["metadatas"]

    # Ricostruisce oggetti Document
    return [
        Document(page_content=docs[i], metadata=metas[i])
        for i in range(len(docs))
    ]

# Carica tutti i documenti in memoria
all_docs = load_all_docs()



# ============================================================
# 3. BM25 KEYWORD SEARCH
# ============================================================

# BM25 è un retriever basato su parole chiave
bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 20  # restituisce fino a 20 documenti



# ============================================================
# 4. CROSS-ENCODER RERANKER
# ============================================================

# Modello supervisionato che valuta coppie (query, documento)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



# ============================================================
# 5. QUERY REWRITE (3 VARIANTI)
# ============================================================

def expand_query(question: str):
    """Genera 3 riscritture della query per aumentare la copertura della ricerca."""
    prompt = f"""
Riscrivi la seguente domanda in 3 modi diversi:
1) una versione più lunga e dettagliata
2) una versione con sinonimi
3) una versione con termini tecnici

Domanda originale:
{question}

Scrivi SOLO le 3 riscritture, una per riga.
"""
    rewritten = llm_rewrite.invoke(prompt).content.strip().split("\n")
    return ["0) " + question] + ([q.strip() for q in rewritten if q.strip()])

# ============================================================
# 6.0 RECIPROCAL RANK FUNCTION
# ============================================================

def rrf_fusion(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc.page_content
            score = 1 / (k + rank + 1)
            if key not in scores:
                scores[key] = [0, doc]
            scores[key][0] += score
    return [d for _, d in sorted(scores.values(), key=lambda x: x[0], reverse=True)]




# ============================================================
# 6.1 HYBRID SEARCH (SEMANTICA + BM25)
# ============================================================


def hybrid_search(queries, top_k=20):
    all_results = []

    for q in queries:
        sem = retriever_semantic.invoke(q)
        bm = bm25.invoke(q)
        fused = rrf_fusion([sem, bm])
        all_results.extend(fused)

    # deduplica mantenendo l'ordine
    seen = set()
    unique_docs = []
    for d in all_results:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)

    return unique_docs[:top_k]



# ============================================================
# 7. RERANKING CON CROSS-ENCODER
# ============================================================

def rerank(query, docs, top_k=5):
    """Ordina i documenti usando un CrossEncoder molto preciso."""
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), reverse=True)
    return [d for _, d in ranked[:top_k]]



# ============================================================
# 8. PROMPT FINALE PER L'LLM
# ============================================================

template = """
Sei un assistente che risponde alle domande sul bando di borsa di studio universitario.
Usa SOLO le informazioni nel CONTENUTO.
Se non trovi la risposta, dì che non è specificato nel bando.

CONTENUTO:
{context}

DOMANDA:
{question}

Rispondi in italiano chiaro e preciso.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)



# ============================================================
# 9. PIPELINE COMPLETA DI RISPOSTA
# ============================================================

def answer_question(question: str):
    # 1. Riscrittura della query
    expanded_queries = expand_query(question)

    # 2. Hybrid search
    candidates = hybrid_search(expanded_queries, top_k=20)

    # 3. Reranking
    top_docs = rerank(question, candidates, top_k=5)

    # 4. Costruzione del contesto per il prompt finale
    context = "\n\n".join(
        f"[Pagina {d.metadata.get('page','N/D')}] {d.page_content}"
        for d in top_docs
    )

    # 5. LLM finale per generare la risposta
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)

    return response.content, top_docs



# ============================================================
# 10. CLI CHAT LOOP
# ============================================================

if __name__ == "__main__":
    print("Chatbot Borsa di Studio DSU Toscana (digita 'exit' per uscire)\n")
    while True:
        q = input("Domanda: ")
        if q.lower() in ["exit", "quit"]:
            break

        answer, docs = answer_question(q)

        print("\nRisposta:\n")
        print(answer)

        print("\nEvidenza dal bando:\n")
        for d in docs:
            page = d.metadata.get("page", "N/D")
            source = d.metadata.get("source", "N/D")
            snippet = d.page_content.replace("\n", " ")[:200]
            print(f"- [Pagina {page}, Fonte: {source}] {snippet}...")

        print("\n" + "-"*50 + "\n")


