from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool 
import subprocess 
from langchain_core.runnables import Runnable 
import os
import numpy as np 
import plotly.graph_objects as go

WOLFRAM = r"...\wolframscript.exe"

def start_session():
    with open('log.txt', 'w', encoding='utf-8') as log:
        log.write("Nuova sessione\n")

llm = ChatOllama(model="llama3.1:8b", temperature=0)

with open("sys_prompt.txt", "r", encoding="utf-8") as f:
    sys_prompt = f.read()


@tool
def wolfram_tool(query: str) -> str:
    """Tool used to call Wolfram Mathematica"""
    try:
        result = subprocess.run(
            [WOLFRAM, "-code", query],
            capture_output=True,
            text=True
        )
    except:
        with open('log.txt', 'a', encoding='utf-8') as log:
            print("Errore nella chiamata a 'wolfram_tool'", file = log)

    with open('log.txt', 'a', encoding='utf-8') as log:
        print("="*50, file = log)
        print("Per rispondere alla domanda dell'utente userò Wolfram Mathematica.", file=log)
        print("Procedo con la chiamata a Wolfram Mathematica...", file=log)
        print(file=log)
        print('Result: ', result, file=log)

    return result.stdout.strip()

PLOT_DIR = "plots"  # cartella dove salvare i file HTML

# crea la cartella se non esiste
os.makedirs(PLOT_DIR, exist_ok=True)

@tool
def plotly_tool(expr: str, t_min: float, t_max: float, filename: str = "plot.html") -> str:
    """
    Tool used to generate plots 
    """
    # forza l'estensione .html 
    if not filename.endswith(".html"): 
        filename += ".html"

    t = np.linspace(t_min, t_max, 500)


    try:
        y = eval(expr)
    except Exception as e:
        return f"Errore nell'espressione Python: {e}"
    
    with open('log.txt', 'a', encoding='utf-8') as log:
        print("="*50, file = log)
        print(f"Per generare il plot richiesto ({expr}) userò il tool 'plotly_plot'", file=log)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines"))

    filepath = os.path.join(PLOT_DIR, filename)
    fig.write_html(filepath)

    return f"Grafico interattivo salvato in: {os.path.abspath(filepath)}"



agent = create_agent(model=llm, 
                     tools=[wolfram_tool, plotly_tool], 
                     system_prompt=sys_prompt)



class StatefulAgent(Runnable):
    def __init__(self, agent):
        self.agent = agent
        self.state = {"messages": []}

    def invoke(self, input, config=None, **kwargs):
        # aggiungi il messaggio dell’utente allo stato
        self.state["messages"].append({"role": "user", "content": input})

        # invoca l’agente con lo stato completo
        result = self.agent.invoke(self.state)

        # aggiorna SOLO i messaggi nello stato
        self.state["messages"] = result["messages"]

        # ritorna il contenuto dell’ultimo messaggio AI
        return result["messages"][-1].content

print("Questo è un chatbot specializzato nella manipolazione matematica. Puoi interrogarlo con domande matematiche complesse.")
print("Scrivi 'exit' per uscire.\n")

start_session()
session = StatefulAgent(agent)

while True:
    domanda = input("> ")

    if domanda.lower() in ["exit", "quit"]:
        break

    risposta = session.invoke(domanda)
    print(risposta)
    print()



