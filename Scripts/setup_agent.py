import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain_community.llms import Ollama
from time import time

s_time = time()
llm = Ollama(model="llama2")
print("Model Loaded")

ori_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
test_url = "https://pandas.pydata.org/docs/user_guide/10min.html#basic-data-structures-in-pandas"

### Construct retriever ###
loader = WebBaseLoader(
    web_paths=(test_url,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("bd-article") # Change according to requirements and dependency of web-page
        )
    ),
)

docs = loader.load()
print(f"Document Loaded, Characters {len(docs[0].page_content)}")

embed_model = embeddings.OllamaEmbeddings(model='nomic-embed-text')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
retriever = vectorstore.as_retriever()
print("Vectorstore Initialised")

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
print("History Line Initialised")

### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
print("RAG Chain Initialised")


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
print("Ynjoy!!")

i_time = time()
response = conversational_rag_chain.invoke(
    {"input": "Give the code to view data."},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]

print(response)
print(f"Your answer took: {time() - s_time} secs")
print(f"Inference took: {time() - i_time} secs")