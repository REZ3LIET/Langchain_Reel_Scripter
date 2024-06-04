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

class AnalyticsRAG:
    def __init__(self, url):
        llm = Ollama(model="llama2")
        print("Model Loaded")

        retriever = self.data_loader(url)
        contextualize_q_prompt = self.get_context_prompt()
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        print("History Line Initialised")

        system_prompt = self.get_system_prompt()
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

        self.chat_history = {}

        self.rag_chat = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        print("Ynjoy!!")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_history:
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]

    def data_loader(self, data_url):
        """
        Loads data from source
        TODO: Integrate csv/json as data source
        """
        loader = WebBaseLoader(
        web_paths=(data_url,),
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
        return retriever

    def get_context_prompt(self):
        contextualize_q_system_prompt = (
            """
            Given a chat history and the latest user question
            which might reference context in the chat history,
            formulate a standalone question which can be understood
            without the chat history. Do NOT answer the question,
            just reformulate it if needed and otherwise return it as is.
            """
        )
        return contextualize_q_system_prompt

    def get_system_prompt(self):
        system_prompt = (
            """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer
            the question. If you don't know the answer, say that you
            don't know. Use three sentences maximum and keep the
            answer concise.

            {context}
            """
        )
        return system_prompt
    
    def chat_with_data(self, usr_prompt):
        response = self.rag_chat.invoke(
            {"input": usr_prompt},
                config={
                    "configurable": {"session_id": "analytics_chat"}
                }
        )["answer"]
        return response

def main():
    url = ""
    chat_agent = AnalyticsRAG(url)
    print("You are now conversing with an Analytics Agent. Good Luck!")
    while True:
        prompt = input("Enter you query|('/exit' to quit session): ")
        if prompt == "/exit":
            print("You will have a fortuious encounter soon, God Speed!")
            break
        response = chat_agent.chat_with_data(prompt)
        print(f"Analytics-Agent: {response}")

if __name__ == "__main__":
    main()
