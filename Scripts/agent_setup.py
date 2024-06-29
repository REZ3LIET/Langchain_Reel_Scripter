"""
Interacts with user to understand audience and content type
and converts the information to json.
"""

import json
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama

class SetupAgent:
    def __init__(self, url):
        llm = Ollama(model="llama2")
        print("Model Loaded")
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.chat_history = {}

        q_chain = qa_prompt | llm
        self.chat_model = RunnableWithMessageHistory(
            q_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        print("Ynjoy!!")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.chat_history:
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]

    def reformat_json(self, json_string):
        data = json.loads(json_string)
        formatted_messages = []
        for message in data['messages']:
            if message['type'] == 'human':
                user_content = message['content']
                ai_response = next((msg['content'] for msg in data['messages'] if msg['type'] == 'ai'), None)
                if ai_response:
                    formatted_messages.append({
                        "user_query": user_content,
                        "ai_response": ai_response
                    })

        # Creating the desired JSON structure
        formatted_data = {
            "messages": formatted_messages
        }
        # Convert Python dictionary back to JSON string
        with open("./agent_setup.json", 'w') as f:
            json.dump(formatted_data, f, indent=2)
        return formatted_data

    def display_history(self):
        history = self.chat_history["acc_setup"].json()
        json = self.reformat_json(history)
        print(history)
        print()
        print(json)

    def get_system_prompt(self):
        system_prompt = (
            """
            You are an account setup assistant. You have to ask various
            questions to user about their audience, type of content they
            want, description of content and other neccessary questions
            whose responses are required for content generation.

            The type of content you will focus on is Reels hosted by
            Instagram.

            Your job is to just understand the general requirments and
            the audience of the user and summarize his requirements at last
            as [Summary:]. These requirements will later be utilized to create
            reels.
            """
        )
        return system_prompt
    
    def agent_chat(self, usr_prompt):
        response = self.chat_model.invoke(
            {"input": usr_prompt},
                config={
                    "configurable": {"session_id": "acc_setup"}
                }
        )
        return response

def main():
    url = ""
    chat_agent = SetupAgent(url)
    print("You are now conversing with an Setup Agent. Good Luck!")
    while True:
        prompt = input("Enter you query|('/exit' to quit session): ")
        if prompt == "/exit":
            print("You will have a fortuitous encounter soon, God Speed!")
            break
        response = chat_agent.agent_chat(prompt)
        print("-"*30)
        print(f"Setup-Agent: {response}")
    chat_agent.display_history()

if __name__ == "__main__":
    main()
