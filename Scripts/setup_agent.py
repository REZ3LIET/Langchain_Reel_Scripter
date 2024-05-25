from langchain_community.llms import Ollama

llm = Ollama(model="tinyllama")

result = llm.invoke("What types of reels are there now on instagram based on types of video content")
print(result)