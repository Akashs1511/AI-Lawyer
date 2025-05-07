
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate

# Uncomment the following if you're NOT using pipenv
#from dotenv import load_dotenv
#load_dotenv()

#Step1: Setup LLM (Use DeepSeek R1 with Groq)
llm_model=ChatGroq(model="deepseek-r1-distill-llama-70b")

#Step2: Retrieve Docs

def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#Step3: Answer Question

custom_prompt_template = """

Use only the following context to answer the user's question.
Do not repeat the question or include any system messages.
If the answer is not in the context, say "I don't know."
Dont provide anything out of the given context.
Context:
{context}

Question:
{question}

Answer in a concise and clear manner:
"""


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    output = chain.invoke({"question": query, "context": context})
    # Assuming the final response is the last line of the output
    response = output.content
    return response
# question="If a government forbids the right to assemble peacefully which articles are violated and why?"
# retrieved_docs=retrieve_docs(question)
# print("AI Lawyer: ",answer_query(documents=retrieved_docs, model=llm_model, query=question))