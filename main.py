import getpass
import os
from helpers import *
import pandas as pd
import numpy as np
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings('ignore')
# load_dotenv()
GEMINI_API_KEY = 'AIzaSyAETnBDnIVlekpZPKw7t9cUisqqIX93sH8'



if __name__ == "__main__":
    main()

df = pd.read_csv("datasets/Loan_default.csv")
for column in df.columns:
  if df[column].isnull().sum() > 0 and not df[column].mode().empty:
    df[column].fillna(df[column].mode().iloc[0], inplace=True)
df = df.iloc[:10,:]
# Define the function to create a prompt

Loan_ID='C1OZ6DPJ8Y'
question = generate_prompt_by_loanid(Loan_ID,df)

documents = []

# Iterate over rows using .rows() method
for i, row_tuple in df.iterrows():
    document = f"id:{i}\ LoanID: {row_tuple.iloc[0]}\ Credit Score: {row_tuple.iloc[4]}\ Debt-to-Income Ratio:{row_tuple.iloc[9]}"
    documents.append(Document(page_content=document))
# display(documents)

llm = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=GEMINI_API_KEY, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key = GEMINI_API_KEY)

persist_directory = 'docs/chroma_rag/'

langchain_chroma = Chroma.from_documents(
    documents=documents,
    collection_name="default_prediction",
    embedding=embeddings,
    persist_directory=persist_directory
)
""""
Financial coefficients required to calculate PD: 
These values may vary according to the FInancial Lending organisation
Intercept = b0
Credit Score = b1
Debt-to-income ratio = b2

PD = 1/(1+e^(-z))
where,The linear combination of borrower characteristics and their corresponding coefficients
z = b0+b1.x1+b2.x2+....+bn.xn 
=>b0+b1*[credit Score]+b2*[Debt-to-income-ratio]
"""
b0 = -1.5
b1 = -0.005
b2 = 0.07

# Define the prompt template
template = """
You are an Credit Risk Expert in Financial Text Data, Analyse the question and get the context and Answer the following:
1.**Instruction :**
    - Predict if the Given Customer is going to Default or not by calculating Default using formula below and consider the return values as flag.
2. **Analysis Criteria:**
   - Assess overall creditworthiness based on payment patterns and credit utilization.
3. **Output Requirements:**
   - If the customer has no defaults:
     - Respond with: "The loan associated with Loan ID [Insert Loan ID] has no recorded defaults. It is safe to proceed with any further actions related to this loan."
   - If there are potential risks (e.g., late payments, high credit utilization):
     - Respond with: "The loan associated with Loan ID [Insert Loan ID] has recorded defaults. This poses a risk for further lending activities and may require immediate attention to mitigate potential financial repercussions"
   - Provide any additional recommendations or next steps if necessary.
4. **Tone:** 
   - Professional, concise, and informative.
5. *Response*
   - The loan associated with Loan ID and may 
  require immediate attention to mitigate potential financial repercussions. 
  The customer has a (high/low) credit score  and a relatively (high/low) debt-to-income ratio. 
  These factors increase the likelihood of default.

Question: {question}
Context: {context}

z = b0+b1*[Inser Credit Score]+b2*[Insert Debt-to-income-ratio]
Formula for Default = 1/(1+e^(-z))
Answer: 
"""

PROMPT = PromptTemplate(input_variables=["context", "query","b0","b1","b2"], template=template)

# Ensure llm and langchain_chroma are properly initialized
context = langchain_chroma.as_retriever(search_kwargs={"k": 1})

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=context, chain_type_kwargs={"prompt": PROMPT}
)
# LonaID=input("Enter LoanID")
# question = data_input
# print(question)


# Run the QA chain
try:
    #result = qa_chain.invoke({"query": question})
    result = qa_chain.invoke(question)
    print(result.get('result'))
except Exception as e:
    print(f"Error encountered: {e}")








