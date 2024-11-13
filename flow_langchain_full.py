import faiss
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
import langchain_community
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from config import OLLAMA_URL
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def processing_df(df):
    """
    Processing dataframe
    :param df: Pandas dataframe
    :return: processed dataframe
    """
    df.fillna("", inplace=True)
    return df
df_train = processing_df(pd.read_csv("data/train.csv", dtype=str))
df_test = processing_df(pd.read_csv("data/test.csv", dtype=str))

df_train.drop_duplicates(inplace=True)


texts = df_train["Response"].values.tolist()
# Declare embedding model
embeddings = OllamaEmbeddings(model="all-minilm:l6-v2", base_url=OLLAMA_URL)

PAIR_FORMAT = """{question} {answer}"""

# Create LangChain Document from training data
docs = [Document(page_content=PAIR_FORMAT.format(question=question, answer=answer),
                  metadata={"info": "pair", "question": question, "answer": answer}
                 ) for question, answer in zip(df_train["Query"].values.tolist(), df_train["Response"].values.tolist())
        ]

# Load dumped vector store
vector_store = FAISS.load_local(
    folder_path="./saved_dir_db_faiss",
    index_name="faiss_pair_collection",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(k=20)
model = OllamaLLM(model="qwen2.5", base_url=OLLAMA_URL)

PROMPT_TEMPLATE = """
Answer the question using only the following questions and answers:
{context}
-------------------------------------------------------------
Answer this question based on the questions and answers above, try to answer correctly and have similar style to above answers:
If you can not found information which can be used to answer, answer is "NOT_FOUND_ANSWER".
Question: {question}
Answer: """


PAIR_METADATA_FORMAT = \
"""
Question: {question},
Answer: {answer},
"""

def format_docs(docs):
    """
    Feed retrieved documents to prompt before inputting to LLM
    :param docs: list LagnChain documents
    :return: str
    """
    def doc_to_text(doc):
        return PAIR_METADATA_FORMAT.format(question=doc.metadata["question"], answer=doc.metadata["answer"])
    return "\n---\n".join([doc_to_text(doc) for doc in docs])

# Prompt template for LLM as QA Assistant
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def identity(x):
    return x

# Retriever Chain
retriever_chain = (
    dict(context=retriever | format_docs, question=RunnablePassthrough())
    | RunnableLambda(identity)
)

# QA Chain
rag_chain: RunnableSerializable[str, str] = (
    retriever_chain
    | prompt_template
    | model
    | StrOutputParser()
)

EXPERT_PROMPT_TEMPLATE = """
Answer the question using only the following questions and answers:
{context}
-------------------------------------------------------------
Try to answer correctly and have similar style to above answers:
You a technology expert, make up answer as best as you can. DO NOT EXPLAIN ANYTHING ELSE
Question: {question}
Answer: """

# Prompt template for LLM as Tech expert
prompt_expert_template = ChatPromptTemplate.from_template(EXPERT_PROMPT_TEMPLATE)

# Tech Expert Chain
expert_chain = (
    retriever_chain
    | prompt_expert_template
    | model
    | StrOutputParser()
)

def inference(question: str) -> str:
    """
    Full flow for inference
    :param question: input question
    :return: output result
    """
    rag_result = rag_chain.invoke(question)
    if "not_found_answer" in rag_result.lower():
        return expert_chain.invoke(question)
    return rag_result

if __name__ == "__main__":
    df_test["Response"] = df_test["Query"].progress_apply(lambda x: inference(x))
    df_test.to_csv("submission/submission_langchain_full.csv", index=False)