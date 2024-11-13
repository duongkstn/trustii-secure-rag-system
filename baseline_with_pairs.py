from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()



def processing_df(df):
    df.fillna("", inplace=True)
    return df
df_train = processing_df(pd.read_csv("data/train.csv", dtype=str))
df_test = processing_df(pd.read_csv("data/test.csv", dtype=str))


texts = df_train["Response"].values.tolist()

embeddings = OllamaEmbeddings(model="all-minilm:l6-v2", base_url="localhost:11434")

# RESPONSE_FORMAT = """Context: {text}"""
PAIR_FORMAT = """Question: {response}\nAnswer: {query}"""

# docs = [Document(page_content=RESPONSE_FORMAT.format(text=text), metadata={"info": "response"}) for text in df_train["Response"].values.tolist()]
docs = [Document(page_content=PAIR_FORMAT.format(response=response, query=query),
                 metadata={"info": "pair"}
                 ) for query, response in zip(df_train["Query"].values.tolist(), df_train["Response"].values.tolist())
        ]
qdrant = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    collection_name="pair_collection",
    path="./saved_dir_db_qdrant"
)


# client = QdrantClient(path="./saved_dir_db_qdrant")
# qdrant = QdrantVectorStore(client=client, collection_name="pair_collection", embedding=embeddings)

PROMPT_TEMPLATE = """
Answer the question using only the following questions and answers:
{context}
-------------------------------------------------------------
Answer this question based on the questions and answers above, try to answer correctly and have similar style to above answers: 
Question: {question}
Answer:  """

model = OllamaLLM(model="qwen2.5", base_url="localhost:11434")

def answer(query):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    search_results = qdrant.similarity_search_with_relevance_scores(query, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in search_results])
    prompt = prompt_template.format(context=context, question=query)

    response = model.predict(prompt)

    print('response')
    print(response)
    return response

if __name__ == "__main__":
    df_test["Response"] = df_test["Query"].progress_apply(lambda x: answer(x))
    df_test.to_csv("submission/submission_pair.csv", index=False)


