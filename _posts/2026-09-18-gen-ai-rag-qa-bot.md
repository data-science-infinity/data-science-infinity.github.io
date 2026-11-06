---
layout: post
title: Building an AI Help-Desk Assistant Using RAG (Retrieval Augmented Generation)
image: "/posts/gen-ai-rag-title-img.png"
tags: [GenAI, RAG, LLMs, Python, LangChain]
---

In this project we build a real, production-style AI assistant for our grocery retail client, capable of answering customer help-desk questions using **Retrieval Augmented Generation (RAG)**.  

We begin by building a core RAG system that loads internal documents, chunks them intelligently, embeds them into a vector database, retrieves relevant content, and generates grounded answers.  

We then extend the assistant by **adding conversational memory**, allowing the model to maintain a short-term personalised dialogue while still respecting strict grounding rules.

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. RAG Overview](#rag-overview)
- [03. Building the Core RAG System](#rag-core)
    - [Secure API Handling](#rag-api)
    - [Document Loading](#rag-docs)
    - [Document Chunking](#rag-chunking)
    - [Embeddings & Vector Store](#rag-embeddings)
    - [LLM Setup](#rag-llm)
    - [Prompt Template](#rag-prompt)
    - [Retriever Setup](#rag-retriever)
    - [Full RAG Pipeline](#rag-pipeline)
- [04. Enhancing the Assistant With Memory](#rag-memory)
- [05. Application & Examples](#rag-application)
- [06. Inspecting the Retrieved Context](#rag-inspection)
- [07. Growth & Next Steps](#growth-next-steps)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, a grocery retailers, operates a busy customer help-desk, answering queries around store hours, product availability, delivery services, loyalty cards, payments, and general store operations.

They need an **AI assistant** that can answer these questions accurately, consistently, and safely, using only approved internal information.

### Actions <a name="overview-actions"></a>

We built a full end-to-end RAG system that:

* Loaded internal help-desk documentation  
* Split it into meaningful chunks  
* Created dense vector embeddings  
* Stored these embeddings in a persistent vector database  
* Retrieved only the most relevant content at query time  
* Generated answers grounded strictly in this retrieved context  

We also extended the project with **conversational memory**, enabling more natural multi-turn interactions while ensuring the assistant never hallucinates.

Internally, we also added monitoring, tracing, and evaluation using LangSmith during development.

### Results <a name="overview-results"></a>

The final assistant:

* Reliably answers customer help-desk questions  
* Grounds every answer in retrieved internal documentation  
* Rejects unsupported questions with a safe fallback message  
* Maintains short-term conversational history for better UX  
* Prevents hallucinations using strict grounding rules  

### Growth/Next Steps <a name="overview-growth"></a>

Potential future enhancements include:

* Ingestion of multiple document types (PDFs, product catalogues)  
* Adding tool use such as SQL lookups for live stock, prices, or loyalty data  
* Adding a real chat interface (frontend + backend)  
* Streaming responses for improved UX  
* Building automated daily document ingestion pipelines  

___

# 01. Data Overview <a name="data-overview"></a>

The dataset contains **many question–answer pairs** taken from ABC Grocery’s internal help-desk documentation.

Each Q&A pair follows a consistent structure, which can be seen below for 5 examples:

```md
### 0001
Q: What is ABC Grocery?
A: ABC Grocery is a family-run supermarket focused on fresh produce, household essentials, and friendly service.

### 0004
Q: What hours are you open on public holidays?
A: Most stores operate reduced hours on public holidays. Please check our store locator for updated hours.

### 0012
Q: Do you offer home delivery?
A: Yes. We offer home delivery 7 days a week. Delivery fees and times depend on location.

### 0020
Q: How do I update my loyalty card details?
A: You can update loyalty details online or by calling our customer support team.

### 0027
Q: Do you sell gluten-free products?
A: Yes. We carry a wide range of gluten-free products across bakery, frozen, snacks, and household aisles.
```

___

# 02. RAG Overview <a name="rag-overview"></a>

Large Language Models are powerful, but they have a key limitation, **their knowledge is fixed at training time**, and they cannot reliably retrieve up-to-date, organisation-specific, or policy-specific information.

A naive solution would be to simply **feed the entire help-desk document into the model on every query**, but this has major drawbacks:

* It is slow  
* It is expensive (token costs scale with document length)  
* It overwhelms the model with irrelevant information  
* It dramatically increases the risk of hallucination  
* It doesn’t scale as documents grow into hundreds of pages  

**RAG solves all of these issues.**

With RAG:

1. We embed the documents into a vector database.  
2. When a user asks a question, we retrieve *only the most relevant chunks*.  
3. We pass this small, focused context into the LLM.  
4. The LLM generates a grounded answer based solely on verified internal content.

This ensures answers are **factual, fast, cheap, and controllable**.

___

# 03. Building the Core RAG System <a name="rag-core"></a>

<br>
## Secure API Handling <a name="rag-api"></a>

We load API keys from a **.env** file. This prevents credentials from being hard-coded directly in the script.

```python
from dotenv import load_dotenv
load_dotenv()
```

---


## Document Loading <a name="rag-docs"></a>

We use LangChain’s `TextLoader` to import our help-desk markdown file.

```python
from langchain_community.document_loaders import TextLoader

raw_filename = 'abc-grocery-help-desk-data.md'
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content
```

<br>
**Why this matters:**  Document loaders standardise the data into LangChain *Document* objects, which makes later steps like chunking and embedding seamless.

---

## Document Chunking <a name="rag-chunking"></a>

We split the markdown by level-3 headers (`###`), where each header introduces a new Q&A pair.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers=True
)

chunked_docs = splitter.split_text(text)
print(len(chunked_docs), "Q/A chunks")
```

<br>
**Why this matters:**  Chunking ensures retrieval focuses on the specific Q&A pair that relates to a user query.  Good chunking dramatically improves retrieval accuracy.

---

## Embeddings & Vector Store <a name="rag-embeddings"></a>

Embeddings convert text into **numeric vectors** that represent meaning.  Documents with similar meaning end up closer together in vector space.

We embed each Q&A chunk and store the embeddings in Chroma:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="abc_vector_db_chroma",
    collection_name="abc_help_qa")
```

<br>
To load later, instead of re-creating from scratch, we can use this code:

```python
vectorstore = Chroma(
    persist_directory="abc_vector_db_chroma",
    collection_name="abc_help_qa",
    embedding_function=embeddings)
```

---

## LLM Setup <a name="rag-llm"></a>

We instantiate the model that will generate the final answer:

```python
from langchain_openai import ChatOpenAI

abc_assistant_llm = ChatOpenAI(model="gpt-5",
                               temperature=0,
                               max_tokens=None,
                               timeout=None,
                               max_retries=1)
```

<br>
A temperature of 0 is essential for help-desk systems where consistency and accuracy matter more than creativity.

---

## Prompt Template <a name="rag-prompt"></a>

The prompt instructs the model to answer **only** using retrieved context, and to avoid hallucination.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(
"""
System Instructions: You are a helpful assistant for ABC Grocery - your job is to find the best solutions & answers for the customer's query.
Answer ONLY using the provided context. If the answer is not in the context, say that you don't have this information and encourage the customer to email human@abc-grocery.com

Context: {context}

Question: {input}

Answer:
"""
)
```

<br>
**Why this matters:**  Prompt templates are the *instructions* that govern how the LLM behaves.  They ensure the assistant is safe, grounded, and consistent.

We have kept this simple here, but have included one important instruction for the LLM: that if the answer is not in the context, to say that it doesn't have this information and to encourage the customer to email human@abc-grocery.com

---

## Retriever Setup <a name="rag-retriever"></a>

We configure how relevant chunks are selected from the vector database:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.25})
```

<br>
We have set this retrieval up in a way where it will retrieve *up to* 6 documents, but only if they meet the specified relevance score threshold of 0.25. 
<br>
This keeps the context focused and prevents irrelevant content from confusing the LLM.

---

## Full RAG Pipeline <a name="rag-pipeline"></a>

This pipeline connects all of the key components of our system, namely:

1. Take in the user query  
2. Retrieve in relevant chunks from the vector database  
3. Format them  
4. Inject them into the prompt template, along with the system instructions and user query 
5. Pass this information to the LLM  
6. Return the answer  

```python
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_answer_chain = (
    {
        "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
        "input": itemgetter("input"),
    }
    | prompt_template
    | abc_assistant_llm
)
```
<br>
This is the *brain* of the system, the end-to-end mechanism that retrieves, processes and then answers!

___

# 04. Enhancing the Assistant With Memory <a name="rag-memory"></a>

In the enhanced version of the RAG system, we introduced **conversational memory**, allowing multi-turn dialogue while still obeying strict grounding rules.

Memory is added through:

```python

# set up the memory store (a unique session for each unique user)
from langchain_community.chat_message_histories import ChatMessageHistory

_session_store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


# create an updated pipeline that feeds memory into the system prompt
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    runnable=rag_answer_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

When adding memory, we also update the system prompt to include a placeholder place for it to be injected.  It is also important to include information in the system instructions about how to make use of this memory, i.e. to only use it for personalisation

___

# 05. Application & Examples <a name="rag-application"></a>

To pass a query into the system, and have a result returned, we use the following code:

```python
query = "What hours are you open on Easter Sunday?"
response = rag_answer_chain.invoke({"input": query})
print(response)
```
<br>
As an illustration, here are two example queries we passed into the system, along with the resulting response:
<br><br>
**Query:** What time can I come into the store today?  
**Response:** Most locations are open 7am-10pm today.  If it's a holiday, hours may vary - please check the Store Locator for your specific store's hours  
<br>
**Query:** What is a baby dolphin called?  
**Response:** I don't have that information in the provided context. Please email human@abc-grocery.com and our team can help.  
<br>
The latter question is important and shows a behaviour that we want, and that we described in the system instructions.  This was a question that was not answerable using the business-specific context documents, and thus it did not create an answer from it's own memory, it provided the default response.

___

# 06. Inspecting the Retrieved Context <a name="rag-inspection"></a>

One of the most important aspects of building safe and reliable RAG systems is the ability to **inspect exactly which documents were used** to produce an answer.  

This helps us confirm that:

* The system is grounding answers in the correct internal documentation  
* No irrelevant or low-quality chunks were retrieved  
* The model is not hallucinating content  
* Retrieval performance is behaving as expected  
* The system is explainable and auditable  

To enable this, we implemented a clever parallel chain that returns **both** the final answer, and the raw retrieved context (the documents)  

The code that enables this behaviour is below:

```python
from langchain_core.runnables import RunnableParallel

# to also bring through context and user query for analysis
rag_with_context = RunnableParallel(answer=rag_answer_chain,
                                    context=itemgetter("input") | retriever,
                                    input=itemgetter("input"))

user_prompt = ("What time can I come into the store today?")

# invoke
response = rag_with_context.invoke({"input": user_prompt})
print(response["answer"].content)
```
<br>
By calling *RunnableParallel* we are able to run multiple pieces of logic at once.  

In this case, **answer** runs the full RAG pipeline, **context** runs the retriever on it's own (allowing us to capture the returned chunks), and **input** returns the original user query.  When we invoke this, we are returned a *dictionary* containing everything we need to inspect what drove the LLM's answer.

This means a single **.invoke()** call returns a *dictionary* containing everything we need:

We inspected these retrieved documents in *LangSmith* allowing us to verify that our vector store, retriever, and chunking strategy were behaving correctly.

This approach is extremely important in real-world RAG systems where explainability, auditability, and debugging retrieval issues are essential.

___

# 07. Growth & Next Steps <a name="growth-next-steps"></a>

Potential future enhancements include:

* Ingestion of multiple data types (PDFs, product catalogues)  
* Integrating SQL tools for real-time store data, delivery slots, or loyalty information  
* Building a production web interface (React + FastAPI)  
* Automated indexing pipelines to detect new documents  
* Response streaming for real-time chat UX  

This project forms a strong foundation for a scalable enterprise help-desk assistant powered by Retrieval Augmented Generation.

___
