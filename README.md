# Domain-Specific PDF RAG Chatbot

A Streamlit app that lets you upload PDF documents and ask questions about
their contents. It finds the most relevant passages using FAISS and shows
you the answer along with the source file and page number — so you always
know where the answer came from.

**Live app:** https://domain-specific-rag-chatbot-4bvyg5dyfdf3birzpxjgbi.streamlit.app/ 

*(hosted on Streamlit Community Cloud — may take a few seconds to wake up if it's been idle)*

## What it does

- Upload one or more PDFs
- Extracts text page by page
- Splits text into overlapping chunks
- Converts chunks into embeddings and indexes them with FAISS
- Finds the most relevant chunks for your question
- Sends only that context to the LLM (no outside knowledge used)
- Shows the answer with source filename + page number
- Says "I could not find this information in the uploaded documents" if nothing relevant is found
- Keeps chat history with a clear-chat option

## How it works

Upload PDF(s) -> Extract text (pypdf) -> Split into chunks
-> Generate embeddings (MiniLM) -> Index in FAISS
-> Question -> Retrieve top matching chunks
-> Relevance check -> Answer with sources (Groq)


## Folder Structure

```
pdf-rag-chatbot/
├── app.py                    # Streamlit interface
├── document_loader.py        # PDF validation + text extraction
├── rag_pipeline.py           # chunking + retrieval + answer flow
├── vector_store.py           # embeddings + FAISS index
├── prompt.py                 # strict answer-only-from-context prompt
├── documents/                # sample PDFs to test with
├── tests/                    # unit tests
├── test_questions.csv        # evaluation sheet (17 questions)
├── architecture.svg          # workflow diagram
├── create_sample_documents.py
├── requirements.txt
└── .env.example
```



## How to run it

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and add your Groq API key (get one free at
[console.groq.com/keys](https://console.groq.com/keys)) — don't share or commit this file.

```bash
streamlit run app.py
```

Then open the link it prints (usually `http://localhost:8501`), upload a
PDF from `documents/`, and start asking questions.

## Note

- This app answers only from the PDFs you upload — it will not use outside
  knowledge, and text inside a PDF cannot override its instructions.
- Answers are estimates based on retrieved text, not guaranteed facts —
  always check the cited page for anything important.
- Uploaded PDFs are processed for the session only and are not stored
  permanently.
