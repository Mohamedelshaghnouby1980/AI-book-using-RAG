# AI Books RAG

I built this project so I can ask questions about any technical book and get a precise answer with a page number, instead of reading the whole thing.

The idea is simple: I load a PDF, the system splits it into chunks and saves them, and when I ask a question it finds the most relevant chunks and sends them to Ollama to generate an answer.

## What I used

- PyMuPDF to read the PDF
- sentence-transformers to convert text into vectors
- ChromaDB to store and search the vectors locally
- Ollama to generate answers on my own machine
- Streamlit for a simple web interface

## How to run it

First download Ollama from ollama.com, then:

```
git clone https://github.com/Mohamedelshaghnouby1980/AI-book-using-RAG.git
cd AI-book-using-RAG
pip install -r requirements.txt
ollama pull llama3.2
```

Put your book in the data/books folder and run:

```
python src/main.py --ingest data/books/yourbook.pdf
python -m streamlit run app.py
```

## What it does

You type a question in the interface, it searches through the book, and returns an answer with the page number it came from.

You can also add any PDF directly from the interface without touching the code.

## Note

The whole thing runs locally with no internet connection needed after setup, and no paid API keys.