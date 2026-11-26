# AstroDocs RAG

A web application that answers questions about the Solar System
using NASA PDF documents and Retrieval-Augmented Generation.

## Student

Ziati Ikhlasse

## Technologies

- Python and FastAPI
- React and Vite
- PyMuPDF
- Sentence Transformers, FAISS and Transformers — planned

## Status

Initial project setup.
# AstroDocs RAG

An astronomy question-answering application that retrieves evidence from NASA PDF documents and returns source excerpts with document names and PDF page numbers.

Built with React, FastAPI, FAISS, and locally running pretrained models.

**Author:** Ikhlasse  
**GitHub:** [ikhlasse07](https://github.com/ikhlasse07)

## Overview

AstroDocs helps users explore a small astronomy document collection through natural-language questions.

The application searches the documents, ranks relevant passages, and checks whether a candidate excerpt answers the question. Accepted answers are copied from the source text and displayed with citations.

The current answer mode is extractive: the application returns source excerpts rather than freely generated summaries.

No paid inference API is required.

## Features

- Astronomy questions over four reference PDFs.
- Hybrid retrieval combining dense search and BM25 lexical search.
- Cross-encoder reranking.
- Language-model checks for whether an excerpt answers the question.
- Exact source-text verification before displaying accepted excerpts.
- Citations containing document names, PDF page numbers, and chunk identifiers.
- Abstention when suitable evidence is not found.
- React interface with an animated space background.
- FastAPI endpoints and interactive API documentation.
- Evaluation with JSON and CSV results.
- Windows launcher for starting the application after setup.

## Document Collection

Place the reference documents in `backend/data/documents/`.

| Document | PDF pages |
|---|---:|
| `Solar_System_Lithograph_Set.pdf` | 42 |
| `44738_Mars2020_Fact_Sheet_REV_BI.pdf` | 2 |
| `Europa-Clipper-Fact-Sheet.pdf` | 2 |
| `nh-fact-sheet-2015_1.pdf` | 2 |
| **Total** | **48** |

The processed collection contains **518 indexed passages**.

Page citations refer to the one-based page position in the PDF file.

The documents include historical descriptions and mission plans. Answers reflect their contents and may not describe the latest mission developments.

## Models and Technologies

| Component | Technology |
|---|---|
| Frontend | React and Vite |
| Backend | FastAPI and Pydantic |
| PDF extraction | PyMuPDF |
| Dense embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector index | FAISS `IndexFlatIP` |
| Lexical retrieval | BM25 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Answerability checking | `Qwen/Qwen2.5-1.5B-Instruct` |

The embedding vectors have **384 dimensions**.

Vectors are normalized before indexing, allowing inner-product search to represent cosine similarity.

The current implementation runs the models on CPU.

## How It Works

### Document preparation

1. Extract text from each PDF while preserving its filename and page number.
2. Split page text into overlapping chunks.
3. Generate normalized embeddings.
4. Build the FAISS index and save the corresponding chunk metadata.

### Question answering

1. Receive and validate the question.
2. Search the FAISS index for dense candidates.
3. Search the same collection using BM25.
4. Merge candidates and rerank them with the cross-encoder.
5. Build candidate excerpts from the retrieved passages.
6. Rank excerpts and check whether they answer the question.
7. Verify that an accepted excerpt occurs in its cited source.
8. Return the excerpt and citation, or abstain.

With the default settings, retrieval considers up to 20 dense candidates and 20 lexical candidates before deduplication and returns four reranked passages.

The answerability stage checks up to three candidate excerpts.

## Main Files

| Path | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI application |
| `backend/app/schemas.py` | Request and response schemas |
| `backend/app/rag/pdf_loader.py` | PDF text extraction |
| `backend/app/rag/chunker.py` | Token-aware text chunking |
| `backend/app/rag/embedder.py` | Embedding model wrapper |
| `backend/app/rag/vector_store.py` | FAISS persistence and validation |
| `backend/app/rag/hybrid_retriever.py` | Dense and lexical retrieval |
| `backend/app/rag/evidence_selector.py` | Excerpt ranking and answerability checks |
| `backend/app/rag/generator.py` | Shared model and source-verification utilities |
| `backend/app/rag/pipeline.py` | Question-answering pipeline |
| `backend/evaluate.py` | API evaluation runner |
| `backend/data/evaluation/questions.json` | Development evaluation questions |
| `frontend/src/` | React interface |
| `Start-AstroDocs.bat` | Windows application launcher |

## Setup on Windows

The project was developed using:

- Python 3.13.2.
- Node.js 24.14.0.
- Windows PowerShell.
- CPU inference.

Internet access is needed to install dependencies and download model files on first use. Model files are cached locally by Hugging Face.

Run the following commands from the repository root.

### 1. Install backend dependencies

```powershell
python -m venv backend\.venv
```

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 2. Install frontend dependencies

```powershell
Set-Location frontend
npm.cmd install
Set-Location ..
```

### 3. Prepare the documents and index

Ensure that the four reference PDFs are in `backend/data/documents/`.

Run the preparation scripts in this order:

```powershell
.\backend\.venv\Scripts\python.exe backend\extract_documents.py
```

```powershell
.\backend\.venv\Scripts\python.exe backend\chunk_documents.py
```

```powershell
.\backend\.venv\Scripts\python.exe backend\build_index.py
```

The index directory should contain:

- `rag_index.faiss`
- `chunks.json`
- `index_config.json`

Rebuild the processed data and index after changing the document collection or chunking configuration.

## Start the Application

### Windows launcher

After completing setup and building the index, double-click:

```text
Start-AstroDocs.bat
```

Keep the launcher window open while using the application.

The launcher starts the backend and frontend and opens the browser after the services are ready. Press Enter in its window to stop services started by the launcher.

### Manual startup

Open two terminals in the repository root.

In the first terminal, start the backend:

```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1
```

Wait for:

```text
Application startup complete.
```

Keep that terminal running.

In the second terminal, start the frontend:

```powershell
Set-Location frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open:

- Application: http://127.0.0.1:5173
- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

Use a single backend worker to avoid loading additional copies of the models.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API information |
| `GET` | `/health` | Readiness, model information, and busy state |
| `POST` | `/ask` | Retrieve evidence and answer a question |

Example request to `POST /ask`:

```json
{
  "question": "What is the temperature of the Sun's photosphere?",
  "top_k": 4,
  "candidate_k": 20
}
```

Example response, showing selected fields:

```json
{
  "question": "What is the temperature of the Sun's photosphere?",
  "answer": "The temperature of the photosphere is about 5,500 degrees Celsius (10,000 degrees Fahrenheit). [1]",
  "status": "verified_excerpt",
  "answer_mode": "extractive",
  "evidence_verified": true,
  "citation_format_ok": true,
  "cited_sources": [
    {
      "source_id": 1,
      "source": "Solar_System_Lithograph_Set.pdf",
      "page": 6
    }
  ]
}
```

### Response statuses

| Status | Meaning |
|---|---|
| `verified_excerpt` | The displayed excerpt matches its cited source text. |
| `abstained` | The pipeline did not accept an excerpt as answering the question. |
| `needs_review` | The pipeline could not complete the required checks successfully. |

`verified_excerpt` confirms source-text provenance. It does not independently prove that an answer is relevant, complete, or scientifically current.

The backend processes one answer request at a time. A concurrent answer request may receive HTTP `503`.

## Example Questions

- What is the temperature of the Sun's photosphere?
- Why do we always see the same face of the Moon from Earth?
- What measurements does the MEDA instrument provide?
- According to the Mars 2020 fact sheet, which crater is the landing site?
- What does the REASON radar study on Europa?
- What does the LORRI instrument on New Horizons do?

Questions outside the supplied evidence should produce an abstention.

## Evaluation

The development dataset contains:

- 12 answerable questions covering the four PDFs.
- 3 questions that cannot be answered from the supplied documents.

The evaluation runner sends requests to the running API and saves progress after each completed question.

### Recorded results

| Automatic check | `baseline_v1` | `improved_v3` |
|---|---:|---:|
| Completed questions | 15 | 15 |
| Reference page retrieved | 11/12 | 12/12 |
| Unanswerable questions correctly abstained | 3/3 | 3/3 |
| Answerable questions incorrectly abstained | 2/12 | 0/12 |
| Literal support among non-abstained responses | 5/10 | 12/12 |
| Mean API response time | 41.067 s | 13.070 s |
| Recorded request errors | 0 | 0 |

These are development-set measurements, not a general answer-accuracy score.

The same questions were used while improving the pipeline. They therefore do not constitute an independent held-out test.

Reference-page retrieval checks whether a retrieved passage comes from the expected page. Literal support checks whether an answer occurs in its cited source. Neither metric alone establishes answer correctness.

Manual inspection found that the final Venus answer identifies heat trapping but begins mid-sentence. Some excerpts also contain additional context beyond the shortest possible answer.

Response times were measured under local execution conditions and exclude initial model loading. They depend on hardware and system load.

### Run a new evaluation

Start the backend first. From the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -u backend\evaluate.py --run-name verification_v1
```

Use a new run name for each changed version of the code or settings.

To resume an interrupted run without changing its configuration:

```powershell
.\backend\.venv\Scripts\python.exe -u backend\evaluate.py --run-name verification_v1 --resume
```

Results are saved in:

```text
backend/data/evaluation/results/
```

Each run produces:

- A JSON file containing responses, reference information, configuration, package versions, and file hashes.
- A CSV file for reviewing answers and recording manual judgments.

## Limitations

- The application searches a fixed collection of four PDFs.
- It does not perform live web searches.
- The language model can incorrectly accept or reject an excerpt.
- The relevance threshold was selected during development and may need recalibration for another corpus.
- Chunk boundaries can produce incomplete sentence fragments.
- Answers may contain more context than necessary.
- The application returns excerpts and does not perform general multi-document synthesis.
- Evaluation currently covers English questions.
- The small development dataset does not establish performance on unseen questions.
- Document facts and mission plans may be outdated.

## Troubleshooting

### Connection refused on port 8000

Start the backend and wait for `Application startup complete` before using the frontend or evaluation script.

### API is busy

Wait for the current request to finish before sending another question or resuming evaluation.

### Slow first startup

The first startup may download model files. Later startups use the local model cache.

### Missing index files

Run extraction, chunking, and index building in that order before starting the backend.

### An evaluation run name already exists

Use `--resume` for an interrupted run with unchanged code and settings, or choose a new run name.

## Acknowledgements

The reference material comes from NASA and NASA/JPL publications.

AstroDocs uses pretrained models distributed through Hugging Face, together with open-source tools including React, Vite, FastAPI, PyMuPDF, Sentence Transformers, and FAISS.

Source documents and pretrained models retain their respective attribution and licensing conditions.