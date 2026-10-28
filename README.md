# Adaptive LLM-Based Parliamentary Agriculture Debate Intelligence System

A Flask research dashboard built from the supplied Excel workbook and the NLP workflow in `Copy_of_nlppro (1).ipynb`.

## What the supplied files contain

- The workbook contains **10,800 records** and 28 columns, including `House`, `debateTitle`, `debateTypeDesc`, `keywordUsed`, `title`, `debateTitleSubject`, `date`, `debateDate`, `year`, `members`, `ministry`, and related parliamentary metadata.
- Workbook columns: `loksabha`, `session`, `dbSlno`, `debateTitle`, `memberName`, `debateType`, `debateTypeDesc`, `debateDate`, `keywordUsed`, `mpPartDetailList`, `House`, `title`, `debateTitleSubject`, `date`, `type`, `sessionNo`, `questionNo`, `questionType`, `resourceId`, `year`, `pageNoFromTo`, `handle`, `members`, `questionerName`, `ministerName`, `language`, `files`, and `ministry`.
- It contains **2,099 Lok Sabha** and **8,701 Rajya Sabha** rows.
- The workbook does not contain a full debate transcript, speech, question, or answer column. The notebook’s unified text therefore consists of whichever supported metadata/text fields are present for a row. Search, topics, frequency, TF-IDF, and entity results on this site are explicitly based on that available record text; they must not be interpreted as analysis of the complete debate speeches.
- The notebook contains preprocessing, rule-based agriculture entity extraction, a `dslim/bert-base-NER` pass over at most 100 records, BERTopic, and Sentence-BERT/FAISS semantic retrieval. It does not save model weights, a BERTopic model, or a FAISS index. The FLAN-T5 heading/import in the notebook is not accompanied by a model-loading or generation implementation.
- Notebook preprocessing lowercases text; removes URLs, HTML, digits, punctuation/non-ASCII letters, and repeated whitespace; tokenizes with NLTK; removes English and custom parliamentary stopwords and tokens shorter than three characters; and applies WordNet lemmatization. It removes empty cleaned text and duplicate `(House, text)` records.
- The notebook’s original in-memory search embeds 250-word chunks with 40-word overlap using normalized `all-MiniLM-L6-v2` vectors, then retrieves with FAISS `IndexFlatIP`. This app exposes that same retrieval through a persistent index built by the initializer. The notebook's BERTopic setup is English with `min_topic_size=20`; the separate NER model is not used as a substitute for the agriculture keyword extraction feature.

## Features

- Overview cards, year-wise trend chart, and Lok Sabha/Rajya Sabha comparison calculated from the Excel file.
- Semantic search using `all-MiniLM-L6-v2` embeddings, normalized vectors, 250-word chunks with 40-word overlap, and a FAISS `IndexFlatIP` index when FAISS is available.
- An interactive question-and-answer experience: Sentence-BERT retrieves matching source records, the fitted BERTopic assignments and agriculture keyword rules summarize that retrieved set, and the dashboard displays supporting record titles, dates, houses, and source text. The response is grounded in the top distinct matches and explicitly is not an exhaustive corpus count.
- Agricultural keyword extraction using the notebook’s CROP, ISSUE, POLICY, and RESOURCE dictionaries.
- BERTopic themes from the notebook’s cleaned text and `language="english", min_topic_size=20` configuration.
- Most frequent cleaned terms and TF-IDF terms using the notebook’s `max_features=5000`, `ngram_range=(1, 2)`, and `min_df=2`.
- Dataset hash validation prevents a saved index or topics file from silently being used with a different workbook.

## Run locally (Windows)

Use Python 3.11 (the version used by the Render blueprint; installing the full ML stack on the Python 3.14 environment may not be supported by all transitive packages).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python init_artifacts.py
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The initialization step downloads the NLTK resources and pretrained Sentence-BERT model, runs the notebook’s BERTopic configuration once on the current workbook, and saves artifacts locally under `artifacts/`. It does not fine-tune Sentence-BERT. Re-run it only after changing the workbook or intentionally rebuilding the index/topics.

`faiss-cpu` is installed on non-Windows platforms. On Windows, the app uses the same normalized inner-product ranking over the saved Sentence-BERT vectors if FAISS is unavailable.

To use a different workbook, set `DATASET_PATH` to its absolute path before running both `init_artifacts.py` and `app.py`. The workbook must contain at least one of the text fields used by the notebook.

## Deploy to Render

The included `render.yaml` configures a public Python web service. During the build, Render installs `requirements.txt`, downloads the required NLTK and pretrained model resources, runs `init_artifacts.py` against the workbook, and then serves the existing Flask site with Gunicorn. The workbook must be committed to the GitHub project; the virtual environment and generated model/index artifacts must not be.

### 1. Create a GitHub repository for this project only

1. Sign in at [github.com](https://github.com/). Create a new repository, for example `agriparliament-dashboard`. A private repository is suitable; connect Render to the same GitHub account.
2. Open PowerShell. This checkout is inside a parent Git repository rooted at `C:\Users\Lokeshwari K`, so **do not run Git commands from your home folder**. Initialize a separate repository from the project directory:

   ```powershell
   $project = "C:\Users\Lokeshwari K\Downloads\nlppro"
   git -C $project init -b main
   git -C $project add .
   git -C $project status --short
   ```

3. Before committing, confirm the status lists only the project source, `templates`, `static`, the notebook, and the Excel workbook. `.gitignore` excludes `.venv`, `artifacts`, Python caches, and `.env`. Do not add credentials or generated model caches.
4. Commit and connect the empty GitHub repository (replace the placeholder with your GitHub username):

   ```powershell
   git -C $project commit -m "Prepare agriculture debate dashboard for deployment"
   git -C $project remote add origin https://github.com/<YOUR-USERNAME>/agriparliament-dashboard.git
   git -C $project push -u origin main
   ```

   If Git prompts for authentication, complete the Git Credential Manager/browser sign-in. Do not put a GitHub password or access token in source files or the remote URL.

### 2. Connect the repository to Render

1. Sign in at [render.com](https://render.com/) and choose **New → Blueprint**.
2. Connect GitHub if prompted, grant Render access to `agriparliament-dashboard`, then select that repository and its `main` branch.
3. Render reads `render.yaml`. Review the service settings and approve creation. The blueprint uses a **Standard** instance because the Sentence-BERT model and BERTopic initialization need more memory than a small free instance generally provides; Render may require a paid plan and billing details.
4. Wait for the build to complete. The first build downloads the pretrained `all-MiniLM-L6-v2` model and NLTK resources and fits the notebook's BERTopic configuration against the actual workbook. This can take several minutes. If it fails, open the Render service's **Events/Logs → Build** and address the first build error; do not remove the artifact-generation command.
5. When Render reports the service is live, open the HTTPS URL shown on the service page. The blueprint uses Gunicorn with one worker and four threads, `/api/health` as its health check, and Render's `PORT` binding through the WSGI server.

No application API keys or passwords are required. The model is downloaded from Hugging Face during the build; the app makes no paid LLM API calls. Search results and the response summary are produced by the actual Sentence-BERT retrieval, BERTopic assignments, and notebook agriculture-keyword rules.

### 3. Verify the public deployment

Replace `<YOUR-RENDER-URL>` with the HTTPS URL from the Render service page:

```text
https://<YOUR-RENDER-URL>/api/health
https://<YOUR-RENDER-URL>/api/analytics
https://<YOUR-RENDER-URL>/api/topics
https://<YOUR-RENDER-URL>/api/search?q=farmer%20crop%20insurance
```

The health endpoint should report `"status":"ok"`. Analytics should report the workbook row count and calculated statistics. Topics should return the generated topic list. Search should return actual records and an evidence-grounded summary. Also open the home page on a phone or a different computer and try one of the example questions.

The current local generated artifacts occupy about 106 MB and are intentionally excluded from Git. Render generates them during each deployment build, so you do not need to upload model weights, embeddings, or the index. They live with that deployment; a new deployment regenerates them. If you change the Excel workbook, keep the same workbook filename/path or update `DATASET_PATH`, then deploy so the artifacts are rebuilt for that data.

If configuring a Web Service manually instead of using the Blueprint, use:

- **Build command:** `pip install -r requirements.txt && python init_artifacts.py`
- **Start command:** `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180 app:app`
- **Health check path:** `/api/health`
- **Python:** 3.11.11 or a compatible 3.11 release

If manually configuring the environment, set `PYTHON_VERSION=3.11.11` and `LOG_LEVEL=INFO`. Do not set `FLASK_DEBUG=true` in production.

## API endpoints

- `GET /api/health` — service and workbook availability.
- `GET /api/analytics` — dataset statistics, trend, house totals, word frequency, TF-IDF, and agricultural entity counts.
- `GET /api/entities` — agricultural entity counts by category.
- `GET /api/topics` — BERTopic labels and representative keywords; returns HTTP 503 with setup guidance until artifacts exist.
- `GET /api/search?q=<question>&limit=5` — semantic search; returns HTTP 503 with setup guidance until artifacts exist.

The dashboard reports setup/model errors instead of showing placeholder statistics or synthetic search results. The conversational response is an evidence-grounded summary built from retrieval, BERTopic, and the notebook's keyword rules; the supplied notebook does not implement a text-generating LLM, and the workbook has no debate transcripts. It therefore does not pretend to generate transcript-level answers.

### Deployment status

This repository contains deployment configuration and a verified local app, but no Render account or GitHub authentication is available from the development environment. The GitHub push, Render account authorization, instance approval/billing, and final public-URL verification must be completed by the project owner using the steps above. Do not treat the localhost URL as a public deployment.
