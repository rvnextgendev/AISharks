# AI Sharks — Streamlit application

Use Python 3.12 or 3.13. Extract this folder and open a terminal inside it.

## Windows setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Open http://localhost:8501 if your browser does not open automatically.
Enter your Groq key in the sidebar. Select Sample pitch, Write a pitch, or Upload .txt.
Click Evaluate my pitch. View the host summary, final decisions, first round, and download JSON.
Uploaded files must be UTF-8 text. The included pitch is fictional demo data.

Optional: copy .env.example to .env and put your key there for local use.
On Streamlit Community Cloud use app.py as the entry point and put
GROQ_API_KEY in app secrets. Never commit real keys.

The backend uses session-scoped credentials, not shared environment mutations for sidebar keys.
Results remain in the browser session and do not trigger fresh API calls on downloads or tab changes.
Each evaluation makes 11 model calls, plus any retries. This is an AI simulation; offers are fictional.
Dependency ranges are supplied, not a verified lockfile. Requires access to the default
openai/gpt-oss-120b model and its strict JSON schema output support on Groq.
Live API evaluation requires your own key; it has not been tested in this environment.

## If widgets show "Failed to fetch dynamically imported module"

Stop all running Streamlit processes with Ctrl+C in their terminals.
Use a fresh virtual environment so Python and Streamlit assets match:

```powershell
py -3.12 -m venv .venv_clean
.venv_clean\Scripts\python -m pip install --no-cache-dir -r requirements.txt
.venv_clean\Scripts\python -m streamlit run app.py --server.port 8503
```

Open http://localhost:8503 in an InPrivate/Incognito window, or hard-refresh
with Ctrl+Shift+R. Use this new port rather than an old app tab.
If the problem continues, inspect the browser Network tab for failed /static/js/
requests and their HTTP status. Check whether a browser extension or local proxy
is blocking those requests. The app now uses native Streamlit theming rather
than injected CSS.
