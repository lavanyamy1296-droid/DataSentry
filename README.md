# DataSentry AI — Intelligent Dataset Auditor

A local-first educational platform that audits CSV datasets, highlights common data-quality issues, recommends preprocessing steps, and compares baseline ML models.

## Features
- CSV upload and dataset profiling
- Missing-value, duplicate-row, constant-column, and likely-ID checks
- Numeric outlier screening using the IQR rule
- Target-column class balance inspection
- Charts and column-level quality summaries
- Preprocessing recommendations
- Baseline classification/regression model comparison
- Downloadable JSON audit report

## Requirements
- Python 3.10+
- Node.js 18+

## 1. Start the backend (Windows PowerShell)
```powershell
cd backend
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

If PowerShell blocks activation, do not activate the environment. Run:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## 2. Start the frontend (open a second terminal)
```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```
Open the Vite URL, normally http://localhost:5173.

## 3. Try the sample data
Use **Load sample dataset** in the dashboard, or upload `sample_data/customer_churn_sample.csv`.

## API
- `GET /api/health`
- `GET /api/sample`
- `POST /api/audit` — multipart field `file`
- `POST /api/model/{dataset_id}` — JSON body `{"target":"Churn"}`
- `GET /api/report/{dataset_id}`

## Notes and limitations
This is a learning/demo tool, not a production data-governance system. IQR outlier flags and likely-ID detection are heuristics and require human review. Model metrics use a single stratified/random holdout split and are not a guarantee of future performance. Review leakage, fairness, domain validity, and cross-validation before using any model for decisions. Uploaded files are held in server memory only and disappear when the backend restarts. Do not upload confidential or personally identifiable information.

## GitHub
Create an empty repository, then from this project root:
```powershell
git init
git branch -M main
git add README.md .gitignore backend frontend sample_data
git commit -m "Initial commit - DataSentry AI"
git remote add origin https://github.com/YOUR-USERNAME/DataSentry-AI.git
git push -u origin main
```
