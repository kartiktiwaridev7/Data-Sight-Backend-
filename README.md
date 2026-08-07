# 🧠 DataSight AI Engine (Backend)

The powerhouse behind the **DataSight Analytics Dashboard**. This repository contains the Python-based machine learning API that processes raw datasets, executes multi-variable regressions, and streams aggregated insights back to the client interface in milliseconds.

🔗 **Frontend Repository:** [DataSight Client](https://github.com/kartiktiwaridev7/Data-Sight)  
🔴 **Live Client Dashboard:** [View Here](https://data-sight.netlify.app)

## 🏗 Architecture & Workflow

This backend acts as an asynchronous, non-blocking data processing engine. Engineered with **FastAPI**, it is designed to securely accept large data payloads from the React frontend, completely bypassing the browser's computational limits.

1. **Ingestion:** Secure POST endpoints (`/analyze/upload`) receive raw data files.
2. **Processing:** Dynamically cleans and structures the incoming data for analysis.
3. **Machine Learning:** Executes statistical regressions and identifies time-series trends on the fly.
4. **Response:** Returns highly optimized, aggregated JSON payloads ready for immediate visualization on the frontend.

## ⚙️ Tech Stack

* **Core Framework:** Python 3 & FastAPI
* **Server:** Uvicorn (ASGI)
* **Security:** Strict CORS middleware locking access exclusively to the verified Netlify production frontend.
* **Deployment Engine:** Render

## 🚀 Local Installation & Development

To run this AI engine locally on your machine for development or testing:

### 1. Clone the Repository
```bash
git clone [https://github.com/kartiktiwaridev7/Data-Sight-Backend-.git](https://github.com/kartiktiwaridev7/Data-Sight-Backend-.git)
cd Data-Sight-Backend-
