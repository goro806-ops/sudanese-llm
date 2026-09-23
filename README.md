# Multi-Regional Sudanese LLM

This project provides a complete multi-regional Sudanese LLM structure with web server API, vector database search (RAG), dialect preprocessing, fine-tuning, and evaluation pipelines.

---

## 🚀 Free Deployment & Training Guide

### License Selection on Hugging Face
When creating a repository or Space on Hugging Face, select **MIT License** (or Apache 2.0). **MIT** is open-source, free for public and commercial use, and permits unlimited sharing.

---

### Option 1: Render / Koyeb / Railway (100% Free Python API Web Server)

Since dynamic backend compute servers (FastAPI/Docker) on Hugging Face require a paid PRO plan, **Render** and **Koyeb** provide 100% free dynamic Python web hosting:

#### Deploying on Render (Free Tier):
1. Sign up for free at [Render.com](https://render.com/).
2. Click **New + -> Web Service** and connect your GitHub repo `goro806-ops`.
3. Choose **Docker** or **Python**.
4. Set Build Command: `pip install -r requirements.txt` (or Docker default).
5. Set Start Command: `uvicorn web_server.app:app --host 0.0.0.0 --port $PORT`
6. Render gives you a free live HTTPS URL (e.g., `https://sudanese-llm.onrender.com`).

---

### Option 2: 1-Click Free GPU Training on Google Colab

Fine-tune `Qwen/Qwen2.5-7B-Instruct` on multi-regional Sudanese dialects (Khartoum, Darfur, Kordofan, Eastern, Northern) using **QLoRA** on a free **NVIDIA T4 GPU**:

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File -> Open Notebook -> GitHub**.
3. Select your username `goro806-ops` and open `notebooks/train_colab.ipynb`.
4. Go to **Runtime -> Change runtime type** and select **T4 GPU**.
5. Click **Runtime -> Run all** (`Ctrl + F9`).

---

### Option 3: Hugging Face Static Space (Browser-Based Frontend)

Hugging Face Static Spaces are 100% free forever for hosting HTML/JS/Gradio-Lite web apps that run client-side in the browser:
1. On Hugging Face, click **New Space**.
2. Select **Static** -> **Gradio-Lite** or **React** or **Transformers.js**.
3. License: Select **MIT**.
4. Point your static UI to call your live Render/Koyeb FastAPI backend endpoint!

---

## 💻 Local Running & Testing

To launch the web server locally:
```bash
uvicorn web_server.app:app --reload --port 7860
```
Open interactive Swagger UI docs in your browser:
👉 `http://localhost:7860/docs`

To run the test suite:
```bash
python3 -m pytest tests/test_project.py
```
