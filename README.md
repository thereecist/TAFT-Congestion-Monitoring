# 🚦 Traffic Intelligence Command Center

An AI-powered traffic analysis dashboard built with Streamlit. Upload a road video and get real-time vehicle detection, tracking, Volume-to-Capacity Ratio (VCR) analysis, and an AI-generated engineering report — all in your browser.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Vehicle Detection** | Custom YOLO models (`.pt`) — swap models from the sidebar |
| **Multi-Tracker** | ByteTrack (fast) or DeepSORT (accurate) |
| **VCR Analysis** | Real-time Volume-to-Capacity Ratio with zone indicators |
| **Live Pie Chart** | SVG donut chart updating every processed frame |
| **AI Report** | Generates a professional traffic engineering report via **Gemini**, **OpenAI**, or **Claude** |
| **PDF Export** | Download a formatted PDF report with charts |
| **Frame Gallery** | Normal Grid / Compact Grid / List view — hover to download any frame |
| **Session History** | Past analyses persist across video uploads |
| **Saved API Keys** | Keys saved locally to `config.json` — no re-entering on restart |
| **Model Manager** | Upload or delete `.pt` model files from the sidebar |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/traffic-intelligence.git
cd traffic-intelligence
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your YOLO model
Place any `.pt` YOLO model file in the project root. It will auto-appear in the sidebar dropdown.

```
traffic-intelligence/
├── app.py
├── requirements.txt
├── your-model.pt        ← put it here
└── ...
```

### 5. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 AI Report API Keys

Get a free API key from one of these providers and paste it into the sidebar:

| Provider | Free Tier | Get Key |
|----------|-----------|---------|
| **Gemini** | ✅ Yes (recommended) | [aistudio.google.com](https://aistudio.google.com) |
| **OpenAI** | ❌ Paid | [platform.openai.com](https://platform.openai.com) |
| **Claude** | ❌ Paid | [console.anthropic.com](https://console.anthropic.com) |

Keys are saved locally to `config.json` (gitignored) — click **💾 Save Key** in the sidebar.

---

## 🐳 Docker (Optional)

```bash
docker build -t traffic-intelligence .
docker run -p 8501:8501 traffic-intelligence
```

---

## ⚙️ Sidebar Controls

| Control | Description |
|---------|-------------|
| **Model selector** | Choose any `.pt` file found in the project root |
| **🗑 Delete model** | Remove a model file from disk |
| **Upload model** | Add a new `.pt` file without leaving the browser |
| **AI Provider** | Gemini / OpenAI / Claude |
| **Confidence threshold** | YOLO detection confidence (0.1–0.95) |
| **Frame skip** | Process every N frames (higher = faster) |
| **Tracker** | ByteTrack (fast) or DeepSORT (accurate) |
| **Analysis resolution** | Lower = faster inference |

---

## 📊 VCR Thresholds

| VCR Range | Status | Meaning |
|-----------|--------|---------|
| < 0.4 | 🟢 Free Flow | Road operating well below capacity |
| 0.4 – 0.6 | 🟡 Stable Flow | Light-to-moderate traffic |
| 0.6 – 0.8 | 🟠 Moderate Congestion | Noticeable slowdowns possible |
| 0.8 – 1.0 | 🟠 Approaching Capacity | Intervention may be needed |
| > 1.0 | 🔴 Over Capacity | Severe congestion, intervention required |

Road capacity baseline: **3000 PCU/hour** (multi-lane urban road).

---

## 📁 Project Structure

```
traffic-intelligence/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container setup
├── README.md
├── .gitignore
└── config.json         # Saved API keys (auto-created, gitignored)
```

---

## 📦 Key Dependencies

- [Streamlit](https://streamlit.io/) — web UI framework
- [Ultralytics](https://ultralytics.com/) — YOLO inference + ByteTrack
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) — DeepSORT tracker
- [Plotly](https://plotly.com/) — interactive charts
- [fpdf2](https://py-pdf.github.io/fpdf2/) — PDF generation
- [google-generativeai](https://ai.google.dev/) / [openai](https://pypi.org/project/openai/) / [anthropic](https://pypi.org/project/anthropic/) — AI report providers

---

## 📝 Notes

- Model files (`.pt`) are gitignored — you must supply your own trained YOLO model.
- Video files are never stored permanently; they are processed from a temp file and deleted after analysis.
- `config.json` stores your saved API keys locally and is gitignored.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
