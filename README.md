# 🚦 Traffic Intelligence Command Center

A real-time traffic video analysis application built with YOLO, ByteTrack/DeepSORT, and Google Gemini AI.

Upload any MP4 traffic video and get:
- **Vehicle detection & tracking** with unique IDs per vehicle
- **Per-class color-coded bounding boxes** (pink = jeepney, blue = sedan, etc.)
- **PCU & VCR calculations** with live donut pie chart
- **AI-generated traffic report** via Google Gemini
- **Frame gallery** (Normal Grid / Compact Grid / List) with per-frame downloads
- **Session history** — past analyses survive across video uploads
- **PDF export** of the full traffic report

---

## ⚡ Quick Start (Local)

### 1. Prerequisites
- Python 3.9+
- A YOLO `.pt` model file (e.g. `exp-2.pt`)
- A [Google Gemini API key](https://aistudio.google.com) (free)

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/traffic-analyzer.git
cd traffic-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Place your model file in this directory
cp /path/to/your/model.pt ./exp-2.pt
```

### 3. Run

```bash
streamlit run app.py
```

Browser opens at **http://localhost:8501**

---

## 🌐 Deploy Online (Free — Streamlit Community Cloud)

1. **Push to GitHub** (public or private):
   ```bash
   git init
   git add app.py requirements.txt .streamlit/ Dockerfile .gitignore README.md
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/traffic-analyzer.git
   git push -u origin main
   ```

2. **Upload your `.pt` model** — since model files are large, do one of:
   - Upload to your repo via GitHub web UI (if < 100 MB), or
   - Use [Git LFS](https://git-lfs.com) for large files, or
   - Host on HuggingFace Hub and download at startup (see note below)

3. **Go to [share.streamlit.io](https://share.streamlit.io)**, sign in with GitHub, and click **New App** → select your repo → `app.py`.

4. **Set Secrets** in the Streamlit Cloud dashboard:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
   *(Optional — users can also enter it directly in the app sidebar)*

5. Your app will be live at `https://YOUR_USERNAME-traffic-analyzer-app-XXXX.streamlit.app`

---

## 🐳 Deploy with Docker

```bash
# Build image
docker build -t traffic-analyzer .

# Copy your model into the container (or mount as volume)
docker run -p 8501:8501 \
  -v /path/to/your/models:/app \
  traffic-analyzer
```

Then open **http://localhost:8501**

### Deploy to Railway / Render / Fly.io

All three support Docker deployments from a GitHub repo:
- [Railway](https://railway.app) → New Project → Deploy from GitHub → auto-detects Dockerfile
- [Render](https://render.com) → New Web Service → Docker → set port 8501
- [Fly.io](https://fly.io) → `fly launch` → follows Dockerfile automatically

---

## ⚙️ Configuration

All settings are in the **sidebar** inside the app. No code changes needed:

| Setting | Description |
|---------|-------------|
| Model | Dropdown of all `.pt` files in the app folder, or upload a new one |
| Tracker | ByteTrack (fast, built-in) or DeepSORT (accurate, slower) |
| Confidence | Detection threshold (0.1–0.95) |
| Frame skip | Process every N frames (higher = faster) |
| Resolution | 320 / 416 / 480 / 640 px (lower = faster) |

---

## 📦 Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLO detection + ByteTrack |
| `deep-sort-realtime` | DeepSORT tracker (optional, falls back to ByteTrack) |
| `google-generativeai` | Gemini AI report generation |
| `streamlit` | Web UI |
| `opencv-python-headless` | Video processing |
| `fpdf2` | PDF export |
| `plotly` | Interactive charts |
| `Pillow` | Frame encoding |

---

## 🎨 Design

Amber/industrial "ops center" aesthetic — Russo One + Share Tech Mono + Barlow Condensed fonts, near-black `#0A0A08` background with amber `#F0A500` accent.

---

## 📄 License

MIT — free to use, modify, and distribute.
