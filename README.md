# 🛡️ Sentinel AI - Cyberbullying & Risk Monitor

Sentinel AI is an autonomous monitoring system designed to detect risk, distress, and cyberbullying in digital conversations. It uses machine learning and NLP to provide real-time analysis and recommendations.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Chrome Browser (for the extension)

### 2. Installation
1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies**:
   ```bash
   pip install -r api/requirements.txt
   pip install streamlit
   ```
3. **Train the models**:
   The project requires pre-trained models. Run the training script first:
   ```bash
   python train_models.py
   ```
   *Note: Ensure you have the dataset files (like `toxic_comments.csv`) in the root directory.*

---

## 🏃 How to Run

The project has two main parts that you can run independently or together:

### A. The API Server (Backend & Chrome Extension Support)
This is the core server that powers the Chrome Extension and the main Dashboard.
1. **Start the server**:
   ```bash
   python api/app.py
   ```
2. **Access the Dashboard**: Open [http://localhost:5000/dashboard](http://localhost:5000/dashboard) in your browser.
3. **Chrome Extension**: 
   - Go to `chrome://extensions/`
   - Enable **Developer mode**.
   - Click **Load unpacked** and select the `extension` folder from this project.

### B. The Streamlit Dashboard (Standalone Manual Input)
This is a dedicated UI for manual message simulation and testing.
1. **Start Streamlit**:
   ```bash
   streamlit run app.py
   ```

---

## 🛑 How to Stop the Server
If you are running the script in a terminal (including PyCharm's terminal), you can stop it by pressing:
**`Ctrl + C`**

This will gracefully shut down the Flask or Streamlit server.

---

## 📁 Project Structure
- `api/app.py`: Flask API server and Dashboard.
- `app.py`: Streamlit standalone dashboard.
- `sentinel_logic.py`: Core AI/NLP logic.
- `extension/`: Chrome Extension for WhatsApp Web/Discord/Instagram.
- `train_models.py`: Script to generate the ML models.

---

## 🛠️ Troubleshooting
- **Script finishes immediately?** Make sure you are running `python api/app.py` for the API, or `streamlit run app.py` for the Streamlit dashboard. Running `python app.py` will not work.
- **Models missing?** Run `python train_models.py` to generate them.
- **Port already in use?** If port 5000 is taken, the API won't start. Stop any other running Flask apps first.
