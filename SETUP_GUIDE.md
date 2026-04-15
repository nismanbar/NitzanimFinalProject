# Sentinel Chat Monitor - Setup Guide

## System Overview

This system has 3 components:
1. **Chrome Extension** - Monitors WhatsApp Web and sends messages for analysis
2. **Backend API** - Analyzes messages using your existing SentinelLogic
3. **Dashboard** - Web interface to view alerts and statistics

## Quick Start

### Step 1: Install API Dependencies
```bash
cd api
pip install -r requirements.txt
```

### Step 2: Train the AI Models
Before starting the API, you must generate the machine learning models:
```bash
# In the root directory (NitzanimFinalProject)
python train_models.py
```
This will create several `.pkl` and `.keras` files in the root folder. These are necessary for the Sentinel AI to function.

### Step 3: Start the API Server
```bash
cd api
python app.py
```
The dashboard will be available at: http://localhost:5000/dashboard

### Step 3: Load the Chrome Extension
1. Open Chrome and go to: `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The Sentinel icon will appear in your toolbar

### Step 4: Use WhatsApp Web
1. Open WhatsApp Web: https://web.whatsapp.com
2. Click the Sentinel icon in your toolbar
3. Toggle monitoring ON
4. View alerts on the dashboard at http://localhost:5000/dashboard

## How It Works

1. The Chrome Extension monitors messages on WhatsApp Web
2. When a new message is detected, it's sent to the Flask API
3. The API uses your existing `SentinelLogic` to analyze the message
4. If risk is detected (score > 45%), an alert appears:
   - On WhatsApp Web (in-page warning)
   - In the extension popup (stats)
   - On the dashboard (full history)

## File Structure

```
NitzanimFinalProject/
├── app.py                    # Original Streamlit app (keep for manual input)
├── sentinel_logic.py         # Core analysis logic (shared)
├── extension/                # Chrome Extension
│   ├── manifest.json
│   ├── content.js           # WhatsApp Web monitoring script
│   ├── background.js        # Service worker
│   ├── popup.html          # Extension popup UI
│   ├── popup.js            # Popup logic
│   ├── styles.css          # Alert styling
│   └── icons/              # Extension icons
├── api/                     # Flask Backend
│   ├── app.py              # API server
│   ├── requirements.txt
│   └── templates/
│       └── dashboard.html  # Dashboard UI
└── ...
```

## Important Notes

- The API must be running for the extension to work
- WhatsApp Web must be open in Chrome for monitoring
- The extension shows alerts directly on the WhatsApp page
- The dashboard shows full history and statistics
- You can use both the Streamlit app AND the API together

## Troubleshooting

**Extension not detecting messages?**
- Make sure WhatsApp Web is fully loaded
- Click the Sentinel icon and toggle monitoring OFF/ON
- Check the browser console for errors

**API not responding?**
- Make sure the Flask server is running
- Check that port 5000 is not blocked

**No alerts showing?**
- Alerts only appear when risk score >= 45%
- Try sending a test message with concerning content
