# 🌍 AI Satellite Image Analyzer

The **AI Satellite Image Analyzer** is a state-of-the-art hybrid computer vision application. It intelligently routes satellite imagery using a custom PyTorch Convolutional Neural Network (CNN) for uniform EuroSAT patches, and dynamically falls back to Google's **Gemini 2.5 Vision AI** for complex, real-world topographical analysis. The application features a high-tech, responsive Glassmorphism UI with a 3D rotating CSS globe.

---

## 🚀 How to Run this Project Locally in VS Code

If you want to run this project on a new computer or present it to an examiner, follow these exact steps.

### 1. Prerequisites

You must have **Node.js** and **Python** installed on your computer.
*   **Node.js**: [Download here](https://nodejs.org/) (Install the "LTS" version).
*   **Python**: [Download here](https://www.python.org/downloads/) (Version 3.10 or higher).

### 2. VS Code Setup

*   Open **VS Code** (Visual Studio Code).
*   Go to `File > Open Folder` and select the `satellite-analyzer` folder.

### 3. API Key Setup (Backend)

This project requires a Google Gemini API Key to analyze complex real-world images.

1.  Go to [Google AI Studio](https://aistudio.google.com/) and create a free API key.
2.  Open the `backend` folder in VS Code.
3.  Create a file named exactly `.env` inside the `backend` folder.
4.  Open `.env` and paste your key like this:
```env
GEMINI_API_KEY=AIzaSyYourSecretKeyGoesHere...
```

### 4. Start the Python Backend

You need to install the AI and server libraries, then start the FastAPI server.

1.  Open a new terminal in VS Code (`Terminal > New Terminal`).
2.  Navigate into the backend folder:
```bash
cd backend
```
3.  Install the dependencies:
```bash
pip install -r requirements.txt
```
4.  Start the backend server:
```bash
python app.py
```
*(Leave this terminal running! It will host the API at `http://localhost:8000`)*

### 5. Start the React Frontend

Now, you need to start the beautiful user interface.

1.  Open a **second, completely new terminal** in VS Code (click the `+` icon in the terminal panel).
2.  Navigate into the frontend folder:
```bash
cd frontend
```
3.  Install the React dependencies:
```bash
npm install
```
4.  Start the frontend application:
```bash
npm start
```
5.  Your browser should automatically open to `http://localhost:3000`.

---

## 🛠️ Tech Stack / Project Details

*   **Frontend**: React.js, Vanilla CSS (Dark Glassmorphism UI, Custom CSS 3D Globe Animation)
*   **Backend**: Python, FastAPI, Uvicorn
*   **AI Models**:
    *   **Custom CNN**: Built with PyTorch, trained on 27,000 EuroSAT images (87% accuracy on 64x64 patches).
    *   **Gemini 2.5 Vision**: Google's multimodal Large Language Model for advanced contextual feature extraction.
*   **Architecture**: "Smart Router" heuristic algorithm that detects image complexity based on color variance and resolution to determine which AI model to invoke.
