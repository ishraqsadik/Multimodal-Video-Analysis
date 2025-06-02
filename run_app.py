#!/usr/bin/env python3
"""
Simple script to run the Multimodal Video Analysis app
"""

import subprocess
import sys
import os

def main():
    print("🎬 Starting Multimodal Video Analysis System...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Run: python setup.py")
        print("   Then edit .env file with your Gemini API key")
        return
    
    # Check if API key is configured
    try:
        with open('.env', 'r') as f:
            content = f.read()
            if 'your_gemini_api_key_here' in content:
                print("⚠️  Please add your Gemini API key to .env file")
                print("   Get one from: https://aistudio.google.com/app/apikey")
                print("   Edit .env and replace 'your_gemini_api_key_here' with your actual key")
                return
    except Exception:
        pass
    
    print("🚀 Launching app...")
    print("📱 App will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped!")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("   Try running: python -m streamlit run app.py")

if __name__ == "__main__":
    main() 