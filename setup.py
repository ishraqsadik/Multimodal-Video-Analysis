#!/usr/bin/env python3
"""
Setup script for Multimodal Video Analysis System
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", 
                              "streamlit", "google-generativeai", "youtube-transcript-api", 
                              "pytube", "python-dotenv"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing requirements. Try running manually:")
        print("   pip install --user streamlit google-generativeai youtube-transcript-api pytube python-dotenv")
        return False

def create_env_file():
    """Create .env file if it doesn't exist."""
    if not os.path.exists('.env'):
        print("🔧 Creating .env file...")
        try:
            with open('.env', 'w', encoding='utf-8') as f:
                f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
            print("✅ .env file created!")
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("ℹ️ .env file already exists.")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Multimodal Video Analysis System")
    print("=" * 50)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Create environment file
    if not create_env_file():
        success = False
    
    if success:
        print("\n🎉 Setup complete!")
        print("\n📋 Next steps:")
        print("1. Get your Gemini API key from: https://aistudio.google.com/app/apikey")
        print("2. Edit .env file and replace 'your_gemini_api_key_here' with your actual key")
        print("3. Run the app: python -m streamlit run app.py")
        print("\n🎬 The app will open in your browser at http://localhost:8501")
        print("\n💡 Note: Use 'python -m streamlit' instead of just 'streamlit' on Windows")
    else:
        print("\n⚠️ Setup had some issues. Please check the errors above.")

if __name__ == "__main__":
    main() 