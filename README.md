# 🎬 Multimodal Video Analysis

A sophisticated YouTube video analysis system powered by Google Gemini AI that provides comprehensive video understanding through transcript analysis, intelligent sectioning, and interactive chat capabilities.

## ✨ Features

### 🎯 Core Capabilities
- **📝 Intelligent Video Sectioning**: AI-powered breakdown of videos into logical sections with clickable timestamps
- **💬 Video Chat Interface**: Interactive Q&A with comprehensive video context and timestamp citations
- **🔍 Content Search**: Search functionality for finding specific moments and topics within videos
- **🎥 Embedded Video Player**: Integrated YouTube player with timestamp navigation
- **📱 Responsive UI**: Clean, modern Streamlit interface optimized for video analysis

### 🤖 AI-Powered Analysis
- **Smart Transcript Processing**: Handles both manual and auto-generated YouTube transcripts
- **Context-Aware Chat**: Uses full video context (15,000+ characters) for accurate responses
- **Timestamp Extraction**: Automatically extracts and validates timestamps from AI responses
- **Rate Limit Handling**: Robust retry logic with exponential backoff for reliable API calls

## 🛠️ Technical Architecture

### Video Processing Pipeline
1. **Metadata Extraction**: Uses `yt-dlp` for robust YouTube video information retrieval
2. **Transcript Analysis**: Leverages `youtube-transcript-api` for comprehensive transcript extraction
3. **AI Sectioning**: Gemini 2.0 Flash-Lite processes transcript chunks to generate meaningful sections
4. **Interactive UI**: Streamlit-based interface with embedded video and clickable timestamps

### Chat System
- **Enhanced Context Building**: Generates video summaries and selects relevant chunks
- **Intelligent Timestamp Detection**: Regex-based extraction of timestamps from AI responses
- **Session Management**: Persistent chat history and video state management
- **Visual Timestamp Buttons**: Clickable buttons that jump to specific video moments

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ishraqsadik/Multimodal-Video-Analysis.git
cd Multimodal-Video-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file or set environment variable
export GEMINI_API_KEY="your_gemini_api_key_here"
```

4. Run the application:
```bash
streamlit run app.py
```

## 📋 Usage Guide

1. **Enter YouTube URL**: Paste any YouTube video URL with available transcripts
2. **Process Video**: Click "Process Video" to analyze content and generate sections
3. **Browse Sections**: Use the "Outline" tab to see AI-generated video breakdown
4. **Interactive Chat**: Ask questions about video content in the "Chat" tab
5. **Search Content**: Use the "Search" tab to find specific topics or moments
6. **Navigate Timestamps**: Click any timestamp button to jump to that moment in the video

## 🏗️ Project Structure

```
├── app.py                 # Main Streamlit application
├── video_processor.py     # YouTube video processing and transcript extraction
├── gemini_client.py       # Gemini AI integration and analysis
├── utils.py              # Utility functions for formatting and validation
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup configuration
└── README.md             # Project documentation
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Supported Video Types
- YouTube videos with English transcripts (manual or auto-generated)
- Educational content, tutorials, presentations work best
- Videos with clear topic transitions provide better sectioning

## 🛣️ Roadmap

### Planned Enhancements (v2.0)
- **Semantic Chunking**: Replace line-based chunking with intelligent semantic segmentation
- **LlamaIndex Integration**: Add semantic search and retrieval capabilities
- **Multimodal Analysis**: Integrate Gemini Video Understanding for visual frame analysis
- **LLM Routing**: Intelligent query routing between different AI models
- **Enhanced Search**: Advanced visual content search with frame extraction
