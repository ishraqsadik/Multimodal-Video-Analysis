import re
import os
from urllib.parse import urlparse, parse_qs
from typing import Optional, Tuple, Dict, List, Any
import streamlit as st

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    if not url:
        return ""
    
    # Handle different YouTube URL formats
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return ""

def is_valid_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    return bool(extract_video_id(url))

def format_timestamp(seconds: float) -> str:
    """Format seconds to timestamp string (MM:SS or HH:MM:SS)."""
    if not isinstance(seconds, (int, float)):
        return "0:00"
    
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

def parse_timestamp(timestamp_str: str) -> float:
    """Convert timestamp string to seconds."""
    parts = timestamp_str.split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0

def create_youtube_timestamp_url(video_id: str, timestamp_seconds: float) -> str:
    """Create YouTube URL with timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp_seconds)}s"

def create_youtube_embed_timestamp_url(video_id: str, timestamp_seconds: float) -> str:
    """Create YouTube embed URL with timestamp for embedded videos."""
    return f"https://www.youtube.com/embed/{video_id}?start={int(timestamp_seconds)}"

def parse_timestamp_from_text(text: str) -> float:
    """Parse timestamp from transcript text like [1:23]."""
    match = re.search(r'\[(\d+):(\d+)\]', text)
    if match:
        minutes, seconds = map(int, match.groups())
        return minutes * 60 + seconds
    return 0.0

def generate_sections_from_transcript(transcript: str, video_id: str) -> List[Dict[str, Any]]:
    """Generate video sections from transcript without heavy AI processing."""
    if not transcript:
        return []
    
    lines = transcript.strip().split('\n')
    sections = []
    current_section = None
    section_threshold = 120  # Start new section every 2 minutes minimum
    
    for line in lines:
        if not line.strip():
            continue
            
        # Extract timestamp and text
        timestamp_match = re.search(r'\[(\d+):(\d+)\]', line)
        if not timestamp_match:
            continue
            
        minutes, seconds = map(int, timestamp_match.groups())
        timestamp = minutes * 60 + seconds
        text = re.sub(r'\[\d+:\d+\]', '', line).strip()
        
        # Start new section if needed
        if (current_section is None or 
            timestamp - current_section['start_time'] >= section_threshold):
            
            # Finalize previous section
            if current_section:
                current_section['end_time'] = timestamp
                sections.append(current_section)
            
            # Start new section
            current_section = {
                'start_time': timestamp,
                'end_time': timestamp + section_threshold,  # Will be updated
                'title': generate_section_title(text),
                'summary': text[:100] + "..." if len(text) > 100 else text,
                'topics': extract_simple_topics(text),
                'content': [text]
            }
        else:
            # Add to current section
            if current_section:
                current_section['content'].append(text)
                current_section['summary'] = ' '.join(current_section['content'][:3])
                if len(current_section['summary']) > 150:
                    current_section['summary'] = current_section['summary'][:150] + "..."
    
    # Finalize last section
    if current_section:
        # Set end time to be reasonable
        current_section['end_time'] = current_section['start_time'] + len(current_section['content']) * 10
        sections.append(current_section)
    
    # Improve section titles and summaries
    sections = improve_section_metadata(sections)
    
    return sections

def generate_section_title(text: str) -> str:
    """Generate a simple section title from text."""
    # Remove common filler words and get key phrases
    words = text.lower().split()
    
    # Common start phrases that might indicate section topics
    key_phrases = []
    
    # Look for topic indicators
    topic_indicators = [
        'introduction', 'intro', 'overview', 'what is', 'definition',
        'example', 'examples', 'demonstration', 'demo',
        'conclusion', 'summary', 'recap', 'final',
        'next', 'now', 'first', 'second', 'third', 'finally',
        'chapter', 'section', 'part'
    ]
    
    for indicator in topic_indicators:
        if indicator in ' '.join(words[:10]):  # Check first 10 words
            key_phrases.append(indicator.title())
    
    if key_phrases:
        return f"Section: {', '.join(key_phrases[:2])}"
    
    # Fallback: use first few meaningful words
    meaningful_words = [w for w in words[:5] if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'they', 'were', 'been']]
    if meaningful_words:
        return ' '.join(meaningful_words[:3]).title()
    
    return "Video Content"

def extract_simple_topics(text: str) -> List[str]:
    """Extract simple topics from text."""
    # Look for potential topics (capitalized words, technical terms)
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # Filter out common words
    common_words = {'The', 'This', 'That', 'And', 'But', 'For', 'You', 'Can', 'Will', 'Now', 'Here', 'There'}
    topics = [word for word in words if word not in common_words]
    
    # Return unique topics, limited to 5
    return list(set(topics))[:5]

def improve_section_metadata(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Improve section titles and metadata."""
    for i, section in enumerate(sections):
        # Improve title based on content
        content_text = ' '.join(section.get('content', []))
        
        # Look for better title indicators
        if 'introduction' in content_text.lower() or i == 0:
            section['title'] = "Introduction"
        elif 'conclusion' in content_text.lower() or i == len(sections) - 1:
            section['title'] = "Conclusion"
        elif 'example' in content_text.lower():
            section['title'] = "Examples & Demonstration"
        elif any(word in content_text.lower() for word in ['explain', 'definition', 'what is']):
            section['title'] = "Explanation & Concepts"
        elif not section['title'] or section['title'] == "Video Content":
            section['title'] = f"Section {i + 1}"
        
        # Ensure reasonable end times
        if i < len(sections) - 1:
            next_start = sections[i + 1]['start_time']
            section['end_time'] = min(section['end_time'], next_start)
    
    return sections

def load_config():
    """Load configuration from environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
    }
    
    if not config['gemini_api_key']:
        st.error("🔑 Gemini API key not found. Please check your .env file.")
        st.stop()
    
    return config

def display_progress(message: str, progress: float = None):
    """Display progress message with optional progress bar."""
    if progress is not None:
        st.progress(progress, text=message)
    else:
        st.info(message) 