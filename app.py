import streamlit as st
import json
import os
from typing import List, Dict, Any
from video_processor import VideoProcessor
from gemini_client import GeminiVideoAnalyzer
from utils import (
    is_valid_youtube_url, 
    load_config, 
    format_timestamp, 
    create_youtube_timestamp_url,
    extract_video_id
)

# Page configuration
st.set_page_config(
    page_title="Multimodal Video Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'video_sections' not in st.session_state:
        st.session_state.video_sections = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'video_info' not in st.session_state:
        st.session_state.video_info = {}
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'current_url' not in st.session_state:
        st.session_state.current_url = ""
    if 'current_timestamp' not in st.session_state:
        st.session_state.current_timestamp = 0
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # Track which tab is active

def display_video_sections(sections: List[Dict], video_id: str):
    """Display video sections as a clean outline with clickable timestamps."""
    st.subheader("📑 Video Outline")
    
    if not sections:
        st.info("No sections available yet.")
        return
    
    # Display as clean outline
    st.markdown("**OUTLINE:**")
    
    for section in sections:
        start_time = section.get('start_time', 0)
        topic = section.get('topic', section.get('title', 'Unknown Topic'))
        timestamp_formatted = format_timestamp(start_time)
        
        # Use very specific ratios and st.write for minimal formatting
        col1, col2 = st.columns([0.12, 0.88])
        
        with col1:
            if st.button(f"[{timestamp_formatted}]", key=f"timestamp_{start_time}", 
                        help="Click to jump to this time in the video"):
                st.session_state.current_timestamp = start_time
                st.rerun()
        
        with col2:
            # Use st.write which typically has less padding than st.markdown
            st.write(f"**{topic}**")
    
    st.markdown("")  # Add spacing

def display_chat_interface(analyzer: GeminiVideoAnalyzer, video_url: str, transcript: str):
    """Display chat interface for video interaction."""
    st.subheader("💬 Chat with Video")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.write(chat['user'])
            
            # Assistant message with functional timestamp buttons
            with st.chat_message("assistant"):
                # Display message with working timestamp buttons
                display_chat_message_with_timestamps(chat['assistant'], i)
    
    # Chat input at the bottom
    user_input = st.chat_input("Ask about the video content, key moments, or get a summary...")
    if user_input:
        with st.spinner("🤔 AI is thinking..."):
            response = analyzer.chat_with_video_content(
                video_url, 
                transcript,
                user_input, 
                st.session_state.chat_history
            )
        
        # Add to chat history - Streamlit will automatically refresh
        st.session_state.chat_history.append({
            'user': user_input,
            'assistant': response
        })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

def display_chat_message_with_timestamps(message: str, message_index: int):
    """Display chat message with timestamp buttons below the text."""
    import re
    
    # First, display the full message text as is
    st.write(message)
    
    # Pattern to find both single timestamps [5:23] and ranges [5:23-5:45]
    timestamp_pattern = r'\[(\d{1,2}):(\d{2})(?:-\d{1,2}:\d{2})?\]'
    
    # Find all timestamps in the message (includes both single and ranges)
    timestamps = re.findall(timestamp_pattern, message)
    
    if timestamps:
        # Create horizontal row of timestamp buttons side by side
        st.markdown("**Jump to timestamps:**")
        
        # Remove duplicates while preserving order
        seen_timestamps = set()
        unique_timestamps = []
        for minutes, seconds in timestamps:
            timestamp_key = (minutes, seconds)
            if timestamp_key not in seen_timestamps:
                seen_timestamps.add(timestamp_key)
                unique_timestamps.append((minutes, seconds))
        
        # Create buttons in a more compact row - limit to 8 per row
        max_per_row = 8
        for row_start in range(0, len(unique_timestamps), max_per_row):
            row_timestamps = unique_timestamps[row_start:row_start + max_per_row]
            row_size = len(row_timestamps)
            
            # Create columns for this row
            cols = st.columns([1] * row_size + [max(1, 8 - row_size)])
            
            for i, (minutes, seconds) in enumerate(row_timestamps):
                minutes = int(minutes)
                seconds = int(seconds)
                total_seconds = minutes * 60 + seconds
                global_index = row_start + i
                
                with cols[i]:
                    if st.button(f"[{minutes}:{seconds:02d}]", 
                               key=f"chat_ts_{message_index}_{global_index}",
                               help="Click to jump to this time in the video"):
                        st.session_state.current_timestamp = total_seconds
                        # Keep rerun for timestamp changes only
                        st.rerun()

def process_inline_timestamps(response: str, message_index: int) -> str:
    """Legacy function - replaced by display_chat_message_with_timestamps."""
    return response

def create_timestamp_handlers():
    """Legacy function - no longer needed."""
    pass

def display_visual_search(analyzer: GeminiVideoAnalyzer, transcript: str, video_id: str):
    """Display simple search interface placeholder."""
    st.subheader("🔍 Video Search")
    
    st.info("🔍 Search for specific content within the video")
    
    search_query = st.text_input(
        "What do you want to find in the video?",
        placeholder="e.g., 'specific topic', 'moment when...', 'explanation of...'",
        key="video_search"
    )
    
    if st.button("🔍 Search in Video", type="primary"):
        if search_query.strip():
            st.info("🔍 Search functionality coming soon...")
        else:
            st.warning("Please enter a search query")

def full_frame_analysis(analyzer: GeminiVideoAnalyzer, video_id: str, query: str):
    """Placeholder for future search implementation."""
    return []

def enhanced_transcript_analysis(analyzer: GeminiVideoAnalyzer, transcript: str, query: str) -> List[Dict]:
    """Placeholder for future search implementation."""
    return []

def combine_and_deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Placeholder for future search implementation."""
    return []

def display_visual_results(results, video_id: str):
    """Display visual search results with thumbnails and timestamp links."""
    if results:
        st.success(f"🎯 Found {len(results)} visual matches!")
        
        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, result in enumerate(results):
            # Get timestamp info
            timestamp = result.get('timestamp', 0)
            confidence = result.get('confidence', 0)
            minutes, seconds = divmod(int(timestamp), 60)
            
            # Create main row with timestamp button and basic info
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Jump to timestamp button - EXACT same pattern as outline
                if st.button(f"[{minutes}:{seconds:02d}]", 
                           key=f"visual_ts_{timestamp}_{i}",
                           help="Click to jump to this time in the video"):
                    st.session_state.current_timestamp = timestamp
                    st.rerun()
            
            with col2:
                description = result.get('description', 'No description available')
                st.markdown(f"**🎬 Match {i+1}:** {description}")
            
            with col3:
                # Confidence indicator
                if confidence >= 8:
                    st.success(f"🟢 {confidence}/10")
                elif confidence >= 6:
                    st.warning(f"🟡 {confidence}/10") 
                else:
                    st.info(f"🔵 {confidence}/10")
            
            # Expandable details
            with st.expander(f"📋 Details for Match {i+1}", expanded=False):
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.markdown(f"**⏰ Time:** {minutes}:{seconds:02d}")
                    st.markdown(f"**🎯 Confidence:** {confidence}/10")
                    
                    # Show source type
                    if result.get('visual_match', False):
                        st.markdown("🎬 **Source:** AI Visual Analysis")
                    elif result.get('fallback_search', False):
                        st.markdown("📝 **Source:** Smart Transcript Search")
                    else:
                        st.markdown("🧠 **Source:** AI Context Analysis")
                
                with col_b:
                    # Show transcript snippet if available
                    if 'transcript_snippet' in result:
                        st.markdown(f"**📄 Transcript Context:**")
                        st.markdown(f"*{result['transcript_snippet'][:150]}...*")
                    
                    # Show actual frame image if available, otherwise YouTube thumbnail
                    frame_path = result.get('frame_path')
                    if frame_path and os.path.exists(frame_path):
                        st.markdown(f"**🎬 Extracted Frame:**")
                        st.image(frame_path, caption=f"Frame at {minutes}:{seconds:02d}", width=300)
                    else:
                        # Fallback to YouTube thumbnail
                        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                        st.image(thumbnail_url, caption=f"Video thumbnail (timestamp: {minutes}:{seconds:02d})", width=200)
            
            st.markdown("---")  # Separator line
    else:
        st.warning("🔍 No visual matches found. Try different search terms or descriptions.")

def process_video_efficiently(video_url: str, video_processor: VideoProcessor, analyzer: GeminiVideoAnalyzer) -> tuple:
    """Process video efficiently to get metadata, transcript, and AI-generated sections."""
    
    # Get video metadata and transcript
    video_id = extract_video_id(video_url)
    
    with st.spinner("🔍 Fetching video information..."):
        metadata = video_processor.get_basic_info(video_url)
        transcript = video_processor.get_transcript(video_url)
    
    # Generate sections using Gemini AI
    sections = []
    if transcript:
        # Get video duration for timestamp validation
        video_duration = metadata.get('duration') if metadata else None
        
        with st.spinner("📑 Analyzing video content for outline..."):
            sections = analyzer.analyze_video_from_transcript(video_url, transcript, video_duration)
            
        if sections:
            st.session_state.sections = sections
        else:
            st.error("❌ Failed to generate outline - please try again")
    else:
        st.warning("⚠️ No transcript available - cannot generate outline")
    
    return metadata, transcript, sections

def main():
    """Main application function."""
    st.title("🎬 Multimodal Video Analysis")
    st.markdown("*Powered by Google Gemini AI*")
    
    # Load configuration
    try:
        config = load_config()
    except:
        st.error("⚠️ Please set up your environment variables. See README.md for instructions.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize components
    video_processor = VideoProcessor()
    analyzer = GeminiVideoAnalyzer(config['gemini_api_key'])
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Enter YouTube URL** and click Process
        2. **Browse Sections** with timestamps
        3. **Chat** about video content
        4. **Search** transcript content
        """)
        
        st.header("ℹ️ Features")
        st.markdown("""
        - 📝 **Transcript Analysis**: Works with available transcripts
        - 🧠 **AI Understanding**: Smart content analysis
        - 💬 **Video Chat**: Ask questions about content
        - 🔍 **Content Search**: Find specific topics
        - 🔗 **Timestamp Links**: Jump to exact moments
        """)
        
        st.header("💡 Tips")
        st.markdown("""
        - Videos with transcripts work best
        - Try educational/tutorial videos
        - Ask specific questions for better responses
        """)
    
    # Main interface
    st.header("🎥 Video Input")
    
    # Video URL input
    col1, col2 = st.columns([3, 1])
    with col1:
        video_url = st.text_input(
            "Enter YouTube video URL:",
            value=st.session_state.current_url,
            placeholder="https://www.youtube.com/watch?v=...",
            key="url_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        process_button = st.button("Process Video 🚀", type="primary")
    
    # Process video when button clicked
    if process_button and video_url.strip():
        if not is_valid_youtube_url(video_url):
            st.error("❌ Please enter a valid YouTube URL")
        else:
            # Reset state for new video
            if video_url != st.session_state.current_url:
                st.session_state.video_processed = False
                st.session_state.video_sections = []
                st.session_state.chat_history = []
                st.session_state.current_url = video_url
                st.session_state.current_timestamp = 0
            
            # Extract video ID for display
            video_id = extract_video_id(video_url)
            
            try:
                # Process video efficiently
                metadata, transcript, sections = process_video_efficiently(video_url, video_processor, analyzer)
                
                # Update session state
                st.session_state.video_metadata = metadata
                st.session_state.video_transcript = transcript
                st.session_state.video_sections = sections
                st.session_state.video_processed = True
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.session_state.video_processed = False
    
    # Display processed video content
    if st.session_state.video_processed and hasattr(st.session_state, 'video_metadata'):
        metadata = st.session_state.video_metadata
        video_id = metadata.get('video_id', '')
        transcript = st.session_state.video_transcript
        
        # Video info
        st.header("📺 Video Information")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Embed YouTube video with current timestamp
            current_time = st.session_state.current_timestamp
            if current_time > 0:
                video_url = f"https://www.youtube.com/embed/{video_id}?start={int(current_time)}&autoplay=1"
            else:
                video_url = f"https://www.youtube.com/embed/{video_id}"
            
            # Use HTML iframe for better control
            iframe_html = f'''
            <iframe width="560" height="315" 
                    src="{video_url}" 
                    title="YouTube video player" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                    allowfullscreen>
            </iframe>
            '''
            st.components.v1.html(iframe_html, height=330)
        
        with col2:
            st.markdown(f"**🎬 Title:** {metadata.get('title', 'Unknown')}")
            st.markdown(f"**📺 Channel:** {metadata.get('author', 'Unknown')}")
        
        # Tabs for different features
        tab1, tab2, tab3 = st.tabs(["📑 Outline", "💬 Chat", "🔍 Search"])
        
        with tab1:
            if st.session_state.video_sections:
                display_video_sections(st.session_state.video_sections, video_id)
            else:
                st.info("No outline generated yet.")
        
        with tab2:
            display_chat_interface(analyzer, st.session_state.current_url, transcript)
        
        with tab3:
            display_visual_search(analyzer, transcript, video_id)

if __name__ == "__main__":
    main() 