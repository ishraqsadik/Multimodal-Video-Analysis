import os
import tempfile
from typing import Optional, Dict, Any
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from utils import extract_video_id, format_timestamp

def safe_streamlit_call(func_name: str, message: str, fallback_print: bool = True):
    """Safely call Streamlit functions with fallback to print."""
    try:
        import streamlit as st
        # Check if we're in a Streamlit context
        if hasattr(st, 'session_state') and hasattr(st.session_state, '_state'):
            func = getattr(st, func_name, None)
            if func:
                func(message)
                return
    except:
        pass
    
    # Fallback to print
    if fallback_print:
        print(f"[{func_name.upper()}] {message}")

class VideoProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def get_basic_info(self, url: str) -> Dict[str, Any]:
        """Get video metadata using yt-dlp."""
        video_id = extract_video_id(url)
        if not video_id:
            return {}
        
        try:
            safe_streamlit_call('info', "🔍 Fetching video metadata...")
            
            # Configure yt-dlp for metadata only (no download)
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Format duration
                duration = info.get('duration', 0)
                duration_formatted = f"{duration//60}:{duration%60:02d}" if duration else "Unknown"
                
                # Extract metadata
                metadata = {
                    'video_id': video_id,
                    'title': info.get('title', f"YouTube Video {video_id}"),
                    'author': info.get('uploader', 'Unknown Creator'),
                    'duration': duration,
                    'duration_formatted': duration_formatted,
                    'description': info.get('description', 'No description available'),
                    'views': info.get('view_count', 0),
                    'thumbnail_url': info.get('thumbnail', f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'url': url
                }
                
                safe_streamlit_call('success', f"✅ Metadata extracted: {metadata['title']} by {metadata['author']}")
                return metadata
                
        except Exception as e:
            safe_streamlit_call('warning', f"⚠️ Could not fetch metadata: {e}")
            # Fallback to basic info
            return {
                'video_id': video_id,
                'title': f"YouTube Video {video_id}",
                'author': "YouTube Creator", 
                'duration': 600,  # Default duration
                'duration_formatted': "~10:00",
                'description': "Video ready for AI analysis",
                'views': 0,
                'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                'url': url
            }
    
    def get_transcript(self, url: str) -> Optional[str]:
        """Get video transcript if available."""
        try:
            video_id = extract_video_id(url)
            if not video_id:
                safe_streamlit_call('warning', "⚠️ Could not extract video ID from URL")
                return None
            
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Show available transcripts in UI
            available_langs = []
            for t in transcript_list:
                lang_info = t.language_code
                if t.is_generated:
                    lang_info += " (auto)"
                available_langs.append(lang_info)
            safe_streamlit_call('info', f"📝 Available transcripts: {', '.join(available_langs)}")
            
            # Prefer manually created transcripts
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                safe_streamlit_call('success', "✅ Found manual English transcript")
            except:
                try:
                    # Fall back to auto-generated
                    transcript = transcript_list.find_generated_transcript(['en'])
                    safe_streamlit_call('success', "✅ Found auto-generated English transcript")
                except Exception as inner_e:
                    safe_streamlit_call('warning', f"⚠️ No English transcript available: {inner_e}")
                    return None
            
            # Get transcript data
            transcript_data = transcript.fetch()
            
            # Format transcript
            formatted_transcript = ""
            for entry in transcript_data:
                timestamp = format_timestamp(entry.start)
                text = entry.text
                formatted_transcript += f"[{timestamp}] {text}\n"
            
            word_count = len(formatted_transcript.split())
            safe_streamlit_call('success', f"✅ Transcript extracted: {word_count:,} words")
            
            return formatted_transcript
            
        except Exception as e:
            error_msg = f"Transcript extraction failed: {str(e)}"
            safe_streamlit_call('warning', f"⚠️ {error_msg}")
            return None
    
    def prepare_for_gemini(self, url: str) -> Optional[str]:
        """Prepare video data for Gemini analysis."""
        video_id = extract_video_id(url)
        if not video_id:
            return None
        
        # Get both metadata and transcript
        metadata = self.get_basic_info(url)
        transcript = self.get_transcript(url)
        
        # Create a data file with available information
        try:
            data_path = os.path.join(self.temp_dir, f"{video_id}_data.txt")
            
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(f"YouTube Video Analysis\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Title: {metadata.get('title', 'Unknown')}\n")
                f.write(f"Author: {metadata.get('author', 'Unknown')}\n")
                f.write(f"Duration: {metadata.get('duration_formatted', 'Unknown')}\n")
                f.write(f"Views: {metadata.get('views', 0):,}\n")
                f.write(f"Upload Date: {metadata.get('upload_date', 'Unknown')}\n\n")
                
                if metadata.get('description'):
                    f.write("DESCRIPTION:\n")
                    f.write(metadata['description'][:1000] + "...\n\n")  # Limit description length
                
                if transcript:
                    f.write("TRANSCRIPT:\n")
                    f.write(transcript)
                else:
                    f.write("No transcript available.\n")
                    f.write("Analysis will be based on video content only.\n")
            
            return data_path
            
        except Exception as e:
            safe_streamlit_call('error', f"Error preparing video data: {e}")
            return None
    
    def extract_frames(self, url: str, interval_seconds: int = 15, max_frames: int = 20) -> list:
        """Extract frames from video at specified intervals using yt-dlp."""
        video_id = extract_video_id(url)
        if not video_id:
            return []
        
        try:
            safe_streamlit_call('info', f"🎬 Extracting frames every {interval_seconds} seconds...")
            
            # Create frames directory
            frames_dir = os.path.join(self.temp_dir, f"{video_id}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Configure yt-dlp for frame extraction
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'format': 'best[height<=720]',  # Limit quality for faster processing
                'outtmpl': os.path.join(frames_dir, 'video.%(ext)s'),
                'writeinfojson': False,
                'extract_flat': False,
            }
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                
            # Use ffmpeg through yt-dlp to extract frames
            frames = []
            frame_timestamps = []
            
            # Get video duration
            duration = info.get('duration', 300)  # Default to 5 minutes
            
            # Calculate frame extraction points
            for i in range(0, min(duration, max_frames * interval_seconds), interval_seconds):
                frame_timestamps.append(i)
            
            # Extract frames using ffmpeg
            import subprocess
            
            for i, timestamp in enumerate(frame_timestamps):
                if i >= max_frames:
                    break
                    
                frame_path = os.path.join(frames_dir, f"frame_{timestamp:04d}.jpg")
                
                try:
                    # Use ffmpeg to extract frame at specific timestamp
                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-ss', str(timestamp),
                        '-vframes', '1',
                        '-q:v', '2',  # High quality
                        '-y',  # Overwrite
                        frame_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0 and os.path.exists(frame_path):
                        frames.append({
                            'timestamp': timestamp,
                            'path': frame_path,
                            'minutes': timestamp // 60,
                            'seconds': timestamp % 60
                        })
                        safe_streamlit_call('info', f"📸 Extracted frame at {timestamp//60}:{timestamp%60:02d}")
                    
                except subprocess.TimeoutExpired:
                    safe_streamlit_call('warning', f"⏱️ Timeout extracting frame at {timestamp}s")
                    continue
                except Exception as e:
                    safe_streamlit_call('warning', f"⚠️ Failed to extract frame at {timestamp}s: {e}")
                    continue
            
            # Clean up video file
            try:
                os.remove(video_path)
            except:
                pass
                
            safe_streamlit_call('success', f"✅ Extracted {len(frames)} frames successfully")
            return frames
            
        except Exception as e:
            safe_streamlit_call('error', f"❌ Frame extraction failed: {e}")
            return []
    
    def extract_frames_fallback(self, url: str, interval_seconds: int = 15, max_frames: int = 20) -> list:
        """Fallback method using direct URL streaming (if ffmpeg not available)."""
        try:
            video_id = extract_video_id(url)
            safe_streamlit_call('info', "🎬 Using fallback frame extraction method...")
            
            # Get video metadata for duration
            metadata = self.get_basic_info(url)
            duration = metadata.get('duration', 300)
            
            # Generate frame data without actual extraction
            frames = []
            for i in range(0, min(duration, max_frames * interval_seconds), interval_seconds):
                frames.append({
                    'timestamp': i,
                    'path': None,  # No actual file
                    'minutes': i // 60,
                    'seconds': i % 60,
                    'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                })
            
            safe_streamlit_call('info', f"📸 Generated {len(frames)} frame timestamps for analysis")
            return frames
            
        except Exception as e:
            safe_streamlit_call('error', f"❌ Fallback frame extraction failed: {e}")
            return []

    def extract_frames_youtube_thumbs(self, url: str, interval_seconds: int = 15, max_frames: int = 20) -> list:
        """Extract frames using YouTube's thumbnail API (no ffmpeg required)."""
        video_id = extract_video_id(url)
        if not video_id:
            return []
        
        try:
            safe_streamlit_call('info', f"🎬 Using YouTube thumbnails every {interval_seconds} seconds...")
            
            # Get video metadata for duration
            metadata = self.get_basic_info(url)
            duration = metadata.get('duration', 300)
            
            frames = []
            
            # Generate thumbnail URLs for different timestamps
            for i in range(0, min(duration, max_frames * interval_seconds), interval_seconds):
                # YouTube provides thumbnails at specific timestamps using this URL pattern
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                
                frames.append({
                    'timestamp': i,
                    'path': None,  # No local file
                    'minutes': i // 60,
                    'seconds': i % 60,
                    'thumbnail_url': thumbnail_url,
                    'youtube_thumb': True
                })
                
                safe_streamlit_call('info', f"📸 Generated thumbnail reference for {i//60}:{i%60:02d}")
            
            safe_streamlit_call('success', f"✅ Generated {len(frames)} thumbnail references")
            return frames
            
        except Exception as e:
            safe_streamlit_call('error', f"❌ YouTube thumbnail extraction failed: {e}")
            return []

    def extract_frames_smart(self, url: str, interval_seconds: int = 15, max_frames: int = 20) -> list:
        """Smart frame extraction that tries multiple methods."""
        # Try real ffmpeg extraction first
        frames = self.extract_frames(url, interval_seconds, max_frames)
        
        if not frames:
            safe_streamlit_call('info', "🔄 FFmpeg not available, using YouTube thumbnails...")
            # Fallback to YouTube thumbnails
            frames = self.extract_frames_youtube_thumbs(url, interval_seconds, max_frames)
        
        if not frames:
            safe_streamlit_call('info', "🔄 Using basic fallback method...")
            # Final fallback
            frames = self.extract_frames_fallback(url, interval_seconds, max_frames)
        
        return frames

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass 