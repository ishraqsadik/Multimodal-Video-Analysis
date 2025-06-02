import google.generativeai as genai
import time
from typing import List, Dict, Any, Optional
import streamlit as st
from utils import format_timestamp, extract_video_id
import re
from PIL import Image
import base64
import io
import os

class GeminiVideoAnalyzer:
    def __init__(self, api_key: str):
        """Initialize Gemini client with API key."""
        genai.configure(api_key=api_key)
        # Use Gemini 2.0 Flash-Lite for higher rate limits (30 RPM vs 15 RPM)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
    def _create_smart_fallback_sections(self, transcript: str, video_id: str, video_duration: int = None) -> List[Dict[str, Any]]:
        """Create intelligent sections without AI as fallback."""
        if not transcript:
            return []
        
        lines = transcript.strip().split('\n')
        sections = []
        
        # Look for natural breaks in content
        current_section = None
        
        for line in lines:
            if not line.strip():
                continue
                
            # Extract timestamp and text
            timestamp_match = re.search(r'\[(\d+):(\d+)\]', line)
            if not timestamp_match:
                continue
                
            minutes, seconds = map(int, timestamp_match.groups())
            timestamp = minutes * 60 + seconds
            
            # Skip timestamps that exceed video duration
            if video_duration and timestamp > video_duration:
                continue
                
            text = re.sub(r'\[\d+:\d+\]', '', line).strip().lower()
            
            # Generic topic detection keywords
            topic_keywords = {
                'introduction': ['hello', 'welcome', 'today', 'going to', 'introduce', 'start'],
                'explanation': ['what is', 'means', 'definition', 'explain', 'understand'],
                'example': ['example', 'for instance', 'let me show', 'demonstration'],
                'discussion': ['discuss', 'talk about', 'look at', 'analyze', 'review'],
                'conclusion': ['conclusion', 'summary', 'that\'s it', 'thank you', 'wrap up']
            }
            
            # Start new section based on topic changes only
            start_new_section = False
            detected_topic = None
            
            if current_section is None:
                start_new_section = True
                detected_topic = "Introduction"
            else:
                # Check for topic change - no minimum time requirement
                for topic_name, keywords in topic_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        start_new_section = True
                        detected_topic = topic_name.title()
                        break
                
                # Create new section every 3 minutes if no topic change detected (fallback only)
                if not start_new_section and timestamp - current_section['start_time'] >= 180:
                    start_new_section = True
                    detected_topic = f"Section {len(sections) + 2}"
            
            if start_new_section:
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'start_time': timestamp,
                    'topic': detected_topic
                }
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        # Clean up section names and validate timestamps
        for i, section in enumerate(sections):
            # Cap timestamp at video duration
            if video_duration:
                section['start_time'] = min(section['start_time'], video_duration)
            
            if i == len(sections) - 1 and 'conclusion' not in section['topic'].lower():
                section['topic'] = "Conclusion"
            elif section['topic'].startswith('Section') and i == 0:
                section['topic'] = "Introduction"
        
        return sections
        
    def analyze_video_from_transcript(self, video_url: str, transcript: Optional[str] = None, video_duration: int = None) -> List[Dict[str, Any]]:
        """Analyze video using transcript and generate intelligent section breakdown."""
        if not transcript:
            return []
        
        import re  # Single import at the top
        
        # Simplified approach: Use larger, meaningful chunks
        lines = transcript.strip().split('\n')
        sections = []
        
        # Debug: Check transcript length for long videos
        total_lines = len(lines)
        if total_lines > 1000:  # Long video
            # Check actual last timestamp in transcript
            last_timestamp = None
            for line in reversed(lines[-50:]):  # Check last 50 lines
                timestamp_match = re.search(r'\[(\d+):(\d+)\]', line)
                if timestamp_match:
                    minutes, seconds = map(int, timestamp_match.groups())
                    last_timestamp = minutes * 60 + seconds
                    break
            
            if last_timestamp:
                if video_duration and last_timestamp > video_duration + 600:  # 10+ minutes beyond
                    video_duration = None  # Disable duration checking for this video
        
        chunk_size = 50 if total_lines < 1000 else 100  # Smaller chunks for short videos, larger for long videos
        processed_chunks = 0
        skipped_chunks = 0
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            if not chunk_text.strip():
                skipped_chunks += 1
                continue
                
            # Extract first timestamp from this chunk
            first_timestamp_match = re.search(r'\[(\d+):(\d+):?(\d+)?\]', chunk_lines[0])
            if not first_timestamp_match:
                skipped_chunks += 1
                continue
                
            # Handle both [MM:SS] and [H:MM:SS] formats
            if first_timestamp_match.group(3):  # [H:MM:SS] format
                hours = int(first_timestamp_match.group(1))
                minutes = int(first_timestamp_match.group(2))
                seconds = int(first_timestamp_match.group(3))
                chunk_start_time = hours * 3600 + minutes * 60 + seconds
            else:  # [MM:SS] format
                minutes = int(first_timestamp_match.group(1))
                seconds = int(first_timestamp_match.group(2))
                chunk_start_time = minutes * 60 + seconds
            
            # Debug: Show all timestamps in this chunk
            chunk_timestamps = re.findall(r'\[(\d+):(\d+):?(\d+)?\]', chunk_text)
            chunk_times_formatted = []
            for match in chunk_timestamps[:5]:  # Show first 5
                if match[2]:  # H:MM:SS
                    chunk_times_formatted.append(f"{match[0]}:{match[1]}:{match[2]}")
                else:  # MM:SS
                    chunk_times_formatted.append(f"{match[0]}:{match[1]}")
            
            # For long videos, be more lenient with duration checking
            if video_duration and chunk_start_time > video_duration:
                # Only skip if way beyond (20% buffer for long videos)
                buffer = video_duration * 0.2 if video_duration > 3600 else 300  # 20% for long videos, 5 min for short
                if chunk_start_time > video_duration + buffer:
                    break
                
            processed_chunks += 1
            
            try:
                # Ask for multiple sections from this larger chunk
                prompt = f"""
                Analyze this transcript chunk and identify the main topics discussed.
                
                IMPORTANT: {"Return EXACTLY 1 section for the main topic in this chunk." if total_lines > 1000 else "Return 1-3 sections for significant topic changes in this chunk."}
                
                CHUNK (starting at {chunk_start_time} seconds):
                {chunk_text}
                
                CRITICAL RULES:
                - ONLY use timestamps that appear EXACTLY in the transcript above
                - Available timestamps: {chunk_times_formatted}
                - DO NOT create, estimate, or modify any timestamps
                - Convert format: [3:24] = 204 seconds, [1:03:45] = 3825 seconds
                - Topic headings must be 5-7 words maximum
                
                Return JSON array with {"EXACTLY 1 object" if total_lines > 1000 else "1-3 objects"}:
                [
                    {{"start_time": 204, "topic": "Challenge Introduction and Setup"}}
                ]
                
                {"RETURN ONLY 1 SECTION with concise 5-7 word topic." if total_lines > 1000 else "Focus on major topic changes with concise 5-7 word topics."}
                """
                
                # Retry logic for rate limits
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.model.generate_content(prompt)
                        response_text = response.text.strip().replace('```json', '').replace('```', '')
                        break  # Success, exit retry loop
                    except Exception as api_error:
                        if "429" in str(api_error) or "quota" in str(api_error).lower():
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                                time.sleep(wait_time)
                                continue
                            else:
                                raise api_error  # Final attempt failed
                        else:
                            raise api_error  # Non-rate-limit error
                
                # Parse JSON response
                import json
                try:
                    chunk_sections = json.loads(response_text)
                    if isinstance(chunk_sections, list):
                        for section in chunk_sections:
                            if isinstance(section, dict) and 'start_time' in section and 'topic' in section:
                                # Handle both timestamp formats: seconds (276) or string ("8:57")
                                start_time = section['start_time']
                                original_start_time = start_time  # For debugging
                                
                                if isinstance(start_time, str):
                                    # Convert "8:57" or "1:14:50" to seconds
                                    try:
                                        if ':' in start_time:
                                            parts = start_time.split(':')
                                            if len(parts) == 3:  # H:MM:SS format
                                                hours = int(parts[0])
                                                minutes = int(parts[1])
                                                seconds = int(parts[2])
                                                start_time = hours * 3600 + minutes * 60 + seconds
                                            elif len(parts) == 2:  # MM:SS format
                                                minutes = int(parts[0])
                                                seconds = int(parts[1])
                                                start_time = minutes * 60 + seconds
                                        else:
                                            start_time = int(start_time)
                                    except Exception as e:
                                        continue  # Skip if conversion fails
                                
                                section['start_time'] = start_time  # Ensure it's stored as seconds
                                
                                # Validate timestamp exists in this chunk
                                chunk_timestamp_seconds = set()
                                for match in chunk_timestamps:
                                    if match[2]:  # H:MM:SS
                                        hours = int(match[0])
                                        minutes = int(match[1])
                                        seconds = int(match[2])
                                        chunk_timestamp_seconds.add(hours * 3600 + minutes * 60 + seconds)
                                    else:  # MM:SS
                                        minutes = int(match[0])
                                        seconds = int(match[1])
                                        chunk_timestamp_seconds.add(minutes * 60 + seconds)
                                
                                if start_time not in chunk_timestamp_seconds:
                                    continue  # Skip this section
                                
                                # Check if timestamp exceeds video duration
                                if video_duration and start_time > video_duration:
                                    overshoot = start_time - video_duration
                                    overshoot_mins = overshoot // 60
                                    overshoot_secs = overshoot % 60
                                    section_time_formatted = f"{start_time//60}:{start_time%60:02d}"
                                
                                sections.append(section)
                except Exception as e:
                    # If JSON parsing fails, skip this chunk
                    continue
                
            except Exception as e:
                # Skip this chunk if AI fails
                continue
        
        # Remove duplicates and sort by time
        unique_sections = []
        seen_times = set()
        for section in sorted(sections, key=lambda x: x['start_time']):
            if section['start_time'] not in seen_times:
                unique_sections.append(section)
                seen_times.add(section['start_time'])
        
        return unique_sections
    
    def chat_with_video_content(self, video_url: str, transcript: Optional[str], user_message: str, chat_history: List[Dict] = None) -> str:
        """Chat about video content with comprehensive context."""
        try:
            # Build comprehensive context
            context = "You are an AI assistant analyzing a YouTube video. Answer questions based on the complete video content.\n\n"
            context += f"Video URL: {video_url}\n\n"
            
            if transcript:
                # Provide much more context using smart chunking
                enhanced_context = self._build_enhanced_transcript_context(transcript, user_message)
                context += enhanced_context
                
                context += "\n\nIMPORTANT INSTRUCTIONS:\n"
                context += "- When referring to specific parts of the video, ALWAYS include the timestamp in [MM:SS] format\n"
                context += "- Use timestamps like: 'At [5:23], the speaker mentions...' or 'Around [12:45], you can see...'\n"
                context += "- Only use timestamps that actually exist in the transcript\n"
                context += "- Be specific about when things happen in the video\n"
                context += "- Draw from the ENTIRE video content, not just the beginning\n\n"
            else:
                context += "No transcript available. Provide general guidance about video analysis.\n\n"
            
            # Include recent chat history for continuity
            if chat_history:
                context += "Previous conversation:\n"
                for msg in chat_history[-5:]:  # Last 5 messages for better continuity
                    context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
            
            context += f"Current question: {user_message}\n\n"
            context += "Provide a comprehensive answer based on the full video content. Include relevant timestamps."
            
            response = self.model.generate_content(context)
            return response.text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _build_enhanced_transcript_context(self, transcript: str, user_message: str) -> str:
        """Build enhanced transcript context with intelligent selection."""
        if not transcript:
            return ""
        
        # Get video summary first
        video_summary = self._generate_video_summary(transcript)
        
        # Get relevant chunks based on user question
        relevant_chunks = self._find_relevant_transcript_chunks(transcript, user_message)
        
        # Build context
        context = f"VIDEO SUMMARY:\n{video_summary}\n\n"
        
        if relevant_chunks:
            context += "RELEVANT TRANSCRIPT SECTIONS:\n"
            for chunk in relevant_chunks:
                context += f"{chunk}\n\n"
        
        # Always include some beginning and end context
        lines = transcript.split('\n')
        beginning = '\n'.join(lines[:20])  # First 20 lines
        ending = '\n'.join(lines[-20:])    # Last 20 lines
        
        context += f"VIDEO BEGINNING:\n{beginning}\n\n"
        context += f"VIDEO ENDING:\n{ending}\n\n"
        
        # If we have space, include more middle content
        if len(context) < 15000:  # Leave room for more content
            middle_start = len(lines) // 3
            middle_end = (2 * len(lines)) // 3
            middle = '\n'.join(lines[middle_start:middle_end])
            context += f"VIDEO MIDDLE SECTION:\n{middle[:5000]}\n\n"
        
        return context
    
    def _generate_video_summary(self, transcript: str) -> str:
        """Generate a summary of the entire video."""
        try:
            # Use first portion of transcript to generate summary
            summary_prompt = f"""
            Analyze this video transcript and provide a comprehensive summary covering:
            1. Main topic/subject
            2. Key points discussed
            3. Overall structure/flow
            4. Important timestamps and topics
            
            TRANSCRIPT (partial):
            {transcript[:8000]}
            
            Provide a detailed summary that captures the essence of the entire video.
            """
            
            response = self.model.generate_content(summary_prompt)
            return response.text
            
        except Exception as e:
            return f"Video about: {transcript[:200]}..."
    
    def _find_relevant_transcript_chunks(self, transcript: str, user_message: str) -> List[str]:
        """Find transcript chunks most relevant to user's question."""
        lines = transcript.split('\n')
        chunks = []
        
        # Create chunks of ~50 lines each
        chunk_size = 50
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i + chunk_size])
            chunks.append(chunk)
        
        # Score chunks based on relevance to user question
        user_words = user_message.lower().split()
        scored_chunks = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Simple relevance scoring
            score = sum(1 for word in user_words if word in chunk_lower and len(word) > 3)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Return top 3 most relevant chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:3]]
    
    def search_transcript_content(self, transcript: Optional[str], query: str) -> List[Dict[str, Any]]:
        """Search for content in transcript."""
        if not transcript:
            return []
        
        try:
            prompt = f"""
            Search through this video transcript for content matching: "{query}"
            
            TRANSCRIPT:
            {transcript}
            
            Find relevant moments and return as JSON:
            [
                {{
                    "timestamp": 45.5,
                    "duration": 10.0,
                    "description": "Description of matching content",
                    "confidence": 8,
                    "text_snippet": "Relevant transcript text"
                }}
            ]
            
            Only include results with confidence >= 6.
            """
            
            response = self.model.generate_content(prompt)
            
            import json
            try:
                results = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
                return results
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            st.error(f"Error searching content: {str(e)}")
            return []

    def analyze_frames_for_visual_search(self, frames: List[Dict], query: str) -> List[Dict]:
        """Analyze extracted video frames using Gemini Vision API to find visual content."""
        results = []
        
        if not frames:
            return results
        
        # Analyze each frame
        for i, frame_data in enumerate(frames):
            try:
                timestamp = frame_data['timestamp']
                frame_path = frame_data.get('path')
                
                if not frame_path or not os.path.exists(frame_path):
                    continue
                
                # Load and prepare image
                image = Image.open(frame_path)
                
                # Create vision analysis prompt
                vision_prompt = f"""
                Analyze this video frame image to determine if it contains: "{query}"
                
                VISUAL ANALYSIS TASK:
                1. Look carefully at all objects, people, scenes, and activities in the image
                2. Determine if "{query}" is visible in any form
                3. Consider partial views, background elements, and context
                4. Rate confidence from 1-10 based on visual evidence
                
                RESPOND WITH:
                - DETECTED: Yes/No (is "{query}" visible?)
                - CONFIDENCE: 1-10 (how certain are you?)
                - DESCRIPTION: Brief description of what you see related to "{query}"
                - CONTEXT: Overall scene description
                
                Only report confidence 7+ as positive matches.
                Be specific about what you observe in the image.
                """
                
                # Analyze frame with Gemini Vision
                response = self.model.generate_content([vision_prompt, image])
                analysis_text = response.text.lower()
                
                # Parse response
                detected = "yes" in analysis_text and "detected:" in analysis_text
                
                # Extract confidence score
                confidence = 0
                confidence_match = re.search(r'confidence[:\s]*(\d+)', analysis_text)
                if confidence_match:
                    confidence = int(confidence_match.group(1))
                
                # Extract description
                description_match = re.search(r'description[:\s]*([^\n]+)', analysis_text, re.IGNORECASE)
                description = description_match.group(1).strip() if description_match else f"Frame analysis for '{query}'"
                
                # Only include high-confidence positive detections
                if detected and confidence >= 7:
                    results.append({
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'description': f"🎬 Visual Detection: {description}",
                        'visual_match': True,
                        'frame_analysis': True,
                        'frame_path': frame_path,
                        'analysis_text': response.text
                    })
                
                # Add small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error analyzing frame at {timestamp}s: {e}")
                continue
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def analyze_frames_batch(self, frames: List[Dict], query: str, batch_size: int = 5) -> List[Dict]:
        """Analyze frames in batches to optimize API usage and speed."""
        results = []
        
        # Check if we have actual frames or just YouTube thumbnails
        has_real_frames = any(frame.get('path') and os.path.exists(frame['path']) for frame in frames)
        
        if not has_real_frames:
            # Use YouTube thumbnails with contextual analysis
            return self.analyze_youtube_thumbnails(frames, query)
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            try:
                # Create batch analysis prompt
                batch_prompt = f"""
                Analyze these {len(batch)} video frames to find visual content: "{query}"
                
                For each frame, determine:
                1. Is "{query}" visible? (Yes/No)
                2. Confidence level (1-10)
                3. Brief description of what you see
                
                Respond in this format for each frame:
                Frame [timestamp]: DETECTED: Yes/No, CONFIDENCE: X, DESCRIPTION: [what you see]
                
                Only report frames with confidence 7+.
                """
                
                # Prepare images for batch processing
                content = [batch_prompt]
                valid_frames = []
                
                for frame_data in batch:
                    frame_path = frame_data.get('path')
                    if frame_path and os.path.exists(frame_path):
                        try:
                            image = Image.open(frame_path)
                            content.append(image)
                            valid_frames.append(frame_data)
                        except:
                            continue
                
                if not valid_frames:
                    continue
                
                # Analyze batch
                response = self.model.generate_content(content)
                batch_results = self._parse_batch_analysis(response.text, valid_frames, query)
                results.extend(batch_results)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in batch analysis: {e}")
                # Fallback to individual analysis
                for frame_data in batch:
                    try:
                        individual_result = self.analyze_frames_for_visual_search([frame_data], query)
                        results.extend(individual_result)
                    except:
                        continue
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def analyze_youtube_thumbnails(self, frames: List[Dict], query: str) -> List[Dict]:
        """Analyze using YouTube thumbnails and transcript-based reasoning."""
        results = []
        
        # Get transcript for contextual analysis
        transcript = st.session_state.get('video_transcript', '') if 'st' in globals() else ''
        
        for frame_data in frames:
            timestamp = frame_data['timestamp']
            
            # Find transcript content around this timestamp
            context_text = self.get_transcript_context(transcript, timestamp)
            
            if context_text:
                # Use AI to determine likelihood of visual content based on context
                confidence = self.estimate_visual_likelihood(context_text, query, timestamp)
                
                if confidence >= 6:
                    results.append({
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'description': f"🎯 Context-based detection: {query} likely visible",
                        'visual_match': False,
                        'context_analysis': True,
                        'transcript_snippet': context_text[:100] + "...",
                        'thumbnail_url': frame_data.get('thumbnail_url')
                    })
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def get_transcript_context(self, transcript: str, timestamp: int) -> str:
        """Get transcript context around a specific timestamp."""
        if not transcript:
            return ""
        
        lines = transcript.split('\n')
        context_lines = []
        
        # Find lines within 30 seconds of target timestamp
        for line in lines:
            timestamp_match = re.search(r'\[(\d+):(\d+)\]', line)
            if timestamp_match:
                minutes, seconds = map(int, timestamp_match.groups())
                line_timestamp = minutes * 60 + seconds
                
                if abs(line_timestamp - timestamp) <= 30:  # Within 30 seconds
                    context_lines.append(line.strip())
        
        return '\n'.join(context_lines)
    
    def estimate_visual_likelihood(self, context_text: str, query: str, timestamp: int) -> int:
        """Estimate likelihood of visual content based on transcript context."""
        try:
            prompt = f"""
            Based on this transcript context around timestamp {timestamp} seconds, 
            estimate the likelihood (1-10) that "{query}" is visually present:
            
            CONTEXT:
            {context_text}
            
            Consider:
            - Direct mentions of "{query}"
            - Related activities or scenarios
            - Visual cues in speech ("look at", "you can see", "here's")
            - Environmental context
            
            Return only a number 1-10 representing visual likelihood.
            """
            
            response = self.model.generate_content(prompt)
            likelihood_text = response.text.strip()
            
            # Extract number from response
            number_match = re.search(r'(\d+)', likelihood_text)
            if number_match:
                return min(10, max(1, int(number_match.group(1))))
            
            return 5  # Default moderate likelihood
            
        except:
            return 5  # Default fallback

    def _parse_batch_analysis(self, analysis_text: str, frames: List[Dict], query: str) -> List[Dict]:
        """Parse batch analysis response into structured results."""
        results = []
        lines = analysis_text.split('\n')
        
        for line in lines:
            if 'frame' in line.lower() and 'detected:' in line.lower():
                try:
                    # Extract timestamp
                    timestamp_match = re.search(r'frame\s*\[?(\d+)\]?', line.lower())
                    if not timestamp_match:
                        continue
                    
                    timestamp = int(timestamp_match.group(1))
                    
                    # Find corresponding frame
                    frame_data = next((f for f in frames if f['timestamp'] == timestamp), None)
                    if not frame_data:
                        continue
                    
                    # Extract detection info
                    detected = 'detected: yes' in line.lower()
                    confidence_match = re.search(r'confidence[:\s]*(\d+)', line.lower())
                    confidence = int(confidence_match.group(1)) if confidence_match else 0
                    
                    # Extract description
                    desc_match = re.search(r'description[:\s]*(.+)', line, re.IGNORECASE)
                    description = desc_match.group(1).strip() if desc_match else f"Visual detection of '{query}'"
                    
                    if detected and confidence >= 7:
                        results.append({
                            'timestamp': timestamp,
                            'confidence': confidence,
                            'description': f"🎬 Visual Detection: {description}",
                            'visual_match': True,
                            'frame_analysis': True,
                            'frame_path': frame_data.get('path'),
                            'batch_analysis': True
                        })
                
                except Exception as e:
                    continue
        
        return results 