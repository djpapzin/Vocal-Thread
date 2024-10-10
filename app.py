import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
from wordcloud import WordCloud
import streamlit as st
from st_copy_to_clipboard import st_copy_to_clipboard
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import uuid
from youtube_transcript_api import YouTubeTranscriptApi
import emoji
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from googleapiclient.errors import HttpError
import elevenlabs
import json
from PIL import Image
import base64

st.set_page_config(page_title="Vocal Thread: Powered by Gemini AI", page_icon="logo.jpeg") # Set page title and favicon


# Load API keys from Streamlit secrets
gemini_api_key = st.secrets["general"]["GEMINI_API_KEY"]
youtube_api_key = st.secrets["general"]["YOUTUBE_API_KEY"]
elevenlabs_api_key = st.secrets["general"]["ELEVENLABS_API_KEY"]

# --- ElevenLabs setup ---

# Get available voices and select the first one as default
try:
    client = elevenlabs.ElevenLabs(api_key=elevenlabs_api_key)
    voices = client.voices.get_all().voices
    default_voice = voices[0].voice_id if voices else None

    # (Optional) Save voices to JSON (excluding problematic keys)
    if voices:
        simplified_voices = []
        for voice in voices:
            voice_dict = voice.__dict__

            keys_to_remove = ["fine_tuning", "voice_verification", "sharing", "samples"] # Add "samples"
          
            for key in keys_to_remove:
                if key in voice_dict:
                    del voice_dict[key]

            simplified_voices.append(voice_dict)

        with open("voices.json", "w") as f:
            json.dump(simplified_voices, f, indent=4)

except Exception as e:
    st.error(f"Error initializing ElevenLabs: {e}")
    default_voice = None


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Create the model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction= """You are a YouTube comment specialist...""", # (Existing system instruction)
)

# Start a chat session
chat_session = model.start_chat()

# --- Initialize Gemini Pro Exp Model ---
gemini_pro_exp_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0801",
    generation_config=generation_config,
)

gemini_pro_exp_chat_session = gemini_pro_exp_model.start_chat()

safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]
# --- End of Gemini Pro Exp Initialization ---

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    patterns = [
        r"(?<=v=)[^&]+",
        r"(?<=be\/)[^?]+",
        r"(?<=embed\/)[^\"?]+",
        r"(?<=youtu.be\/)[^\"?]+",
        r"(?<=youtube.com/live/)[^?]+"
    ]
    for pattern in patterns:
        video_id = re.search(pattern, url)
        if video_id:
            return video_id.group(0)
    return None

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Function to # Function to scrape YouTube comments
def scrape_youtube_comments(youtube_api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
    comments = []
    try:
        next_page_token = None
        page_count = 0
        progress_bar = st.progress(0, text="Scrutinizing comments...")
        while True:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,  # Max results per page (Maximum is 100)
                pageToken=next_page_token,
                order="time"  # Order by time to potentially improve consistency
            )
            response = request.execute()

            # Extract comments and replies
            for item in response["items"]:
                # Extract top-level comment
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(extract_comment_data(comment))

                # Extract replies (if any)
                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        reply_comment = reply["snippet"]
                        comments.append(extract_comment_data(reply_comment))


            # Check for next page
            if "nextPageToken" in response:
                next_page_token = response["nextPageToken"]
            else:
                break  # Exit loop if no next page

            page_count += 1
            progress_bar.progress(min(page_count / 10, 1.0), text=f"Scrutinizing comments... (Page {page_count})")

        df = pd.DataFrame(comments)
        df["Sentiment"] = df["Comment"].apply(analyze_sentiment)
        total_comments = len(comments)
        return df, total_comments

    except HttpError as e:
        logging.error(f"HTTP error occurred: {e}")
        st.error(f"HTTP error occurred: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error scraping comments: {e}")
        st.error(f"Error scraping comments: {e}")
        return None, None
    
def extract_comment_data(comment):
    return {
        "Comment": comment["textDisplay"],
        "Likes": comment["likeCount"],
        "Name": comment["authorDisplayName"],
        "Time": comment["publishedAt"],
        "Reply Count": comment.get("totalReplyCount", 0)
    }

# Function to generate a word cloud
def generate_word_cloud(text, stopwords=None, colormap='viridis', contour_color='steelblue'):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, colormap=colormap, contour_color=contour_color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# Function to export visualization
def export_visualization(fig, filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    st.success(f"Visualization saved as {filename}")


# Function to get trending videos
def get_trending_videos(youtube_api_key):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
    request = youtube.videos().list(part="snippet,statistics", chart="mostPopular", regionCode="US", maxResults=10)
    response = request.execute()
    videos = []
    for item in response["items"]:
        video = {
            "videoId": item["id"],
            "title": item["snippet"]["title"],
            "channelTitle": item["snippet"]["channelTitle"],
            "viewCount": item["statistics"].get("viewCount", 0),
            "likeCount": item["statistics"].get("likeCount", 0),
            "commentCount": item["statistics"].get("commentCount", 0)
        }
        videos.append(video)
    return videos

# Function to display video metadata
def display_video_metadata(video):
    st.write("Video Title:", video["title"])
    st.write("Channel Title:", video["channelTitle"])
    st.write("View Count:", video["viewCount"])
    st.write("Like Count:", video["likeCount"])
    st.write("Comment Count:", video["commentCount"])


# Function to summarize comments
def summarize_comments(comments):
    summary = None  # Initialize summary to None
    audio_bytes = None  # Initialize audio_bytes to None
    if not comments:
        return "No comments to summarize.", audio_bytes

    all_comments = "\n\n".join(comments)
    prompt = f"""
    Summarize the following YouTube comments into a concise overview, aiming for a length between 150 and 360 words.  Focus on the main topics and sentiment expressed in the comments.

    Comments:
    {all_comments}

    Summary:
    """
    try:
        response = gemini_pro_exp_chat_session.send_message(prompt)
        summary = response.text.strip()

        if default_voice and summary:  # Check both default_voice and summary
            try:
                text_to_read = re.sub(r"[*_`\[\]()#+\-=|{}.!?:;<>~]", "", summary)
                audio_generator = client.generate(text=text_to_read, voice=default_voice)
                audio_bytes = b"".join(list(audio_generator))
            except Exception as e:
                logging.error(f"Error generating ElevenLabs audio: {e}")
                st.error(f"Error generating ElevenLabs audio: {e}")

    except Exception as e:  # Catch summarization errors
        logging.error(f"Error summarizing comments: {e}")
        st.error(f"Error summarizing comments: {e}")  # Display error in Streamlit
        summary = "Error generating summary."  # Assign a value to summary in case of error

    return summary, audio_bytes  # Return summary (even if it's an error message)

# Function to get top comments by likes
def get_top_comments_by_likes(df, top_n=3):
    top_comments = df.nlargest(top_n, "Likes")
    return top_comments[["Name", "Comment", "Likes"]]

# Function to perform in-depth analysis with Gemini Pro Exp
def in_depth_analysis(comments):
    if not comments:
        return "No comments to analyze."

    # Process comments within a thread as a single unit
    threads = []
    current_thread = []
    for comment in comments:
        if comment.startswith("@"):  # Assuming replies start with "@"
            current_thread.append(comment)
        else:
            if current_thread:
                threads.append("\n".join(current_thread))
            current_thread = [comment]
    if current_thread:
        threads.append("\n".join(current_thread))

    all_comments = "\n\n---\n\n".join(threads)
    prompt = f"Provide an in-depth analysis of the following YouTube comment threads, focusing on the overall sentiment, key themes and topics, and any interesting patterns or insights you can identify:\n\n{all_comments}"
    try:
        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error performing in-depth analysis: {e}")
        st.error(f"Error performing in-depth analysis: {e}")  # Display error message to the user
        return "Error performing in-depth analysis. Please try again later."

# Function to perform comparative analysis
def comparative_analysis(dfs, video_ids):
    if not dfs:
        return "No data to compare."

    all_comments = []
    for i, df in enumerate(dfs):
        comments = "\n\n".join(df["Comment"].tolist())
        all_comments.append(f"Comments for Video {video_ids[i]}:\n{comments}")

    all_comments_str = "\n\n---\n\n".join(all_comments)
    prompt = f"Compare and contrast the comments across the following YouTube videos, focusing on the overall sentiment, key themes and topics, and any interesting patterns or insights you can identify:\n\n{all_comments_str}"
    try:
        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error performing comparative analysis: {e}")
        return "Error performing comparative analysis."

# Function to generate video summary
def generate_video_summary(youtube_api_key, video_id, comments):
    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        video_details = response["items"][0]["snippet"]

        title = video_details["title"]
        description = video_details["description"]

        # Get transcript
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
        except Exception as e:
            logging.error(f"Error fetching transcript: {e}")
            transcript_text = "Transcript not available."

        all_comments = "\n\n".join(comments)

        prompt = f"""
        Generate a comprehensive summary of the YouTube video with the following title, description, and transcript:

        Title: {title}
        Description: {description}
        Transcript: {transcript_text}

        Consider the following comments from viewers:

        {all_comments}

        The summary should include:

        * Key topics covered in the video
        * Main points discussed
        * Overall sentiment of the viewers based on the comments
        * Any interesting patterns or insights from the comments

        Please provide a concise and informative summary.
        """

        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()

    except Exception as e:
        logging.error(f"Error generating video summary: {e}")
        st.error(f"Error generating video summary: {e}")  # Display the error to the user
        return "Error generating video summary."


def chat_with_comments(df, question, chat_history, top_n=3):  # Add chat_history, top_n
    try:
        comment_embeddings = []
        for comment in df["Comment"]:
            embedding = genai.embed_content(
                model="models/embedding-004",  # Use embedding-004
                content=comment,
                task_type="RETRIEVAL_DOCUMENT"
            )
            comment_embeddings.append(embedding["embedding"])

        question_embedding = genai.embed_content(
            model="models/embedding-004",  # Use embedding-004
            content=question,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]

        similarities = cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(comment_embeddings))

        # Get indices of top N similar comments
        top_indices = np.argsort(similarities[0])[-top_n:]  
        top_comments = df["Comment"].iloc[top_indices].tolist()

        # Include chat history in the prompt
        context = "\n\n".join(top_comments)  # Use top N comments
        prompt = f"""
        You are a helpful AI assistant. Answer the question based on the context provided and the chat history.

        Chat History:
        {chat_history}

        Context from YouTube Comments:
        {context}

        Question:
        {question}

        Answer:
        """
        response = gemini_pro_exp_chat_session.send_message(prompt)
        return response.text.strip()

    except Exception as e:
        logging.error(f"Error in chat_with_comments: {e}")
        return "Error answering your question. Please try again later." # Or raise the exception for higher-level handling.

def generate_in_depth_analysis(comments):
    """Generates in-depth analysis using Gemini Pro Exp."""

    prompt = f"""
    Please provide an in-depth analysis of the following YouTube comments:

    {comments}

    Your analysis should include:
    * Key themes and topics discussed in the comments
    * Sentiment analysis of the comments (overall sentiment and distribution)
    * Identification of any controversial or polarizing topics
    * Insights into the audience's opinions and perspectives
    * Any other relevant observations or insights
    """

    response = gemini_pro_exp_chat_session.send_message(prompt)
    return response.text

# --- Function to perform common analysis tasks ---
def analyze_comments(df, video_id):
    # Sentiment Analysis Visualization
    with st.expander("Sentiment Analysis", expanded=False):
        sentiment_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)
        export_visualization(fig, "sentiment_analysis.png")

    # Generate Word Cloud
    with st.expander("Word Cloud", expanded=False):
        all_comments = ' '.join(df['Comment'])
        generate_word_cloud(all_comments)

    # In-Depth Analysis with Gemini Pro Exp
    with st.expander("In-Depth Analysis (Gemini Pro Exp)", expanded=False):
        try:
            in_depth_analysis = generate_in_depth_analysis(df["Comment"].tolist())
            st.write(in_depth_analysis)

            # Add copy button
            st_copy_to_clipboard(in_depth_analysis, key="in_depth_analysis_copy_button")

        except Exception as e:
            st.error(f"Error generating in-depth analysis: {e}")

    # Video Summary
    with st.expander("Video Summary (Gemini Pro Exp)", expanded=False):
        summary = generate_video_summary(youtube_api_key, video_id, df["Comment"].tolist())
        st.write(summary)

    # Chat with Comments (moved outside of any expander)
    st.subheader("Chat with Comments")
    user_question = st.chat_input("Ask a question about the comments:")
    if user_question:
        try:
            with st.spinner("Thinking..."):
                answer = chat_with_comments(df, user_question, st.session_state.chat_history)
                st.session_state.chat_history += f"User: {user_question}\nAI: {answer}\n"
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def scrape_live_chat_messages(youtube_api_key, live_chat_id):
    """Scrapes live chat messages from a YouTube live stream."""

    youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
    messages = []

    try:
        next_page_token = None
        while True:
            request = youtube.liveChatMessages().list(
                liveChatId=live_chat_id,
                part="snippet,authorDetails",
                maxResults=2000,  # Adjust as needed
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response["items"]:
                snippet = item["snippet"]
                author_details = item["authorDetails"]
                messages.append({
                    "Message": snippet["displayMessage"],
                    "Author": author_details["displayName"],
                    "Time": snippet["publishedAt"],
                    # Add other relevant fields as needed
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        df = pd.DataFrame(messages)
        return df

    except HttpError as e:
        logging.error(f"HTTP error occurred: {e}")
        st.error(f"HTTP error occurred: {e}")
        return None
    except Exception as e:
        logging.error(f"Error scraping live chat messages: {e}")
        st.error(f"Error scraping live chat messages: {e}")
        return None

def get_live_chat_id(youtube_api_key, video_id):
    """Retrieves the live chat ID for a YouTube live stream."""

    youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)

    try:
        request = youtube.videos().list(
            part="liveStreamingDetails",
            id=video_id
        )
        response = request.execute()

        live_streaming_details = response["items"][0].get("liveStreamingDetails")
        if live_streaming_details:
            return live_streaming_details.get("activeLiveChatId")
        else:
            return None  # Not a live stream or no active chat

    except HttpError as e:
        logging.error(f"HTTP error occurred: {e}")
        st.error(f"HTTP error occurred: {e}")
        return None
    except Exception as e:
        logging.error(f"Error getting live chat ID: {e}")
        st.error(f"Error getting live chat ID: {e}")
        return None
    
# Function to convert an image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load the logo and convert to base64
logo_path = "logo.jpeg"  # Assuming the correct logo file is available
logo_base64 = get_base64_image(logo_path)

# Display logo, title, and description
st.markdown(
    f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="data:image/jpeg;base64,{logo_base64}" style="border-radius: 50%; width: 150px; height: 150px; margin-bottom: 10px;">
        <h1 style="font-size:36px; text-align: center;">🎤 Vocal Thread 💬</h1>
        <h2 style="font-size:16px; text-align: center; font-style: italic; margin-top: -30px;">Where Comments Become Conversations</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = pd.DataFrame()
if "chat_history" not in st.session_state:  # Initialize chat history here
    st.session_state.chat_history = ""

# --- Single Video Analysis ---
st.header("Single Video Analysis")
video_url = st.text_input("Enter YouTube video URL")

# Scrape Comments Button
if st.button("Scrutinize Comments"):
    video_id = extract_video_id(video_url)
    if video_id:
        with st.spinner("Scrutinizing comments..."):
            progress_bar = st.progress(0, text="Scrutinizing comments...")
            df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
            progress_bar.progress(1.0, text="Scrutinizing comments...")
            if df is None or total_comments is None:
                st.error("Error scraping comments. Please try again.")
            else:
                st.success(f"Scrutinizing Complete! Total Comments: {total_comments}")
                st.session_state['df'] = df.copy()
                st.session_state['filtered_df'] = df.copy()
                st.warning("This video is not a live stream or does not have an active live chat.")

                # Get video details
                youtube = build('youtube', 'v3', developerKey=youtube_api_key, cache_discovery=False)
                request = youtube.videos().list(part="snippet,statistics", id=video_id)
                response = request.execute()
                video_details = response["items"][0]

                # Display video thumbnail and details
                st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
                st.write(f"**Title:** {video_details['snippet']['title']}")
                st.write(f"**Views:** {video_details['statistics']['viewCount']}")
                st.write(f"**Likes:** {video_details['statistics']['likeCount']}")

                # Comments Summary
                with st.expander("Comments Summary", expanded=True):
                    summary, audio_bytes = summarize_comments(df["Comment"].tolist())
                    try:
                        if summary:  # This check is now always safe
                            if audio_bytes:  # Check if audio generation was successful - NOW AT THE TOP
                                st.audio(audio_bytes, format="audio/mpeg")

                            sentiment = analyze_sentiment(summary)
                            emoji_for_sentiment = emoji.emojize(
                                ":thumbs_up:" if sentiment == "Positive"
                                else ":thumbs_down:" if sentiment == "Negative"
                                else ":neutral_face:"
                            )
                            summary_with_emoji = f"{emoji_for_sentiment} {summary}"
                            st.write(summary_with_emoji)  # Display the summary with emoji

                    except Exception as e:
                        st.error(f"Error summarizing comments: {e}")

                # Perform common analysis tasks
                analyze_comments(df, video_id)
                
                # Reset Chat History after new comments are scrutinized (Added)
                st.session_state.chat_history = ""
                
                

# --- Comparative Analysis ---
st.header("Comparative Analysis")
video_urls = st.text_area("Enter YouTube video URLs (one per line)")
video_urls = video_urls.strip().splitlines()

if st.button("Compare"):
    video_ids = []
    dfs = []
    for url in video_urls:
        video_id = extract_video_id(url)
        if video_id:
            video_ids.append(video_id)
            with st.spinner(f"Scrutinizing comments for video {video_id}..."):
                df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
                if df is not None:
                    dfs.append(df)

    if dfs:
        with st.spinner("Performing comparative analysis..."):
            analysis = comparative_analysis(dfs, video_ids)
            st.write(analysis)

# --- Trending Videos ---
st.header("Trending Videos")
trending_videos = get_trending_videos(youtube_api_key)
if trending_videos:
    video_selection = st.selectbox("Select a trending video", [f"{video['title']} (by {video['channelTitle']})" for video in trending_videos])
    selected_video = next(video for video in trending_videos if f"{video['title']} (by {video['channelTitle']})" == video_selection)
    
    # Display video thumbnail
    st.image(f"https://img.youtube.com/vi/{selected_video['videoId']}/hqdefault.jpg")

    # Display video metadata below the thumbnail
    display_video_metadata(selected_video)
    
    if st.button("Scrutinize Comments", key=f"scrutinize_comments_{selected_video['videoId']}"):
        video_id = selected_video['videoId']
        with st.spinner("Scrutinizing comments..."):
            progress_bar = st.progress(0, text="Scrutinizing comments...")
            df, total_comments = scrape_youtube_comments(youtube_api_key, video_id)
            progress_bar.progress(1.0, text="Scrutinizing comments...")
            if df is None or total_comments is None:
                st.error("Error scraping comments. Please try again.")
            else:
                st.success(f"Scraping complete! Total Comments: {total_comments}")
                st.session_state['df'] = df.copy()  # Add this line to store the DataFrame in session state
                st.session_state['filtered_df'] = df.copy()  # Add this line to store the DataFrame in session state
                # Perform common analysis tasks
                analyze_comments(df, video_id)
                
                # Reset chat history (Added)
                st.session_state.chat_history = ""

                # Comments Summary
                with st.expander("Comments Summary", expanded=True):
                    try:
                        summary = summarize_comments(df["Comment"].tolist())
                        sentiment = analyze_sentiment(summary)  # Analyze sentiment of the summary
                        emoji_for_sentiment = emoji.emojize(
                            ":thumbs_up:" if sentiment == "Positive"
                            else ":thumbs_down:" if sentiment == "Negative"
                            else ":neutral_face:"
                        )
                        st.write(f"{emoji_for_sentiment} {summary}")  # Add emoji to the summary
                    except Exception as e:
                        st.error(f"Error summarizing comments: {e}")

# Add this at the end of your app.py file
st.markdown("<h5 style='text-align: center;'>Vocal Thread 2024 &copy;</h5>", unsafe_allow_html=True)