import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from detection import analyze_text as analyze_text_model, analyze_image as analyze_image_model, analyze_audio as analyze_audio_model, analyze_video as analyze_video_model


# ==================== RESULTS TRACKING SYSTEM ====================

def initialize_session_state():
    """Initialize session state for tracking results"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'image_analysis_result' not in st.session_state:
        st.session_state.image_analysis_result = None
    if 'show_heatmap' not in st.session_state:
        st.session_state.show_heatmap = False
    if 'audio_analysis_result' not in st.session_state:
        st.session_state.audio_analysis_result = None

def add_result(content_type, ai_score, confidence, interpretation, features_dict=None):
    """Add analysis result to tracking list"""
    result = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'content_type': content_type,
        'ai_score_percent': round(ai_score * 100, 2),
        'confidence': round(confidence * 100, 2),
        'interpretation': interpretation,
        'detection_type': 'AI-Generated' if ai_score > 0.5 else 'Real/Natural',
        'features': json.dumps(features_dict) if features_dict else 'N/A'
    }
    st.session_state.analysis_results.append(result)
    return result

def get_results_dataframe():
    """Convert results to DataFrame for export"""
    if not st.session_state.analysis_results:
        return None
    return pd.DataFrame(st.session_state.analysis_results)

def export_to_csv():
    """Export results to CSV format"""
    df = get_results_dataframe()
    if df is None:
        return None
    return df.to_csv(index=False).encode('utf-8')

def export_to_excel():
    """Export results to Excel format"""
    df = get_results_dataframe()
    if df is None:
        return None
    
    try:
        import openpyxl
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analysis Results')
            
            # Format the Excel file
            worksheet = writer.sheets['Analysis Results']
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).map(len).max(), len(col))
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        output.seek(0)
        return output.getvalue()
    except ImportError:
        # Fallback to None if openpyxl not available
        return None

# ==================== TEXT DETECTION (RULE-BASED) ====================

def compute_lexical_richness(text: str) -> float:
    """Type-to-Token Ratio (TTR) – Higher = more varied vocabulary"""
    tokens = [t.lower().strip('.,!?;:"\'()[]{}') for t in text.split() if t.strip()]
    if len(tokens) < 3:
        return 0.5
    unique = len(set(tokens))
    ttr = unique / len(tokens)
    adjusted_ttr = ttr * 1.2 if len(tokens) < 50 else ttr
    return float(min(adjusted_ttr, 1.0))


def compute_repetition_score(text: str) -> float:
    """Measure trigram repetition – Higher = AI-like"""
    tokens = text.split()
    if len(tokens) < 5:
        return 0.0
    n = len(tokens)
    trigrams = []
    seen_trigrams = set()
    repeats = 0
    for i in range(n - 2):
        tri = tuple(tokens[i:i+3])
        trigrams.append(tri)
        if tri in seen_trigrams:
            repeats += 1
        else:
            seen_trigrams.add(tri)
    if not trigrams:
        return 0.0
    rep_ratio = repeats / len(trigrams)
    return float(min(rep_ratio, 1.0))


def compute_sentence_structure_variance(text: str) -> float:
    """AI has uniform sentence lengths – Humans have natural variation"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 2:
        return 0.5
    lengths = [len(s.split()) for s in sentences]
    if not lengths or len(set(lengths)) < 2:
        return 0.0
    variance = np.var(lengths)
    mean_length = np.mean(lengths)
    if mean_length > 0:
        cv = np.sqrt(variance) / mean_length
    else:
        cv = 0.0
    human_like_score = min(cv / 0.6, 1.0)
    return float(human_like_score)


def compute_punctuation_naturalness(text: str) -> float:
    """Check naturalness – AI is formal/uniform, humans are varied"""
    total_chars = len(text)
    if total_chars < 20:
        return 0.5
    periods = text.count('.')
    exclamations = text.count('!')
    questions = text.count('?')
    commas = text.count(',')
    colons = text.count(':')
    semicolons = text.count(';')
    dashes = text.count('-') + text.count('–')
    total_punct = periods + exclamations + questions + commas + colons + semicolons + dashes
    if total_punct == 0:
        return 0.3
    period_ratio = periods / max(1, total_punct)
    variety_score = 1.0 - period_ratio
    has_emotion = 1.0 if (exclamations + questions) > 0 else 0.5
    naturalness = 0.6 * variety_score + 0.4 * has_emotion
    return float(min(max(naturalness, 0.0), 1.0))


def compute_word_length_distribution(text: str) -> float:
    """AI uses consistent word lengths – Humans vary more"""
    words = [w.strip('.,!?;:') for w in text.split() if w.strip()]
    if len(words) < 5:
        return 0.5
    word_lengths = [len(w) for w in words]
    mean_len = np.mean(word_lengths)
    var_len = np.var(word_lengths)
    if mean_len > 0:
        cv = np.sqrt(var_len) / mean_len
    else:
        cv = 0.0
    human_like = min(cv / 0.6, 1.0)
    return float(human_like)


def analyze_text(text: str) -> dict:
    """Comprehensive text analysis following CORE RULES"""
    return analyze_text_model(text)



# ==================== IMAGE DETECTION (CONTENT-BASED) ====================

# Using detection.analyze_image instead.

# ==================== AUDIO DETECTION ====================

def analyze_audio(audio_file) -> dict:
    """Analyze audio file using the unified detection module"""
    return analyze_audio_model(audio_file)

# ==================== VIDEO DETECTION ====================

def analyze_video(video_file) -> dict:
    """Analyze video file using the unified detection module"""
    import tempfile
    import os
    
    try:
        # Read file into bytes
        video_bytes = video_file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            result = analyze_video_model(tmp_path)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
    except Exception as e:
        return {
            "ai_like": 0.5, "human_like": 0.5,
            "label": "Analysis Error",
            "interpretation": f"Error analyzing video: {str(e)}"
        }




st.set_page_config(page_title="AI Content Detector", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
initialize_session_state()

st.title("🔍 AI Content Detection System")
st.markdown("**Text & Image Analysis — Grammarly-Style UI**")
st.markdown("---")


def draw_confidence_gauge(ai_pct: int) -> None:
    """Draw Grammarly-style semi-circular gauge"""
    color = "red" if ai_pct > 70 else ("yellow" if ai_pct > 40 else "green")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ai_pct,
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'darkgray'},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 33], 'color': "#90EE90"},
                {'range': [33, 66], 'color': "#FFD700"},
                {'range': [66, 100], 'color': "#FF6B6B"}
            ],
        },
        number={'suffix': '%', 'font': {'size': 48, 'color': color}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=320, margin={'l': 20, 'r': 20, 't': 40, 'b': 20})
    st.plotly_chart(fig, use_container_width=True)


def display_unified_report(result: dict, content_type: str) -> None:
    """Standardized result display in the original simple style"""
    ai_pct = int(round(result.get('ai_like', 0.5) * 100))
    human_pct = 100 - ai_pct
    
    # Final Label Logic
    label = result.get('label')
    if not label:
        label = "AI Generated" if ai_pct > 50 else "Human Generated"
    
    # Error state override
    is_error = label == "Analysis Error"
    if is_error:
        ai_pct = 0
        human_pct = 0

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Confidence Meter")
        draw_confidence_gauge(ai_pct)
        
    with col2:
        st.subheader("Result Breakdown")
        st.write(f"### Final Prediction: **{label}**")
        
        if is_error:
            st.write("🔵 AI Confidence → **N/A**")
            st.write("🟢 Human Confidence → **N/A**")
        else:
            st.write(f"🔵 AI Confidence → **{ai_pct}%**")
            st.write(f"🟢 Human Confidence → **{human_pct}%**")
        
        st.markdown("---")
        
        # Adjust subheader based on content type
        if content_type == 'Text':
            st.subheader("Interpretation & Patterns")
        else:
            st.subheader("Interpretation")
            
        st.info(result.get('interpretation', 'No interpretation available.'))
        
        if 'patterns' in result and result['patterns']:
            st.write("**Detected Indicators:**")
            for p in result['patterns']:
                st.write(f"- {p}")
        
        if content_type == 'Video' and 'frame_count' in result:
             st.caption(f"💡 Analyzed {result['frame_count']} key frames.")

def show_heatmap(heatmap: np.ndarray) -> None:
    """Display explainability heatmap"""
    fig, ax = plt.subplots(figsize=(8, 8))
    hm_display = (heatmap * 255).astype(np.uint8)
    sns.heatmap(hm_display, cmap='RdYlGn_r', cbar=True, xticklabels=False, yticklabels=False, ax=ax, cbar_kws={'label': 'AI Suspicion Level'})
    ax.set_title('Explainability Heatmap\n(Red = AI-Generated Regions, Green = Natural Areas)', fontsize=12, fontweight='bold')
    st.pyplot(fig, use_container_width=True)


# ==================== TABS ====================

tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "🎵 Audio Analysis", "🎬 Video Analysis"])

# ==================== TEXT TAB ====================
with tab1:
    st.header("Text AI Content Detection")
    st.markdown("Analyze text for AI-generated language patterns. **Follows CORE RULES:** Mistakes ≠ AI | Polished language = AI-assisted | Mixed content = probabilistic")
    
    text_input = st.text_area(
        "Enter text to analyze:",
        height=300,
        placeholder="Paste your text here. Works best with 50+ words for high confidence.",
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Analyze Text", key="text_btn", use_container_width=True):
        if text_input.strip():
            with st.spinner("Analyzing linguistic patterns..."):
                result = analyze_text(text_input)
                display_unified_report(result, 'Text')
                
                # Track result
                add_result(
                    content_type='Text',
                    ai_score=result['ai_like'],
                    confidence=result.get('confidence', 0.5),
                    interpretation=result['interpretation'],
                    features_dict=result.get('features', {})
                )
                
                with st.expander("📊 Technical Feature Breakdown", expanded=False):
                    feat = result.get('features', {})
                    st.markdown("<div style='padding: 15px; background-color: #f0f2f6; border-radius: 8px;'>", unsafe_allow_html=True)
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.metric("Lexical Richness", f"{feat.get('lexical_richness', 0):.3f}")
                        st.caption("📊 Higher = more varied vocabulary")
                    with col_f2:
                        st.metric("Repetition Score", f"{feat.get('repetition', 0):.3f}")
                        st.caption("📈 Higher = more repetitive")
                    col_f3, col_f4 = st.columns(2)
                    with col_f3:
                        st.metric("Sentence Variance", f"{feat.get('sentence_variance', 0):.3f}")
                        st.caption("📖 Higher = more varied lengths")
                    with col_f4:
                        st.metric("Punctuation", f"{feat.get('punctuation', 0):.3f}")
                        st.caption("✨ Higher = more natural")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Results are probabilistic and based on linguistic patterns. Short or manually typed text may have lower accuracy. For reference only.")

# ==================== IMAGE TAB ====================
with tab2:
    st.header("Image AI Content Detection")
    st.markdown("Analyze images for AI-generated visual patterns. **Content-based only** — camera, upload, URL don't matter. Only visual patterns analyzed.")
    
    image_input = st.file_uploader(
        "Upload an image (PNG, JPG, BMP, WebP):",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed"
    )
    
    if image_input is not None:
        img = Image.open(io.BytesIO(image_input.read()))
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Analyze Image", key="image_btn", use_container_width=True):
                with st.spinner("Analyzing visual patterns with Deep Learning..."):
                    result = analyze_image_model(img)
                    st.session_state.image_analysis_result = result
                    
                    # Track result
                    add_result(
                        content_type='Image',
                        ai_score=result['ai_like'],
                        confidence=result.get('confidence', 0.5),
                        interpretation=result['interpretation'],
                        features_dict=result.get('features', {})
                    )
        
        # Display results if they exist
        if st.session_state.image_analysis_result is not None:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            result = st.session_state.image_analysis_result
            display_unified_report(result, 'Image')
            
            if result.get('heatmap') is not None:
                with st.expander("🌡️ Show Explainability Heatmap", expanded=False):
                    show_heatmap(result['heatmap'])
                    st.caption("**Red regions** = model detected AI-generated patterns (over-smooth texture, unnatural gradients). **Green regions** = natural, realistic areas.")
            else:
                st.warning("⚠️ Explainability heatmap unavailable: Model error state.")
        else:
            st.info("👆 Click button to analyze")
    else:
        st.info("👆 Upload an image to analyze")
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Results are probabilistic and based on visual content patterns. Detection accuracy depends on image quality and resolution. For reference only.")

# ==================== AUDIO TAB ====================
with tab3:
    st.header("Audio AI Content Detection")
    st.markdown("Analyze audio for AI-generated synthesis patterns. Detects spectral flatness, MFCC consistency, and energy distribution.")
    
    if not AUDIO_AVAILABLE:
        st.error("❌ Audio analysis requires librosa and soundfile. Install with: `pip install librosa soundfile`")
    else:
        audio_input = st.file_uploader(
            "Upload an audio file (WAV, MP3, OGG):",
            type=["wav", "mp3", "ogg", "flac"],
            label_visibility="collapsed"
        )
        
        if audio_input is not None:
            st.audio(audio_input, format="audio/wav")
            
            if st.button("🔍 Analyze Audio", key="audio_btn", use_container_width=True):
                with st.spinner("Analyzing audio spectral and temporal patterns..."):
                    result = analyze_audio(audio_input)
                    st.session_state.audio_analysis_result = result
                    
                    if "error" not in result:
                        # Track result
                        add_result(
                            content_type='Audio',
                            ai_score=result['ai_like'],
                            confidence=result.get('confidence', 0.5),
                            interpretation=result['interpretation'],
                            features_dict=result.get('features', {})
                        )
            
            # Display results if they exist
            if st.session_state.audio_analysis_result is not None:
                result = st.session_state.audio_analysis_result
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    display_unified_report(result, 'Audio')
                    
                    with st.expander("📊 Technical Feature Breakdown", expanded=False):
                        feat = result.get('features', {})
                        st.markdown("<div style='padding: 15px; background-color: #f0f2f6; border-radius: 8px;'>", unsafe_allow_html=True)
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            st.metric("Spectral Flatness", f"{feat.get('spectral_flatness', 0):.3f}")
                            st.caption("🎵 Higher = flatter spectrum")
                        with col_f2:
                            st.metric("MFCC Variance", f"{feat.get('mfcc_variance', 0):.3f}")
                            st.caption("📊 Higher = uniform")
                        col_f3, col_f4 = st.columns(2)
                        with col_f3:
                            st.metric("ZCR Stability", f"{feat.get('zcr_stability', 0):.3f}")
                            st.caption("🔄 Higher = stable")
                        with col_f4:
                            st.metric("Energy Uniformity", f"{feat.get('energy_uniformity', 0):.3f}")
                            st.caption("⚡ Higher = uniform")
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("👆 Click button to analyze")
        else:
            st.info("👆 Upload an audio file to analyze")
        
        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** Audio analysis detects synthesis artifacts and spectral patterns. Results are probabilistic. For reference only.")

# ==================== VIDEO TAB ====================
with tab4:
    st.header("Video AI Content Detection")
    st.markdown("Analyze video for AI-generated synthesis patterns. Detects frame consistency, deepfake artifacts, and facial anomalies.")
    
    video_input = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV):",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )
    
    if video_input is not None:
        # Show file info instead of preview
        st.info(f"✅ **Video loaded:** {video_input.name} ({video_input.size / (1024*1024):.2f} MB)")
        
        st.write("")
        if st.button("🔍 Analyze Video", key="video_btn", use_container_width=True):
            with st.spinner("Analyzing video frames..."):
                # Reset pointer before analysis
                video_input.seek(0)
                result = analyze_video(video_input)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Track result
                    add_result(
                        content_type='Video',
                        ai_score=result['ai_like'],
                        confidence=result.get('confidence', 0.5),
                        interpretation=result['interpretation'],
                        features_dict=result.get('features', {})
                    )
                    
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    display_unified_report(result, 'Video')
                    
                    with st.expander("📊 Technical Feature Breakdown", expanded=False):
                        feat = result.get('features', {})
                        st.markdown("<div style='padding: 15px; background-color: #f0f2f6; border-radius: 8px;'>", unsafe_allow_html=True)
                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            st.metric("Frame Consistency", f"{feat.get('frame_consistency', 0):.3f}")
                            st.caption("🎬 Higher = consistent motion")
                        with col_f2:
                            st.metric("Compression Artifacts", f"{feat.get('compression_artifacts', 0):.3f}")
                            st.caption("📊 Higher = more artifacts")
                        col_f3, col_f4 = st.columns(2)
                        with col_f3:
                            st.metric("Flicker Detection", f"{feat.get('flicker_detection', 0):.3f}")
                            st.caption("⚡ Higher = more flicker")
                        with col_f4:
                            st.metric("Face Consistency", f"{feat.get('face_consistency', 0):.3f}")
                            st.caption("👤 Higher = consistent")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display Master Heatmap
                    heatmap_overlay = result.get('heatmap_overlay', None)
                    if heatmap_overlay is not None:
                        st.markdown("---")
                        st.subheader("🔥 Master Heatmap — AI Artifact Detection")
                        st.caption("Aggregated GradCAM heatmap across all sampled frames. Red/hot regions indicate areas where the model detected the strongest AI-generation patterns.")
                        st.image(heatmap_overlay, caption="Master Heatmap Overlay (aggregated across sampled frames)", use_container_width=True)
    else:
        st.info("👆 Upload a video file to analyze")
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Video analysis detects synthesis and deepfake artifacts. Results are probabilistic. For reference only.")

st.markdown("---")

# ==================== RESULTS DOWNLOAD SECTION ====================
st.subheader("📥 Download Analysis Results")

df = get_results_dataframe()
if df is not None and len(df) > 0:
    st.write(f"**Total analyses: {len(df)}**")
    
    # Display results table
    st.dataframe(df, use_container_width=True)
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download as CSV", use_container_width=True, key='csv_btn'):
            csv_data = export_to_csv()
            st.download_button(
                label="📊 Get CSV File",
                data=csv_data,
                file_name=f"ai_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key='csv_download'
            )
    
    with col2:
        if st.button("📋 Download as Excel", use_container_width=True, key='excel_btn'):
            excel_data = export_to_excel()
            if excel_data:
                st.download_button(
                    label="📋 Get Excel File",
                    data=excel_data,
                    file_name=f"ai_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key='excel_download'
                )
            else:
                st.info("Excel export requires openpyxl. Use CSV instead.")
    
    with col3:
        if st.button("🗑️ Clear All Results", use_container_width=True):
            st.session_state.analysis_results = []
            st.success("Results cleared!")
            st.rerun()
else:
    st.info("No analyses yet. Run analyses to see results here.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px;'>
    <p><strong>AI Content Detection System</strong><br/>
    Text & Image Analysis | Probabilistic Results | For Educational & Reference Use Only</p>
</div>
""", unsafe_allow_html=True)
