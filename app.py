"""
Voice Maker - PDF to TXT & Excel to TTS
"""

import asyncio
import io
import zipfile
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import edge_tts
import easyocr
import numpy as np
from PIL import Image

# TTS 음성 설정
VOICES = {
    "English (US)": "en-US-AriaNeural",
    "English (UK)": "en-GB-SoniaNeural",
    "German": "de-DE-KatjaNeural",
    "Korean": "ko-KR-SunHiNeural",
    "Japanese": "ja-JP-NanamiNeural",
    "Chinese": "zh-CN-XiaoxiaoNeural",
    "Turkish": "tr-TR-EmelNeural",
}

# OCR 리더 (캐시)
@st.cache_resource
def get_ocr_reader(langs):
    return easyocr.Reader(langs, gpu=False)


async def generate_audio(text: str, voice: str) -> bytes:
    """텍스트를 음성으로 변환"""
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


def extract_text_from_pdf(pdf_file, use_ocr=False, ocr_lang="ko") -> str:
    """PDF에서 텍스트 추출"""
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    all_text = []
    
    for page_num, page in enumerate(doc):
        text = ""
        
        if not use_ocr:
            # 일반 텍스트 추출
            text = page.get_text("text")
        
        # OCR 사용 또는 텍스트가 없을 때
        if use_ocr or not text.strip():
            # 페이지를 이미지로 변환
            mat = fitz.Matrix(2, 2)  # 해상도 2배
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # PIL 이미지로 변환
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            
            # OCR 실행
            lang_map = {"ko": ["ko", "en"], "en": ["en"], "de": ["de", "en"], "ja": ["ja", "en"], "ch": ["ch_sim", "en"]}
            langs = lang_map.get(ocr_lang, ["ko", "en"])
            
            reader = get_ocr_reader(langs)
            results = reader.readtext(img_array)
            
            # 결과 정렬 (위에서 아래로)
            results.sort(key=lambda x: x[0][0][1])
            text = "\n".join([r[1] for r in results])
        
        if text.strip():
            all_text.append(f"--- 페이지 {page_num + 1} ---\n{text.strip()}")
    
    doc.close()
    
    if not all_text:
        return "⚠️ 텍스트를 추출할 수 없습니다."
    
    return "\n\n".join(all_text)


def main():
    st.set_page_config(page_title="Voice Maker", page_icon="🎙️")
    st.title("🎙️ Voice Maker")

    tab1, tab2 = st.tabs(["📄 PDF → TXT", "🔊 Excel → TTS"])

    # ========================================
    # Part 1: PDF → TXT
    # ========================================
    with tab1:
        st.header("PDF를 TXT로 변환")
        
        col1, col2 = st.columns(2)
        with col1:
            use_ocr = st.checkbox("🔍 OCR 사용 (이미지 PDF용)", value=False)
        with col2:
            ocr_lang = st.selectbox("OCR 언어", ["ko", "en", "de", "ja", "ch"], 
                                    format_func=lambda x: {"ko": "한국어", "en": "영어", "de": "독일어", "ja": "일본어", "ch": "중국어"}[x])
        
        if use_ocr:
            st.info("⚠️ OCR은 첫 실행시 모델 다운로드로 시간이 걸릴 수 있습니다.")
        
        pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], key="pdf")
        
        if pdf_file:
            with st.spinner("PDF 처리 중... (OCR 사용시 시간이 걸릴 수 있습니다)"):
                text = extract_text_from_pdf(pdf_file, use_ocr=use_ocr, ocr_lang=ocr_lang)
            
            st.text_area("추출된 텍스트", text, height=300)
            
            # 다운로드 버튼
            filename = pdf_file.name.replace(".pdf", ".txt")
            st.download_button(
                label="📥 TXT 다운로드",
                data=text.encode("utf-8"),
                file_name=filename,
                mime="text/plain"
            )

    # ========================================
    # Part 2: Excel → TTS
    # ========================================
    with tab2:
        st.header("Excel을 음성 파일로 변환")
        
        # 음성 선택
        voice_name = st.selectbox("음성 선택", list(VOICES.keys()))
        voice = VOICES[voice_name]
        
        st.markdown("""
        **Excel 형식:**
        | 1번 컬럼 (파일명) | 2번 컬럼 (내용) |
        |------------------|-----------------|
        | 001 | Hello world |
        | 002 | Good morning |
        """)
        
        excel_file = st.file_uploader("Excel 파일 업로드", type=["xlsx"], key="excel")
        
        if excel_file:
            # Excel 읽기 (헤더 없이)
            df = pd.read_excel(excel_file, header=None, engine="openpyxl")
            
            # 첫 번째 행이 헤더처럼 보이면 스킵 옵션
            if st.checkbox("첫 번째 행이 헤더인 경우 스킵", value=False):
                df = df.iloc[1:]
            
            # 빈 행 제거
            df = df.dropna(subset=[0, 1])
            
            st.write(f"**총 {len(df)}개 항목**")
            st.dataframe(df.head(10))
            
            if st.button("🚀 음성 생성", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                # ZIP 파일 생성
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, (_, row) in enumerate(df.iterrows()):
                        filename = str(row[0]).strip()  # 1번 컬럼 = 파일명
                        content = str(row[1]).strip()   # 2번 컬럼 = 내용
                        
                        status.text(f"생성 중: {filename}.mp3")
                        
                        # 음성 생성
                        audio_data = asyncio.run(generate_audio(content, voice))
                        
                        # ZIP에 추가
                        zf.writestr(f"{filename}.mp3", audio_data)
                        
                        progress.progress((i + 1) / len(df))
                
                status.text("✅ 완료!")
                
                # 다운로드
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 ZIP 다운로드",
                    data=zip_buffer.getvalue(),
                    file_name="audio_files.zip",
                    mime="application/zip"
                )


if __name__ == "__main__":
    main()
