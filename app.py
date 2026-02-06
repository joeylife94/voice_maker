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
from lingua import Language, LanguageDetectorBuilder
from xml.sax.saxutils import escape as xml_escape
import re


def looks_like_ssml(text: str) -> bool:
    t = (text or "").lstrip()
    return t.startswith("<speak") and "</speak>" in t


def lang_code_from_voice(voice: str) -> str | None:
    v = (voice or "").strip()
    if len(v) < 2:
        return None
    # Voice names look like: de-DE-KatjaNeural
    prefix = v.split("-")[0].lower()
    if prefix in {"en", "de", "ko", "ja", "zh", "tr"}:
        return prefix
    return None


def char_lang_group(ch: str) -> str | None:
    if not ch:
        return None
    cp = ord(ch)

    # Hangul syllables
    if 0xAC00 <= cp <= 0xD7A3:
        return "ko"
    # Hiragana/Katakana
    if 0x3040 <= cp <= 0x30FF:
        return "ja"
    # CJK Unified Ideographs
    if 0x4E00 <= cp <= 0x9FFF:
        return "zh"

    # Latin letters (includes German umlauts and ß)
    if (
        "A" <= ch <= "Z"
        or "a" <= ch <= "z"
        or ch in "ÄÖÜäöüß"
    ):
        return "latin"

    return None


def split_runs_by_script(text: str) -> list[tuple[str, str]]:
    """Split text into runs by script group.

    Returns list of (group, run_text) where group is one of:
    - ko / ja / zh / latin
    - other (digits, punctuation, whitespace-only, etc.)
    """
    text = text or ""
    runs: list[tuple[str, str]] = []

    current_group: str | None = None
    current: list[str] = []

    def flush():
        nonlocal current_group, current
        if current:
            g = current_group or "other"
            runs.append((g, "".join(current)))
        current_group = None
        current = []

    for ch in text:
        g = char_lang_group(ch)

        # Keep digits/punctuation/whitespace attached to current run if possible.
        if g is None:
            if current:
                current.append(ch)
            else:
                current_group = "other"
                current.append(ch)
            continue

        if current_group is None:
            current_group = g
            current.append(ch)
            continue

        if current_group == g:
            current.append(ch)
            continue

        # group switch
        flush()
        current_group = g
        current.append(ch)

    flush()
    return runs


def build_ssml_by_script_voice_switch(text: str, *, base_voice: str, break_ms: int = 200) -> tuple[str | None, list[str]]:
    """Auto-build SSML with <voice> switches based on script changes.

    - Latin segments use base_voice (user-selected)
    - ko/ja/zh segments use mapped voices
    Returns (ssml_or_none, used_lang_codes)
    """
    runs = split_runs_by_script(text)
    if not runs:
        return None, []

    base_lang = lang_code_from_voice(base_voice)
    used: list[str] = []

    def voice_for_group(group: str) -> str:
        if group == "latin":
            return base_voice
        if group in VOICE_BY_LANG:
            return VOICE_BY_LANG[group]
        return base_voice

    voices_in_order: list[str] = []
    voice_chunks: list[str] = []

    for group, chunk in runs:
        if not chunk:
            continue

        voice = voice_for_group(group)
        voices_in_order.append(voice)

        if group == "latin":
            if base_lang and base_lang not in used:
                used.append(base_lang)
        elif group in {"ko", "ja", "zh"}:
            if group not in used:
                used.append(group)

        voice_chunks.append(f"<voice name=\"{voice}\">{xml_escape(chunk)}</voice>")

    # If everything resolves to the same voice, don't use SSML.
    if len(set(voices_in_order)) <= 1:
        return None, used

    brk = f"<break time=\"{int(break_ms)}ms\"/>" if break_ms and break_ms > 0 else ""
    ssml = (
        "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\">"
        + brk.join(voice_chunks)
        + "</speak>"
    )
    return ssml, used

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

VOICE_BY_LANG = {
    "en": VOICES["English (US)"],
    "de": VOICES["German"],
    "ko": VOICES["Korean"],
    "ja": VOICES["Japanese"],
    "zh": VOICES["Chinese"],
    "tr": VOICES["Turkish"],
}

@st.cache_resource
def get_language_detector():
    return (
        LanguageDetectorBuilder.from_languages(
            Language.ENGLISH,
            Language.GERMAN,
            Language.KOREAN,
            Language.JAPANESE,
            Language.CHINESE,
            Language.TURKISH,
        )
        .with_preloaded_language_models()
        .build()
    )


def detect_best_lang_code(
    text: str,
    *,
    min_len: int = 10,
    min_conf: float = 0.45,
    min_margin: float = 0.08,
) -> str | None:
    text = (text or "").strip()
    if len(text) < min_len:
        return None

    detector = get_language_detector()
    confidences = detector.compute_language_confidence_values(text)
    if not confidences:
        return None

    confidences = sorted(confidences, key=lambda x: x.value, reverse=True)
    best = confidences[0]
    if best.value < min_conf:
        return None

    if len(confidences) > 1:
        second = confidences[1]
        if (best.value - second.value) < min_margin:
            # Ambiguous / mixed
            return None

    lang = best.language
    return {
        Language.ENGLISH: "en",
        Language.GERMAN: "de",
        Language.KOREAN: "ko",
        Language.JAPANESE: "ja",
        Language.CHINESE: "zh",
        Language.TURKISH: "tr",
    }.get(lang)


def split_into_segments(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]
def build_ssml_with_auto_voices(text: str, default_voice: str) -> tuple[str, list[str]]:
    segments = split_into_segments(text)
    if not segments:
        return (text or ""), []

    used_langs: list[str] = []
    voice_chunks: list[str] = []
    for segment in segments:
        lang_code = detect_best_lang_code(segment, min_len=6, min_conf=0.40, min_margin=0.06)
        voice = VOICE_BY_LANG.get(lang_code or "", default_voice)
        if lang_code and lang_code not in used_langs:
            used_langs.append(lang_code)
        voice_chunks.append(f"<voice name=\"{voice}\">{xml_escape(segment)}</voice>")

    ssml = (
        "<speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\">"
        + "<break time=\"200ms\"/>".join(voice_chunks)
        + "</speak>"
    )
    return ssml, used_langs

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


def extract_text_from_pdf(pdf_file, use_ocr: bool = False, ocr_langs: list[str] | None = None) -> str:
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
            if not ocr_langs:
                ocr_langs = ["ko", "en"]
            easyocr_langs = ["ch_sim" if x == "zh" else x for x in ocr_langs]
            if "en" not in easyocr_langs:
                easyocr_langs.append("en")
            
            reader = get_ocr_reader(tuple(easyocr_langs))
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
        
        use_ocr = st.checkbox("🔍 OCR 사용 (이미지 PDF용)", value=False)

        ocr_langs: list[str] | None = None
        if use_ocr:
            st.caption("혼합 언어 PDF라면 자동(혼합) 모드를 추천합니다.")
            ocr_mode = st.radio("OCR 언어 모드", ["자동(혼합)", "직접 선택"], horizontal=True)
            if ocr_mode == "자동(혼합)":
                # Too many languages can reduce accuracy; start small and let user add extras if needed.
                ocr_langs = ["ko", "de", "en"]
                with st.expander("추가 언어 포함(선택)"):
                    extra = st.multiselect(
                        "추가 언어",
                        options=["ja", "zh", "tr"],
                        default=[],
                        format_func=lambda x: {
                            "ja": "일본어",
                            "zh": "중국어",
                            "tr": "터키어",
                        }[x],
                    )
                    if extra:
                        ocr_langs = list(dict.fromkeys(ocr_langs + extra))
                st.info("자동(혼합)은 기본 3개 언어로 시작합니다. 필요하면 추가 언어를 포함하세요.")
            else:
                ocr_langs = st.multiselect(
                    "OCR 언어(복수 선택 가능)",
                    options=["ko", "en", "de", "ja", "zh", "tr"],
                    default=["ko", "en"],
                    format_func=lambda x: {
                        "ko": "한국어",
                        "en": "영어",
                        "de": "독일어",
                        "ja": "일본어",
                        "zh": "중국어",
                        "tr": "터키어",
                    }[x],
                )
        
        if use_ocr:
            st.warning("⚠️ OCR은 첫 실행시 모델 다운로드로 시간이 걸릴 수 있습니다.")
        
        pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], key="pdf")
        
        if pdf_file:
            with st.spinner("PDF 처리 중... (OCR 사용시 시간이 걸릴 수 있습니다)"):
                text = extract_text_from_pdf(pdf_file, use_ocr=use_ocr, ocr_langs=ocr_langs)
            
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
        voice_name = st.selectbox("기본 음성 선택 (자동 감지 실패/기타 언어 fallback)", list(VOICES.keys()))
        voice = VOICES[voice_name]

        auto_mixed_ssml = st.checkbox("🌍 혼합 언어면 자동으로 SSML 삽입(추천)", value=True)
        break_ms = 200
        if auto_mixed_ssml:
            break_ms = int(st.slider("언어 전환 쉬는 시간(ms)", min_value=0, max_value=800, value=200, step=50))

        tts_auto = st.checkbox("🌐 언어 자동 감지 (행별, 같은 스크립트 언어 구분용)", value=False)

        allow_ssml_input = st.checkbox("🧩 Excel 셀에 SSML 직접 입력(고급)", value=False)
        if allow_ssml_input:
            st.info(
                "2번 컬럼에 `<speak ...>...</speak>` 형태로 SSML을 넣으면 그대로 합성합니다. "
                "(예: `<voice name=\"de-DE-KatjaNeural\">Ich zahle ...</voice><break time=\"200ms\"/><voice name=\"ko-KR-SunHiNeural\">...</voice>`)"
            )
        
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
                errors: list[str] = []
                
                # ZIP 파일 생성
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, (_, row) in enumerate(df.iterrows()):
                        filename = str(row[0]).strip()  # 1번 컬럼 = 파일명
                        content = str(row[1]).strip()   # 2번 컬럼 = 내용
                        
                        status.text(f"생성 중: {filename}.mp3")
                        
                        # 음성 생성
                        text_for_tts = content
                        used_langs: list[str] = []
                        chosen_voice = voice

                        is_ssml = allow_ssml_input and looks_like_ssml(content)
                        if is_ssml:
                            status.text(f"생성 중: {filename}.mp3 (SSML)")
                            try:
                                audio_data = asyncio.run(generate_audio(content, chosen_voice))
                            except Exception as e:
                                errors.append(f"{filename}: SSML 처리 실패 ({e})")
                                progress.progress((i + 1) / len(df))
                                continue
                            zf.writestr(f"{filename}.mp3", audio_data)
                            progress.progress((i + 1) / len(df))
                            continue

                        if tts_auto:
                            lang_code = detect_best_lang_code(content)
                            if lang_code:
                                used_langs = [lang_code]
                                chosen_voice = VOICE_BY_LANG.get(lang_code, voice)

                        # Mixed-script auto SSML (German+Korean 등) — no manual SSML needed.
                        if auto_mixed_ssml:
                            ssml, ssml_langs = build_ssml_by_script_voice_switch(content, base_voice=chosen_voice, break_ms=break_ms)
                            if ssml:
                                text_for_tts = ssml
                                used_langs = list(dict.fromkeys((used_langs or []) + ssml_langs))

                        if used_langs:
                            status.text(f"생성 중: {filename}.mp3 (감지: {', '.join(used_langs)})")

                        try:
                            audio_data = asyncio.run(generate_audio(text_for_tts, chosen_voice))
                        except Exception:
                            # Fallback: never corrupt the user's content; retry with plain text.
                            audio_data = asyncio.run(generate_audio(content, chosen_voice))
                        
                        # ZIP에 추가
                        zf.writestr(f"{filename}.mp3", audio_data)
                        
                        progress.progress((i + 1) / len(df))
                
                status.text("✅ 완료!")

                if errors:
                    st.warning("일부 항목에서 오류가 발생했습니다. 아래 목록을 확인하세요.")
                    st.code("\n".join(errors))
                
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
