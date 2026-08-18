"""
Voice Maker - PDF to TXT & Excel to multilingual TTS
"""

import io
import zipfile

import easyocr
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tts_core import (
    VOICES,
    build_voice_segments,
    is_tts_header_row,
    run_segmented_audio,
    safe_output_name,
)


@st.cache_resource
def get_ocr_reader(langs):
    return easyocr.Reader(langs, gpu=False)


def extract_text_from_pdf(
    pdf_file,
    use_ocr: bool = False,
    ocr_langs: list[str] | None = None,
) -> str:
    """Extract text from a PDF, optionally using OCR for image-based pages."""
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []

    try:
        for page_num, page in enumerate(doc):
            text = ""

            if not use_ocr:
                text = page.get_text("text")

            if use_ocr or not text.strip():
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                image = Image.open(io.BytesIO(img_data))
                img_array = np.array(image)

                selected_langs = list(ocr_langs or ["ko", "en"])
                easyocr_langs = [
                    "ch_sim" if lang == "zh" else lang
                    for lang in selected_langs
                ]
                if "en" not in easyocr_langs:
                    easyocr_langs.append("en")

                reader = get_ocr_reader(tuple(easyocr_langs))
                results = reader.readtext(img_array)
                results.sort(key=lambda item: item[0][0][1])
                text = "\n".join(item[1] for item in results)

            if text.strip():
                all_text.append(
                    f"--- 페이지 {page_num + 1} ---\n{text.strip()}"
                )
    finally:
        doc.close()

    if not all_text:
        return "⚠️ 텍스트를 추출할 수 없습니다."

    return "\n\n".join(all_text)


def render_pdf_tab() -> None:
    st.header("PDF를 TXT로 변환")

    use_ocr = st.checkbox("🔍 OCR 사용 (이미지 PDF용)", value=False)
    ocr_langs: list[str] | None = None

    if use_ocr:
        st.caption("혼합 언어 PDF라면 자동(혼합) 모드를 추천합니다.")
        ocr_mode = st.radio(
            "OCR 언어 모드",
            ["자동(혼합)", "직접 선택"],
            horizontal=True,
        )

        if ocr_mode == "자동(혼합)":
            ocr_langs = ["ko", "de", "en"]

            with st.expander("추가 언어 포함(선택)"):
                extra = st.multiselect(
                    "추가 언어",
                    options=["ja", "zh", "tr"],
                    default=[],
                    format_func=lambda lang: {
                        "ja": "일본어",
                        "zh": "중국어",
                        "tr": "터키어",
                    }[lang],
                )

                if extra:
                    ocr_langs = list(dict.fromkeys(ocr_langs + extra))

            st.info(
                "자동(혼합)은 기본 3개 언어로 시작합니다. "
                "필요하면 추가 언어를 포함하세요."
            )
        else:
            ocr_langs = st.multiselect(
                "OCR 언어(복수 선택 가능)",
                options=["ko", "en", "de", "ja", "zh", "tr"],
                default=["ko", "en"],
                format_func=lambda lang: {
                    "ko": "한국어",
                    "en": "영어",
                    "de": "독일어",
                    "ja": "일본어",
                    "zh": "중국어",
                    "tr": "터키어",
                }[lang],
            )

        st.warning(
            "⚠️ OCR은 첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다."
        )

    pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], key="pdf")

    if not pdf_file:
        return

    with st.spinner("PDF 처리 중... (OCR 사용 시 시간이 걸릴 수 있습니다)"):
        text = extract_text_from_pdf(
            pdf_file,
            use_ocr=use_ocr,
            ocr_langs=ocr_langs,
        )

    st.text_area("추출된 텍스트", text, height=300)

    filename = pdf_file.name.rsplit(".", 1)[0] + ".txt"
    st.download_button(
        label="📥 TXT 다운로드",
        data=text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
    )


def render_tts_tab() -> None:
    st.header("Excel을 다국어 음성 파일로 변환")

    voice_name = st.selectbox(
        "기본 음성 선택 (감지 실패 시 fallback)",
        list(VOICES.keys()),
    )
    base_voice = VOICES[voice_name]

    auto_voice_switch = st.checkbox(
        "🌍 언어 자동 감지 + 혼합 언어 음성 전환 (추천)",
        value=True,
    )

    break_ms = 200
    if auto_voice_switch:
        st.caption(
            "SSML을 사용하지 않습니다. 언어별 음성을 따로 생성한 뒤 "
            "한 MP3로 결합합니다."
        )
        break_ms = int(
            st.slider(
                "언어 전환 쉬는 시간(ms)",
                min_value=0,
                max_value=800,
                value=200,
                step=50,
            )
        )

    st.markdown(
        """
        **Excel 형식:**

        | 1번 컬럼 (파일명) | 2번 컬럼 (내용) |
        |---|---|
        | 001 | I have to go to work. 나는 출근해야 해. |
        | 002 | Ich trinke Kaffee. 커피를 마셔요. |

        `Filename | Text` 또는 `파일명 | 내용` 헤더는 자동으로 인식해 제외합니다.
        """
    )

    excel_file = st.file_uploader(
        "Excel 파일 업로드",
        type=["xlsx"],
        key="excel",
    )

    if not excel_file:
        return

    df = pd.read_excel(excel_file, header=None, engine="openpyxl")

    header_detected = False
    if not df.empty and is_tts_header_row(df.iloc[0].tolist()):
        df = df.iloc[1:].reset_index(drop=True)
        header_detected = True

    df = df.dropna(subset=[0, 1])

    if header_detected:
        st.caption("✅ 첫 행의 Voice Maker 헤더를 자동 인식해 제외했습니다.")

    st.write(f"**총 {len(df):,}개 항목**")
    st.dataframe(df.head(10))

    if not st.button("🚀 음성 생성", type="primary"):
        return

    if df.empty:
        st.warning("변환할 행이 없습니다.")
        return

    progress = st.progress(0)
    status = st.empty()
    errors: list[str] = []
    used_raw_concat = False
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zf:
        for i, (_, row) in enumerate(df.iterrows()):
            filename = safe_output_name(row[0])
            content = str(row[1]).strip()

            if not content:
                errors.append(f"{filename}: 내용이 비어 있습니다.")
                progress.progress((i + 1) / len(df))
                continue

            if content.lstrip().startswith("<speak"):
                errors.append(
                    f"{filename}: custom SSML 입력은 지원하지 않습니다. "
                    "일반 텍스트로 넣어주세요."
                )
                progress.progress((i + 1) / len(df))
                continue

            if auto_voice_switch:
                segments, used_langs = build_voice_segments(
                    content,
                    base_voice=base_voice,
                )
            else:
                segments = [(base_voice, content, None)]
                used_langs = []

            suffix = (
                f" (감지: {', '.join(used_langs)})"
                if used_langs
                else ""
            )
            status.text(f"생성 중: {filename}.mp3{suffix}")

            try:
                audio_data, raw_fallback = run_segmented_audio(
                    segments,
                    break_ms=break_ms if auto_voice_switch else 0,
                )
                used_raw_concat = used_raw_concat or raw_fallback
                zf.writestr(f"{filename}.mp3", audio_data)
            except Exception as exc:
                errors.append(f"{filename}: TTS 처리 실패 ({exc})")

            progress.progress((i + 1) / len(df))

    status.text("✅ 완료!")

    if used_raw_concat:
        st.warning(
            "ffmpeg를 사용할 수 없어 일부 혼합 언어 파일은 "
            "무음 간격 없이 MP3 스트림을 직접 연결했습니다. "
            "정확한 전환 간격이 필요하면 ffmpeg를 설치하세요."
        )

    if errors:
        st.warning("일부 항목에서 오류가 발생했습니다.")
        st.code("\n".join(errors))

    zip_buffer.seek(0)
    st.download_button(
        label="📥 ZIP 다운로드",
        data=zip_buffer.getvalue(),
        file_name="audio_files.zip",
        mime="application/zip",
    )


def main() -> None:
    st.set_page_config(page_title="Voice Maker", page_icon="🎙️")
    st.title("🎙️ Voice Maker")

    tab1, tab2 = st.tabs(["📄 PDF → TXT", "🔊 Excel → TTS"])

    with tab1:
        render_pdf_tab()

    with tab2:
        render_tts_tab()


if __name__ == "__main__":
    main()
