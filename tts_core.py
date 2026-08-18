from __future__ import annotations

import asyncio
import io
import re
from functools import lru_cache

import edge_tts
from lingua import Language, LanguageDetectorBuilder


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


def lang_code_from_voice(voice: str) -> str | None:
    voice = (voice or "").strip()
    if len(voice) < 2:
        return None
    prefix = voice.split("-")[0].lower()
    return prefix if prefix in VOICE_BY_LANG else None


def char_lang_group(ch: str) -> str | None:
    if not ch:
        return None

    cp = ord(ch)

    if (
        0xAC00 <= cp <= 0xD7A3
        or 0x1100 <= cp <= 0x11FF
        or 0x3130 <= cp <= 0x318F
        or 0xA960 <= cp <= 0xA97F
        or 0xD7B0 <= cp <= 0xD7FF
    ):
        return "ko"

    if 0x3040 <= cp <= 0x30FF:
        return "ja"

    if 0x4E00 <= cp <= 0x9FFF:
        return "zh"

    if (
        "A" <= ch <= "Z"
        or "a" <= ch <= "z"
        or ch in "ÄÖÜäöüßÇĞİÖŞÜçğıöşü"
    ):
        return "latin"

    return None


def split_runs_by_script(text: str) -> list[tuple[str, str]]:
    """Split text into script runs while keeping punctuation with nearby text."""
    text = text or ""
    runs: list[tuple[str, str]] = []
    current_group: str | None = None
    current: list[str] = []
    pending_prefix: list[str] = []

    def flush() -> None:
        nonlocal current_group, current
        if current:
            runs.append((current_group or "other", "".join(current)))
        current_group = None
        current = []

    for ch in text:
        group = char_lang_group(ch)

        if group is None:
            if current:
                current.append(ch)
            else:
                pending_prefix.append(ch)
            continue

        if current_group is None:
            current_group = group
            if pending_prefix:
                current.extend(pending_prefix)
                pending_prefix = []
            current.append(ch)
            continue

        if current_group == group:
            current.append(ch)
            continue

        flush()
        current_group = group
        if pending_prefix:
            current.extend(pending_prefix)
            pending_prefix = []
        current.append(ch)

    if pending_prefix:
        if current:
            current.extend(pending_prefix)
        elif runs:
            last_group, last_text = runs[-1]
            runs[-1] = (last_group, last_text + "".join(pending_prefix))
        else:
            current_group = "other"
            current = pending_prefix

    flush()
    return runs


def split_into_segments(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


@lru_cache(maxsize=1)
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
    min_len: int = 6,
    min_conf: float = 0.40,
    min_margin: float = 0.06,
) -> str | None:
    text = (text or "").strip()
    if len(text) < min_len:
        return None

    confidences = get_language_detector().compute_language_confidence_values(text)
    if not confidences:
        return None

    confidences = sorted(confidences, key=lambda item: item.value, reverse=True)
    best = confidences[0]

    if best.value < min_conf:
        return None

    if len(confidences) > 1 and (best.value - confidences[1].value) < min_margin:
        return None

    return {
        Language.ENGLISH: "en",
        Language.GERMAN: "de",
        Language.KOREAN: "ko",
        Language.JAPANESE: "ja",
        Language.CHINESE: "zh",
        Language.TURKISH: "tr",
    }.get(best.language)


def voice_for_lang(lang_code: str | None, base_voice: str) -> str:
    if not lang_code:
        return base_voice

    if lang_code == lang_code_from_voice(base_voice):
        return base_voice

    return VOICE_BY_LANG.get(lang_code, base_voice)


def _lang_for_script_run(
    group: str,
    chunk: str,
    *,
    base_voice: str,
    japanese_context: bool = False,
) -> str | None:
    base_lang = lang_code_from_voice(base_voice)

    if japanese_context and group in {"ja", "zh"}:
        return "ja"

    if group in {"ko", "ja", "zh"}:
        return group

    if group == "latin":
        detected = detect_best_lang_code(
            chunk,
            min_len=3,
            min_conf=0.34,
            min_margin=0.04,
        )
        if detected in {"en", "de", "tr"}:
            return detected
        return base_lang

    return base_lang


def _append_voice_segment(
    segments: list[tuple[str, str, str | None]],
    *,
    voice: str,
    text: str,
    lang_code: str | None,
) -> None:
    text = (text or "").strip()
    if not text:
        return

    if segments and segments[-1][0] == voice:
        prev_voice, prev_text, prev_lang = segments[-1]
        separator = "" if prev_text.endswith((" ", "\n")) else " "
        segments[-1] = (prev_voice, prev_text + separator + text, prev_lang or lang_code)
        return

    segments.append((voice, text, lang_code))


def build_voice_segments(
    text: str,
    *,
    base_voice: str,
) -> tuple[list[tuple[str, str, str | None]], list[str]]:
    """Plan per-language TTS segments without custom SSML.

    Returns a list of ``(voice, text, lang_code)`` tuples plus the language
    codes encountered in playback order.
    """
    segments: list[tuple[str, str, str | None]] = []
    used_langs: list[str] = []
    base_lang = lang_code_from_voice(base_voice)

    for sentence in split_into_segments(text):
        groups = {
            group
            for ch in sentence
            if (group := char_lang_group(ch)) is not None
        }

        # Japanese commonly mixes kana and kanji. Treat kana+kanji as a single
        # Japanese segment instead of switching the kanji run to Chinese.
        if "ja" in groups and groups.issubset({"ja", "zh"}):
            lang_code = "ja"
            _append_voice_segment(
                segments,
                voice=voice_for_lang(lang_code, base_voice),
                text=sentence,
                lang_code=lang_code,
            )
            if lang_code not in used_langs:
                used_langs.append(lang_code)
            continue

        if len(groups) <= 1:
            group = next(iter(groups), None)

            if group in {"ko", "ja"}:
                lang_code = group
            elif group == "zh":
                lang_code = detect_best_lang_code(
                    sentence,
                    min_len=3,
                    min_conf=0.32,
                    min_margin=0.03,
                )
                if lang_code not in {"zh", "ja"}:
                    lang_code = "zh"
            elif group == "latin":
                lang_code = detect_best_lang_code(
                    sentence,
                    min_len=3,
                    min_conf=0.34,
                    min_margin=0.04,
                )
                if lang_code not in {"en", "de", "tr"}:
                    lang_code = base_lang
            else:
                lang_code = base_lang

            _append_voice_segment(
                segments,
                voice=voice_for_lang(lang_code, base_voice),
                text=sentence,
                lang_code=lang_code,
            )
            if lang_code and lang_code not in used_langs:
                used_langs.append(lang_code)
            continue

        japanese_context = "ja" in groups
        for group, chunk in split_runs_by_script(sentence):
            lang_code = _lang_for_script_run(
                group,
                chunk,
                base_voice=base_voice,
                japanese_context=japanese_context,
            )
            _append_voice_segment(
                segments,
                voice=voice_for_lang(lang_code, base_voice),
                text=chunk,
                lang_code=lang_code,
            )
            if lang_code and lang_code not in used_langs:
                used_langs.append(lang_code)

    if not segments and (text or "").strip():
        _append_voice_segment(
            segments,
            voice=base_voice,
            text=text,
            lang_code=base_lang,
        )
        if base_lang:
            used_langs.append(base_lang)

    return segments, used_langs


async def generate_audio(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    if not audio_data:
        raise RuntimeError("TTS returned no audio data.")

    return audio_data


def concatenate_mp3(
    audio_parts: list[bytes],
    *,
    break_ms: int = 200,
) -> tuple[bytes, bool]:
    """Combine MP3 segments.

    Returns ``(audio_bytes, used_raw_fallback)``. pydub/ffmpeg is preferred
    because it can insert an exact silence gap. If ffmpeg is unavailable, raw
    MP3 frame concatenation keeps the batch usable but omits the requested gap.
    """
    audio_parts = [part for part in audio_parts if part]
    if not audio_parts:
        raise ValueError("No audio segments to concatenate.")
    if len(audio_parts) == 1:
        return audio_parts[0], False

    try:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=max(0, int(break_ms)))

        for index, part in enumerate(audio_parts):
            if index > 0 and break_ms > 0:
                combined += silence
            combined += AudioSegment.from_file(io.BytesIO(part), format="mp3")

        output = io.BytesIO()
        combined.export(output, format="mp3")
        return output.getvalue(), False
    except Exception:
        return b"".join(audio_parts), True


async def generate_segmented_audio(
    segments: list[tuple[str, str, str | None]],
    *,
    break_ms: int = 200,
) -> tuple[bytes, bool]:
    if not segments:
        raise ValueError("No TTS segments were planned.")

    audio_parts = []
    for voice, text, _lang_code in segments:
        audio_parts.append(await generate_audio(text, voice))

    return concatenate_mp3(audio_parts, break_ms=break_ms)


def run_segmented_audio(
    segments: list[tuple[str, str, str | None]],
    *,
    break_ms: int = 200,
) -> tuple[bytes, bool]:
    return asyncio.run(generate_segmented_audio(segments, break_ms=break_ms))


def safe_output_name(value: object) -> str:
    name = str(value).strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip(" .")
    return (name or "audio")[:120]
