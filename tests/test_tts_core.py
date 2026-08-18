from unittest.mock import patch

from tts_core import (
    VOICES,
    build_voice_segments,
    is_tts_header_row,
    safe_output_name,
    split_runs_by_script,
)


def test_mixed_english_korean_uses_two_voices():
    def fake_detect(text, **_kwargs):
        if "Hello" in text:
            return "en"
        return None

    with patch("tts_core.detect_best_lang_code", side_effect=fake_detect):
        segments, used = build_voice_segments(
            "Hello world. 안녕하세요.",
            base_voice=VOICES["English (US)"],
        )

    assert [segment[0] for segment in segments] == [
        VOICES["English (US)"],
        VOICES["Korean"],
    ]
    assert used == ["en", "ko"]


def test_japanese_kana_and_kanji_stay_on_japanese_voice():
    segments, used = build_voice_segments(
        "東京に行きます。",
        base_voice=VOICES["English (US)"],
    )

    assert len(segments) == 1
    assert segments[0][0] == VOICES["Japanese"]
    assert used == ["ja"]


def test_script_split_preserves_punctuation():
    runs = split_runs_by_script("Hello, 한국어!")
    assert runs == [
        ("latin", "Hello, "),
        ("ko", "한국어!"),
    ]


def test_output_name_never_contains_path_separators():
    name = safe_output_name("../../unsafe\\name")
    assert "/" not in name
    assert "\\" not in name
    assert name


def test_standard_excel_header_is_detected():
    assert is_tts_header_row(["Filename", "Text"])
    assert is_tts_header_row(["파일명", "내용"])


def test_real_data_row_is_not_treated_as_header():
    assert not is_tts_header_row(["CORE_001_01", "I think this is better."])
