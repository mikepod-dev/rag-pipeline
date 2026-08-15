from pipeline import chunk_text, validate_query


def test_chunk_text_basic_splitting():
    text = " ".join([f"word{i}" for i in range(250)])
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) == 4


def test_chunk_text_overlap_preserves_boundary_content():
    text = " ".join([f"word{i}" for i in range(150)])
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert "word95" in chunks[0]
    assert "word95" in chunks[1]


def test_chunk_text_short_input_single_chunk():
    text = "just a few words here"
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty_string():
    chunks = chunk_text("", chunk_size=100, overlap=30)
    assert chunks == []


def test_validate_query_rejects_empty_string():
    is_valid, error = validate_query("")
    assert is_valid is False
    assert "empty" in error.lower()


def test_validate_query_rejects_whitespace_only():
    is_valid, error = validate_query("     ")
    assert is_valid is False


def test_validate_query_rejects_oversized_input():
    is_valid, error = validate_query("a" * 501)
    assert is_valid is False
    assert "long" in error.lower()


def test_validate_query_accepts_normal_question():
    is_valid, error = validate_query(
        "What year did the French officer bring coffee to the Americas?"
    )
    assert is_valid is True
    assert error is None


def test_validate_query_accepts_exactly_500_chars():
    is_valid, error = validate_query("a" * 500)
    assert is_valid is True
