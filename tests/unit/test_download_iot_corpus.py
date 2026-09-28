from scripts.download_iot_corpus import MIN_PDF_BYTES, _looks_like_pdf, _safe_url


def test_looks_like_pdf_accepts_leading_whitespace(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"\n %PDF-1.7\n" + b"0" * MIN_PDF_BYTES)

    assert _looks_like_pdf(path)


def test_looks_like_pdf_rejects_html(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"<html>not a pdf</html>" + b"0" * MIN_PDF_BYTES)

    assert not _looks_like_pdf(path)


def test_safe_url_encodes_spaces_without_double_encoding() -> None:
    url = "http://example.test/files/My Paper%20v2.pdf?download=1&name=My Paper"

    assert _safe_url(url) == (
        "http://example.test/files/My%20Paper%20v2.pdf?download=1&name=My%20Paper"
    )
