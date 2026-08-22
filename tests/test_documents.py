from jarvis.modules.documents import DocumentSkill, extract_text


def _build_minimal_pdf(text: str) -> bytes:
    """A syntactically real, minimal single-page PDF with visible text --
    not mocked, so extract_text() is tested against pypdf's actual parser,
    not just our own code's assumptions about what it returns."""
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/MediaBox[0 0 300 144]/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    stream = f"BT /F1 18 Tf 10 100 Td ({text}) Tj ET".encode()
    objects.append(b"<</Length " + str(len(stream)).encode() + b">>\nstream\n" + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj".encode() + body + b"endobj\n"
    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF".encode()
    return bytes(out)


def _write_pdf(tmp_path, text="Hello World Test PDF") -> str:
    path = tmp_path / "test.pdf"
    path.write_bytes(_build_minimal_pdf(text))
    return str(path)


def _write_docx(tmp_path, paragraphs) -> str:
    import docx
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    path = tmp_path / "test.docx"
    doc.save(str(path))
    return str(path)


def test_extract_text_from_real_pdf(tmp_path):
    path = _write_pdf(tmp_path)
    result = extract_text(path)
    assert result["ok"] is True
    assert "Hello World Test PDF" in result["content"]
    assert result["truncated"] is False


def test_extract_text_from_real_docx(tmp_path):
    path = _write_docx(tmp_path, ["First paragraph of the report.", "Second paragraph with details."])
    result = extract_text(path)
    assert result["ok"] is True
    assert "First paragraph of the report." in result["content"]
    assert "Second paragraph with details." in result["content"]


def test_extract_text_missing_file(tmp_path):
    result = extract_text(str(tmp_path / "nope.pdf"))
    assert result["ok"] is False
    assert result["reason"] == "no such file"


def test_extract_text_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain text file")
    result = extract_text(str(path))
    assert result["ok"] is False
    assert "unsupported extension" in result["reason"]


def test_extract_text_empty_docx_reports_no_extractable_text(tmp_path):
    path = _write_docx(tmp_path, [])
    result = extract_text(path)
    assert result["ok"] is False
    assert "no extractable text" in result["reason"]


def test_extract_text_truncates_long_content(tmp_path):
    long_text = "word " * 5000  # well over 12,000 chars
    path = _write_docx(tmp_path, [long_text])
    result = extract_text(path, max_chars=100)
    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["content"]) == 100


def test_skill_matches_known_triggers():
    sk = DocumentSkill()
    assert sk.matches("read document report.pdf") is True
    assert sk.matches("read pdf report.pdf") is True
    assert sk.matches("read docx notes.docx") is True
    assert sk.matches("tell me a joke") is False


def test_skill_handle_no_path_prompts():
    sk = DocumentSkill()
    reply = sk.handle("read document")
    assert "Read which document" in reply


def test_skill_handle_reads_real_pdf(tmp_path):
    path = _write_pdf(tmp_path, "Quarterly Report Summary")
    sk = DocumentSkill()
    reply = sk.handle(f"read document {path}")
    assert "Quarterly Report Summary" in reply


def test_skill_handle_reports_missing_file_clearly(tmp_path):
    sk = DocumentSkill()
    reply = sk.handle(f"read document {tmp_path / 'missing.pdf'}")
    assert "Couldn't read" in reply
    assert "no such file" in reply
