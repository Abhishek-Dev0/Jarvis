from jarvis.modules.fileread import FileReadSkill, list_files, read_file


def test_read_file_returns_real_content():
    result = read_file("jarvis/modules/fileread.py")
    assert result["ok"] is True
    assert "class FileReadSkill" in result["content"]


def test_read_file_rejects_path_escaping_the_repo():
    result = read_file("../../../../etc/passwd")
    assert result["ok"] is False
    assert "escapes" in result["reason"]


def test_read_file_excludes_security_data():
    result = read_file("jarvis/data/security/passphrase.hash")
    assert result["ok"] is False
    assert "excluded" in result["reason"]


def test_read_file_excludes_memory_json():
    result = read_file("jarvis/data/memory.json")
    assert result["ok"] is False
    assert "excluded" in result["reason"]


def test_read_file_reports_missing_file_clearly():
    result = read_file("jarvis/modules/definitely_not_a_real_file.py")
    assert result["ok"] is False
    assert "no such file" in result["reason"]


def test_read_file_truncates_long_files():
    result = read_file("jarvis/modules/fileread.py", max_chars=50)
    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["content"]) == 50


def test_list_files_finds_real_python_modules():
    files = list_files("jarvis/modules/*.py")
    assert "jarvis/modules/fileread.py" in files
    assert "jarvis/modules/security.py" not in files  # security.py is one level up, not in modules/


def test_list_files_excludes_security_directory():
    files = list_files("jarvis/data/security/**")
    assert files == []


def test_list_files_respects_max_results():
    files = list_files("**/*.py", max_results=3)
    assert len(files) <= 3


def test_skill_matches_read_and_list_triggers():
    sk = FileReadSkill()
    assert sk.matches("read file jarvis/security.py") is True
    assert sk.matches("list files matching *.py") is True
    assert sk.matches("show me jarvis/README.md") is True
    assert sk.matches("tell me a joke") is False


def test_skill_handle_read_real_file():
    sk = FileReadSkill()
    reply = sk.handle("read file jarvis/modules/fileread.py")
    assert "class FileReadSkill" in reply


def test_skill_handle_read_denies_security_file_with_clear_message():
    sk = FileReadSkill()
    reply = sk.handle("read file jarvis/data/security/passphrase.hash")
    assert "Couldn't read" in reply
    assert "excluded" in reply


def test_skill_handle_read_no_path_prompts():
    sk = FileReadSkill()
    assert sk.handle("read file") == "Read which file?"


def test_skill_handle_list_no_matches():
    sk = FileReadSkill()
    reply = sk.handle("list files matching *.definitely-not-a-real-extension")
    assert "No files matching" in reply
