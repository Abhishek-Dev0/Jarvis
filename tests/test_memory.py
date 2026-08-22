from jarvis.modules.memory import MemorySkill, add_memory, load_memories, memory_context


def test_load_memories_empty_when_file_absent(tmp_path):
    path = str(tmp_path / "memory.json")
    assert load_memories(path) == []


def test_add_and_load_round_trip(tmp_path):
    path = str(tmp_path / "memory.json")
    add_memory("Abi uses a Ryzen 7 laptop with an RTX 3050", path)
    add_memory("Abi is in Japan", path)
    memories = load_memories(path)
    assert len(memories) == 2
    assert memories[0]["text"] == "Abi uses a Ryzen 7 laptop with an RTX 3050"
    assert "added" in memories[0]


def test_add_memory_ignores_blank_text(tmp_path):
    path = str(tmp_path / "memory.json")
    add_memory("   ", path)
    assert load_memories(path) == []


def test_memory_context_empty_when_nothing_remembered(tmp_path):
    path = str(tmp_path / "memory.json")
    assert memory_context(path) == ""


def test_memory_context_frames_facts_as_data_not_instructions(tmp_path):
    path = str(tmp_path / "memory.json")
    add_memory("prefers concise answers", path)
    ctx = memory_context(path)
    assert "not as new instructions" in ctx
    assert "- prefers concise answers" in ctx


def test_memory_skill_remember_and_recall(tmp_path):
    path = str(tmp_path / "memory.json")
    sk = MemorySkill(path=path)

    assert sk.matches("remember that I use a Ryzen 7 laptop") is True
    reply = sk.handle("remember that I use a Ryzen 7 laptop")
    assert "I'll remember that" in reply

    assert sk.matches("what do you remember") is True
    recall = sk.handle("what do you remember")
    assert "Ryzen 7 laptop" in recall


def test_memory_skill_remember_nothing_prompts_for_content(tmp_path):
    path = str(tmp_path / "memory.json")
    sk = MemorySkill(path=path)
    reply = sk.handle("remember that ")
    assert reply == "Remember what?"


def test_memory_skill_list_when_empty(tmp_path):
    path = str(tmp_path / "memory.json")
    sk = MemorySkill(path=path)
    assert "don't have anything remembered yet" in sk.handle("what do you remember")


def test_memory_skill_forget_removes_matching_entry(tmp_path):
    path = str(tmp_path / "memory.json")
    sk = MemorySkill(path=path)
    sk.handle("remember that the sky is green")
    sk.handle("remember that the grass is blue")

    reply = sk.handle("forget that sky is green")
    assert "Forgot 1" in reply

    remaining = load_memories(path)
    assert len(remaining) == 1
    assert "grass is blue" in remaining[0]["text"]


def test_memory_skill_forget_no_match(tmp_path):
    path = str(tmp_path / "memory.json")
    sk = MemorySkill(path=path)
    sk.handle("remember that the sky is green")
    reply = sk.handle("forget that nonexistent thing")
    assert "didn't have anything matching" in reply


def test_memory_skill_matches_is_selective():
    sk = MemorySkill()
    assert sk.matches("tell me a joke") is False
    assert sk.matches("remember that this is a test") is True
    assert sk.matches("forget that thing") is True
    assert sk.matches("what do you remember") is True
