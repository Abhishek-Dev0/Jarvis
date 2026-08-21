from jarvis.core.data import format_chat_turns


def test_format_chat_turns_preserves_order_and_eos():
    turns = [
        ("hello there", "hi! how can I help?"),
        ("what is 2 + 2", "4"),
    ]

    text = format_chat_turns(turns)

    assert text == (
        "<|user|>hello there<|assistant|>hi! how can I help?<|eos|>"
        "<|user|>what is 2 + 2<|assistant|>4<|eos|>"
    )


def test_format_chat_turns_ignores_empty_turns():
    turns = [
        ("", ""),
        ("  hello  ", "  world  "),
    ]

    text = format_chat_turns(turns)

    assert text == "<|user|>hello<|assistant|>world<|eos|>"
