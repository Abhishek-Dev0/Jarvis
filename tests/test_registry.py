from jarvis.modules.base import Registry, SkillModule


class _FakeSkill(SkillModule):
    def __init__(self, name, trigger):
        self.name = name
        self.description = "test skill"
        self.trigger = trigger

    def matches(self, text):
        return text.strip().lower().startswith(self.trigger)

    def handle(self, text):
        return f"{self.name} handled it"


def test_disabled_skill_is_skipped_by_find_skill():
    registry = Registry()
    skill = _FakeSkill("greet", "hello")
    registry.register(skill)

    assert registry.find_skill("hello there") is skill

    skill.enabled = False
    assert registry.find_skill("hello there") is None

    skill.enabled = True
    assert registry.find_skill("hello there") is skill


def test_disabled_skill_does_not_block_a_lower_priority_match():
    registry = Registry()
    high = _FakeSkill("high", "hello")
    high.priority = 10
    low = _FakeSkill("low", "hello")
    low.priority = 1
    registry.register(high)
    registry.register(low)

    assert registry.find_skill("hello") is high
    high.enabled = False
    assert registry.find_skill("hello") is low
