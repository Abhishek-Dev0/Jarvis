"""theme.py — one cyberpunk-adjacent dark stylesheet for the whole app.
Applied once via QApplication.setStyleSheet(STYLESHEET) in app.py.

Scoped deliberately to what QSS can actually do well: a deep matte-black
palette, neon cyan/green accents, rounded corners, and real hover/focus
states. No animated flicker/glow (QSS has no keyframe animation — that
needs QPropertyAnimation/custom painting, a bigger lift than this pass) —
the "glow" here is a static colored border + a lighter background tint on
hover, not a moving effect."""

BG = "#0a0a0f"
BG_LIGHT = "#12121a"
BG_LIGHTER = "#1a1a26"
BORDER = "#2a2a3a"
FG = "#e4e4ec"
FG_DIM = "#7d7d90"
ACCENT = "#00d4ff"
ACCENT_HOVER = "#3ddfff"
ACCENT_DIM = "rgba(0, 212, 255, 0.12)"
GOOD = "#00e6a0"
GOOD_DIM = "rgba(0, 230, 160, 0.14)"
BAD = "#ff3b6b"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Cascadia Code", "Consolas", sans-serif;
    font-size: 13px;
    color: {FG};
}}
QMainWindow, QWidget {{
    background-color: {BG};
}}
QToolBar {{
    background-color: {BG_LIGHT};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px;
    spacing: 10px;
}}
QToolBar QLabel#TitleLabel {{
    font-size: 17px;
    font-weight: 600;
    color: {ACCENT};
    padding: 0 8px;
    letter-spacing: 1px;
}}
QListWidget {{
    background-color: {BG_LIGHT};
    border: none;
    border-right: 1px solid {BORDER};
    outline: none;
}}
QListWidget::item {{
    padding: 6px 8px;
    border-bottom: 1px solid {BORDER};
}}
QListWidget::item:selected {{
    background-color: {BG_LIGHTER};
}}
QListWidget#SkillList::item {{
    padding: 0px;
    border-bottom: 1px solid {BORDER};
}}
QTabWidget::pane {{
    border: none;
    background-color: {BG};
}}
QTabBar::tab {{
    background-color: {BG_LIGHT};
    color: {FG_DIM};
    padding: 9px 18px;
    border: none;
    border-right: 1px solid {BORDER};
}}
QTabBar::tab:hover {{
    color: {FG};
    background-color: {BG_LIGHTER};
}}
QTabBar::tab:selected {{
    background-color: {BG};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QTextEdit, QPlainTextEdit, QListView {{
    background-color: {BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QLineEdit {{
    background-color: {BG_LIGHTER};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QLineEdit:hover {{
    border: 1px solid {FG_DIM};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
    background-color: {BG_LIGHT};
}}
QPushButton {{
    background-color: {BG_LIGHTER};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 16px;
}}
QPushButton:hover {{
    background-color: {ACCENT_DIM};
    border-color: {ACCENT};
    color: {ACCENT_HOVER};
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    color: {BG};
}}
QPushButton:disabled {{
    color: {FG_DIM};
    background-color: {BG_LIGHT};
    border-color: {BORDER};
}}
QPushButton#MicButton[recording="true"] {{
    background-color: {BAD};
    border-color: {BAD};
    color: {FG};
}}
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BG_LIGHTER};
}}
QCheckBox::indicator:checked {{
    background-color: {GOOD};
    border-color: {GOOD};
}}
QComboBox {{
    background-color: {BG_LIGHTER};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 10px;
}}
QComboBox:hover {{
    border: 1px solid {ACCENT};
}}
QLabel#SkillDescription {{
    color: {FG_DIM};
    font-size: 11px;
}}
QLabel#StatusOk {{
    color: {GOOD};
}}
QLabel#StatusBad {{
    color: {BAD};
}}
QScrollBar:vertical {{
    background: {BG};
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    min-height: 24px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
QStatusBar {{
    color: {FG_DIM};
    background-color: {BG_LIGHT};
    border-top: 1px solid {BORDER};
}}
"""
