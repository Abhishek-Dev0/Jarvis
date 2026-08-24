"""theme.py — one VS-Code-dark-inspired stylesheet for the whole app.
Applied once via QApplication.setStyleSheet(STYLESHEET) in app.py."""

BG = "#1e1e1e"
BG_LIGHT = "#252526"
BG_LIGHTER = "#2d2d30"
BORDER = "#3c3c3c"
FG = "#cccccc"
FG_DIM = "#8a8a8a"
ACCENT = "#0e78c9"
ACCENT_HOVER = "#1a8cdf"
GOOD = "#4ec9b0"
BAD = "#f14c4c"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Cascadia Code", sans-serif;
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
    padding: 4px;
    spacing: 8px;
}}
QToolBar QLabel#TitleLabel {{
    font-size: 16px;
    font-weight: 600;
    color: {FG};
    padding: 0 8px;
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
QTabWidget::pane {{
    border: none;
    background-color: {BG};
}}
QTabBar::tab {{
    background-color: {BG_LIGHT};
    color: {FG_DIM};
    padding: 8px 16px;
    border: none;
    border-right: 1px solid {BORDER};
}}
QTabBar::tab:selected {{
    background-color: {BG};
    color: {FG};
    border-bottom: 2px solid {ACCENT};
}}
QTextEdit, QPlainTextEdit, QListView {{
    background-color: {BG};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}
QLineEdit {{
    background-color: {BG_LIGHTER};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton {{
    background-color: {BG_LIGHTER};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {ACCENT_HOVER};
}}
QPushButton:disabled {{
    color: {FG_DIM};
    background-color: {BG_LIGHT};
}}
QPushButton#MicButton[recording="true"] {{
    background-color: {BAD};
    border-color: {BAD};
}}
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
}}
QComboBox {{
    background-color: {BG_LIGHTER};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
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
    background: {FG_DIM};
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
"""
