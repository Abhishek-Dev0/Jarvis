import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless-safe before any Qt import

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from jarvis.gui.markets_tab import MarketsTab
from jarvis.gui.theme import STYLESHEET


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLESHEET)
    yield app


def _pump(ms=500):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _fake_ohlcv(n=80):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    trend = np.arange(n, dtype=float) * 0.3
    wiggle = (np.arange(n) % 5 - 2).astype(float)
    close = pd.Series(100.0 + trend + wiggle, index=idx)
    volume = pd.Series(np.full(n, 1_000_000.0), index=idx)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)


def test_tab_builds_with_empty_watchlist(qapp, tmp_path, monkeypatch):
    from jarvis.modules import market_analysis
    monkeypatch.setattr(market_analysis, "_WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    tab = MarketsTab()
    assert tab.watchlist_widget.count() == 0


def test_loading_a_symbol_renders_chart_and_info(qapp, tmp_path, monkeypatch):
    from jarvis.modules import market_analysis
    monkeypatch.setattr(market_analysis, "_WATCHLIST_PATH", str(tmp_path / "watchlist.json"))

    df = _fake_ohlcv()
    monkeypatch.setattr(market_analysis, "fetch_history", lambda symbol, period="2y": df)

    tab = MarketsTab()
    tab.symbol_input.setText("AAPL")
    tab._on_load_clicked()
    _pump(1500)

    text = tab.info_view.toPlainText()
    assert "RSI(14)" in text
    assert "MACD" in text
    assert "Educational only" in text
    assert len(tab.figure.axes) == 4  # price, volume, RSI, MACD subplots


def test_load_failure_shown_in_info_view(qapp, tmp_path, monkeypatch):
    from jarvis.modules import market_analysis
    monkeypatch.setattr(market_analysis, "_WATCHLIST_PATH", str(tmp_path / "watchlist.json"))

    def boom(symbol, period="2y"):
        raise ValueError("no data returned for 'ZZZZ'")

    monkeypatch.setattr(market_analysis, "fetch_history", boom)

    tab = MarketsTab()
    tab.symbol_input.setText("ZZZZ")
    tab._on_load_clicked()
    _pump(1000)

    assert "Couldn't load data" in tab.info_view.toPlainText()


def test_add_and_remove_watchlist_updates_the_list_widget(qapp, tmp_path, monkeypatch):
    from jarvis.modules import market_analysis
    monkeypatch.setattr(market_analysis, "_WATCHLIST_PATH", str(tmp_path / "watchlist.json"))

    tab = MarketsTab()
    tab.symbol_input.setText("AAPL")
    tab._on_add_watch_clicked()
    assert [tab.watchlist_widget.item(i).text() for i in range(tab.watchlist_widget.count())] == ["AAPL"]

    tab.watchlist_widget.setCurrentRow(0)
    tab._on_remove_watch_clicked()
    assert tab.watchlist_widget.count() == 0


def test_clicking_a_watchlist_item_loads_it(qapp, tmp_path, monkeypatch):
    from jarvis.modules import market_analysis
    monkeypatch.setattr(market_analysis, "_WATCHLIST_PATH", str(tmp_path / "watchlist.json"))
    market_analysis.add_to_watchlist("MSFT")

    df = _fake_ohlcv()
    monkeypatch.setattr(market_analysis, "fetch_history", lambda symbol, period="2y": df)

    tab = MarketsTab()
    item = tab.watchlist_widget.item(0)
    tab._on_watchlist_item_clicked(item)
    _pump(1500)

    assert tab.symbol_input.text() == "MSFT"
    assert "RSI(14)" in tab.info_view.toPlainText()


class _FakeSignal:
    """Minimal connect()/emit() stand-in for a Qt Signal -- instance-level,
    no shared state across fake-worker instances."""

    def __init__(self):
        self._slot = None

    def connect(self, slot):
        self._slot = slot

    def emit(self, *args):
        if self._slot is not None:
            self._slot(*args)


def test_news_category_button_populates_results(qapp, monkeypatch):
    from jarvis.gui import markets_tab

    fake_results = [{"title": "Fake headline", "url": "https://example.com/a", "snippet": "snippet text"}]

    class _FakeWorker:
        def __init__(self, query):
            self.query = query
            self.finished_ok = _FakeSignal()
            self.failed = _FakeSignal()
            self.finished = _FakeSignal()

        def start(self):
            self.finished_ok.emit(fake_results)
            self.finished.emit()

    monkeypatch.setattr(markets_tab, "NewsSearchWorker", _FakeWorker)

    tab = MarketsTab()
    tab._run_news_search("stock market news today")

    assert tab.news_results.count() == 1
    assert "Fake headline" in tab.news_results.item(0).text()


def test_news_search_failure_shown_in_results_list(qapp, monkeypatch):
    from jarvis.gui import markets_tab

    class _FakeWorker:
        def __init__(self, query):
            self.finished_ok = _FakeSignal()
            self.failed = _FakeSignal()
            self.finished = _FakeSignal()

        def start(self):
            self.failed.emit("network error")
            self.finished.emit()

    monkeypatch.setattr(markets_tab, "NewsSearchWorker", _FakeWorker)

    tab = MarketsTab()
    tab._run_news_search("stock market news today")

    assert "network error" in tab.news_results.item(0).text()


def test_news_double_click_opens_url(qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidgetItem
    from jarvis.gui import markets_tab

    opened = []
    monkeypatch.setattr(markets_tab.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))

    tab = MarketsTab()
    item = QListWidgetItem("Some headline")
    item.setData(Qt.ItemDataRole.UserRole, "https://example.com/article")
    tab._on_news_item_double_clicked(item)

    assert opened == ["https://example.com/article"]
