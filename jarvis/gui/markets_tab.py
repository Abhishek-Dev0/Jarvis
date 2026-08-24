"""markets_tab.py — the GUI's "Markets" tab: charts + technical indicators
for a symbol, purely informational. Deliberately does not recommend,
predict, or suggest anything — see modules/market_analysis.py's own
docstring for why (a strategy that backtests well isn't evidence it will
perform well going forward, and this tab doesn't even go that far: it just
shows what an indicator IS doing, described in the same "traditionally
read as..." language a textbook would use, never "buy"/"sell").

Reuses modules/market_analysis.py for everything (fetch_history, the
indicator functions, the watchlist store) -- this file is presentation
only, no market logic of its own."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .theme import ACCENT, BAD, BG, BG_LIGHT, BORDER, FG, FG_DIM, GOOD
from .workers import MarketDataWorker, NewsSearchWorker

try:
    from ..modules import market_analysis
except ImportError:  # pragma: no cover - legacy direct execution
    from modules import market_analysis

_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
_MA_COLORS = [(20, ACCENT), (50, GOOD), (200, "#c586c0")]

# News panel's canned category buttons -- covers the sectors asked for
# (general market, stocks, crypto, options) via live search, not a
# perpetual background poller (kept on-demand/refresh-on-click, matching
# the "on-demand, not 24/7" design this whole tab already uses -- see
# modules/market_analysis.py's market_news()).
_NEWS_CATEGORIES = {
    "Market": "stock market news today",
    "Stocks": "stock market news today S&P 500 Nasdaq",
    "Crypto": "cryptocurrency market news today bitcoin ethereum",
    "Options": "options market news today",
}


class MarketsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: MarketDataWorker | None = None
        self._news_worker: NewsSearchWorker | None = None
        self._build_ui()
        self._refresh_watchlist()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        left = QWidget()
        left.setMaximumWidth(220)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Symbol (AAPL, BTC-USD)...")
        self.symbol_input.returnPressed.connect(self._on_load_clicked)
        left_layout.addWidget(self.symbol_input)

        self.period_combo = QComboBox()
        self.period_combo.addItems(_PERIODS)
        self.period_combo.setCurrentText("1y")
        left_layout.addWidget(self.period_combo)

        row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load_clicked)
        row.addWidget(load_btn)
        watch_btn = QPushButton("+ Watch")
        watch_btn.clicked.connect(self._on_add_watch_clicked)
        row.addWidget(watch_btn)
        left_layout.addLayout(row)

        left_layout.addWidget(QLabel("Watchlist"))
        self.watchlist_widget = QListWidget()
        self.watchlist_widget.itemClicked.connect(self._on_watchlist_item_clicked)
        left_layout.addWidget(self.watchlist_widget)

        remove_btn = QPushButton("Remove from watchlist")
        remove_btn.clicked.connect(self._on_remove_watch_clicked)
        left_layout.addWidget(remove_btn)

        layout.addWidget(left)

        right_tabs = QTabWidget()
        right_tabs.addTab(self._build_chart_pane(), "Chart")
        right_tabs.addTab(self._build_news_pane(), "News")
        layout.addWidget(right_tabs, stretch=1)

    def _build_chart_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 8))
        self.figure.set_facecolor(BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        self.info_view = QTextEdit()
        self.info_view.setReadOnly(True)
        self.info_view.setMaximumHeight(150)
        self.info_view.setPlainText("Load a symbol to see its chart and indicator readings.")
        layout.addWidget(self.info_view)
        return pane

    def _build_news_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)

        categories_row = QHBoxLayout()
        for label, query in _NEWS_CATEGORIES.items():
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, q=query: self._run_news_search(q))
            categories_row.addWidget(btn)
        categories_row.addStretch()
        layout.addLayout(categories_row)

        search_row = QHBoxLayout()
        self.news_query_input = QLineEdit()
        self.news_query_input.setPlaceholderText("Or search news for a symbol/topic...")
        self.news_query_input.returnPressed.connect(self._on_news_custom_search)
        search_row.addWidget(self.news_query_input)
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._on_news_custom_search)
        search_row.addWidget(search_btn)
        layout.addLayout(search_row)

        self.news_results = QListWidget()
        self.news_results.itemDoubleClicked.connect(self._on_news_item_double_clicked)
        layout.addWidget(self.news_results, stretch=1)

        note = QLabel("Live web search results — headlines and snippets only, no rating or "
                       "analysis layered on top. Double-click a result to open it in your browser.")
        note.setObjectName("SkillDescription")
        note.setWordWrap(True)
        layout.addWidget(note)
        return pane

    # ------------------------------------------------------------ loading

    def _on_load_clicked(self):
        symbol = self.symbol_input.text().strip().upper()
        if not symbol or self._worker is not None:
            return
        period = self.period_combo.currentText()
        self.info_view.setPlainText(f"Loading {symbol} ({period})...")

        worker = MarketDataWorker(symbol, period)
        worker.finished_ok.connect(self._on_data_loaded)
        worker.failed.connect(self._on_load_failed)
        worker.finished.connect(lambda: setattr(self, "_worker", None))
        self._worker = worker
        worker.start()

    def _on_load_failed(self, error: str):
        self.info_view.setPlainText(f"Couldn't load data: {error}")

    def _on_data_loaded(self, df: pd.DataFrame):
        self._plot(df)
        self._update_info(df)

    # --------------------------------------------------------------- chart

    def _plot(self, df: pd.DataFrame):
        close = df["Close"]
        x = df.index

        self.figure.clear()
        gs = self.figure.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.15)
        ax_price = self.figure.add_subplot(gs[0])
        ax_vol = self.figure.add_subplot(gs[1], sharex=ax_price)
        ax_rsi = self.figure.add_subplot(gs[2], sharex=ax_price)
        ax_macd = self.figure.add_subplot(gs[3], sharex=ax_price)

        ax_price.plot(x, close, color=FG, linewidth=1.0, label="Close")
        for window, color in _MA_COLORS:
            if len(close) >= window:
                ax_price.plot(x, market_analysis.sma(close, window), linewidth=0.8,
                               color=color, alpha=0.85, label=f"SMA{window}")
        upper, _middle, lower = market_analysis.bollinger_bands(close)
        ax_price.fill_between(x, lower, upper, color=ACCENT, alpha=0.08, label="Bollinger")
        ax_price.legend(loc="upper left", fontsize=7, facecolor=BG_LIGHT,
                         labelcolor=FG, edgecolor=BORDER)
        ax_price.set_ylabel("Price", color=FG_DIM, fontsize=8)

        if "Volume" in df.columns:
            ax_vol.bar(x, df["Volume"], color=FG_DIM, width=1.0)
        ax_vol.set_ylabel("Vol", color=FG_DIM, fontsize=8)

        rsi_vals = market_analysis.rsi(close)
        ax_rsi.plot(x, rsi_vals, color=ACCENT, linewidth=0.8)
        ax_rsi.axhline(70, color=BAD, linewidth=0.6, linestyle="--")
        ax_rsi.axhline(30, color=GOOD, linewidth=0.6, linestyle="--")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", color=FG_DIM, fontsize=8)

        macd_line, signal_line, hist = market_analysis.macd(close)
        ax_macd.plot(x, macd_line, color=ACCENT, linewidth=0.8, label="MACD")
        ax_macd.plot(x, signal_line, color="#c586c0", linewidth=0.8, label="Signal")
        hist_colors = [GOOD if v >= 0 else BAD for v in hist.fillna(0)]
        ax_macd.bar(x, hist, color=hist_colors, width=1.0, alpha=0.5)
        ax_macd.legend(loc="upper left", fontsize=7, facecolor=BG_LIGHT,
                        labelcolor=FG, edgecolor=BORDER)
        ax_macd.set_ylabel("MACD", color=FG_DIM, fontsize=8)

        for ax in (ax_price, ax_vol, ax_rsi, ax_macd):
            ax.set_facecolor(BG)
            ax.tick_params(colors=FG_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
        for ax in (ax_price, ax_vol, ax_rsi):
            ax.tick_params(labelbottom=False)

        self.figure.set_facecolor(BG)
        self.canvas.draw()

    # ---------------------------------------------------------------- info

    def _update_info(self, df: pd.DataFrame):
        close = df["Close"]
        latest_close = close.iloc[-1]

        lines = []

        latest_rsi = market_analysis.rsi(close).iloc[-1]
        if pd.notna(latest_rsi):
            state = "overbought" if latest_rsi > 70 else "oversold" if latest_rsi < 30 else "neutral"
            lines.append(f"RSI(14): {latest_rsi:.1f} -- traditionally, above 70 is read as "
                          f"overbought and below 30 as oversold ({state} range right now).")

        macd_line, signal_line, _hist = market_analysis.macd(close)
        latest_macd, latest_signal = macd_line.iloc[-1], signal_line.iloc[-1]
        if pd.notna(latest_macd) and pd.notna(latest_signal):
            rel = "above" if latest_macd > latest_signal else "below"
            mood = "bullish" if rel == "above" else "bearish"
            lines.append(f"MACD: {latest_macd:.2f} is {rel} its signal line ({latest_signal:.2f}) "
                          f"-- traditionally read as {mood} momentum.")

        upper, _middle, lower = market_analysis.bollinger_bands(close)
        band_span = upper.iloc[-1] - lower.iloc[-1]
        if pd.notna(band_span) and band_span > 0:
            band_pos = (latest_close - lower.iloc[-1]) / band_span * 100
            lines.append(f"Price sits at {band_pos:.0f}% of the Bollinger Band range "
                          f"(0%=lower band, 100%=upper band) -- extremes here are traditionally "
                          f"read as an unusually stretched move.")

        latest_vol = market_analysis.volatility(close).iloc[-1]
        if pd.notna(latest_vol):
            lines.append(f"10-day annualized volatility: {latest_vol * 100:.1f}%.")

        lines.append("")
        lines.append("Educational only -- not investment advice, not a recommendation to buy or sell.")
        self.info_view.setPlainText("\n".join(lines))

    # ---------------------------------------------------------- watchlist

    def _refresh_watchlist(self):
        self.watchlist_widget.clear()
        for symbol in market_analysis.load_watchlist():
            self.watchlist_widget.addItem(symbol)

    def _on_add_watch_clicked(self):
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            return
        market_analysis.add_to_watchlist(symbol)
        self._refresh_watchlist()

    def _on_remove_watch_clicked(self):
        item = self.watchlist_widget.currentItem()
        if item is None:
            return
        market_analysis.remove_from_watchlist(item.text())
        self._refresh_watchlist()

    def _on_watchlist_item_clicked(self, item):
        self.symbol_input.setText(item.text())
        self._on_load_clicked()

    # ------------------------------------------------------------- news

    def _on_news_custom_search(self):
        query = self.news_query_input.text().strip()
        if query:
            self._run_news_search(f"{query} news")

    def _run_news_search(self, query: str):
        if self._news_worker is not None:
            return
        self.news_results.clear()
        self.news_results.addItem("Searching...")

        worker = NewsSearchWorker(query)
        worker.finished_ok.connect(self._on_news_results)
        worker.failed.connect(self._on_news_failed)
        worker.finished.connect(lambda: setattr(self, "_news_worker", None))
        self._news_worker = worker
        worker.start()

    def _on_news_failed(self, error: str):
        self.news_results.clear()
        self.news_results.addItem(f"Search failed: {error}")

    def _on_news_results(self, results: list):
        self.news_results.clear()
        if not results:
            self.news_results.addItem("No results.")
            return
        for r in results:
            item = QListWidgetItem(f"{r['title']}\n{r['snippet'][:160]}")
            item.setData(Qt.ItemDataRole.UserRole, r["url"])
            self.news_results.addItem(item)

    def _on_news_item_double_clicked(self, item):
        url = item.data(Qt.ItemDataRole.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(url))
