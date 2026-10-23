from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TRADING_DAYS_3Y = 252 * 3
WARMUP_CALENDAR_DAYS = 1100


class BasePredictor:

    name = "base_predictor"
    warmup_days = WARMUP_CALENDAR_DAYS

    def prepare(self, close: pd.Series, ohlc: pd.DataFrame | None = None) -> None:
        self._reset()

    def _reset(self) -> None:
        pass

    def decide(self, ctx) -> float:
        raise NotImplementedError


class ARIMAStrategy(BasePredictor):

    name = "arima"

    def __init__(self, order: tuple[int, int, int] = (5, 1, 0), train_window: int = TRADING_DAYS_3Y):
        self.order, self.train_window = order, train_window

    def _reset(self) -> None:
        self._fit = None

    def decide(self, ctx) -> float:
        if getattr(ctx, "phase", "test") == "warmup":
            return 0.0
        close = ctx.ohlc["close"].dropna()
        if len(close) < max(self.order[0] + self.order[1] + 2, 30):
            return ctx.current_weight
        try:
            from statsmodels.tsa.arima.model import ARIMA

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if getattr(self, "_fit", None) is None:
                    _pre = close.iloc[:-1]
                    _ws = getattr(self, "_warmup_start", None)
                    _pre = _pre[_pre.index >= _ws] if _ws is not None else _pre.iloc[-self.train_window :]
                    train = _pre.to_numpy(dtype=float)
                    self._fit = ARIMA(train, order=self.order).fit()
                forecast = float(self._fit.apply(close.to_numpy(dtype=float)).forecast(steps=1)[0])
        except Exception:
            return ctx.current_weight
        if not np.isfinite(forecast):
            return ctx.current_weight
        if forecast > ctx.close:
            return 1.0
        if forecast < ctx.close:
            return 0.0
        return ctx.current_weight


class _ClassifierPredictor(BasePredictor):

    FEATURES = ("open", "high", "low", "close", "volume")

    def __init__(self, train_window: int = TRADING_DAYS_3Y, seed: int = 0):
        self.train_window = train_window
        self.seed = seed

    def _reset(self) -> None:
        self._model = None
        self._cols: list[str] | None = None

    def _fit(self, X: np.ndarray, y: pd.Series) -> None:
        raise NotImplementedError

    def _prob(self, row: np.ndarray) -> float:
        raise NotImplementedError

    def _train(self, ohlc: pd.DataFrame) -> None:
        cols = [c for c in self.FEATURES if c in ohlc.columns]
        df = ohlc[cols].copy()
        df["__target__"] = (df["close"].pct_change().shift(-1) > 0).astype(int)
        _ws = getattr(self, "_warmup_start", None)
        df = (df[df.index >= _ws] if _ws is not None else df.iloc[-self.train_window :]).dropna()
        if len(df) < 30:
            return
        y = df["__target__"]
        if y.nunique() < 2:
            return
        self._cols = cols
        self._fit(df[cols].to_numpy(dtype=float), y)

    def decide(self, ctx) -> float:
        if getattr(ctx, "phase", "test") == "warmup":
            return 0.0
        if getattr(self, "_model", None) is None:
            self._train(ctx.ohlc.iloc[:-1])
        if self._model is None or not self._cols:
            return ctx.current_weight
        row = ctx.ohlc.iloc[[-1]][self._cols].to_numpy(dtype=float)
        return 1.0 if self._prob(row) >= 0.5 else 0.0


class XGBoostStrategy(_ClassifierPredictor):

    name = "xgboost"

    def __init__(self, train_window: int = TRADING_DAYS_3Y, params: dict | None = None, num_boost_round: int = 10):
        super().__init__(train_window)
        self.params = params if params is not None else {"verbosity": 0, "seed": 0}
        self.num_boost_round = num_boost_round

    def _fit(self, X, y) -> None:
        import xgboost as xgb

        dtrain = xgb.DMatrix(X, label=y.to_numpy(dtype=float))
        self._model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)

    def _prob(self, row) -> float:
        import xgboost as xgb

        return float(self._model.predict(xgb.DMatrix(row))[0])


class SVMStrategy(_ClassifierPredictor):

    name = "svm"

    def _reset(self) -> None:
        super()._reset()
        self._scaler = None

    def _fit(self, X, y) -> None:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(X)
        self._model = SVC(kernel="rbf", probability=True, random_state=self.seed)
        self._model.fit(self._scaler.transform(X), y.to_numpy(dtype=int))

    def _prob(self, row) -> float:
        return float(self._model.predict_proba(self._scaler.transform(row))[0][1])


class NaiveBayesStrategy(_ClassifierPredictor):

    name = "naive_bayes"

    def _fit(self, X, y) -> None:
        from sklearn.naive_bayes import GaussianNB

        self._model = GaussianNB()
        self._model.fit(X, y.to_numpy(dtype=int))

    def _prob(self, row) -> float:
        return float(self._model.predict_proba(row)[0][1])


PREDICTOR_STRATEGIES = {cls.name: cls for cls in (ARIMAStrategy, XGBoostStrategy, SVMStrategy, NaiveBayesStrategy)}
