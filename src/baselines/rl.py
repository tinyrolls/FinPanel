from __future__ import annotations

import numpy as np
import pandas as pd

INDICATORS = ("macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma")
HMAX = 1000
INITIAL_AMOUNT = 100_000.0
COST_PCT = 0.0049
REWARD_SCALING = 1e-4
TRADING_DAYS_3Y = 252 * 3
WARMUP_CALENDAR_DAYS = 1200
TOTAL_TIMESTEPS = 5000
SEED = 7


def _indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    from stockstats import StockDataFrame as Sdf

    df = ohlc.copy()
    df.columns = [str(c).lower() for c in df.columns]
    sdf = Sdf.retype(df[["open", "high", "low", "close", "volume"]].copy())
    out = pd.DataFrame(index=ohlc.index)
    for name in INDICATORS:
        out[name] = pd.to_numeric(sdf[name], errors="coerce").to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def _apply_action(cash: float, shares: float, price: float, action: float) -> tuple[float, float, int]:
    n = int(float(np.clip(action, -1.0, 1.0)) * HMAX)
    if n < 0:
        sell = min(-n, int(shares))
        return cash + sell * price * (1 - COST_PCT), shares - sell, sell
    if n > 0:
        affordable = int(cash // (price * (1 + COST_PCT))) if price > 0 else 0
        buy = min(n, affordable)
        return cash - buy * price * (1 + COST_PCT), shares + buy, buy
    return cash, shares, 0


def _make_env(prices: np.ndarray, ind: np.ndarray):
    import gymnasium as gym
    from gymnasium import spaces

    class StockTradingEnv(gym.Env):
        metadata: dict = {}

        def __init__(self, prices: np.ndarray, ind: np.ndarray):
            super().__init__()
            self.prices, self.ind, self.T = prices, ind, len(prices)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3 + ind.shape[1],), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def _obs(self) -> np.ndarray:
            return np.array([self.cash, self.prices[self.day], self.shares, *self.ind[self.day]], dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.day, self.cash, self.shares = 0, INITIAL_AMOUNT, 0.0
            return self._obs(), {}

        def step(self, action):
            price = self.prices[self.day]
            begin = self.cash + self.shares * price
            self.cash, self.shares, traded = _apply_action(self.cash, self.shares, price, action[0])
            self.day += 1
            done = self.day >= self.T - 1
            end = self.cash + self.shares * self.prices[self.day]
            penalty = -0.1 * (1 - traded / HMAX)
            reward = (end - begin) * REWARD_SCALING + penalty
            return self._obs(), float(reward), done, False, {}

    return StockTradingEnv(prices, ind)


class _RLStrategy:

    name = "rl_base"
    algo = None
    algo_kwargs: dict = {}
    warmup_days = WARMUP_CALENDAR_DAYS

    seed = SEED

    def __init__(self, seed: int | None = None, train_window: int = TRADING_DAYS_3Y, total_timesteps: int = TOTAL_TIMESTEPS):
        if seed is not None:
            self.seed = seed
        self.train_window, self.total_timesteps = train_window, total_timesteps

    def prepare(self, close: pd.Series, ohlc: pd.DataFrame | None = None) -> None:
        self._model = None
        self._ind_full = _indicators(ohlc) if ohlc is not None else None
        self._prices_full = (ohlc["close"] if ohlc is not None else close).astype(float)
        self._sh_cash, self._sh_shares = INITIAL_AMOUNT, 0.0

    def _train(self, before_date) -> None:
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.utils import set_random_seed

        ind = self._ind_full[self._ind_full.index < before_date]
        prices = self._prices_full.reindex(ind.index)
        df = ind.join(prices.rename("__price__")).dropna()
        _ws = getattr(self, "_warmup_start", None)
        df = df[df.index >= _ws] if _ws is not None else df.iloc[-self.train_window :]
        if len(df) < 60:
            return
        p = df["__price__"].to_numpy(dtype=np.float32)
        X = df[list(INDICATORS)].to_numpy(dtype=np.float32)
        kwargs = dict(self.algo_kwargs)
        if kwargs.get("action_noise") == "normal":
            kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
        set_random_seed(self.seed)
        env = _make_env(p, X)
        model = self.algo("MlpPolicy", env, seed=self.seed, verbose=0, **kwargs)
        model.learn(total_timesteps=self.total_timesteps)
        self._model = model

    def decide(self, ctx) -> float:
        if getattr(ctx, "phase", "test") == "warmup":
            return 0.0
        if self._ind_full is None:
            return ctx.current_weight
        if getattr(self, "_model", None) is None:
            try:
                self._train(ctx.date)
            except Exception:
                self._model = None
        if getattr(self, "_model", None) is None:
            return ctx.current_weight
        try:
            ind_t = self._ind_full.loc[ctx.date, list(INDICATORS)].to_numpy(dtype=np.float32)
        except KeyError:
            return ctx.current_weight
        if np.isnan(ind_t).any():
            return ctx.current_weight
        price = float(ctx.close)
        obs = np.array([self._sh_cash, price, self._sh_shares, *ind_t], dtype=np.float32)
        action, _ = self._model.predict(obs, deterministic=True)
        self._sh_cash, self._sh_shares, _ = _apply_action(self._sh_cash, self._sh_shares, price, action[0])
        nav = self._sh_cash + self._sh_shares * price
        return float(np.clip((self._sh_shares * price) / nav, 0.0, 1.0)) if nav > 0 else 0.0


def _algos() -> dict:
    from stable_baselines3 import A2C, PPO, SAC, TD3

    return {
        "a2c": (A2C, {"n_steps": 100, "learning_rate": 1e-5, "ent_coef": 0.1, "vf_coef": 0.5, "max_grad_norm": 0.5, "gae_lambda": 0.95, "gamma": 0.99}),
        "ppo": (PPO, {"n_steps": 2048, "batch_size": 64, "n_epochs": 10, "learning_rate": 2.5e-4, "ent_coef": 0.1, "clip_range": 0.2, "gae_lambda": 0.95, "gamma": 0.99}),
        "sac": (SAC, {"learning_rate": 2e-2, "buffer_size": 1_000_000, "batch_size": 256, "learning_starts": 100, "ent_coef": 0.1, "tau": 0.005, "gamma": 0.99, "action_noise": "normal"}),
        "td3": (TD3, {"learning_rate": 3e-2, "buffer_size": 1_000_000, "tau": 0.005, "gamma": 0.99, "policy_delay": 2, "target_policy_noise": 0.5, "target_noise_clip": 0.5, "action_noise": "normal"}),
    }


def _build() -> dict:
    out = {}
    for name, (algo, kw) in _algos().items():
        out[name] = type(
            f"{name.upper()}Strategy", (_RLStrategy,),
            {"name": name, "algo": staticmethod(algo), "algo_kwargs": kw},
        )
    return out


RL_NAMES = ("a2c", "ppo", "sac", "td3")
_CACHE: dict = {}


def get_rl_strategy(name: str):
    if not _CACHE:
        _CACHE.update(_build())
    return _CACHE[name]
