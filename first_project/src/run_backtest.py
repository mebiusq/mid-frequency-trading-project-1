import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def run_backtest(cfg):  # вместо фиксированного коэффициента, будем пересчитывать бета каждое окно, чтобы учитывать изменения рынка
    test = pd.read_csv('data/backtest_data.csv', index_col=0, parse_dates=True)
    with open('artifacts/best_params.json', 'r') as f:
        pairs = json.load(f)

    pnl_portfolio = pd.Series(0.0, index=test.index)
    cap_per_pair = cfg['initial_capital'] / max(1, len(pairs))
    for p in pairs:
        y, x = test[p['y']], test[p['x']]

        cov_roll = y.rolling(window=cfg['rolling_window']).cov(x)  # rolling OLS для динамического подсчета беты
        var_roll = x.rolling(window=cfg['rolling_window']).var()
        beta = cov_roll / var_roll
        beta = beta.fillna(method='bfill')

        spread = y - beta * x  # нормализация
        mean_roll = spread.rolling(window=cfg['z_window']).mean()
        std_roll = spread.rolling(window=cfg['z_window']).std()
        z = (spread - mean_roll) / std_roll  # насколько текущее расхождение аномально

        signals = pd.Series(0, index=test.index)  # большой блок для определения сигналов
        pos = 0
        pos_array = np.zeros(len(z))
        z_arr = z.values

        for i in range(len(z_arr)):  # разные проверки
            zval = z_arr[i]
            if np.isnan(zval):
                continue
            if abs(zval) > cfg['stop_loss']:
                pos = 0
            elif pos == 0:  # вне рынка
                if zval < -cfg['entry_threshold']:
                    pos = 1  # шорт
                elif zval > cfg['entry_threshold']:
                    pos = -1  # лонг
            elif pos != 0:  # в рынке
                if abs(zval) < cfg['exit_threshold']:
                    pos = 0  # возврат к среднему
                if pos == 1 and zval > cfg['entry_threshold']:
                    pos = -1
                if pos == -1 and zval < -cfg['entry_threshold']:
                    pos = 1

            pos_array[i] = pos

        positions = pd.Series(pos_array, index=test.index)
        ret_y = y.pct_change().fillna(0)  # считаем доходность
        ret_x = x.pct_change().fillna(0)
        strategy_ret = positions.shift(1) * (0.5 * ret_y - 0.5 * ret_x)
        turnover = positions.diff().abs().fillna(0)  # учитываем транзакционные издержки
        costs = turnover * cfg['transaction_cost']
        net = strategy_ret - costs

        pnl_portfolio += (net * cap_per_pair)

    equity = cfg['initial_capital'] + pnl_portfolio.cumsum()

    rets = equity.pct_change().fillna(0)
    total_ret = (equity.iloc[-1] - cfg['initial_capital']) / cfg['initial_capital']
    cumulative = (1 + rets).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    sharpe = np.sqrt(252) * rets.mean() / rets.std()

    metrics = {
        "final_capital": float(equity.iloc[-1]),
        "total_return": float(total_ret),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd)
    }
    with open('artifacts/backtest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    plt.figure(figsize=(12, 6))
    plt.plot(equity)
    plt.title('Изменение капитала во времени')
    plt.grid(True)
    plt.savefig('artifacts/equity_curve_image.png')
