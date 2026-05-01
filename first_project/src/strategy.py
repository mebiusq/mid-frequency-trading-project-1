import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import json
import os
from first_project.src.utils import calculate_half_life


def train_strategy(cfg):  # ищем коинтегрированные пары на трейне, учитывая есть ли долгосрочная связь и достаточно ли быстрая она
    train = pd.read_csv('data/train_data.csv', index_col=0, parse_dates=True)
    cols = train.columns
    candidates = []

    for i in range(len(cols)):  # не очень эффективно, но перебираем все пары
        for j in range(i + 1, len(cols)):
            s1, s2 = train[cols[i]], train[cols[j]]
            score, pval, _ = coint(s1, s2)  # тест, H0: нет коинтеграции

            if pval < 0.05:  # если H0 отвергается, коинтеграция есть, проверяем её скорость
                mdl = sm.OLS(s1, sm.add_constant(s2)).fit()
                spread = s1 - mdl.params[1] * s2
                hl = calculate_half_life(spread)

                if hl < cfg['max_half_life']:  # отсеиваем пары где не достаточно быстрая связь
                    candidates.append({
                        'y': cols[i],
                        'x': cols[j],
                        'pvalue': float(pval),
                        'half_life': float(hl),
                        'correlation': float(s1.corr(s2))
                    })

    candidates.sort(key=lambda x: x['pvalue'])  # сортируем, потом убираем тикеры, чтобы каждый был только в 1 паре
    used = set()
    pairs = []
    for c in candidates:
        if c['y'] not in used and c['x'] not in used:
            pairs.append(c)
            used.add(c['y'])
            used.add(c['x'])
            if len(pairs) >= 5:
                break

    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/best_params.json', 'w') as f:
        json.dump(pairs, f, indent=4)