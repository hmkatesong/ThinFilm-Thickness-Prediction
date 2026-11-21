"""모델. 모두 fit_predict(Xtr, ytr, Xva, yva, X_eval_list, seed) -> [예측...] 형태.

Xva/yva는 early stopping과 Ridge alpha 선택에만 쓴다. 최종 점수는 run_exp.py가 따로 계산한다.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def fit_mean(Xtr, ytr, Xva, yva, evals, seed):
    m = ytr.mean(axis=0)
    return [np.tile(m, (len(X), 1)) for X in evals], {}


def fit_knn(Xtr, ytr, Xva, yva, evals, seed):
    """최근접 이웃 1개: 가장 비슷한 학습 곡선의 두께를 그대로 답함 = '외우기' 기준선."""
    m = KNeighborsRegressor(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Xtr, ytr)
    return [m.predict(X) for X in evals], {}


def _prep():
    return [SimpleImputer(strategy="constant", fill_value=-1), StandardScaler()]


def fit_ridge(Xtr, ytr, Xva, yva, evals, seed):
    best = None
    for a in [0.1, 1, 10, 100]:
        m = make_pipeline(*_prep(), Ridge(alpha=a)).fit(Xtr, ytr)
        s = mae(yva, m.predict(Xva))
        if best is None or s < best[0]:
            best = (s, a, m)
    return [best[2].predict(X) for X in evals], {"alpha": best[1]}


def fit_mlp(Xtr, ytr, Xva, yva, evals, seed):
    """sklearn MLP (256, 256). 직접 정한 흔한 설정 (sklearn 기본값은 (100,)).

    입력과 출력 모두 표준화. early stopping은 학습 데이터 안의 10%로 한다 (Xva는 안 씀).
    """
    ym, ys = ytr.mean(0), ytr.std(0)
    m = make_pipeline(
        *_prep(),
        MLPRegressor(
            hidden_layer_sizes=(256, 256),
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        ),
    ).fit(Xtr, (ytr - ym) / ys)
    return [m.predict(X) * ys + ym for X in evals], {"n_iter": int(m[-1].n_iter_)}


LGBM_LR = 0.1  # LightGBM 기본값. run_exp.py --lr 로 바꿀 수 있다 (E04, E05는 0.05)


def fit_lgbm(Xtr, ytr, Xva, yva, evals, seed):
    """층마다 LightGBM 하나씩. 흔한 시작 설정 + valid MAE(l1)로 early stopping.

    first_metric_only=True: 지표 목록이 [l1, l2]라서, 이게 없으면 l2가 먼저 정체돼도 멈춘다.
    E01~E25는 이 옵션을 넣기 전 코드다. 5,000그루 상한까지 간 층은 결과가 같고,
    상한 전에 멈춘 E15 layer_1, E16 layer_2·layer_4만 다시 돌리면 조금 다를 수 있다.
    """
    import lightgbm as lgb

    preds = [np.zeros((len(X), ytr.shape[1])) for X in evals]
    iters = []
    for j in range(ytr.shape[1]):
        m = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=LGBM_LR,
            num_leaves=63,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            n_jobs=8,
            verbose=-1,
        )
        m.fit(
            Xtr, ytr[:, j], eval_set=[(Xva, yva[:, j])], eval_metric="l1",
            callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
        )
        iters.append(int(m.best_iteration_))
        for p, X in zip(preds, evals):
            p[:, j] = m.predict(X, num_iteration=m.best_iteration_)
    return preds, {"best_iter": iters}


MODELS = {"mean": fit_mean, "knn": fit_knn, "ridge": fit_ridge, "mlp": fit_mlp, "lgbm": fit_lgbm}
