# 평가 세트 고정 -> _work/eval_sets.npz  (2026-11-18 생성. 다시 실행하면 같은 세트인지 확인만 한다)
#   python make_eval_sets.py            확인 (파일이 있으면 덮어쓰지 않음)
#   python make_eval_sets.py --write    파일이 없을 때 새로 만들기
import json, os, sys
import numpy as np
from common import WORK
sys.stdout.reconfigure(encoding="utf-8")

sp = np.load(os.path.join(WORK, "split.npz"))
valid, test = sp["valid"], sp["test"]
# 04_train.py가 best 모델 선택에 쓰는 Valid 1만 개 (default_rng(0) 첫 호출과 같은 값)
select = np.sort(np.random.default_rng(0).choice(valid, 10000, replace=False))
pool = np.setdiff1d(valid, select)
# 05_eval.py 예전 방식으로 이미 들여다본 세트: 1,000개(seed 5, 쌍 나누기 설계), 3,000개(seed 11)
p1000 = np.random.default_rng(5).choice(pool, 1000, replace=False)
p3000 = np.random.default_rng(11).choice(pool, 3000, replace=False)
practice = np.union1d(p1000, p3000)
mock_nn = np.setdiff1d(pool, practice)
mock_refine = np.sort(np.random.default_rng(2026).choice(mock_nn, 10000, replace=False))
test_final = np.sort(np.random.default_rng(2027).choice(test, 20000, replace=False))
for x, y in [(select, practice), (select, mock_nn), (practice, mock_nn)]:
    assert len(np.intersect1d(x, y)) == 0
assert len(np.setdiff1d(mock_refine, mock_nn)) == 0
sets = dict(select=select, practice=practice, practice_1000=np.sort(p1000), practice_3000=np.sort(p3000),
            mock_nn=mock_nn, mock_refine=mock_refine, test_final=test_final)
info = {"select": "04_train.py best 모델 선택용", "practice": "연습용: 실패 분석·설계에 자유롭게 사용",
        "mock_nn": "모의고사(신경망만): Valid - select - practice", "mock_refine": "모의고사(정련 포함): mock_nn 중 1만 개",
        "test_final": "최종 평가: Test 중 2만 개. 마지막에 한 번만"}
out = os.path.join(WORK, "eval_sets.npz")
if os.path.exists(out):
    old = np.load(out)
    same = all(np.array_equal(old[k], v) for k, v in sets.items())
    print("기존 파일과 같은가:", same)
elif "--write" in sys.argv:
    np.savez(out, **sets, info=np.array(json.dumps(info, ensure_ascii=False)))
    print("저장:", out)
print({k: len(v) for k, v in sets.items()})
