# 11-20 다음 작업 1~3번 실행 기록. 로그: _work\runs\*.log, 끝나면 _work\runs\S4_<Which>.txt
#   -Which step1  : 후보 여러 개를 정련 출발점으로 (mock_refine 1만 개)
#                   a) B 후보 4개 전부  b) A 답 + B 후보 4개  c) A만 (새 시간 측정 방식 기준선)
#   -Which step2  : 곡선은 맞는 오답 분석 (practice, 단일 스레드)
#   -Which step3  : 보정 신경망 학습 -> 평가
param([string]$Which)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'C:\Users\USER\anaconda3\python.exe'
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
switch ($Which) {
  'step1' {
    & $py 05_eval.py --runs s3_resnet_k4h --all-cands --fast --tag s4_b_allcands *> _work\runs\eval_s4_b_allcands.log
    & $py 05_eval.py --runs s3_resnet --extra-run s3_resnet_k4h --all-cands --tag s4_a_plus_b *> _work\runs\eval_s4_a_plus_b.log
    & $py 05_eval.py --runs s3_resnet --fast --tag s4_a_base *> _work\runs\eval_s4_a_base.log
  }
  'step2' {
    $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
    & $py diag_wrong.py --run s3_resnet *> _work\diag\wrong.log
  }
  'step3' {
    & $py 06_train_corrector.py --base s3_resnet --name s4_corr *> _work\runs\s4_corr.log
    & $py compare_nn.py s3_resnet s3_resnet_k4h s4_corr --out compare_s4 *> _work\runs\compare_s4.log
    & $py 05_eval.py --runs s4_corr --fast --tag s4_corr *> _work\runs\eval_s4_corr.log
  }
}
'done' | Out-File "_work\runs\S4_$Which.txt"
