# 3단계 학습과 함께 돌리는 학습 없는 작업 (단일 스레드라 학습을 크게 방해하지 않음)
#   3-1: 현재 모델(fno_synth)에 새 정련 규칙(판정 기준 0.0120, 재료 맞바꿈 출발점) -> 모의고사 1만 개
#   3-3: TMM과 데이터의 미세한 차이 (diag_noise.py)
param([string]$Which)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'C:\Users\USER\anaconda3\python.exe'
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
switch ($Which) {
  'refine' { & $py 05_eval.py --runs fno_synth --set mock_refine --tag fno_synth_newrules *> _work\runs\eval_s31.log }
  'noise'  { & $py diag_noise.py *> _work\diag\noise.log }
}
'done' | Out-File "_work\runs\S3_$Which.txt"
