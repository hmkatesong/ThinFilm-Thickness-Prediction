# 3단계 이어서 하기 (11-19 15:55에 B 학습이 5에폭에서 멈춤). A(s3_resnet)는 이미 끝남.
#   -Which train : B 학습 -> 신경망 비교 -> B 정련 평가
#   -Which evalA : A 정련 평가 (단일 스레드, B 학습과 동시에)
param([string]$Which)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'C:\Users\USER\anaconda3\python.exe'
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
if ($Which -eq 'train') {
  $common = '--mode', 'fno', '--arch', 'resnet', '--epochs', '30', '--synth-frac', '0.5', '--mse-w', '0.1', '--select', 'over20'
  & $py 04_train.py @common --name s3_resnet_k4h --k 4 --synth-hard 0.5 *> _work\runs\s3_resnet_k4h.log
  & $py compare_nn.py fno_synth s2_resnet s3_resnet s3_resnet_k4h --out compare_s3 *> _work\runs\compare_s3.log
  & $py 05_eval.py --runs s3_resnet_k4h --set mock_refine --tag s3_k4h *> _work\runs\eval_s3_k4h.log
  'done' | Out-File _work\runs\S3_DONE.txt
} elseif ($Which -eq 'evalA') {
  $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
  & $py 05_eval.py --runs s3_resnet --set mock_refine --tag s3_resnet *> _work\runs\eval_s3_resnet.log
  'done' | Out-File _work\runs\S3_evalA.txt
}
