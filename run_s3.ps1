# 3단계: ResNet 30에폭 두 개 학습 -> 신경망만 비교(mock_nn) -> 정련 포함 평가(mock_refine). 로그: _work\runs\*.log
#   A. s3_resnet       : 2단계와 같은 구조, 30에폭 (새 기준선)
#   B. s3_resnet_k4h   : 후보 4개 출력 + 어려운 조합 합성 50%
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'C:\Users\USER\anaconda3\python.exe'
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
Remove-Item _work\runs\S3_DONE.txt -ErrorAction SilentlyContinue
$common = '--mode', 'fno', '--arch', 'resnet', '--epochs', '30', '--synth-frac', '0.5', '--mse-w', '0.1', '--select', 'over20'
& $py 04_train.py @common --name s3_resnet *> _work\runs\s3_resnet.log
& $py 04_train.py @common --name s3_resnet_k4h --k 4 --synth-hard 0.5 *> _work\runs\s3_resnet_k4h.log
& $py compare_nn.py fno_synth s2_resnet s3_resnet s3_resnet_k4h --out compare_s3 *> _work\runs\compare_s3.log
& $py 05_eval.py --runs s3_resnet s3_resnet_k4h --set mock_refine --tag s3 *> _work\runs\eval_s3.log
'done' | Out-File _work\runs\S3_DONE.txt
