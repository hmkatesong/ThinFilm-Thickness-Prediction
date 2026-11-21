# B+C 재학습 -> 세 모델을 같은 Valid 1000개로 정련까지 평가. 로그: _work\runs\*.log
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'C:\Users\USER\anaconda3\python.exe'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item _work\runs\V2_DONE.txt -ErrorAction SilentlyContinue
& $py 04_train.py --mode fno --name fno_synth --synth-frac 0.5 --mse-w 0.1 --select over20 *> _work\runs\fno_synth.log
& $py 05_eval.py --runs fno fno_tmm fno_synth --set valid --n 1000 *> _work\runs\eval_v2.log
'done' | Out-File _work\runs\V2_DONE.txt
