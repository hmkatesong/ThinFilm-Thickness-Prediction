# FNO 단독 -> FNO+TMM 순서로 학습. 로그: _work\runs\<이름>.log
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'C:\Users\USER\anaconda3\python.exe'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force _work\runs | Out-Null
& $py 04_train.py --mode fno --name fno *> _work\runs\fno.log
& $py 04_train.py --mode fno_tmm --name fno_tmm *> _work\runs\fno_tmm.log
'done' | Out-File _work\runs\ALL_DONE.txt
