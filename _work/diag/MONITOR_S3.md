# 3단계 감시 계획 (2026-11-19 13:50 시작)
폴더: C:\Users\USER\Desktop\CUAI\2\project\layer_fno_tmm

## 실행 중
- run_s3.ps1: s3_resnet(30ep) -> s3_resnet_k4h(30ep, k=4, synth-hard 0.5) -> compare_nn.py(mock_nn) -> 05_eval.py mock_refine 두 모델
  - 로그 _work\runs\s3_resnet.log, s3_resnet_k4h.log, compare_s3.log, eval_s3.log / 끝나면 _work\runs\S3_DONE.txt
  - 결과 _work\diag\compare_s3.json, _work\eval_mock_refine_10000_s3.json
- run_s3_side.ps1 refine: fno_synth + 새 정련 규칙, mock_refine -> _work\runs\eval_s31.log, _work\eval_mock_refine_10000_fno_synth_newrules.json, 끝나면 S3_refine.txt
- run_s3_side.ps1 noise: diag_noise.py -> _work\diag\noise.log, noise.json, 끝나면 S3_noise.txt

## 비교 기준 (어제 값, 예전 판정 0.0105)
- fno_synth mock_refine: 최종 0.353 nm, 수렴 99.87%(실패 13), 정련 196 ms/샘플
- 신경망만 mock_nn: fno_synth over20 28.7%, MAE 9.81 / s2_resnet(10ep) 25.96%, 9.37, split13 13.3, split24 16.3

## 보고할 것 (쉬운 한국어, 표)
- 3-1: 새 규칙 효과 (최종 MAE, 수렴(새 기준/예전 기준), 실패 수, n_wrong_but_converged, ms/샘플)
- 3-2: A vs B vs fno_synth: over20, oracle_over20, MAE, split13/24, 구간별, 학습 시간 / 정련 포함 최종 MAE·수렴·실패·ms/샘플
- 3-3: 파장별 평균 잔차가 노이즈 대비 큰가, 이웃 상관, 평균 잔차 빼기로 정답 출발 정련 MAE·이론한계 배수가 줄었나(특히 0-400)
- 다음: 4단계(보정 신경망) 진입 여부 판단

## 중간 결과 메모 (15:25)
- 3-1 끝: fno_synth + 새 규칙 mock_refine 1만: 최종 0.303 nm(어제 0.353), 중앙 0.178, 실패 0(TOL 0.012), 예전 기준 수렴 99.93%(어제 99.87%), 329 ms/샘플(어제 196, 맞바꿈 출발점 탓).
  주의: n_wrong_but_converged=25 (곡선은 0.012 안인데 한 층 5nm 넘게 틀림), 그중 over20 6개. 예전 기준 0.0105 넘는 건 7개뿐 -> 18개 이상은 곡선이 노이즈 수준으로 맞는 오답 = 곡선으로 구분 못하는 경우일 수 있음. 보고 때 '믿을 만함' 표시의 한계로 설명, 다음 분석 후보 (연습용 세트에서 확인).
- 3-3 끝: 평균 잔차 빼기 효과 없음 (0.2736 -> 0.2738, 얇은 쪽 1.95배 그대로). noise.json의 autocorr 값도 보고.
- s3_resnet ep24: 고르기용 over20 13.0%, MAE 6.18 (fno_synth 30ep 약 29%).

## 11-20 09:45 재시작
- 11-19 15:55에 B(s3_resnet_k4h) 학습이 5에폭에서 멈춤(프로세스 사라짐, 원인 불명: PC 절전/종료 또는 런타임 재시작 추정). 멈춘 로그는 s3_resnet_k4h_stopped_1119.log.
- A(s3_resnet)는 30에폭 완료. mock_nn: over20 10.03%, MAE 5.15, median 2.81, split13 7.61/sum13 6.17, split24 8.43/sum24 7.82, 구간별 over20 30.3/13.4/3.8/0.5%. (fno_synth 28.7%, s2_resnet 25.96%)
- run_s3_rest.ps1로 다시 실행: train(B 학습 -> compare_s3 -> B 정련 평가, 끝나면 S3_DONE.txt), evalA(A 정련 평가, 끝나면 S3_evalA.txt).
- 결과 파일: _work\eval_mock_refine_10000_s3_resnet.json, _work\eval_mock_refine_10000_s3_k4h.json, _work\diag\compare_s3.json
- 완료 마커: S3_DONE.txt + S3_evalA.txt (S3_refine, S3_noise는 이미 있음)
