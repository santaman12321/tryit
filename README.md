# tryit — 미국 급등주(동전주·소형주) 자동매매 리서치 & 실행 프레임워크

한국 투자자 기준(국내 증권사 해외주식 API: **공매도 불가**, 매매수수료 **편도 0.25%**)으로
블로그·유튜브·해외 커뮤니티에서 공유되는 '급등주 찾기 공식' 을 모아 정량 규칙으로 바꾸고,
미국 전체 상장 보통주(약 5,000종목) 일봉 3년 + 1시간봉 2년 + 15분봉 60일로 검증한 뒤,
살아남는 규칙만 자동매매(한국투자증권 Open API 어댑터, 기본 dry-run)로 넘기는 구조다.

**최종 권고 모델** → [`reports/best_model/README.md`](reports/best_model/README.md)
- 급등주·데이트레이딩 대신 **저회전 추세 모델 2개의 50/50 블렌드**가 가장 강했다: (1) S&P 1500 돌파 추세추종(20일 고가·거래량·정배열·레짐 필터·ATR 트레일, 10종목), (2) QQQ 변동성 타게팅 30% + 200일선(TQQQ 실행).
  2024-01~2026-09 국내 증권사 비용 기준 총 +135%(CAGR 37%), MDD −16%, 샤프 1.42, 세 해 각각 +32~34%. ETF 레그는 2005년부터 20년 검증(CAGR 22%, MDD −39%).
- 단, 돌파 레그는 상위 10건 거래가 전체 손익(추세추종의 본질), 검증 기간이 강세장, 생존 편향 존재. 실행: `scripts/run_daily.py --strategy breakout --universe sp1500 ...` + `scripts/etf_signal.py`.

**공개 자동매매 모델과의 비교** → [`reports/community_compare/README.md`](reports/community_compare/README.md), 출처 [`docs/community_models.md`](docs/community_models.md)
- GitHub·Reddit·Composer·국내 유튜브/블로그의 모델 17종(HFEA, 9Sig, TQQQ FTLT, 200일선 스위칭, 듀얼모멘텀 GEM, 라오어 무한매수법·밸류리밸런싱, 변동성 돌파, SMA/MACD/볼린저/RSI/슈퍼트렌드/EMA 봇)을 규칙대로 재현해 같은 비용으로 비교했다.
- 2024~2026 구간에서 우리 블렌드(+135%, MDD −16%)보다 총수익이 높은 것은 3배 레버리지에 상시 노출되는 모델(9Sig +190%, TQQQ FTLT +165~210%, 200일선→TQQQ +123%, SMA 50/200 골든크로스 종목 +238%)뿐이며 MDD가 −44~−73%다. 위험 대비로는 우리 블렌드가 앞선다.
- 변동성 돌파(K=0.5)·무한매수법·RSI/볼린저 봇은 국내 수수료에서 손실 또는 저수익. FTLT 류의 장기 CAGR 46~64%는 2020년 한 해(+841~2273%)가 만든 숫자다.

**급등주·데이트레이딩 검증 결과** → [`reports/surge_study/FINDINGS.md`](reports/surge_study/FINDINGS.md)
- 공식 15종 + 커뮤니티 글 4종 + 전략 클래스 5종 = **24개 모델, 1,779개 설정을 같은 조건(수수료 편도 0.25%, 슬리피지, 유동성 제한)으로 경쟁**시켰다.
  학습(2023-11~2025-09)에서 고른 최적 설정을 검증(2025-09~2026-09)에 적용하면 24개 중 18개가 기대값 음수로 뒤집힌다.
- 검증에서 살아남은 6개 모델·104개 설정도 강건성 점검에서 무너진다: 승률 60%대 설정(장대양봉+최대거래량 초소형주 / Warrior 스캐너 + 익절 15%)은
  익절 체결 조건을 '고가+3%'로만 바꿔도 학습 수익이 -21~-61%, '종가 확인' 체결이면 -92~-97%. 눌림목 계열(+160%)은 검증 전반 +0.8%/후반 -41%, 상위 10건 손익 비중 249%(로또 거래 의존).
- '이유 없는 급등'(실적 무관)은 실적 급등보다 더 나쁘고(3일 보유 -1.3~-1.8%), 회전율 100%↑ 급등은 3일 중앙값 -7~-9%. 급등 다음날 고가의 58%(1시간봉)/35%(15분 첫 봉)는 개장 직후.
- 장중 진입×청산×손절 조합 300개(1시간봉·15분봉) 중 두 기간 모두 양(+)은 0개. **결론: 국내 계좌로 미국 동전주·소형주 급등주를 매수하는 자동매매는 검증을 통과한 설정이 없다.**
- **딥러닝/ML 데이트레이딩(대형주 포함 4,259종목, 1시간봉)** → [`reports/dl_daytrading/README_open.md`](reports/dl_daytrading/README_open.md): 문헌 설계(횡단면 순위 목표, LightGBM+GRU 앙상블, 시간 분할)로 테스트 IC +0.03~+0.05(t≈4~6)의 **유의한 예측력**은 확인.
  하지만 시가 매수·종가 매도 상위 10종목의 거래당 총수익 +0.4~+0.7%가 국내 증권사 왕복 비용 0.7%와 같거나 낮아 비용후 −0.3~−0.1%/거래. 병목은 모델이 아니라 수수료 구조(왕복 0.2% 이하면 비용후 양수 기대).

## 구조

```
quant/            핵심 패키지
  data.py         유니버스(S&P1500 / 미국 전체 보통주+시총·국가), Yahoo 일봉 수집·캐시, 패널 변환
  indicators.py   SMA/EMA/RSI/ATR 등 (look-ahead 없음)
  screens.py      급등주 공식 15종 → 불리언 신호 패널 (docs/surge_formulas.md 와 1:1)
  events.py       이벤트 스터디(신호 이후 N일 수익률·승률·t값, 구간별)
  intraday.py     1시간/15분봉 경로 시뮬레이션(손절·익절·트레일)
  backtest.py     포트폴리오 백테스터(다음날 시가 체결, 슬리피지·정률 수수료·유동성 제한, 손절/익절/트레일/시간청산)
  metrics.py      CAGR·샤프·MDD·승률·PF 등
  report.py       차트·마크다운
  strategies/     pullback(눌림목 평균회귀), breakout(돌파 추세추종), gap_day(갭 단타), surge(급등 추격/눌림), formula(공식 래퍼)
  broker.py       DryRun / KIS(한국투자증권 해외주식) 주문 어댑터
  live.py         일일 주문 계획(백테스터와 같은 Strategy 사용)
scripts/
  fetch_data.py       일봉 수집  (--universe all|sp1500)
  fetch_intraday.py   1h(730일)/15m(60일) 봉 수집
  fetch_earnings.py   실적 발표일 수집(재료 유무 구분용)
  event_study.py      공식별 이벤트 스터디 + 적절값 탐색 → reports/surge_study/README.md
  intraday_study.py   장중 진입×청산×손절 조합 검증 → reports/surge_study/intraday_{1h,15m}/
  run_backtest.py     포트폴리오 백테스트 → reports/<기간>/
  sensitivity.py      파라미터 격자(여러 기간 동시)
  tournament.py       모델 토너먼트(모든 모델 파라미터 탐색 → 학습 선택 → 검증 평가, 샤딩 지원) → reports/surge_study/tournament/
  robustness.py       생존 설정 강건성 점검(검증 전/후반, 익절 체결 가정 변형, 이상치 의존도)
  make_findings.py    최종 보고서 생성 → reports/surge_study/FINDINGS.md
  run_daily.py        매일 장 마감 후 신호→주문 계획/제출
  dl/               딥러닝/ML 데이트레이딩: dataset.py(1시간봉+일봉 맥락 샘플), models.py(LightGBM·GRU·규칙 베이스라인·평가)
scripts/
  dl_train.py         모델 학습·평가 (--decision h1|open) → reports/dl_daytrading/README_{h1,open}.md
  dl_exits.py         ML 상위 K 종목에 장중 손절/익절/트레일 적용 비교
  etf_strategies.py   ETF 장기(2005~) 추세·듀얼모멘텀·변동성타게팅 검증
  ml_rank_train.py    20일 보유용 ML 랭커(walk-forward) → data/cache/ml_rank_pred.parquet
  best_model_report.py / finalist_eval.py  수익 극대화 후보 리더보드·최종 비교·블렌드
  etf_signal.py       ETF 레그 오늘의 목표 비중
docs/surge_formulas.md  수집한 공식·팁 카탈로그(출처 포함)
docs/daytrading_dl_research.md  데이트레이딩·딥러닝 문헌 조사와 설계 근거
tests/                  백테스터 체결 규칙 단위 테스트
```

## 설치 / 실행

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 엔진 검증

# 1) 데이터
python scripts/fetch_data.py --universe all --start 2023-06-01      # 약 5,000 종목 일봉 (~5분)
python scripts/fetch_intraday.py --symbols symbols.txt --interval 1h # 필요 시
python scripts/fetch_earnings.py --symbols symbols.txt

# 2) 공식 검증
python scripts/event_study.py --train 2023-11-01:2025-09-17 --test 2025-09-18:2026-09-17 --out reports/surge_study
python scripts/intraday_study.py --interval 1h  --out reports/surge_study/intraday_1h
python scripts/intraday_study.py --interval 15m --split 2026-08-06 --out reports/surge_study/intraday_15m

# 3) 모델 토너먼트 + 강건성 + 보고서
python scripts/tournament.py --shard 0/3 --out reports/surge_study/tournament   # 0/3,1/3,2/3 병렬
python scripts/tournament.py --merge --out reports/surge_study/tournament
python scripts/robustness.py --top 6 --out reports/surge_study/tournament
python scripts/make_findings.py

# 4) 포트폴리오 백테스트 / 격자
python scripts/run_backtest.py --strategy all --universe all --start 2025-09-18 --end 2026-09-17
python scripts/run_backtest.py --strategy formula --param screen=ross_5pillars --param stop_pct=0.05 --param max_hold_days=3
python scripts/sensitivity.py --strategy formula --base '{"screen":"pullback_lowvol"}' \
   --grid '{"stop_pct":[0.05,0.1],"max_hold_days":[3,5]}' --window train=2023-11-01:2025-09-17 --window test=2025-09-18:2026-09-17 \
   --out reports/surge_study/grid_pullback_lowvol

# 5) 딥러닝/ML 데이트레이딩 (대형주 포함 전 종목, 1시간봉)
python scripts/fetch_intraday.py --symbols symbols.txt --interval 1h --merge
python scripts/dl_train.py --decision h1 --out reports/dl_daytrading      # 10:30 결정 → 종가 청산
python scripts/dl_train.py --decision open --out reports/dl_daytrading    # 시가 결정 → 종가 청산
python scripts/dl_exits.py --decision h1 --out reports/dl_daytrading

# 6) 자동매매 (기본은 계획만 출력; KIS 모의투자로 먼저)
python scripts/run_daily.py --strategy breakout --universe sp1500 --broker dryrun
KIS_APP_KEY=... KIS_APP_SECRET=... KIS_ACCOUNT=... KIS_PAPER=1 python scripts/run_daily.py --strategy breakout --broker kis --live
```

비용 기본값: 수수료 편도 0.25% (`--commission-pct`), 슬리피지 편도 10bp(주가 ≥ $5) / 50bp(동전주), 포지션 ≤ 20일 평균 거래대금의 1%.

## 한계 (반드시 읽을 것)

- **생존 편향**: 유니버스가 '현재 상장' 종목이라 지난 1~3년 사이 상장폐지된 동전주(대부분 급락)가 빠져 있다. 롱 결과가 실제보다 좋게 나온다.
- 일봉/1시간봉으로는 장중 고가·저가 순서를 알 수 없어 같은 봉의 손절·익절은 손절 우선으로 처리했다(보수적). 15분봉은 Yahoo 한계로 최근 60일만.
- 프리마켓·뉴스·공시(희석) 데이터가 없어 '재료' 는 실적 발표일로만 근사했다. float 는 발행주식수로 대체.
- 근사 시총 = 현재 발행주식수 × 당시 종가(증자/자사주 무시).
- KIS 어댑터의 tr_id/필드는 문서 기준 초안이며 모의투자로 검증 후 사용할 것. 미국 주식은 지정가만 지원.
