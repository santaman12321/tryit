# tryit — 미국 급등주(동전주·소형주) 자동매매 리서치 & 실행 프레임워크

한국 투자자 기준(국내 증권사 해외주식 API: **공매도 불가**, 매매수수료 **편도 0.25%**)으로
블로그·유튜브·해외 커뮤니티에서 공유되는 '급등주 찾기 공식' 을 모아 정량 규칙으로 바꾸고,
미국 전체 상장 보통주(약 5,000종목) 일봉 3년 + 1시간봉 2년 + 15분봉 60일로 검증한 뒤,
살아남는 규칙만 자동매매(한국투자증권 Open API 어댑터, 기본 dry-run)로 넘기는 구조다.

**결론 요약(2026-09-18 기준)** → [`reports/surge_study/FINDINGS.md`](reports/surge_study/FINDINGS.md)
- 15개 공식 × 진입/청산/손절 변형 수백 개를 학습(2023-11~2025-09)·검증(2025-09~2026-09) 두 구간에 돌렸을 때,
  **비용(수수료 0.25%×2 + 슬리피지) 차감 후 두 구간 모두 기대값이 양(+)인 롱 규칙은 없었다.** 최고 승률도 45% 안팎.
- 급등 다음날 고가의 58%(1시간봉)/35%(15분봉 첫 봉)가 개장 직후에 찍히고 66%는 시가보다 낮게 마감 → '추격 매수' 구조 자체가 불리하다.
- 1달러 미만·초소형주 구간의 높은 '평균'은 상위 2% 로또(+2,000%대) 종목이 만든 착시(중앙값·절사평균은 음수, 생존 편향 포함).

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
  run_daily.py        매일 장 마감 후 신호→주문 계획/제출
docs/surge_formulas.md  수집한 공식·팁 카탈로그(출처 포함)
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

# 3) 포트폴리오 백테스트 / 격자
python scripts/run_backtest.py --strategy all --universe all --start 2025-09-18 --end 2026-09-17
python scripts/run_backtest.py --strategy formula --param screen=ross_5pillars --param stop_pct=0.05 --param max_hold_days=3
python scripts/sensitivity.py --strategy formula --base '{"screen":"pullback_lowvol"}' \
   --grid '{"stop_pct":[0.05,0.1],"max_hold_days":[3,5]}' --window train=2023-11-01:2025-09-17 --window test=2025-09-18:2026-09-17 \
   --out reports/surge_study/grid_pullback_lowvol

# 4) 자동매매 (기본은 계획만 출력; KIS 모의투자로 먼저)
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
