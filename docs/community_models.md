# 공개·배포된 자동매매 모델 목록 (재현 대상과 출처)

## A. 레버리지 ETF 계열 (Reddit r/LETFs · Bogleheads · Composer · 국내 유튜브/블로그)

| 모델 | 출처 | 공개된 규칙 | 공개된 성과 주장 | 우리 재현 |
|---|---|---|---|---|
| HFEA (Hedgefundie's Excellent Adventure) | [Bogleheads 스레드](https://www.bogleheads.org/forum/viewtopic.php?t=326588), [요약](https://www.optimizedportfolio.com/hedgefundie-adventure/) | UPRO 55% / TMF 45%, 분기 리밸런스 | 1987~2019 시뮬레이션 CAGR 20%대, 2022년 −60%대 | `community_models.py: hfea` |
| 9Sig (Kelly Letter) | [Jason Kelly](https://jasonkelly.com/resources/strategies/), [BestFolio 재현](https://bestfolio.app/blog/kelly-signal-danger-v2) | TQQQ/AGG 60/40, 분기마다 주식 목표치 9% 성장, 잉여 매도·부족 매수, 매수 상한 채권의 90%, 채권 하한 10%, 채권>30%면 60/40 리셋, 30다운 룰 | 2010~2026 CAGR 39%, MDD −72% (BestFolio), 1999년 포함 시 −99.7% | `nine_sig` |
| TQQQ For The Long Term (FTLT) | [Composer 원본](https://www.composer.trade/trading-strategies/tqqq-for-the-long-term-original-WDQzBV4Mse7Zypxk4LtC), Reddit r/Composer | SPY>200일선이면 TQQQ(RSI10>79면 UVXY), 아니면 TQQQ RSI10<31→TECL, SPY RSI10<30→UPRO, TQQQ>20일선→TQQQ, 아니면 채권 | 2021-09~2026-09 연 160%, 샤프 2.09, MDD −51% (Composer 표시, 인샘플 포함) | `tqqq_ftlt` (공유본 트리 근사) |
| SPY 200일선 → 레버리지 스위칭 | Gayed, *Leverage for the Long Run* (2016); 국내 유튜브 다수 | SPY>200일선이면 3배 ETF, 아니면 현금 | 1928~2020 시뮬레이션 CAGR 20%대 | `sma200_switch` (일간) |
| 듀얼 모멘텀 GEM | Antonacci; [강환국 번역·소개](https://www.yes24.com/Product/goods/59395635), [게으른 퀀트](https://lazyquant.xyz/docs/detail/%EC%9E%90%EC%82%B0%EB%B0%B0%EB%B6%84/15) | 월간, SPY vs EFA 12개월 수익률 상위, 둘 다 T-bill 이하면 채권 | 1974~2013 CAGR 17%, MDD −18% (책) | `gem` |
| 라오어 무한매수법 v2.2 | [나무위키](https://namu.wiki/w/%EB%9D%BC%EC%98%A4%EC%96%B4%EC%9D%98%20%EB%AF%B8%EA%B5%AD%EC%A3%BC%EC%8B%9D%20%EB%AC%B4%ED%95%9C%EB%A7%A4%EC%88%98%EB%B2%95), [v2.2/v3.0 정리](https://www.pbdfinance.com/2024/09/v22-v30.html), [백테스트](https://penglab.net/strategies/infinite-buying) | 40분할, T=누적매수/1회분, 전반(T<20) 절반 평단 LOC + 절반 평단×(1+(10−T/2)%) LOC, 후반 (10−T/2)% LOC, 매도 1/4 (10−T/2)% LOC + 3/4 +10% 지정가 | 저자: 2010년대 TQQQ 연 30~40%대 | `infinite_buying` (40회 소진 시 전량 매도 후 재시작) |
| 라오어 밸류 리밸런싱(VR) | [교보문고](https://product.kyobobook.co.kr/detail/S000061695672), [리뷰](https://theorydb.github.io/review/2022/10/15/review-book-value-rebalancing/) | 목표가치 V 를 일정 비율로 키우고 평가금이 V×(1±15%) 밴드 밖이면 매수/매도 | 1971~2021 나스닥 12,846회 백테스트 (책) | `value_rebalancing` (V 연 15%, ±15%, 2주 — 근사) |
| 변동성 돌파 (Larry Williams) | [조코딩/스팀잇 봇](https://steemit.com/sctcrypto/@anpigon/23vt7e), [GitHub hunmin815/autoTrade](https://github.com/hunmin815/autoTrade), [알고랩](https://algolab.co.kr/blog/algorithmic-trading-strategies-30-complete-guide) | 목표가 = 시가 + K×(전일 고가−저가), K=0.5, 돌파 시 매수·다음날 시가 매도 | 코인에서 유명, 주식 성과 공개 드묾 | `vol_breakout` (TQQQ/QQQ/SPY) |

## B. 기술적 지표 봇 (GitHub 인기 저장소·freqtrade·국내 봇 표준)

| 모델 | 출처 | 규칙 | 우리 재현 (`quant/strategies/technical.py`, S&P 1500) |
|---|---|---|---|
| 이동평균 골든크로스 | [je-suis-tm/quant-trading](https://github.com/je-suis-tm/quant-trading), 알고랩 1번 | MA(20)>MA(60) 매수, 데드크로스 매도 | `sma_cross` (20/60, 50/200) |
| MACD 신호선 크로스 | je-suis-tm MACD, 알고랩 3번 | MACD(12,26)>신호선(9) 매수 | `macd` |
| 볼린저 평균회귀 | je-suis-tm Bollinger, 알고랩 5번 | 하단(20, 2σ) 이탈 + RSI<30 매수, 중심선 매도 | `bollinger_rsi` |
| RSI 과매도 | 알고랩 6·9번 | RSI(14)<30 매수, >70 매도 | `rsi_1430` |
| 슈퍼트렌드 | [TradingView](https://www.tradingview.com/script/341KOBXa-Supertrend-EMA-Strategy-TheHeroBoy/), [backtrader 예제](https://github.com/akshaymehara/Backtrader-backtesting) | Supertrend(10,3) 상향 전환 매수 | `supertrend` |
| EMA 9/21 + RSI 필터 | 알고랩 2번, freqtrade 샘플 | EMA9>EMA21 이고 RSI<70 매수, 데드크로스 매도 | `ema_rsi` |
| Donchian/52주 신고가/거래량 돌파 | 알고랩 6·14·15·16번 | 20일/252일 고가 돌파 | 앞 절의 `breakout`, `high_breakout` |

## C. ML/DRL 프레임워크 (공개 성적만 인용, 우리 ML 랭커와 비교)

| 프레임워크 | 공개 성적 | 비고 |
|---|---|---|
| Microsoft Qlib, LightGBM + Alpha158 (CSI300) | IC 0.040~0.045, 연 초과수익 9~13%, IR 1.0~1.6, MDD −10% ([벤치마크](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)) | 우리 20일 ML 랭커: walk-forward IC 0.017~0.062 (평균 ~0.05), 같은 급 |
| FinRL DRL 앙상블 (DJIA 30) | 2016~2020 연 52.6%·샤프 2.81 ([논문](https://arxiv.org/pdf/2111.09395)); FinRL-Meta 재현 연 25.9%·샤프 1.53 vs DJIA 19.7% ([FinRL-Meta](https://arxiv.org/pdf/2211.03107)) | 비용 전, 재현마다 크게 다름 |
| freqtrade 공개 전략 (53k★) | 문서 스스로 "공개 전략 대부분은 여러 국면에서 수익이 안 난다, 백테스트로 성과를 판단하지 말라" 경고 ([문서](https://www.freqtrade.io/en/stable/strategy-101/)) | 암호화폐 전용 |

## D. 비교 원칙
모든 재현은 같은 비용(편도 0.25% 수수료 + 슬리피지)으로 2024-01~2026-09 와 가능한 최장 기간(2010~) 두 구간에서 우리 최종 모델과 나란히 놓는다. 공개된 '주장 수치' 는 인샘플·비용 전·기간 선택 편향이 흔하므로 별도 열로 표기한다.
