# 데이트레이딩 전략·딥러닝 문헌 조사 (모델 설계의 근거)

목표: 대형주를 포함한 미국 전 종목을 대상으로, 공개된 데이트레이딩 전략과 딥러닝/ML 연구에서 **이미 검증된 설계·파라미터**를 차용해 모델을 만든다.
구현: `quant/dl/`, `scripts/dl_train.py`, 결과: `reports/dl_daytrading/`.

## 1. 학술 문헌 — 무엇이 실제로 재현되는가

| 출처 | 핵심 결과 | 우리가 차용한 것 |
|---|---|---|
| Fischer & Krauss (2018) *Deep learning with LSTM networks for financial market predictions*, EJOR ([논문](https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652), [요약](https://www.semanticscholar.org/paper/701e59358fda0d865d7b26cd954a93a5ad20fd13)) | S&P 500 구성종목 1992~2015, 240일 수익률 시퀀스 → LSTM(25 units) 으로 다음날 횡단면 중앙값 상회 확률 예측, 상위 10/하위 10 롱숏. **비용 전** 일 0.46%, 샤프 5.8. **2010년 이후 초과수익 소멸(비용 후 0 근처)** | 횡단면 순위 목표값, 상위 K 동일비중, 시퀀스 모델(GRU), 시간 기반 분할 |
| Ghosh, Neufeld & Sahoo (2021) *Forecasting directional movements of stock prices for intraday trading using LSTM and random forests* ([arXiv](https://arxiv.org/abs/2004.10178)) | S&P 500 1993~2018, 시가→종가·종가→시가 수익률 다중 특징, 매일 상위 10 매수·하위 10 공매도(시가 진입·종가 청산). 비용 전 일 0.64%(LSTM), 0.54%(RF) | '시가 매수·종가 매도' 결정 시점(`--decision open`), 오버나잇/장중 수익률을 특징으로 분리 |
| Gao, Han, Li & Zhou (2018) *Market intraday momentum*, JFE ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866)) | SPY 1993~2013: 전일 종가→10:00 첫 30분 수익률이 15:30→16:00 마지막 30분 수익률을 예측, R² 1.6%(12번째 30분 추가 시 2.6%). 변동성·거래량 큰 날, 거시 뉴스일에 강함. 16개국 확장(Li et al. 2022) | 결정 시점을 첫 1시간봉 마감(10:30)으로, 첫 1시간 수익률·SPY 첫 1시간 수익률을 특징으로. 규칙 베이스라인 '장중 모멘텀' |
| Zarattini, Barbon & Aziz (2024) *A Profitable Day Trading Strategy for the U.S. Equity Market* ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284), [QuantConnect 재현](https://www.quantconnect.com/research/18444/opening-range-breakout-for-stocks-in-play/), [Concretum](https://concretumgroup.com/a-profitable-day-trading-strategy-for-the-u-s-equity-market/)) | 7,000 종목 2016~2023, **첫 5분 상대거래량(직전 14일 대비) 상위 20 = stocks in play** 에 5분 ORB, 손절 ATR 기반, 종가 청산, 거래당 1% 리스크. 총 1,600%+, 샤프 2.81, 승률 약 17%. 플레인 ORB 는 약하고 **stocks-in-play 선별이 성과의 대부분** | 첫 1시간 상대거래량(`h1_rvol`) 특징, 규칙 베이스라인 'stocks-in-play' |
| Zarattini & Aziz (2024) *Beat the Market: An Effective Intraday Momentum Strategy for the S&P500 ETF* ([리뷰](https://quantmacro.substack.com/p/paper-review-an-effective-intraday)) | SPY 노이즈 밴드(변동성 조정) 돌파 진입, VWAP 트레일링, 종가 청산, 변동성 타게팅. 승률 약 43%, VIX>20 에서 강함 | 지수 ETF 전용이라 직접 차용하지 않음(개별주 모델과 비교용 참고) |
| Lou, Polk & Skouras (2019) *A tug of war: Overnight versus intraday expected returns*, JFE ([논문](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650)) | 14개 전략의 수익이 오버나잇 또는 장중 한쪽에서만 나고 부호가 반대. 소형주 모멘텀은 장중, 대형주 모멘텀은 오버나잇. 기관은 장중에 모멘텀 반대로 거래 | 장중(데이트레이딩) 롱온리는 구조적 역풍 가능 → 같은 예측을 오버나잇/1일 보유에도 적용해 비교 |
| López de Prado (2018) *Advances in Financial Machine Learning* ([노트](https://reasonabledeviations.com/notes/adv_fin_ml/)) | 트리플 배리어 라벨, 메타라벨링, 퍼지드 K-폴드 + 엠바고, 특징 중요도(MDI/MDA) | 시간 기반 분할 + 5거래일 엠바고, 라벨 클리핑, 특징 중요도 보고 |
| Guijarro-Ordonez, Pelger & Zanotti (2021) *Deep Learning Statistical Arbitrage* ([arXiv](https://arxiv.org/pdf/2106.04028), [2016~2024 재현](https://arxiv.org/html/2412.11432v1)) | 요인 잔차에 CNN+Transformer, 일별 리밸런싱, 비용 전 샤프 4+ (재현은 10+ 지만 과최적화·비용 의심) | 보유 기간이 주 단위라 데이트레이딩엔 직접 차용 안 함 |
| Kaggle Jane Street / Optiver 상위 솔루션 ([정리](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-01-30-Kaggle-exp-s2/), [Optiver 1위](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)) | LightGBM + 롤링 통계 특징, NN 과 앙상블, 시간 기반 검증이 핵심 | LightGBM(맥락+시퀀스 요약) + GRU 앙상블(순위 평균) |

## 2. 실무 커뮤니티 규칙 (앞선 급등주 카탈로그와 공유)

- SmallCapLab: 갭업의 66% 는 시가 아래 마감, 고가의 47% 는 개장 15분 안. Warrior: RVOL ≥ 5, $2~$20, 9:30~11:30 집중. 장중 U자형 변동성(개장·마감에 집중).
- VWAP 회귀(QuantConnect 2022, 100 NASDAQ 유동주): 2σ 밴드 역추세 승률 61~63%, 손익비 1.4~1.5.
- 갭 페이드(SMB/QuantifiedStrategies): 지수 갭의 70~76% 메워짐, 5%↑ 큰 갭은 10~30% 만.

## 3. 설계 결정 (문헌 → 구현)

| 항목 | 결정 | 근거 |
|---|---|---|
| 유니버스 | 미국 전 종목 중 20일 평균 거래대금 ≥ $2M, 주가 ≥ $2 (대형주 포함 약 3,000~4,000 종목) | 대형주 포함 요구, 체결 현실성 |
| 봉 | 1시간봉(최근 2년, Yahoo 한계) + 일봉 맥락 | 분봉 1년치는 무료로 없음 |
| 결정 시점 | (a) 10:30 첫 1시간봉 마감 후 → 종가 청산, (b) 시가 → 종가 청산 | Gao et al., Zarattini(첫 구간 정보), Krauss/Ghosh(시가→종가) |
| 목표값 | 일자별 횡단면 순위(결정→종가 수익률, ±50% 클리핑) | Fischer & Krauss 횡단면 분류, 시장 방향 제거 |
| 입력 | 최근 35개 1시간봉(로그수익률·범위·거래량 z·시간대) + 17개 일봉/시장 맥락(갭, 첫1h 수익률·상대거래량, 5/20/60일 수익률, ATR%, 52주고가, 거래대금, 가격, 시총, SPY) | 문헌 특징 합집합 |
| 모델 | LightGBM(early stopping) / GRU 64 + MLP / 순위 평균 앙상블 / 규칙 5종 | Kaggle 상위권 관행, Krauss 계열 |
| 검증 | 시간 분할(학습 ~2025-06, 검증 2025-07~09, 테스트 2025-10~2026-09), 엠바고 5일 | AFML |
| 비용 | 수수료 편도 0.25% + 슬리피지 편도 0.1%(동전주 0.5%) | 국내 증권사 해외주식 기준 |
| 지표 | 일별 IC·t, 십분위 스프레드, 상위 K 포트폴리오(총/비용후), 손익분기 비용, 시총 구간별, 다른 청산 비교 | 문헌 관행 |

## 4. 미리 알아야 할 한계

- 문헌의 데이트레이딩 초과수익은 대부분 **비용 전** 이고 2010년 이후 급격히 줄었다(Fischer & Krauss). 국내 증권사 왕복 0.5% 수수료는 미국 기관 비용의 10배 이상이다.
- Zarattini 계열 성과는 5분봉·ATR 손절·레버리지 등 실행 세부에 민감하고, 재현자들도 "체결 가정 단순화" 를 경고한다.
- 1시간봉으로는 5분 ORB 나 마지막 30분 진입을 흉내낼 수 없다(분봉이 필요).
