# 급등주 찾기 공식 카탈로그 (블로그·유튜브·해외 커뮤니티 수집본)

목적: 시총 작은 동전주·소형주가 "이유 없이" 급등락하는 패턴을 노리는 공식들을 모아, 일봉 데이터로 검증 가능한 **정량 규칙** 으로 바꾼다.
구현은 `quant/screens.py`, 검증은 `scripts/event_study.py` → `reports/surge_study/README.md`.

> 공통 한계: 대부분의 공식은 '뉴스/재료', '프리마켓 고가 돌파', '1분봉 패턴' 처럼 일봉으로 검증할 수 없는 요소를 포함한다.
> 여기서는 **일봉으로 재현 가능한 부분만** 규칙화했다. float(유통주식수)는 데이터가 없어 발행주식수(= 현재 시총/현재가)로 대신했다(float ≤ 발행주식수이므로 보수적).

## 1. 급등일(당일 급등 + 거래량 폭발) 정의 계열 → 다음날 추격 매수 검증

| # | 이름 / 출처 | 원문 조건 | 일봉 규칙 (`screens.py`) |
|---|---|---|---|
| 1 | **Ross Cameron 5 Pillars** (Warrior Trading; TradingView 지표 [Ross Cameron 5 Pillars Filter](https://www.tradingview.com/script/mbqMf3pF-Ross-Cameron-5-Pillars-Filter/), [Ross-Style Momentum Study](https://www.tradingview.com/script/aU6C5iwn-Ross-Style-Momentum-Study/)) | ① RVOL ≥ 5배(30일 평균) ② 당일 +10% 이상(15%↑면 뉴스 가능성) ③ 뉴스 재료 ④ 주가 $1~$20 ⑤ float < 10M(~20M) | `ross_5pillars`: ret1≥10%, vol/30일평균≥5, $1≤종가≤$20, 발행주식수≤20M, 거래대금≥$1M |
| 2 | **Warrior Trading 모멘텀 스캐너(ToS)** ([useThinkScript](https://usethinkscript.com/threads/free-warrior-trading-momentum-scanner-for-thinkorswim.5733/)) | +5%↑, RVOL 1.5, $1~$30, 20만주↑, 양봉·종가가 봉 상단 30%, 9EMA>20EMA, 7~12시 | `warrior_scanner`: ret1≥5%, vol/20일평균≥1.5, $1~$30, 거래량≥20만, close_pos≥0.7, 양봉, EMA9>EMA20 |
| 3 | **Finviz 급등주 스크리너** (국내 블로그 [미국 급등주 빠르게 찾기](https://digitalnomad.hoyabusiness.com/entry/%EB%AF%B8%EA%B5%AD-%EA%B8%89%EB%93%B1%EC%A3%BC-%EB%88%84%EA%B5%AC%EB%B3%B4%EB%8B%A4-%EB%B9%A0%EB%A5%B4%EA%B2%8C-%EC%B0%BE%EA%B8%B0-%EA%B2%80%EC%83%89-%EB%85%B8%ED%95%98%EC%9A%B0%EC%99%80-%ED%88%AC%EC%9E%90%EC%A0%84%EB%9E%B5)) | +20%↑(모멘텀판 +15%), 평균거래량 10만↑, $5↑, 거래량 200만↑, >SMA20/50, RSI>60; 손절 5~7% | `finviz_top_gainer`: ret1≥20%, 20일평균거래량≥10만, $5↑, 거래량≥200만, 종가>SMA20&SMA50, RSI14≥60 |
| 4 | **국내 블로그 자동 스크리너 7가지** ([급등주 초기에 찾는 7가지 방법](https://digitalnomad.hoyabusiness.com/entry/%EB%AF%B8%EA%B5%AD-%EC%A3%BC%EC%8B%9D-%EA%B8%89%EB%93%B1%EC%A3%BC-%EC%B4%88%EA%B8%B0%EC%97%90-%EC%B0%BE%EB%8A%94-7%EA%B0%80%EC%A7%80-%EB%B0%A9%EB%B2%95)) | 프리마켓 +5%·거래량 동반 / 거래량 3~5배 / $1↑, +5%↑, 50만주↑, RVOL≥2, RSI≤70 / 숏플로트 20%↑ | `kr_screener_basic`: ret1≥5%, 거래량≥50만, vol/20일평균≥2, RSI14≤70, $1↑ |
| 5 | **국내 HTS 조건검색식 예시** (증권사 도움말 [급등주탐색](http://www.myasset.com/mynetwhelp/2741.html), [종합종목검색](https://securities.miraeasset.com/kairos/0544.htm)) | 시총 1.5조 이하, 거래량 100만↑, 거래대금 30억↑, 25봉 최고종가 -3% 이내, 연중 신고가, 정배열, 전일 동시간 거래량 200%↑ | `kr_newhigh_alignment`: 시총≤$1B, 거래량≥100만, 거래대금≥$2M, 종가≥25일 최고종가×0.97, 52주 신고가, SMA5>SMA20>SMA60, 전일比 거래량≥2배 |
| 6 | **장대양봉 + 최근 최대 거래량** (다음카페 [급등주 발굴법](https://m.cafe.daum.net/stockpapa/1wTi/7426), [기호일보 거래량 패턴](https://www.kihoilbo.co.kr/news/articleView.html?idxno=495201)) | "최근 최대 거래량으로 20%↑ 장대양봉 = 시세의 시작", 거래량 3배↑ 급증, 10배↑ 3일 연속은 실패 신호, 고점 -10% 매도 | `big_candle_maxvol`: ret1≥20%, 거래량≥직전 20일 최대, close_pos≥0.6, $1↑, 거래대금≥$2M, 시총≤$2B |
| 7 | **횡보 후 거래량 3배** (같은 카페) | 2개월↑ 횡보(박스) 후 거래량 3배 급증에 편승(3~8배) | `volume_3x_after_base`: 직전 40일 박스폭≤25%, vol/20일평균≥3, ret1≥5%, 시총≤$2B |

## 2. 눌림목 계열 → 급등 후 조정 시 매수

| # | 이름 / 출처 | 원문 조건 | 일봉 규칙 |
|---|---|---|---|
| 8 | **장대양봉 후 거래량 마름** (다음카페 [세력 매집패턴](https://m.cafe.daum.net/stockpapa/1wTi/775), 텐인텐 [매집 vs 물량 떠넘김](https://m.cafe.daum.net/10in10/FlzD/245135)) | 장대양봉 이후 3~5봉 안에 거래량이 장대양봉의 10% 이내로 줄고 주가가 안 빠지고 버티면 급등 임박 | `pullback_lowvol`: 5일 내 급등일(+15%, 5배) 존재, 오늘 거래량≤급등일의 20%, 종가≥급등종가×0.9, 급등 전 종가 위 |
| 9 | **5/10일선 눌림 + 거래량 감소** (브런치 [눌림목 매매 완전 해부](https://brunch.co.kr/@hasowon73/57), [f-daily 눌림목](https://f-daily.co.kr/%EC%A6%9D%EA%B6%8C/%EB%88%8C%EB%A6%BC%EB%AA%A9-%EB%A7%A4%EB%A7%A4-%EB%B0%A9%EB%B2%95/)) | 5·10일선 지지, 조정 중 거래량 감소, 손절 5일선 이탈, 1차 익절 15%·30% 분할 | `pullback_ma`: 10일 내 급등일 존재, 저가≤SMA10≤종가, 거래량≤급등일의 50%, 급등종가 아래·급등 전 종가 위 |
| 10 | **스레드 미국 급등주 눌림 검색식** ([@sec_stock](https://www.threads.com/@sec_stock/post/DHcW-EwpV7V/)) | 시총 ₩2.5조 이하, $0.001~$30, 30거래일 내 90%↑ 상승, 30일 내 일 거래대금 ₩900억↑, 종가 5일선 돌파 | `threads_us_pullback`: 시총≤$1.8B, ≤$30, 30일 저점比 +90%, 30일 내 최대 거래대금≥$65M, 종가 SMA5 상향 돌파 |

## 3. 매집·거래량 바닥 계열 → 급등 '전' 포착 시도

| # | 이름 / 출처 | 원문 조건 | 일봉 규칙 |
|---|---|---|---|
| 11 | **변동폭 없는 거래량 폭발(손바뀜)** ([@jtdj_official2](https://www.threads.com/@jtdj_official2/post/DNuoplz5JV5/)) | "주가 변동폭 거의 없는데 말도 안 되는 거래량 동반 = 세력 손바뀜 흔적" (수치 없음) | `quiet_volume_spike`: |ret1|≤3%, vol/20일평균≥3, $1↑, 시총≤$2B |
| 12 | **거래량 바닥 후 첫 급증** (부자아빠 [급등주 발굴법2](https://m.cafe.daum.net/stockpapa/1wTi/7427), [i-whale 기준봉 거래량비율](https://i-whale.com/entry/%EA%B8%B0%EA%B0%84%EB%82%B4-%EA%B8%B0%EC%A4%80%EB%B4%89-%EA%B1%B0%EB%9E%98%EB%9F%89%EB%B9%84%EC%9C%A8%EB%A1%9C-%EA%B2%80%EC%83%89%EC%8B%9D-%EB%A7%8C%EB%93%A4%EA%B8%B0-%EC%A3%BC%EA%B0%80%EB%B9%84%EA%B5%90)) | 거래량이 바닥(조용)일 때 1차 매수, 거래 터지면 추격 | `dryup_first_spike`: 직전 20일 최저 거래량≤60일 평균의 50%, 오늘 vol/20일평균≥3, ret1≥3%, close_pos≥0.6 |

## 4. 오버나잇·멀티데이 계열

| # | 이름 / 출처 | 원문 조건 | 일봉 규칙 |
|---|---|---|---|
| 13 | **First Green Day** (Tim Sykes [FGD setup](https://www.timothysykes.com/blog/first-green-day-setup/), [FGD OTC](https://www.timothysykes.com/blog/first-green-day-otc/)) | 하락해 있던 종목이 대량 거래·큰 % 상승·고가 근처 마감 → 마감 직전 매수, 다음날 아침 갭업에 매도 | `first_green_day` (종가 진입): 전일 종가≤60일 최고종가×0.7, ret1≥15%, vol≥5배, close_pos≥0.7, $0.5~$20 → 다음날 시가 매도 |
| 14 | **멀티데이 러너 day-2 효과** ([tradethematrix](https://www.tradethematrix.net/post/small-cap-stocks-multi-day-runners), StocksToTrade [Multi-Day Runs](https://stockstotrade.com/multi-day-runs/)) | 1일차 대급등 후 2~3일차에 안 빠지고 버티면 진짜 러너(평균 2주, 50~500%), 700%↑·10일↑면 고점 임박 | `multiday_runner_day2`: 전일 +30%·10배 급등, 오늘 종가≥전일 종가 → 다음날 시가 진입 |

## 5. 갭 계열 (단타)

| # | 이름 / 출처 | 원문 조건 | 일봉 규칙 |
|---|---|---|---|
| 15 | **Gap & Go** ([Scanz](https://scanz.com/gap-and-go-strategy/), [TradeZella](https://www.tradezella.com/blog/gap-and-go-strategy), [Bullish Bears](https://bullishbears.com/gap-and-go-strategy/)) | 프리마켓 갭 ≥10%(완화판 2~5%), PM 거래량 10만↑, $2~$20, 저유통 선호, 시가에 프리마켓 고가 돌파 시 진입 | `gap_and_go` (시가 진입): 갭≥10%, 시가 $2~$20, 직전 20일 평균 거래대금≥$1M → 당일 종가 매도 |

## 5-2. 장중(분봉) 진입·청산 규칙 — 1시간봉으로 판단, 15분봉으로 청산

| # | 이름 / 출처 | 원문 조건 | 검증 규칙 (`scripts/intraday_study.py`) |
|---|---|---|---|
| 16 | **시초가 매수 금지 → 첫 봉 확인 후 진입** (국내 단타 커뮤니티 통설, [단타 나무위키](https://namu.wiki/w/%EB%8B%A8%ED%83%80), [스레드 9시~9시20분 매수·10시~10시30분 매도](https://www.threads.com/@letmetakeyououtto/post/DRbYEcTgcf5/)) | 개장 직후 변동성 구간을 피해 첫 봉이 양봉으로 확정된 뒤 진입, 오전 중 청산 | `h1_green`: 첫 1시간봉 양봉 + 봉 상단 60% 마감 → 10:30 진입; `h1_green_strong`: + 첫 1시간 +3%↑ |
| 17 | **Opening Range Breakout(ORB)** ([Zarattini·Barbon·Aziz 2024 SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284), [QuantConnect 재현](https://www.quantconnect.com/research/18444/opening-range-breakout-for-stocks-in-play/), [danfin 요약](https://danfin.net/opening-range-breakout-research)) | 개장 후 n분 범위(5~60분) 고가 돌파 시 진입, 범위 저가 손절, 종가 청산. '거래량이 평소보다 많은 종목(stocks in play)' 에 한정할 때만 유의미. 승률은 낮고 소수 대박에 의존 | `orb60`: 2시간째에 첫 1시간 고가 돌파 시 그 가격에 진입 |
| 18 | **첫 눌림 매수** (Warrior Trading 'first pullback', 국내 '눌림목 단타') | 첫 상승 후 첫 되돌림이 시가/저점을 지키면 진입 | `h1_pull`: 첫 봉 양봉, 둘째 봉 음봉이지만 첫 봉 시가 위 마감 → 11:30 진입 |
| — | **청산 규칙** (공통) | 시간 청산(12:30 / 14:30 / 종가), 손절 2~5%, 익절 3~10%, 트레일 3~5% | 1시간봉·15분봉의 저가/고가로 판정, 같은 봉에 손절·익절 동시 발생 시 손절 우선 |

## 5-3. '조합' 밖의 요소 — 커뮤니티가 말하는 매매 타이밍·재료·환경 팁

> 토스증권 커뮤니티(미국주식이야기)의 "제 매매타이밍(방식) 공유드립니다" 글은 앱 전용이라 웹에서 읽을 수 없었다(검색·직접 접속 모두 실패).
> 같은 성격의 '개인 방식 공유' 글과 해외 커뮤니티 가이드에서 수치화 가능한 팁을 모았다. 검증 결과는 `reports/surge_study/README.md` 의 '적절값 탐색' 표(회전율·레짐·실적·중국·요일)와 장중 스터디(VWAP) 참고.

| # | 팁 / 출처 | 내용 | 검증 방법 |
|---|---|---|---|
| 19 | **프리마켓 급등주의 70%는 개미 진입가 아래로** ([newthinb 개인 방법 공유](https://newthinb.com/%EB%AF%B8%EA%B5%AD-%EA%B8%89%EB%93%B1%EC%A3%BC-%EB%AF%B8%EB%A6%AC-%EC%95%84%EB%8A%94-%EB%B2%95-%EA%B0%9C%EC%9D%B8-%EB%B0%A9%EB%B2%95-%EA%B3%B5%EC%9C%A0-%ED%88%AC%EC%9E%90%EB%8A%94-%EA%B2%B0%EA%B5%AD/)) | 프리장 초반 급등의 70%는 되돌림. 집중 시간대는 한국시간 17~19시(프리장 초반)와 21:40~본장 초반. **종가배팅(종배)**: 윗꼬리 짧은 급등주를 마감 직전 매수해 다음날 시가 매도. 적자기업·바이오 잡주·**중국 주식 절대 배제**, 거래량 없는 상승률은 패스, 장대음봉이면 즉시 매도 | 종가배팅 = `first_green_day`/오버나잇 표(ret_gap), 중국 기업 = `china` 구간, 거래량 = `vol_ratio`·`turnover` 구간 |
| 20 | **재료(뉴스) 있는 급등만** (Ross Cameron 3번째 기둥, trademomentum, [PEAD 문헌](https://quantpedia.com/strategies/post-earnings-announcement-effect), [실적 갭 지속 통계](https://papertradingjournal.com/2026/05/08/post-earnings-momentum-statistics/)) | 실적 서프라이즈·FDA·계약 같은 재료가 있는 급등은 이어지고(PEAD: 60거래일까지 드리프트, 첫 1~10일이 가장 강함, 연 15% 롱숏), 이유 없는 급등은 되돌린다. 8~15% 실적 갭이 20%+ 갭보다 지속성 좋음 | 실적발표일(yfinance) ±1거래일 안의 급등 = `earnings=1` 구간 vs `earnings=0`(이유 없는 급등) |
| 21 | **Float rotation** ([Scanz](https://scanz.com/how-to-find-low-float-stocks/), [TradingView Float Rotation Tracker](https://www.tradingview.com/script/I30clPFx-Float-Rotation-Tracker)) | 당일 거래량 ÷ 유통주식수 ≥ 1 이면 '전량 손바뀜' = 극단적 모멘텀 | `turnover` = 거래량/발행주식수 구간 (float 데이터가 없어 발행주식수로 대체, 보수적) |
| 22 | **시장 환경** (Warrior 'strong market days', 국내 '지수 좋을 때만 단타') | 지수(특히 소형주 지수)가 추세 위일 때만 급등주가 이어진다 | `spy_above200`, `iwm_above50` 구간 |
| 23 | **VWAP 규칙** ([snappchart VWAP 4 setups](https://www.snappchart.app/blog/strategy-playbooks/vwap-momentum-trading-strategy), [humbledtrader](https://www.humbledtrader.com/blog/how-to-find-stocks-to-trade/)) | VWAP 위에서만 롱, VWAP 기울기 상승일 때만. 'VWAP 회복(reclaim)' 봉 종가 매수, 손절은 회복봉 저가 | 장중 스터디 진입 `h1_vwap`, `vwap_reclaim` |
| 24 | **시간대** ([trademomentum 초보 가이드](https://www.trademomentum.org/blog/small-cap-trading-guide-for-beginners), [SmallCapLab](https://www.smallcaplab.com/research)) | 9:30~11:30 에 거래량과 기회가 몰림, 11:30 이후 축소. 고가의 절반은 개장 15~30분 안 | 장중 스터디 '고가/저가 시간대' 표, 청산 시각(12:30/14:30/종가) 비교 |
| 25 | **리스크 규칙** (trademomentum, Warrior) | 거래당 계좌의 1~2% 손실 한도, 일 손실 2~3%면 중단, 손익비 2:1 이상, 지정가만 사용(스프레드), 수수료 무시 금지, 40%+ 오른 뒤 추격 금지 | 백테스터 `sizing="risk"`, 손절/익절 격자, 비용 모델(수수료 0.25% 편도) |
| 26 | **요일/계절성** (커뮤니티 통설: 월요일 갭, 금요일 차익실현) | 요일별 급등 지속성이 다르다는 주장 | `weekday` 구간 |
| 27 | **희석(오퍼링) 리스크** (Warrior, Sykes) | 급등 직후 유상증자/ATM 발표가 잦은 종목(특히 바이오·중국 소형주)은 다음날 급락. S-1/424B 공시 확인 | 공시 데이터가 없어 정량 검증 불가 → 실전 체크리스트로만 유지 |

## 5-4. 사용자가 공유한 커뮤니티 글 2건 (MHTML 저장본에서 추출)

### A. 토스증권 미국주식이야기 — "제 매매타이밍(방식) 공유드립니다" (닉네임 '못버틸거면급등주때려쳐라', 2025-12-08, 승률 약 60% 주장)

거래량 기반 단기 패턴 매매. **첫 거래량 급등 당일에는 절대 진입하지 않는다**가 핵심.

| 원문 규칙 | 정량화 (`screens.py`) |
|---|---|
| 관심종목: 최근 3개월 이상 작은 양봉/음봉으로 조용히 횡보 + 일 거래대금 10억(≈$0.7M) 이하 유지(저유동성) | `_first_spike`: 직전 60거래일 박스폭 ≤ 50%, 60일 평균 거래대금 ≤ $2M(미국 스케일로 완화), 그 상태에서 60일 내 최대 거래량이면서 20일 평균의 5배↑ + \|당일 수익률\| ≥ 5% 인 날 = '첫 거래량' |
| 첫 거래량 폭발은 설거지일 가능성이 가장 높다 → 첫날 패스. **두 번째 거래량이 터지는 순간이 진입** | `toss_second_volume`: 첫 거래량 후 2~30거래일 안에 '급등 전 20일 평균' 의 3배↑ 거래량(단, 첫날 거래량보다는 작게)이 다시 터지는 날 → 다음날 시가 진입 |
| 첫날 양봉 폭등 → 다음날 (1) 강한 음봉 + 거래량 급감 = 눌림목 진입(주력) / (2) 강한 양봉 + 거래량 1.5~2배 = 단타(본인은 안 함) | `toss_pullback_after_green`: 첫날 +10%↑ 양봉 급등, 다음날 -3%↓ 음봉 + 거래량 ≤ 첫날의 50% → 그 다음날 시가 진입 |
| 첫날 음봉 마감 → (1) 거래량 완전 소멸 + 음봉 지속 → 조정 끝나고 거래량 다시 붙는 순간(또는 첫 음봉 종가 -40% 지점) 진입 = 가장 많이 먹는 구간 / (2) 거래량 유지 + 약한 움직임 = 설거지 → 패스 / (3) 거래량 증가 + 약한 양봉 → 진입 후 보유, 음봉 전환 또는 거래량 ×2 전까지 보유 | `toss_dryup_after_red`: 첫날 종가<시가 급등 후 거래량 ≤ 첫날의 20% 로 마르며 하락 지속 → 종가 ≤ 첫날 종가×0.6 또는 거래량이 전일 2배↑ 인 날 → 다음날 시가 진입. `toss_weak_green_rising_vol`: 첫 급등 후 10일 내 '거래량 증가 + 0<수익률≤5% 약한 양봉' → 진입, 청산 규칙 `exit_rule="toss"`(음봉 또는 거래량 전일 2배) |
| 손익비·진입가·부분손절·익절 시나리오를 미리 계획, 하이리스크 하이리턴 | 손절/익절/보유일 격자 (`sensitivity.py`) |

### B. 디시 해외주식 갤러리 — "저만의 급등주 매매법 공유" (GLBC, 2024-06-29)

프리마켓 뉴스·수급 중심의 재량 매매. 정량화 가능한 부분만 검증.

| 원문 규칙 | 검증/반영 |
|---|---|
| 한국시간 17~20시(프리장 초반) 절대 매매 금지, 20시부터 급등 상위 10종목 캔들 관찰, 21시 이후 뉴스·수급·시총으로 선정. 21시부터 수급이 순간적으로 몰리는 종목이 좋고 프리장부터 거래량 오른 종목은 별로 | 프리마켓 데이터가 없어 직접 검증 불가. 본장 데이터로는 `gap` 구간(갭이 이미 큰 종목 = 프리장부터 오른 종목)의 성과로 간접 확인 |
| 시총 ₩100억대(≈$7M)가 가장 좋고 ₩1,000억(≈$70M) 미만 OK, 그 이상은 무거워 5% 먹기도 힘듦 | `mcap` 구간(< $50M) 의 성과. 이스라엘·바이오 배제는 `country`, `sector` 로 확인 가능(카탈로그 27번 참조) |
| 시드의 1/3만 진입, -5% 마다 나머지를 1/4씩 추가매수(물타기), 급등하면 떨어지는 구간에서 익절, **추격매수 절대 금지** | 백테스터는 단일 진입만 지원(물타기는 손실 확대 위험이 커 미구현). '추격 금지' = 시가/첫봉 추격 진입의 음(-) 기대값으로 확인됨 |
| 손절가 도달 시 실패 인정하고 매도, 목표/손절 후 주식창 닫기, 애프터장 금지, **오버나잇(수면매매) 금지** — 본장 개장~11:30 / 13~14시 / 15:30~마감 에 상승 구간이 있으니 자기 전 전량 매도 | 장중 스터디 '봉별 수익률' 표로 세 구간이 실제로 양(+)인지 확인, 청산 시각별(12:30/14:30/종가) 비교 |
| 본장 직전 목표 달성 시 종료, 미달 시 익절 후 본장 재진입(개장 직후 올리는 척하다 빼고 다시 올림) | 15분봉 '첫 눌림'(`h1_pull`), VWAP 회복(`vwap_reclaim`) 진입으로 근사 검증 |

## 6. 공식이 아니라 '통계' — 외부에서 이미 측정된 것들

| 출처 | 내용 |
|---|---|
| [SmallCapLab 리서치](https://www.smallcaplab.com/research) (2023~, 갭업 이벤트 3,000건↑) | 소형주 갭업의 **66%가 시가보다 낮게 마감**, VWAP 아래 마감 73.8%. 고가의 46.6%는 개장 15분 안에, 69.6%는 10:30 전. 고가 대비 20~40% 붕괴가 최빈(43%). $5~$10·$0.5 미만 종목 페이드 75%. 다음날 시가가 전일 종가 위인 경우 25.4%. |
| [SMB Gap Study](https://www.smbtraining.com/blog/gap-study-33) (S&P500, 2년) | 갭의 76%가 전일 범위로 되돌아옴(gap fill). 5%↑ 큰 갭은 9.6~30%만 메워짐 → 큰 갭은 뉴스성. |
| [Reversals and the returns to liquidity provision](https://mysimon.rochester.edu/novy-marx/research/RRLP.pdf), [Short-term reversal & news](https://www.sciencedirect.com/science/article/abs/pii/S0378426621000261) | 마이크로캡 단기 반전(1개월) 효과 월 2.04%(t=10). 유동성 수요로 인한 급등은 되돌리고, 뉴스에 의한 급등은 이어지는 경향. |
| [Login Market 거래량 급등 동전주 30종목](https://www.loginmarket.co.kr/2024/02/coin30.html) (KR) | 거래대금 급증 동전주 30종목 1개월 평균 **-5.6%**, 1년 -73.9%~+16.5%. |
| [Paper Trading Journal 페니스톡 통계](https://papertradingjournal.com/2026/05/02/penny-stock-statistics/) | 급등 시 거래량 5~20배 후 급감, 스프레드 2~10%, 랠리는 며칠~몇 주 안에 되돌림이 다수, 장기 승자 <10%. |

## 7. 검증 방법

1. **이벤트 스터디**: 각 스크린이 켜진 (일자, 종목)에 대해 다음날 시가 매수 후 당일 종가/1·2·3·5·10일 보유 수익률, 오버나잇(종가→시가), 갭(시가→종가)을 계산. 비용 = 국내 증권사 해외주식 수수료 편도 0.25% x2 + 슬리피지 왕복 20bp(동전주 100bp) 차감 후 평균·중앙값·승률·t-통계량. **공매도는 국내 증권사 해외주식 API 로 불가능하므로 롱(매수)만 검증한다.**
2. **적절값 탐색**: '급등일' 전체 이벤트를 상승률·거래량배수·가격·시총·발행주식수·마감위치·연속상승일·30일 상승폭·52주고가 대비 구간으로 쪼개 어느 구간에서 비용후 기대값이 양(+)인지 확인. 학습 구간(2023-11~2025-09)에서 고른 뒤 검증 구간(2025-09~2026-09, 최근 1년)에서 재확인.
3. **장중 스터디**: 급등 다음날의 1시간봉(최근 2년)·15분봉(최근 60일, Yahoo 제공 한계) 경로로 진입(시가/첫봉 확인/ORB/첫 눌림) × 청산(시간/손절/익절/트레일) 조합의 승률·기대값을 계산.
4. **포트폴리오 백테스트**: 살아남은 규칙만 `quant/backtest.py` 로 자금·슬롯·슬리피지·수수료·유동성 제한을 걸고 1년 검증.
