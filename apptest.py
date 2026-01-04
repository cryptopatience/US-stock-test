import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
warnings.filterwarnings('ignore')

# OpenAI와 Gemini 임포트 (선택적)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="Anchored VWAP 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== AI 설정 ====================
# Gemini 초기화
GEMINI_ENABLED = False
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_ENABLED = True
except Exception as e:
    st.warning(f"⚠️ Gemini AI 초기화 실패: {e}")

# OpenAI 초기화
OPENAI_ENABLED = False
OPENAI_CLIENT = None

try:
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_CLIENT = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        OPENAI_ENABLED = True
except Exception as e:
    st.warning(f"⚠️ OpenAI 초기화 실패: {e}")

# 모델 설정
OPENAI_MODEL_MARKET = st.secrets.get("OPENAI_MODEL_MARKET", "gpt-4o-mini")
OPENAI_MODEL_STOCK = st.secrets.get("OPENAI_MODEL_STOCK", "gpt-4o-mini")
OPENAI_MODEL_CHAT = st.secrets.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

# ==================== 로그인 시스템 ====================
def check_password():
    """비밀번호 확인 및 로그인 상태 관리"""
    if st.session_state.get('password_correct', False):
        return True
    
    st.title("🔒 US Stock VWAP 분석 시스템 로그인")
    st.markdown("### S&P 500 시총 상위 30개 종목 분기별 VWAP 분석")
    
    with st.form("credentials"):
        username = st.text_input("아이디 (ID)", key="username")
        password = st.text_input("비밀번호 (Password)", type="password", key="password")
        submit_btn = st.form_submit_button("로그인", type="primary")
    
    if submit_btn:
        if username in st.secrets["passwords"] and password == st.secrets["passwords"][username]:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("😕 아이디 또는 비밀번호가 올바르지 않습니다.")
    
    return False

if not check_password():
    st.stop()

# ==================== 로그아웃 버튼 ====================
with st.sidebar:
    st.success(f"✅ 로그인 성공!")
    if st.button("🚪 로그아웃"):
        st.session_state['password_correct'] = False
        st.rerun()

# ==================== 유틸리티 함수 ====================
@st.cache_data(ttl=3600)
def get_current_quarter_start():
    """현재 분기 시작일 계산 (최소 5거래일 이상 보장)"""
    now = datetime.now()
    
    # 현재 분기 시작일 계산
    current_quarter_start_month = ((now.month - 1) // 3) * 3 + 1
    current_quarter_start_date = datetime(now.year, current_quarter_start_month, 1)

    # 현재 날짜로부터 현재 분기 시작일까지의 일수 계산 (대략적인 일수)
    days_since_quarter_start = (now - current_quarter_start_date).days

    # 현재 분기가 시작된 지 5일 미만이면 이전 분기 사용
    if days_since_quarter_start < 5: 
        if current_quarter_start_month == 1:  # Q1인 경우, 전년도 Q4
            quarter_start_to_use = datetime(now.year - 1, 10, 1)
        else:  # Q2, Q3, Q4인 경우
            quarter_start_to_use = datetime(now.year, current_quarter_start_month - 3, 1)
    else:  # 현재 분기가 5일 이상 진행되었으면 현재 분기 사용
        quarter_start_to_use = current_quarter_start_date
        
    return quarter_start_to_use


@st.cache_data(ttl=3600)
def get_top_30_tickers():
    """실시간 시가총액 상위 30개 종목 수집 (방어 코드 포함)"""
    sp500_major_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'LLY',
        'V', 'UNH', 'XOM', 'WMT', 'JPM', 'MA', 'JNJ', 'PG', 'AVGO', 'HD',
        'CVX', 'MRK', 'COST', 'ABBV', 'KO', 'PEP', 'NFLX', 'BAC', 'CRM', 'TMO',
        'ORCL', 'ACN', 'CSCO', 'AMD', 'MCD', 'ABT', 'DIS', 'ADBE', 'WFC', 'NKE',
        'PM', 'TXN', 'DHR', 'INTU', 'VZ', 'CMCSA', 'QCOM', 'NEE', 'UNP', 'HON',
        'AMGN', 'LOW', 'RTX', 'BMY', 'UPS', 'SPGI', 'BLK', 'COP', 'SBUX', 'ELV',
        'IBM', 'AMAT', 'CAT', 'GE', 'DE', 'PLD', 'AXP', 'MDLZ', 'LMT', 'GILD',
        'SYK', 'ADI', 'BKNG', 'ISRG', 'MMC', 'VRTX', 'TJX', 'CVS', 'AMT', 'CI',
        'ZTS', 'PGR', 'REGN', 'MO', 'CB', 'DUK', 'BDX', 'SO', 'SCHW', 'ETN',
        'INTC', 'NOW', 'BSX', 'SLB', 'EOG', 'ITW', 'PNC', 'USB', 'AON', 'GD'
    ]
    
    market_caps = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(sp500_major_tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', 0)
            
            if market_cap > 0:
                market_caps.append({
                    'Ticker': ticker,
                    'Market_Cap': market_cap,
                    'Company': info.get('longName', ticker),
                    'Sector': info.get('sector', 'N/A')
                })
            
            progress_bar.progress((idx + 1) / len(sp500_major_tickers))
            status_text.text(f"수집 중: {ticker} ({idx+1}/{len(sp500_major_tickers)})")
        except:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not market_caps:
        st.error("❌ Yahoo Finance 데이터 수집 실패 (API 차단 또는 네트워크 오류)")
        return pd.DataFrame(columns=['Ticker', 'Market_Cap', 'Company', 'Sector'])
    
    df_market_cap = pd.DataFrame(market_caps)
    
    if not df_market_cap.empty and 'Market_Cap' in df_market_cap.columns:
        df_market_cap = df_market_cap.sort_values('Market_Cap', ascending=False).head(30)
    
    return df_market_cap


def calculate_anchored_vwap(df):
    """Anchored VWAP 계산 (분기 시작부터 누적)"""
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Volume'] = df['Typical_Price'] * df['Volume']

    # 누적 계산 (Anchored to Quarter Start)
    df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['Anchored_VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']

    return df


@st.cache_data(ttl=1800)
def get_quarterly_vwap_analysis(ticker):
    """분기별 Anchored VWAP 분석"""
    try:
        # 정확한 분기 시작일
        quarter_start = get_current_quarter_start()
        end_date = datetime.now()

        stock = yf.Ticker(ticker)
        df = stock.history(start=quarter_start, end=end_date)

        if df.empty or len(df) < 5:
            return None

        # Anchored VWAP 계산
        df = calculate_anchored_vwap(df)

        # 분석 데이터
        current_price = df['Close'].iloc[-1]
        current_vwap = df['Anchored_VWAP'].iloc[-1]

        # VWAP 위에서 거래된 일수
        above_vwap_ratio = (df['Close'] > df['Anchored_VWAP']).sum() / len(df) * 100

        # 최근 평균
        recent_5days_avg = df['Close'].tail(5).mean()
        recent_10days_avg = df['Close'].tail(10).mean()

        # 추세 강도
        recent_20 = df['Close'].tail(min(20, len(df)))
        uptrend_strength = (recent_20.diff() > 0).sum() / len(recent_20) * 100 if len(recent_20) > 1 else 50

        # 거래량 분석
        recent_volume = df['Volume'].tail(5).mean()
        avg_volume = df['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        # 회사 정보
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')

        # 분기 시작가 대비 변화율
        quarter_start_price = df['Close'].iloc[0]
        quarter_return = ((current_price - quarter_start_price) / quarter_start_price * 100)

        # 분기 번호 계산 (quarter_start 기준)
        quarter_num = (quarter_start.month - 1) // 3 + 1

        return {
            'Ticker': ticker,
            'Company': company_name,
            'Sector': sector,
            'Quarter': f'{quarter_start.year} Q{quarter_num}',
            'Quarter_Start_Date': quarter_start.strftime('%Y-%m-%d'),
            'Trading_Days': len(df),
            'Current_Price': round(current_price, 2),
            'Anchored_VWAP': round(current_vwap, 2),
            'Quarter_Start_Price': round(quarter_start_price, 2),
            'Quarter_Return_%': round(quarter_return, 2),
            'Price_vs_VWAP_%': round((current_price - current_vwap) / current_vwap * 100, 2),
            'Above_VWAP_Days_%': round(above_vwap_ratio, 1),
            'Recent_5D_Avg': round(recent_5days_avg, 2),
            'Recent_10D_Avg': round(recent_10days_avg, 2),
            'Uptrend_Strength_%': round(uptrend_strength, 1),
            'Volume_Ratio': round(volume_ratio, 2),
            'Is_Above_VWAP': current_price > current_vwap,
            'Strong_Position': (current_price > current_vwap) and (recent_5days_avg > current_vwap) and (above_vwap_ratio > 60),
            'Buy_Signal_Score': 0
        }

    except Exception as e:
        st.warning(f"Error processing {ticker}: {str(e)}")
        return None


def calculate_buy_score(row):
    """매수 신호 점수 계산"""
    score = 0

    # VWAP 위 기본점수
    if row['Is_Above_VWAP']:
        score += 30

    # VWAP 대비 프리미엄 (0-5% 이상적)
    price_diff = row['Price_vs_VWAP_%']
    if 0 < price_diff <= 5:
        score += 20
    elif 5 < price_diff <= 10:
        score += 10
    elif price_diff > 10:
        score += 5

    # VWAP 위 거래 일수
    if row['Above_VWAP_Days_%'] >= 80:
        score += 20
    elif row['Above_VWAP_Days_%'] >= 60:
        score += 15
    elif row['Above_VWAP_Days_%'] >= 40:
        score += 10

    # 추세 강도
    if row['Uptrend_Strength_%'] >= 60:
        score += 15
    elif row['Uptrend_Strength_%'] >= 50:
        score += 10

    # 거래량
    if row['Volume_Ratio'] >= 1.2:
        score += 15
    elif row['Volume_Ratio'] >= 1.0:
        score += 10

    return min(score, 100)


@st.cache_data(ttl=1800)
def get_comprehensive_analysis(ticker):
    """종목별 가치평가, 수익성, 재무, 투자의견 종합 분석"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

        # 안전한 값 추출 함수
        def safe_get(key, default='N/A', multiplier=1, format_str=None):
            value = info.get(key)
            if value is None or value == 'N/A':
                return default
            try:
                if format_str:
                    return format_str.format(value * multiplier)
                return value * multiplier
            except:
                return default

        # 투자의견 한글 변환
        recommendation_map = {
            'buy': '매수',
            'strong buy': '적극 매수',
            'hold': '보유',
            'sell': '매도',
            'strong sell': '적극 매도'
        }
        rec_key = info.get('recommendationKey', 'N/A').lower()
        recommendation_kr = recommendation_map.get(rec_key, rec_key.upper())

        # 목표주가 상승여력 계산
        target_price = safe_get('targetMeanPrice', 0)
        upside = 'N/A'
        if target_price and target_price > 0 and current_price > 0:
            upside = f"{((target_price / current_price) - 1) * 100:.2f}%"

        return {
            "종목명": info.get('longName', ticker),
            "섹터": info.get('sector', 'N/A'),
            "산업": info.get('industry', 'N/A'),
            "현재가": f"${current_price:.2f}" if current_price else "N/A",
            "시가총액": f"${safe_get('marketCap', 0) / 1e9:.2f}B" if safe_get('marketCap', 0) else "N/A",
            "PER": f"{safe_get('trailingPE', 0):.2f}" if safe_get('trailingPE') != 'N/A' else "N/A",
            "Forward PER": f"{safe_get('forwardPE', 0):.2f}" if safe_get('forwardPE') != 'N/A' else "N/A",
            "PBR": f"{safe_get('priceToBook', 0):.2f}" if safe_get('priceToBook') != 'N/A' else "N/A",
            "PEG": f"{safe_get('pegRatio', 0):.2f}" if safe_get('pegRatio') != 'N/A' else "N/A",
            "ROE": safe_get('returnOnEquity', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "영업이익률": safe_get('operatingMargins', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "순이익률": safe_get('profitMargins', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "부채비율": f"{safe_get('debtToEquity', 0):.2f}%" if safe_get('debtToEquity') != 'N/A' else "N/A",
            "유동비율": f"{safe_get('currentRatio', 0):.2f}" if safe_get('currentRatio') != 'N/A' else "N/A",
            "매출성장률": safe_get('revenueGrowth', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "이익성장률": safe_get('earningsGrowth', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "배당수익률": safe_get('dividendYield', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "배당성향": safe_get('payoutRatio', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "투자의견": recommendation_kr,
            "목표주가": f"${target_price:.2f}" if target_price else "N/A",
            "상승여력": upside,
            "애널리스트수": safe_get('numberOfAnalystOpinions', default='N/A')
        }
    except Exception as e:
        return {"Error": f"분석 실패: {str(e)}"}


def get_quarterly_anchors(start_date, end_date):
    """1년간의 모든 분기 시작일 계산"""
    quarters = []
    current = start_date

    while current <= end_date:
        year = current.year
        month = current.month

        quarter_start_month = ((month - 1) // 3) * 3 + 1
        quarter_start = datetime(year, quarter_start_month, 1)

        if quarter_start not in [q[0] for q in quarters] and quarter_start >= start_date:
            quarter_num = (quarter_start_month - 1) // 3 + 1
            quarters.append((quarter_start, f"Q{quarter_num} {year}"))

        if month >= 10:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, quarter_start_month + 3, 1)

    return quarters


def calculate_multiple_anchored_vwaps(df, anchor_points):
    """여러 분기의 Anchored VWAP 계산 (timezone 안전)"""
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

    vwap_columns = {}

    # Timezone 확인
    df_tz = df.index.tz if hasattr(df.index, 'tz') else None

    for anchor_date, quarter_label in anchor_points:
        try:
            # Timezone 처리
            if df_tz is not None:
                anchor_date_tz = pd.Timestamp(anchor_date).tz_localize(df_tz)
            else:
                anchor_date_tz = pd.Timestamp(anchor_date)

            # 필터링
            mask = df.index >= anchor_date_tz
            if mask.sum() == 0:
                continue

            df_period = df[mask].copy()

            # VWAP 계산
            tp_volume = (df_period['Typical_Price'] * df_period['Volume']).cumsum()
            cumulative_volume = df_period['Volume'].cumsum()
            vwap = tp_volume / cumulative_volume

            # 확장
            vwap_full = pd.Series(index=df.index, dtype=float)
            vwap_full[mask] = vwap.values

            vwap_columns[quarter_label] = vwap_full

        except Exception as e:
            continue

    return vwap_columns


# ==================== AI 분석 함수 ====================

def prepare_market_data_for_ai(df_results, above_vwap_stocks, below_vwap_stocks):
    """AI 분석을 위한 시장 데이터 준비"""
    
    quarter_start = get_current_quarter_start()
    quarter_num = (quarter_start.month - 1) // 3 + 1
    
    market_summary = {
        "분석_기준일": datetime.now().strftime('%Y-%m-%d'),
        "분기": f"{quarter_start.year} Q{quarter_num}",
        "분기_시작일": quarter_start.strftime('%Y-%m-%d'),
        "전체_종목수": len(df_results),
        "VWAP_위_종목수": len(above_vwap_stocks),
        "VWAP_아래_종목수": len(below_vwap_stocks),
        "평균_매수점수": float(above_vwap_stocks['Buy_Signal_Score'].mean()) if len(above_vwap_stocks) > 0 else 0,
    }
    
    # TOP 10 종목
    top_10 = above_vwap_stocks.head(10)[
        ['Ticker', 'Company', 'Sector', 'Current_Price', 'Anchored_VWAP', 
         'Price_vs_VWAP_%', 'Quarter_Return_%', 'Above_VWAP_Days_%', 
         'Uptrend_Strength_%', 'Volume_Ratio', 'Buy_Signal_Score']
    ].to_dict('records')
    
    # 약세 종목
    weak_stocks = below_vwap_stocks.head(10)[
        ['Ticker', 'Company', 'Sector', 'Current_Price', 'Anchored_VWAP',
         'Price_vs_VWAP_%', 'Quarter_Return_%']
    ].to_dict('records')
    
    return {
        "market_summary": market_summary,
        "top_performers": top_10,
        "weak_performers": weak_stocks
    }


def get_openai_market_analysis(market_data):
    """OpenAI를 활용한 시장 종합 분석"""
    if not OPENAI_ENABLED:
        return "❌ OpenAI가 비활성화되어 있습니다. secrets.toml에 OPENAI_API_KEY를 추가하세요."

    try:
        prompt = f"""
당신은 미국 주식시장 전문 퀀트/매크로 애널리스트입니다.
아래는 S&P 500 시총 상위 30개 종목의 Quarterly Anchored VWAP + 매수신호 점수 요약 데이터입니다.

[데이터]
{json.dumps(market_data, ensure_ascii=False, indent=2)}

[요청]
다음 항목을 포함하여 실행 가능한 투자 전략 리포트를 작성하세요:

1) **시장 전반 진단 (Market Diagnosis)**
   - VWAP 위/아래 비중으로 시장 심리 해석
   - 매수신호 점수 분포 분석 (평균, 최고, 최저)
   - 섹터별 강약 패턴
   - 분기말(quarter-end) 효과 및 왜곡 가능성

2) **매수/매도 우선순위 (Trading Priorities)**
   
   **Top 3 강력 매수 후보 (Strong Buy)**
   - 종목명, 현재가, VWAP 대비 위치
   - 매수 근거 (정량적 지표 중심)
   - 예상 목표가 및 수익률
   
   **Top 3 눌림목 대기 종목 (Buy on Dip)**
   - 적정 매수 가격대
   - 트리거 조건
   
   **Top 3 매도/경계 종목**
   - 약세 전환 신호
   - 리스크 요인

3) **리스크 관리 전략 (Risk Management)**
   - 변동성 대응: 고변동성 섹터 주의사항
   - 섹터 편중: 포트폴리오 분산 제안
   - 분기말 왜곡: 데이터 신뢰도 검증 방법
   - 손절 라인: VWAP 기준 손절 설정

4) **실행 플랜 (Execution Plan)**
   
   **(i) 평균회귀 전략 (Mean Reversion)**
   - 진입 조건: "만약 [종목]이 VWAP 대비 X% 하락하면"
   - 분할 매수: 1차/2차/3차 진입 가격
   - 손절 조건: "VWAP -Y% 이탈 시"
   - 익절 조건: "VWAP 복귀 + Z% 도달 시"
   
   **(ii) 추세 추종 전략 (Trend Following)**
   - 진입 조건: "VWAP 돌파 + 거래량 증가 확인"
   - 추격 매수: VWAP 상단 돌파 시점
   - 손절 조건: "VWAP 하향 이탈 시"
   - 익절 조건: "목표 수익률 달성 또는 모멘텀 약화"

5) **단기/중기 전망 (Outlook)**
   - 1-2주 전망: 주요 이벤트 및 변수
   - 1-3개월 전망: 구조적 트렌드
   - 모니터링 체크리스트

**작성 가이드**:
- 숫자와 구간을 적극 활용 (예: "VWAP +3.5% 지점", "손절 -2%")
- 조건문으로 명확한 규칙 제시 (if-then 형식)
- 과장하지 말고 데이터 기반으로 서술
- 실무에 바로 적용 가능한 수준으로 구체화

**길이**: 900-1300단어
**언어**: 한국어
"""

        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL_MARKET,
            messages=[
                {"role": "system", "content": "너는 규율 있는 퀀트 애널리스트다. 과장하지 말고 숫자 기반으로 말한다. 실행 가능한 매매 규칙을 조건문으로 명확히 제시한다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ OpenAI 분석 실패: {str(e)}\n\n힌트: openai 라이브러리 버전을 확인하세요. pip install --upgrade openai"


def get_gemini_market_analysis(market_data):
    """Gemini AI를 활용한 시장 종합 분석"""
    if not GEMINI_ENABLED:
        return "❌ Gemini AI가 비활성화되어 있습니다. secrets.toml에 GEMINI_API_KEY를 추가하세요."
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # 데이터 요약 생성
        data_summary = f"""
# 시장 데이터 요약
- 분석 기준일: {market_data['market_summary']['분석_기준일']}
- 분기: {market_data['market_summary']['분기']}
- 전체 종목수: {market_data['market_summary']['전체_종목수']}
- VWAP 위 종목: {market_data['market_summary']['VWAP_위_종목수']}개
- VWAP 아래 종목: {market_data['market_summary']['VWAP_아래_종목수']}개
- 평균 매수점수: {market_data['market_summary']['평균_매수점수']:.2f}

# TOP 10 강세 종목
{json.dumps(market_data['top_performers'], ensure_ascii=False, indent=2)}

# TOP 10 약세 종목
{json.dumps(market_data['weak_performers'], ensure_ascii=False, indent=2)}
"""
        
        prompt = f"""
당신은 미국 주식시장 전문 퀀트 애널리스트입니다. 
아래 S&P 500 시총 상위 30개 종목의 Anchored VWAP + 매수신호 점수 분석 데이터를 바탕으로 심층 분석을 제공하세요.

## 분석 데이터
{data_summary}

## 요청사항
다음 내용을 포함하여 전문적인 분석 리포트를 작성하세요:

1. **시장 전반 진단 (Market Overview)**
   - 현재 시장 국면 (강세/중립/약세)
   - VWAP 기준 시장 구조 분석
   - 섹터별 강약 분석
   - 시장 참여자 행동 분석 (기관 매수세 vs 약세)

2. **매수/매도 우선순위 (Trading Priorities)**
   
   **즉시 매수 추천 (Strong Buy)**
   - 매수점수 80점 이상 종목 분석
   - 각 종목별 구체적 매수 근거
   - 예상 수익률 및 목표가
   
   **눌림목 대기 매수 (Buy on Dip)**
   - 매수점수 60-80점 종목 분석
   - 최적 진입 가격대
   - 트리거 조건
   
   **매도/관망 추천 (Sell/Hold)**
   - VWAP 아래 종목 리스크 분석
   - 약세 전환 징후 종목
   - 포지션 축소 고려 종목

3. **리스크 관리 (Risk Management)**
   - 고변동성 섹터 주의사항
   - 분기말(quarter-end) 효과 분석
   - 포트폴리오 분산 제안
   - 손절라인 설정 가이드

4. **타이밍 전략 (Timing Strategy)**
   
   **단기 트레이딩 (1-4주)**
   - 진입/청산 시그널
   - 스윙 트레이딩 전략
   - VWAP 기준 매매 규칙
   
   **중기 투자 (1-3개월)**
   - 포지션 빌딩 전략
   - 분할 매수/매도 계획
   - 리밸런싱 타이밍

5. **시나리오 분석 (Scenario Analysis)**
   
   **Bull Case (강세 시나리오 60%)**
   - 트리거 이벤트
   - 수혜 종목
   - 목표 수익률
   
   **Base Case (중립 시나리오 30%)**
   - 예상 흐름
   - 대응 전략
   
   **Bear Case (약세 시나리오 10%)**
   - 위험 신호
   - 방어 전략
   - 헤지 방안

6. **향후 전망 (Outlook)**
   - 단기 (1-2주) 전망
   - 중기 (1-3개월) 전망
   - 주요 모니터링 포인트
   - 구조적 성장 스토리

7. **실행 가능한 액션 플랜 (Action Plan)**
   - 즉시 실행 항목
   - 모니터링 체크리스트
   - 포트폴리오 최적 구성비

**분석 스타일**: 
- 월스트리트 리서치 리포트 수준의 전문성
- 정량적 데이터와 정성적 판단의 조화
- 구체적이고 실행 가능한 인사이트
- 숫자와 통계를 적극 활용

**길이**: 1200-1800단어
**언어**: 한국어
**톤**: 전문적이면서도 실무적
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Gemini AI 분석 실패: {str(e)}\n\n힌트: 모델명 'gemini-2.0-flash-exp' 또는 'gemini-1.5-flash'를 확인하세요."


def get_openai_stock_analysis(ticker, stock_data, fundamental_data):
    """OpenAI를 활용한 개별 종목 분석"""
    if not OPENAI_ENABLED:
        return "❌ OpenAI가 비활성화되어 있습니다. secrets.toml에 OPENAI_API_KEY를 추가하세요."
    
    try:
        # 종목 데이터 페이로드 구성
        stock_payload = {
            "ticker": ticker,
            "technical": stock_data,
            "fundamental": fundamental_data
        }
        
        prompt = f"""
너는 단일 종목 트레이딩(스윙/포지션) 전문 퀀트다.
아래 종목의 Anchored VWAP + 매수신호 점수 + 펀더멘털 정보를 기반으로 '지금 이 자리에서 할 수 있는 행동' 중심으로 분석해라.

[종목 데이터]
{json.dumps(stock_payload, ensure_ascii=False, indent=2)}

[요청]
다음 항목을 포함하여 실행 가능한 트레이딩 플랜을 작성하세요:

1) **현재 위치 해석 (Current Position Analysis)**
   - VWAP 대비 괴리율 의미
   - 매수신호 점수 해석
   - 분기 초/말 왜곡 가능성 코멘트
   - 현재 구간 특성 (과매수/정상/과매도)

2) **평균회귀 시나리오 (Mean Reversion Strategy)**
   
   **진입 규칙**
```
   IF [현재가] < [VWAP] - X% THEN
       1차 매수: [구체적 가격]
       수량: 계획 자금의 Y%
```
   
   **분할 매수 규칙**
```
   IF [현재가] < [VWAP] - X2% THEN
       2차 매수: [구체적 가격]
       수량: 계획 자금의 Y2%
```
   
   **무효화(손절) 규칙**
```
   IF [현재가] < [VWAP] - Z% OR [기술적 구조 붕괴] THEN
       전량 손절
       손실: 최대 -W%로 제한
```
   
   **익절 규칙**
```
   IF [현재가] >= [VWAP] + P% THEN
       1차 익절: 50% 물량
   IF [현재가] >= [VWAP] + P2% THEN
       전량 익절
```

3) **추세 추종 시나리오 (Trend Following Strategy)**
   
   **VWAP 돌파 매수**
```
   IF [현재가] > [VWAP] AND [거래량] > [평균 거래량] * 1.5 THEN
       진입 매수
       목표: VWAP + Q%
```
   
   **VWAP 이탈 손절**
```
   IF [현재가] < [VWAP] - R% THEN
       추세 무효화 → 손절
```
   
   **추격 익절**
      
