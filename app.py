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
    """현재 분기 시작일 계산"""
    now = datetime.now()
    quarter = (now.month - 1) // 3
    quarter_start_month = quarter * 3 + 1
    quarter_start = datetime(now.year, quarter_start_month, 1)
    return quarter_start

@st.cache_data(ttl=3600)
def get_top_30_tickers():
    """실시간 시가총액 상위 30개 종목 수집"""
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
    
    df_market_cap = pd.DataFrame(market_caps)
    df_market_cap = df_market_cap.sort_values('Market_Cap', ascending=False).head(30)
    
    return df_market_cap

def calculate_anchored_vwap(df):
    """Anchored VWAP 계산"""
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['Anchored_VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
    return df

@st.cache_data(ttl=1800)
def get_quarterly_vwap_analysis(ticker):
    """분기별 Anchored VWAP 분석"""
    try:
        quarter_start = get_current_quarter_start()
        end_date = datetime.now()
        quarter_num = (quarter_start.month - 1) // 3 + 1
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=quarter_start, end=end_date)
        
        if df.empty or len(df) < 5:
            return None
        
        df = calculate_anchored_vwap(df)
        
        current_price = df['Close'].iloc[-1]
        current_vwap = df['Anchored_VWAP'].iloc[-1]
        above_vwap_ratio = (df['Close'] > df['Anchored_VWAP']).sum() / len(df) * 100
        recent_5days_avg = df['Close'].tail(5).mean()
        recent_10days_avg = df['Close'].tail(10).mean()
        
        recent_20 = df['Close'].tail(min(20, len(df)))
        uptrend_strength = (recent_20.diff() > 0).sum() / len(recent_20) * 100 if len(recent_20) > 1 else 50
        
        recent_volume = df['Volume'].tail(5).mean()
        avg_volume = df['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        
        quarter_start_price = df['Close'].iloc[0]
        quarter_return = ((current_price - quarter_start_price) / quarter_start_price * 100)
        
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
            'Is_Above_VWAP': current_price > current_vwap
        }
    except Exception as e:
        st.warning(f"Error processing {ticker}: {str(e)}")
        return None

def calculate_buy_score(row):
    """매수 신호 점수 계산"""
    score = 0
    
    if row['Is_Above_VWAP']:
        score += 30
    
    price_diff = row['Price_vs_VWAP_%']
    if 0 < price_diff <= 5:
        score += 20
    elif 5 < price_diff <= 10:
        score += 10
    elif price_diff > 10:
        score += 5
    
    if row['Above_VWAP_Days_%'] >= 80:
        score += 20
    elif row['Above_VWAP_Days_%'] >= 60:
        score += 15
    elif row['Above_VWAP_Days_%'] >= 40:
        score += 10
    
    if row['Uptrend_Strength_%'] >= 60:
        score += 15
    elif row['Uptrend_Strength_%'] >= 50:
        score += 10
    
    if row['Volume_Ratio'] >= 1.2:
        score += 15
    elif row['Volume_Ratio'] >= 1.0:
        score += 10
    
    return min(score, 100)

@st.cache_data(ttl=1800)
def get_comprehensive_analysis(ticker):
    """종목별 펀더멘털 분석"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
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
        
        recommendation_map = {
            'buy': '매수',
            'strong buy': '적극 매수',
            'hold': '보유',
            'sell': '매도',
            'strong sell': '적극 매도'
        }
        rec_key = info.get('recommendationKey', 'N/A').lower()
        recommendation_kr = recommendation_map.get(rec_key, rec_key.upper())
        
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
            "매출성장률": safe_get('revenueGrowth', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "이익성장률": safe_get('earningsGrowth', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "배당수익률": safe_get('dividendYield', default='N/A', multiplier=100, format_str="{:.2f}%"),
            "투자의견": recommendation_kr,
            "목표주가": f"${target_price:.2f}" if target_price else "N/A",
            "상승여력": upside
        }
    except Exception as e:
        return {"Error": f"분석 실패: {str(e)}"}

def get_quarterly_anchors(start_date, end_date):
    """분기 시작일 계산"""
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
    """여러 분기의 Anchored VWAP 계산"""
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    vwap_columns = {}
    df_tz = df.index.tz if hasattr(df.index, 'tz') else None
    
    for anchor_date, quarter_label in anchor_points:
        try:
            if df_tz is not None:
                anchor_date_tz = pd.Timestamp(anchor_date).tz_localize(df_tz)
            else:
                anchor_date_tz = pd.Timestamp(anchor_date)
            
            mask = df.index >= anchor_date_tz
            if mask.sum() == 0:
                continue
            
            df_period = df[mask].copy()
            tp_volume = (df_period['Typical_Price'] * df_period['Volume']).cumsum()
            cumulative_volume = df_period['Volume'].cumsum()
            vwap = tp_volume / cumulative_volume
            
            vwap_full = pd.Series(index=df.index, dtype=float)
            vwap_full[mask] = vwap.values
            vwap_columns[quarter_label] = vwap_full
        except:
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
        model = genai.GenerativeModel('gemini-2.5-flash')
        
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
        return f"❌ Gemini AI 분석 실패: {str(e)}\n\n힌트: 모델명 'gemini-2.5-flash' 또는 'gemini-1.5-flash'를 확인하세요."

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
   ```
   IF [목표가 달성] OR [모멘텀 약화] THEN
       단계적 익절
   ```

4) **펀더멘털 체크 (Fundamental Check)**
   - PER/PBR/ROE 기준 밸류에이션 평가
   - 성장성 지표 (매출/이익 성장률)
   - 재무 건전성 (부채비율)
   - 월가 컨센서스와의 정합성

5) **리스크 요인 & 대응 (Risk Factors)**
   - 주요 리스크 3가지
   - 각 리스크별 대응 방안
   - 포지션 사이징 제안

6) **체크리스트 (5줄)**
   추가 확인할 것 (가격/변동성/구간 중심):
   - [ ] 항목 1
   - [ ] 항목 2
   - [ ] 항목 3
   - [ ] 항목 4
   - [ ] 항목 5

**작성 가이드**:
- 모든 매매 규칙을 IF-THEN 조건문으로 명확히 작성
- 구체적인 숫자와 가격대 제시
- 뉴스/감정이 아닌 가격/구조 중심
- 한국어, 간결하지만 구체적으로
"""

        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL_STOCK,
            messages=[
                {"role": "system", "content": "너는 매매 규칙을 조건문으로 명확히 쓰는 퀀트 트레이더다. 감정이 아닌 숫자와 규칙으로 말한다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=2500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ OpenAI 개별 종목 분석 실패: {str(e)}"

def get_gemini_stock_analysis(ticker, stock_data, fundamental_data):
    """Gemini AI를 활용한 개별 종목 분석"""
    if not GEMINI_ENABLED:
        return "❌ Gemini AI가 비활성화되어 있습니다. secrets.toml에 GEMINI_API_KEY를 추가하세요."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 종목 데이터 페이로드 구성
        stock_payload = {
            "ticker": ticker,
            "technical": stock_data,
            "fundamental": fundamental_data
        }
        
        prompt = f"""
당신은 월스트리트 톱티어 투자은행의 수석 애널리스트입니다.
다음 종목에 대한 프로페셔널 리서치 리포트를 작성하세요.

# 종목: {ticker}

## 분석 데이터
{json.dumps(stock_payload, ensure_ascii=False, indent=2)}

# 리서치 리포트 구성

## Executive Summary
- **투자 의견**: BUY / HOLD / SELL
- **목표 주가**: $XX.XX
- **상승/하락 여력**: +XX% / -XX%
- **핵심 투자 포인트**:
  1. [포인트 1]
  2. [포인트 2]
  3. [포인트 3]

## 1. 비즈니스 & 산업 분석
**핵심 사업 구조**
- 주요 비즈니스 모델
- 수익 구조 및 마진

**경쟁 우위 요소**
- 차별화 포인트
- 진입장벽

**산업 내 포지셔닝**
- 시장 점유율
- 경쟁사 대비 우위

## 2. 재무 분석

**수익성 분석**
- 영업이익률: {fundamental_data.get('영업이익률', 'N/A')}
- 순이익률: {fundamental_data.get('순이익률', 'N/A')}
- ROE: {fundamental_data.get('ROE', 'N/A')}
- 평가: [우수/양호/보통/미흡]

**성장성 분석**
- 매출 성장률: {fundamental_data.get('매출성장률', 'N/A')}
- 이익 성장률: {fundamental_data.get('이익성장률', 'N/A')}
- 향후 성장 동력

**재무 건전성**
- 부채비율: {fundamental_data.get('부채비율', 'N/A')}
- 현금 흐름 상태
- 재무 리스크 평가

## 3. 밸류에이션

**멀티플 분석**
- PER: {fundamental_data.get('PER', 'N/A')} → [저평가/적정/고평가]
- PBR: {fundamental_data.get('PBR', 'N/A')} → [저평가/적정/고평가]
- PEG: {fundamental_data.get('PEG', 'N/A')} → [성장성 대비 평가]

**동종 업체 대비**
- 섹터 평균 PER과 비교
- 프리미엄/디스카운트 정당성

**목표주가 산출**
- 방법론: [PER 기반 / DCF / 유사기업 비교]
- 목표 멀티플
- 목표주가: $XX.XX
- 상승여력: {fundamental_data.get('상승여력', 'N/A')}

## 4. 기술적 분석 (VWAP 기반)

**현재 포지션**
- 현재가: ${stock_data.get('Current_Price', 'N/A')}
- Anchored VWAP: ${stock_data.get('Anchored_VWAP', 'N/A')}
- VWAP 대비: {stock_data.get('Price_vs_VWAP_%', 'N/A')}%
- 해석: [강세/중립/약세]

**지지/저항 레벨**
- 1차 지지: VWAP - 2% = $XX.XX
- 2차 지지: VWAP - 5% = $XX.XX
- 1차 저항: VWAP + 3% = $XX.XX
- 2차 저항: VWAP + 7% = $XX.XX

**모멘텀 지표**
- 추세 강도: {stock_data.get('Uptrend_Strength_%', 'N/A')}%
- 거래량 비율: {stock_data.get('Volume_Ratio', 'N/A')}x
- VWAP 위 거래일: {stock_data.get('Above_VWAP_Days_%', 'N/A')}%
- 평가: [강함/보통/약함]

## 5. 투자 전략

**매수 전략**
- **최적 진입 가격**: $XX.XX - $XX.XX
- **포지션 사이징**: 포트폴리오의 X-Y%
- **분할 매수 플랜**:
  * 1차: VWAP -2% 도달 시 → 40%
  * 2차: VWAP -4% 도달 시 → 30%
  * 3차: VWAP -6% 도달 시 → 30%

**리스크 관리**
- **손절선**: VWAP -8% = $XX.XX (엄격 준수)
- **포지션 관리**: [섹터] 비중 전체의 15% 이하
- **헤지 전략**: [옵션 전략 / 관련 ETF 매도 / 현금 보유]

**익절 전략**
- **1차 목표**: +10% = $XX.XX → 50% 익절
- **2차 목표**: +20% = $XX.XX → 30% 익절
- **최종 목표**: 목표주가 도달 → 잔량 익절

## 6. 시나리오 분석

**상승 시나리오 (Upside Case, 40%)**
- **트리거 이벤트**:
  * [실적 서프라이즈]
  * [신제품 성공]
  * [밸류에이션 재평가]
- **목표 수익률**: +25-35%
- **기간**: 3-6개월

**기본 시나리오 (Base Case, 50%)**
- **예상 흐름**: VWAP 중심 박스권
- **목표 수익률**: +10-15%
- **기간**: 2-4개월

**하락 시나리오 (Downside Case, 10%)**
- **리스크 요인**:
  * [매크로 악화]
  * [실적 쇼크]
  * [경쟁 심화]
- **최대 손실**: -8% (손절 시)
- **대응**: 즉시 손절, 재진입 기회 포착

## 7. 투자 타임라인

**단기 (1-3개월)**
- **주요 이벤트**: [실적발표일], [신제품 출시]
- **전술적 트레이딩**:
  * VWAP 돌파 시 추격 매수 고려
  * VWAP 하단 터치 시 분할 매수
- **모니터링 지표**:
  * 일일 VWAP 추이
  * 거래량 패턴
  * 섹터 상대강도

**중장기 (6-12개월)**
- **구조적 테마**: [AI 혁명 / 디지털 전환 / 친환경]
- **전략적 홀딩**: 
  * 핵심 포지션으로 보유
  * 조정 시 추가 매수
- **기대 수익률**: +20-30%

## 8. 리스크 체크리스트

**재무 리스크**
- [ ] 부채비율 증가 추이 확인
- [ ] 현금흐름 악화 징후 모니터링
- [ ] 이익 품질 검증 (일회성 항목 제외)

**사업 리스크**
- [ ] 주요 고객사 매출 의존도
- [ ] 신제품 출시 일정 점검
- [ ] 경쟁사 동향 추적

**시장 리스크**
- [ ] 섹터 로테이션 가능성
- [ ] 매크로 변수 (금리, 환율)
- [ ] 기술적 지지선 이탈 여부

## 9. 액션 플랜 (Action Items)

**즉시 실행**
1. 현재가와 VWAP 괴리율 확인
2. 분할 매수 가격대 알림 설정
3. 손절 주문 사전 입력

**모니터링**
1. 일일: VWAP 대비 종가 위치, 거래량
2. 주간: 추세 강도, 섹터 상대성과
3. 월간: 실적 추정치 변화, 애널리스트 의견

**정기 리뷰**
- 분기 실적 발표 후 투자의견 재검토
- VWAP 재설정 시점 (분기 시작) 전략 조정
- 목표가 도달 시 익절 후 재진입 기회 평가

---

**보고서 작성일**: {datetime.now().strftime('%Y-%m-%d')}
**애널리스트**: Gemini AI Quant Division
**평점**: [투자의견 요약]

**면책조항**: 본 리포트는 투자 판단의 참고자료이며, 최종 투자 책임은 투자자 본인에게 있습니다.

---

**작성 가이드**:
- 골드만삭스/모건스탠리 수준의 리서치 퀄리티
- 정량적 근거와 정성적 판단의 균형
- 실무에 바로 적용 가능한 구체성
- 한글, 전문적이면서도 명확한 문체
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Gemini 분석 오류: {str(e)}"

# ==================== 메인 앱 ====================
st.title("📊 US stock 분기 VWAP 분석 ")
st.markdown("### S&P 500 시가총액 상위 30개 종목 분기별 분석")

# 사이드바
with st.sidebar:
    st.markdown("---")
    st.header("⚙️ 설정")
    
    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.info("""
    **Anchored VWAP**
    - 분기 시작일부터 누적된 거래량 가중 평균 가격
    - VWAP 위: 기관/대량 매수세 우위
    - VWAP 아래: 약세 구간
    """)
    
    # AI 상태 표시
    st.markdown("---")
    st.markdown("**🤖 AI 분석 엔진**")
    
    col1, col2 = st.columns(2)
    with col1:
        if OPENAI_ENABLED:
            st.success("✅ OpenAI")
            st.caption(f"모델: {OPENAI_MODEL_MARKET}")
        else:
            st.error("❌ OpenAI")
    
    with col2:
        if GEMINI_ENABLED:
            st.success("✅ Gemini")
            st.caption("모델: 2.0-flash")
        else:
            st.error("❌ Gemini")
    
    if not OPENAI_ENABLED and not GEMINI_ENABLED:
        st.warning("💡 secrets.toml에 API 키 추가 필요")

# 분기 정보
quarter_start = get_current_quarter_start()
quarter_num = (quarter_start.month - 1) // 3 + 1

st.info(f"""
**📍 분석 기준**  
- 분기: {quarter_start.year} Q{quarter_num}  
- Anchor Point: {quarter_start.strftime('%Y-%m-%d')}  
- 경과일: {(datetime.now() - quarter_start).days}일
""")

# 데이터 수집
with st.spinner("📡 시가총액 데이터 수집 중..."):
    df_market_cap = get_top_30_tickers()

st.success(f"✅ 상위 30개 종목 수집 완료!")

# 시가총액 테이블
with st.expander("📋 시가총액 상위 30개 종목 보기"):
    df_display = df_market_cap.copy()
    df_display['Market_Cap_B'] = (df_display['Market_Cap'] / 1e9).round(2)
    st.dataframe(
        df_display[['Ticker', 'Company', 'Sector', 'Market_Cap_B']],
        use_container_width=True,
        hide_index=True
    )

# VWAP 분석
top_30_tickers = df_market_cap['Ticker'].tolist()

with st.spinner("📊 Anchored VWAP 분석 중..."):
    results = []
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(top_30_tickers):
        result = get_quarterly_vwap_analysis(ticker)
        if result:
            results.append(result)
        progress_bar.progress((idx + 1) / len(top_30_tickers))
    
    progress_bar.empty()

df_results = pd.DataFrame(results)
df_results['Buy_Signal_Score'] = df_results.apply(calculate_buy_score, axis=1)

above_vwap_stocks = df_results[df_results['Is_Above_VWAP'] == True].copy()
above_vwap_stocks = above_vwap_stocks.sort_values('Buy_Signal_Score', ascending=False)

below_vwap_stocks = df_results[df_results['Is_Above_VWAP'] == False].copy()
below_vwap_stocks = below_vwap_stocks.sort_values('Price_vs_VWAP_%')

# 탭 구성
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 TOP 5 추천",
    "📊 전체 분석",
    "📈 차트",
    "💼 펀더멘털",
    "🤖 OpenAI 분석",
    "🧠 Gemini AI 분석",
    "💬 AI 챗팅",
    "📋 투자 전략"
])

with tab1:
    st.header("🏆 TOP 5 투자 추천 종목")
    
    top_5_recommendations = above_vwap_stocks.head(5)
    
    for idx, row in top_5_recommendations.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.subheader(f"{row['Ticker']} - {row['Company']}")
                st.caption(f"섹터: {row['Sector']}")
            
            with col2:
                st.metric("현재가", f"${row['Current_Price']}")
                st.metric("Anchored VWAP", f"${row['Anchored_VWAP']}")
            
            with col3:
                score = row['Buy_Signal_Score']
                if score >= 80:
                    st.success(f"⭐ {score}/100")
                    st.caption("💚 강력 매수")
                elif score >= 60:
                    st.warning(f"⭐ {score}/100")
                    st.caption("💛 눌림목 대기")
                else:
                    st.info(f"⭐ {score}/100")
                    st.caption("💙 보통")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("VWAP 대비", f"{row['Price_vs_VWAP_%']:+.2f}%")
            col2.metric("분기 수익률", f"{row['Quarter_Return_%']:+.2f}%")
            col3.metric("VWAP 위 거래일", f"{row['Above_VWAP_Days_%']:.1f}%")
            col4.metric("거래량 비율", f"{row['Volume_Ratio']:.2f}x")
            
            st.markdown("---")

with tab2:
    st.header("📊 전체 종목 분석 결과")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ VWAP 위 종목")
        st.dataframe(
            above_vwap_stocks[[
                'Ticker', 'Company', 'Current_Price', 'Anchored_VWAP',
                'Price_vs_VWAP_%', 'Quarter_Return_%', 'Buy_Signal_Score'
            ]],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("⚠️ VWAP 아래 종목")
        st.dataframe(
            below_vwap_stocks[[
                'Ticker', 'Company', 'Current_Price', 'Anchored_VWAP',
                'Price_vs_VWAP_%', 'Quarter_Return_%'
            ]],
            use_container_width=True,
            hide_index=True
        )

with tab3:
    st.header("📈 인터랙티브 차트")
    
    # 매수 신호 점수
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=above_vwap_stocks['Ticker'],
        x=above_vwap_stocks['Buy_Signal_Score'],
        orientation='h',
        marker=dict(
            color=above_vwap_stocks['Buy_Signal_Score'],
            colorscale='RdYlGn',
            showscale=True
        ),
        text=above_vwap_stocks['Buy_Signal_Score'],
        textposition='auto'
    ))
    fig1.update_layout(
        title=f'매수 신호 점수 ({quarter_start.year} Q{quarter_num})',
        xaxis_title='매수 신호 점수',
        yaxis_title='종목',
        height=600
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # VWAP 대비 가격
    colors = ['green' if x > 0 else 'red' for x in df_results['Price_vs_VWAP_%']]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=df_results.sort_values('Price_vs_VWAP_%', ascending=False)['Ticker'],
        x=df_results.sort_values('Price_vs_VWAP_%', ascending=False)['Price_vs_VWAP_%'],
        orientation='h',
        marker=dict(color=colors),
        text=df_results.sort_values('Price_vs_VWAP_%', ascending=False)['Price_vs_VWAP_%'].round(1),
        textposition='auto'
    ))
    fig2.add_vline(x=0, line_dash="dash", line_color="black")
    fig2.update_layout(
        title='Anchored VWAP 대비 가격 위치',
        xaxis_title='VWAP 대비 차이 (%)',
        yaxis_title='종목',
        height=800
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 종목별 상세 차트
    st.subheader("📊 종목별 1년 차트 + 분기별 VWAP")
    
    selected_ticker = st.selectbox(
        "종목 선택",
        top_5_recommendations['Ticker'].tolist()
    )
    
    if selected_ticker:
        with st.spinner(f"{selected_ticker} 차트 생성 중..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            stock = yf.Ticker(selected_ticker)
            df_1year = stock.history(start=start_date, end=end_date)
            
            if not df_1year.empty:
                quarter_anchors = get_quarterly_anchors(start_date, end_date)
                vwap_dict = calculate_multiple_anchored_vwaps(df_1year, quarter_anchors)
                
                fig = go.Figure()
                
                # 캔들스틱
                fig.add_trace(go.Candlestick(
                    x=df_1year.index,
                    open=df_1year['Open'],
                    high=df_1year['High'],
                    low=df_1year['Low'],
                    close=df_1year['Close'],
                    name='Price'
                ))
                
                # VWAP 라인
                colors_vwap = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
                for idx_vwap, (quarter_label, vwap_series) in enumerate(vwap_dict.items()):
                    color = colors_vwap[idx_vwap % len(colors_vwap)]
                    vwap_clean = vwap_series.dropna()
                    if len(vwap_clean) > 0:
                        fig.add_trace(go.Scatter(
                            x=vwap_clean.index,
                            y=vwap_clean,
                            mode='lines',
                            name=f'VWAP {quarter_label}',
                            line=dict(color=color, width=2)
                        ))
                
                # 거래량
                fig.add_trace(go.Bar(
                    x=df_1year.index,
                    y=df_1year['Volume'],
                    name='Volume',
                    marker_color='rgba(128, 128, 128, 0.3)',
                    yaxis='y2'
                ))
                
                company_info = above_vwap_stocks[above_vwap_stocks['Ticker'] == selected_ticker].iloc[0]
                
                fig.update_layout(
                    title=f"{selected_ticker} - {company_info['Company']}",
                    xaxis=dict(rangeslider=dict(visible=False)),
                    yaxis=dict(title='가격 (USD)', side='right'),
                    yaxis2=dict(
                        title='거래량',
                        overlaying='y',
                        side='left',
                        showgrid=False
                    ),
                    height=700,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("💼 펀더멘털 분석")
    
    for idx, ticker in enumerate(top_5_recommendations['Ticker'].tolist(), 1):
        with st.expander(f"📊 {ticker} 상세 분석", expanded=(idx == 1)):
            analysis = get_comprehensive_analysis(ticker)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**기본 정보**")
                st.write(f"종목명: {analysis['종목명']}")
                st.write(f"섹터: {analysis['섹터']}")
                st.write(f"현재가: {analysis['현재가']}")
                st.write(f"시가총액: {analysis['시가총액']}")
            
            with col2:
                st.markdown("**가치평가**")
                st.write(f"PER: {analysis['PER']}")
                st.write(f"PBR: {analysis['PBR']}")
                st.write(f"PEG: {analysis['PEG']}")
                st.write(f"ROE: {analysis['ROE']}")
            
            with col3:
                st.markdown("**성장성 & 투자의견**")
                st.write(f"매출성장률: {analysis['매출성장률']}")
                st.write(f"이익성장률: {analysis['이익성장률']}")
                st.write(f"투자의견: {analysis['투자의견']}")
                st.write(f"상승여력: {analysis['상승여력']}")

with tab5:
    st.header("🤖 OpenAI 퀀트 분석")
    
    if not OPENAI_ENABLED:
        st.warning("⚠️ OpenAI API가 연결되지 않았습니다. secrets.toml에 OPENAI_API_KEY를 추가해주세요.")
        st.info("""
        **OpenAI API 키 발급 방법:**
        1. https://platform.openai.com 접속
        2. API Keys 메뉴에서 새 키 생성
        3. secrets.toml에 추가: `OPENAI_API_KEY = "sk-..."`
        """)
    else:
        st.success(f"✅ OpenAI 연결됨 (모델: {OPENAI_MODEL_MARKET})")
        st.info("💡 규율 있는 퀀트 관점의 실행 가능한 매매 전략을 제공합니다.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analysis_type = st.radio(
            "분석 유형 선택",
            ["🌍 시장 종합 분석", "📊 개별 종목 분석"],
            horizontal=True
        )
    
    with col2:
        if st.button("🚀 AI 분석 실행", type="primary", use_container_width=True):
            if analysis_type == "🌍 시장 종합 분석":
                with st.spinner("🤖 OpenAI가 시장을 분석하고 있습니다..."):
                    market_data = prepare_market_data_for_ai(df_results, above_vwap_stocks, below_vwap_stocks)
                    analysis_result = get_openai_market_analysis(market_data)
                    st.session_state['openai_market_analysis'] = analysis_result
            else:
                st.session_state['openai_show_stock_selector'] = True
    
    if analysis_type == "🌍 시장 종합 분석":
        if 'openai_market_analysis' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 OpenAI 시장 분석 리포트")
            st.markdown(st.session_state['openai_market_analysis'])
            
            # 다운로드 버튼
            st.download_button(
                label="📥 분석 리포트 다운로드",
                data=st.session_state['openai_market_analysis'],
                file_name=f"OpenAI_Market_Analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    else:  # 개별 종목 분석
        if st.session_state.get('openai_show_stock_selector', False):
            selected_ticker = st.selectbox(
                "분석할 종목 선택",
                above_vwap_stocks['Ticker'].tolist(),
                key="openai_stock_selector"
            )
            
            if st.button("🔍 선택 종목 분석", type="primary"):
                with st.spinner(f"🤖 OpenAI가 {selected_ticker}를 분석하고 있습니다..."):
                    # 종목 데이터 준비
                    stock_data = above_vwap_stocks[above_vwap_stocks['Ticker'] == selected_ticker].iloc[0].to_dict()
                    fundamental_data = get_comprehensive_analysis(selected_ticker)
                    
                    analysis_result = get_openai_stock_analysis(selected_ticker, stock_data, fundamental_data)
                    st.session_state[f'openai_stock_analysis_{selected_ticker}'] = analysis_result
        
        # 분석 결과 표시
        for key in list(st.session_state.keys()):
            if key.startswith('openai_stock_analysis_'):
                ticker = key.replace('openai_stock_analysis_', '')
                st.markdown("---")
                st.markdown(f"### 📊 {ticker} 종목 분석 리포트")
                st.markdown(st.session_state[key])
                
                st.download_button(
                    label=f"📥 {ticker} 분석 다운로드",
                    data=st.session_state[key],
                    file_name=f"OpenAI_{ticker}_Analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key=f"download_openai_{ticker}"
                )

with tab6:
    st.header("🧠 Gemini AI 심층 분석")
    
    if not GEMINI_ENABLED:
        st.warning("⚠️ Gemini API가 연결되지 않았습니다. secrets.toml에 GEMINI_API_KEY를 추가해주세요.")
        st.info("""
        **Gemini API 키 발급 방법:**
        1. https://makersuite.google.com/app/apikey 접속
        2. Create API key 클릭
        3. secrets.toml에 추가: `GEMINI_API_KEY = "..."`
        4. 무료 할당량: 분당 15회, 일당 1,500회
        """)
    else:
        st.success("✅ Gemini 연결됨 (모델: gemini-2.5-flash)")
        st.info("💡 월스트리트 리서치 리포트 수준의 심층적인 투자 인사이트를 제공합니다.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analysis_type_gemini = st.radio(
            "분석 유형 선택",
            ["🌍 시장 종합 분석", "📊 개별 종목 분석"],
            horizontal=True,
            key="gemini_analysis_type"
        )
    
    with col2:
        if st.button("🚀 AI 분석 실행", type="primary", use_container_width=True, key="gemini_analyze"):
            if analysis_type_gemini == "🌍 시장 종합 분석":
                with st.spinner("🧠 Gemini AI가 시장을 분석하고 있습니다..."):
                    market_data = prepare_market_data_for_ai(df_results, above_vwap_stocks, below_vwap_stocks)
                    analysis_result = get_gemini_market_analysis(market_data)
                    st.session_state['gemini_market_analysis'] = analysis_result
            else:
                st.session_state['gemini_show_stock_selector'] = True
    
    if analysis_type_gemini == "🌍 시장 종합 분석":
        if 'gemini_market_analysis' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Gemini AI 시장 분석 리포트")
            st.markdown(st.session_state['gemini_market_analysis'])
            
            st.download_button(
                label="📥 분석 리포트 다운로드",
                data=st.session_state['gemini_market_analysis'],
                file_name=f"Gemini_Market_Analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key="download_gemini_market"
            )
    
    else:  # 개별 종목 분석
        if st.session_state.get('gemini_show_stock_selector', False):
            selected_ticker_gemini = st.selectbox(
                "분석할 종목 선택",
                above_vwap_stocks['Ticker'].tolist(),
                key="gemini_stock_selector"
            )
            
            if st.button("🔍 선택 종목 분석", type="primary", key="gemini_stock_analyze"):
                with st.spinner(f"🧠 Gemini AI가 {selected_ticker_gemini}를 분석하고 있습니다..."):
                    stock_data = above_vwap_stocks[above_vwap_stocks['Ticker'] == selected_ticker_gemini].iloc[0].to_dict()
                    fundamental_data = get_comprehensive_analysis(selected_ticker_gemini)
                    
                    analysis_result = get_gemini_stock_analysis(selected_ticker_gemini, stock_data, fundamental_data)
                    st.session_state[f'gemini_stock_analysis_{selected_ticker_gemini}'] = analysis_result
        
        # 분석 결과 표시
        for key in list(st.session_state.keys()):
            if key.startswith('gemini_stock_analysis_'):
                ticker = key.replace('gemini_stock_analysis_', '')
                st.markdown("---")
                st.markdown(f"### 📊 {ticker} 종목 분석 리포트")
                st.markdown(st.session_state[key])
                
                st.download_button(
                    label=f"📥 {ticker} 분석 다운로드",
                    data=st.session_state[key],
                    file_name=f"Gemini_{ticker}_Analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    key=f"download_gemini_{ticker}"
                )

with tab7:
    st.header("💬 AI 챗팅 - 투자 Q&A")
    
    st.info("""
    **💡 사용 방법**
    - 분석 결과에 대해 자유롭게 질문하세요
    - 특정 종목에 대한 추가 정보 요청
    - 투자 전략 및 리스크 관리 상담
    - 기술적/펀더멘털 지표 해석
    """)
    
    # AI 선택
    col1, col2 = st.columns([1, 3])
    
    with col1:
        ai_engine = st.radio(
            "AI 엔진 선택",
            ["🤖 OpenAI", "🧠 Gemini"],
            key="chat_ai_engine"
        )
    
    with col2:
        if ai_engine == "🤖 OpenAI" and not OPENAI_ENABLED:
            st.warning("⚠️ OpenAI가 비활성화되어 있습니다.")
        elif ai_engine == "🧠 Gemini" and not GEMINI_ENABLED:
            st.warning("⚠️ Gemini가 비활성화되어 있습니다.")
        else:
            st.success(f"✅ {ai_engine} 사용 가능")
    
    # 채팅 히스토리 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 대화 초기화 버튼
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ 대화 초기화", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📊 현재 분석 컨텍스트 제공", type="secondary"):
            # 현재 분석 데이터를 컨텍스트로 추가
            context_message = f"""
현재 분석 상황:
- 분기: {quarter_start.year} Q{quarter_num}
- VWAP 위 종목: {len(above_vwap_stocks)}개
- VWAP 아래 종목: {len(below_vwap_stocks)}개
- TOP 5 추천: {', '.join(above_vwap_stocks.head(5)['Ticker'].tolist())}
- 평균 매수점수: {above_vwap_stocks['Buy_Signal_Score'].mean():.1f}점

이 데이터를 바탕으로 질문에 답변해주세요.
"""
            st.session_state.chat_history.append({
                "role": "system",
                "content": context_message
            })
            st.success("✅ 현재 분석 데이터가 대화 컨텍스트에 추가되었습니다.")
    
    # 채팅 히스토리 표시
    st.markdown("---")
    st.markdown("### 💬 대화 내역")
    
    chat_container = st.container()
    
    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                avatar = "🤖" if ai_engine == "🤖 OpenAI" else "🧠"
                with st.chat_message("assistant", avatar=avatar):
                    st.markdown(message["content"])
    
    # 사용자 입력
    st.markdown("---")
    
    # 빠른 질문 버튼
    st.markdown("**💡 빠른 질문**")
    quick_questions = [
        "TOP 5 종목 중 가장 매수하기 좋은 종목은?",
        "VWAP 아래 종목들은 언제 매수해야 할까?",
        "현재 시장 심리는 어떤가요?",
        "리스크 관리 전략을 알려주세요",
        "분기말 효과를 고려한 전략은?"
    ]
    
    cols = st.columns(3)
    for idx, question in enumerate(quick_questions):
        col_idx = idx % 3
        with cols[col_idx]:
            if st.button(f"💭 {question}", key=f"quick_q_{idx}", use_container_width=True):
                st.session_state.pending_question = question
    
    # 채팅 입력
    user_input = st.chat_input("💬 질문을 입력하세요...", key="chat_input")
    
    # 빠른 질문 처리
    if 'pending_question' in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
    
    if user_input:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # AI 응답 생성
        with st.spinner(f"{ai_engine}가 답변을 생성하고 있습니다..."):
            if ai_engine == "🤖 OpenAI" and OPENAI_ENABLED:
                # OpenAI 챗봇
                try:
                    # 시스템 메시지 구성
                    system_message = f"""
당신은 S&P 500 상위 30개 종목의 Anchored VWAP 분석 전문가입니다.
현재 분기: {quarter_start.year} Q{quarter_num}
분석 대상: 시가총액 상위 30개 종목

사용자의 질문에 대해:
1. 데이터 기반으로 정확하게 답변
2. 구체적인 숫자와 종목명 제시
3. 실행 가능한 조언 제공
4. 간결하고 명확하게 설명

답변은 한국어로 작성하세요.
"""
                    
                    # 대화 히스토리 구성 (최근 10개만)
                    messages = [{"role": "system", "content": system_message}]
                    messages.extend(st.session_state.chat_history[-10:])
                    
                    response = OPENAI_CLIENT.chat.completions.create(
                        model=OPENAI_MODEL_CHAT,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    ai_response = response.choices[0].message.content
                    
                except Exception as e:
                    ai_response = f"❌ OpenAI 응답 생성 실패: {str(e)}"
            
            elif ai_engine == "🧠 Gemini" and GEMINI_ENABLED:
                # Gemini 챗봇
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # 대화 컨텍스트 구성
                    context = f"""
당신은 S&P 500 상위 30개 종목의 Anchored VWAP 분석 전문가입니다.

현재 분석 상황:
- 분기: {quarter_start.year} Q{quarter_num}
- VWAP 위 종목: {len(above_vwap_stocks)}개
- VWAP 아래 종목: {len(below_vwap_stocks)}개
- TOP 5 추천: {', '.join(above_vwap_stocks.head(5)['Ticker'].tolist())}

사용자의 질문에 대해:
1. 현재 분석 데이터를 참고하여 답변
2. 구체적이고 실용적인 조언 제공
3. 투자 리스크를 반드시 언급
4. 간결하고 명확하게 설명

답변은 한국어로 작성하세요.
"""
                    
                    # 대화 히스토리를 하나의 프롬프트로 구성
                    conversation = context + "\n\n"
                    for msg in st.session_state.chat_history[-10:]:
                        if msg["role"] == "user":
                            conversation += f"\n사용자: {msg['content']}\n"
                        elif msg["role"] == "assistant":
                            conversation += f"\n어시스턴트: {msg['content']}\n"
                    
                    response = model.generate_content(conversation)
                    ai_response = response.text
                    
                except Exception as e:
                    ai_response = f"❌ Gemini 응답 생성 실패: {str(e)}"
            
            else:
                ai_response = "❌ 선택한 AI 엔진이 비활성화되어 있습니다."
        
        # AI 응답 추가
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })
        
        # 페이지 새로고침으로 대화 표시
        st.rerun()
    
    # 대화 내보내기
    if st.session_state.chat_history:
        st.markdown("---")
        
        # 대화 내역 텍스트 생성
        chat_text = f"# AI 챗팅 대화 내역\n\n"
        chat_text += f"**일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        chat_text += f"**AI 엔진**: {ai_engine}\n"
        chat_text += f"**분기**: {quarter_start.year} Q{quarter_num}\n\n"
        chat_text += "---\n\n"
        
        for idx, message in enumerate(st.session_state.chat_history, 1):
            if message["role"] == "user":
                chat_text += f"## 👤 사용자 (메시지 {idx})\n{message['content']}\n\n"
            elif message["role"] == "assistant":
                chat_text += f"## 🤖 AI 응답 (메시지 {idx})\n{message['content']}\n\n"
        
        st.download_button(
            label="💾 대화 내역 저장",
            data=chat_text,
            file_name=f"AI_Chat_History_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

with tab8:
    st.header("📋 투자 전략 가이드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "VWAP 위 종목",
            f"{len(above_vwap_stocks)}개",
            f"{len(above_vwap_stocks)/len(df_results)*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "강력 매수 (80점↑)",
            f"{len(above_vwap_stocks[above_vwap_stocks['Buy_Signal_Score'] >= 80])}개"
        )
    
    with col3:
        st.metric(
            "양호 매수 (60점↑)",
            f"{len(above_vwap_stocks[above_vwap_stocks['Buy_Signal_Score'] >= 60])}개"
        )
    
    st.markdown("---")
    
    st.markdown("""
    ### 💡 투자 전략
    
    **1. 💚 강력 매수 (80점 이상)**
    - 현재가가 Anchored VWAP 위에서 안정적
    - 즉시 매수 검토 가능
    - 단, VWAP +5% 이상이면 눌림목 대기 권장
    
    **2. 💛 눌림목 대기 (60-80점)**
    - 기본적으로 좋은 신호
    - VWAP 근처까지 조정 시 매수
    - 손절라인: VWAP -2% 이탈 시
    
    **3. 💙 보통 (60점 미만)**
    - 추가 확인 필요
    - 다른 기술적 지표와 병행 분석
    
    **4. ⚠️ VWAP 아래 종목**
    - 매수 비추천
    - VWAP 돌파 확인 후 재검토
    
    **5. 📊 펀더멘털 체크포인트**
    - PEG Ratio < 1: 성장 대비 저평가
    - ROE > 15%: 우수한 수익성
    - 부채비율 < 100%: 안정적 재무구조
    - 월가 컨센서스 '매수' 이상 권장
    """)
    
    st.markdown("---")
    
    immediate_buy = above_vwap_stocks[above_vwap_stocks['Buy_Signal_Score'] >= 80]
    if not immediate_buy.empty:
        st.success(f"**🎯 즉시 매수 검토:** {', '.join(immediate_buy['Ticker'].tolist())}")
    
    wait_for_dip = above_vwap_stocks[
        (above_vwap_stocks['Buy_Signal_Score'] >= 60) &
        (above_vwap_stocks['Buy_Signal_Score'] < 80)
    ]
    if not wait_for_dip.empty:
        st.warning(f"**💡 눌림목 대기:** {', '.join(wait_for_dip['Ticker'].tolist())}")
    
    if not below_vwap_stocks.empty:
        st.error(f"**⚠️ 매수 비추천:** {', '.join(below_vwap_stocks['Ticker'].tolist())}")

# 푸터
st.markdown("---")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("데이터 출처: Yahoo Finance | 분석 기준: Anchored VWAP | AI: OpenAI GPT-4, Google Gemini")
