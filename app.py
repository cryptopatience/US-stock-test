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
# OpenAI 설정
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

# Gemini 설정
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

# ==================== 로그인 시스템 ====================
def check_password():
    """비밀번호 확인 및 로그인 상태 관리"""
    if st.session_state.get('password_correct', False):
        return True
    
    st.title("🔒 Anchored VWAP 분석 시스템 로그인")
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
    try:
        if not openai_client:
            return "⚠️ OpenAI API 키가 설정되지 않았습니다. secrets.toml에 OPENAI_API_KEY를 추가해주세요."
        
        prompt = f"""
당신은 세계 최고 수준의 퀀트 애널리스트이자 투자 전문가입니다.
다음 S&P 500 시총 상위 30개 종목의 Anchored VWAP 분석 데이터를 바탕으로 종합적인 시장 분석을 제공해주세요.

# 시장 데이터
{json.dumps(market_data, ensure_ascii=False, indent=2)}

# 분석 요구사항

## 1. 현재 시장 상태 분석 (Market Overview)
- VWAP 기준 시장 강도 평가
- 섹터별 강약 분석
- 전반적인 시장 심리 및 트렌드

## 2. 매수 추천 종목 (Top Buy Recommendations)
- 즉시 매수 가능 종목 (강력 매수)
- 눌림목 대기 후 매수 종목
- 각 종목별 구체적인 매수 이유와 근거

## 3. 매도/관망 추천 종목 (Sell/Hold Recommendations)
- 매도 검토 종목
- 관망 추천 종목
- 리스크 요인

## 4. 매수/매도 타이밍 전략
- 진입 시점 (Entry Points)
- 손절 라인 (Stop Loss)
- 목표가 (Target Price)
- 리스크 관리 전략

## 5. 단기 전망 (1-3개월)
- 예상 시나리오
- 주요 모니터링 지표
- 위험 요인

## 6. 장기 전망 (6-12개월)
- 구조적 트렌드
- 장기 투자 전략
- 포트폴리오 구성 제안

한글로 전문적이고 구체적으로 작성해주세요.
각 섹션을 명확히 구분하고, 데이터 기반의 정량적 분석과 정성적 인사이트를 균형있게 제시하세요.
"""
        
        # OpenAI 최신 API (v1.0+)
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "당신은 세계 최고의 퀀트 애널리스트이자 투자 전문가입니다. 데이터 기반의 정확하고 실용적인 투자 인사이트를 제공합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ OpenAI 분석 오류: {str(e)}\n\n힌트: openai 라이브러리 버전을 확인하세요. pip install --upgrade openai"

def get_gemini_market_analysis(market_data):
    """Gemini AI를 활용한 시장 종합 분석"""
    try:
        if not GEMINI_API_KEY:
            return "⚠️ Gemini API 키가 설정되지 않았습니다. secrets.toml에 GEMINI_API_KEY를 추가해주세요."
        
        # gemini-2.0-flash-exp 모델 사용
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
당신은 글로벌 투자은행의 수석 애널리스트입니다.
다음 S&P 500 시총 상위 30개 종목의 Anchored VWAP 분석 데이터를 바탕으로 심층적인 시장 분석을 제공해주세요.

# 시장 데이터
{json.dumps(market_data, ensure_ascii=False, indent=2)}

# 분석 프레임워크

## 1. 현재 시장 상태 진단 (Current Market Diagnosis)
- VWAP 기반 시장 구조 분석
- 섹터 로테이션 패턴
- 시장 참여자 행동 분석 (기관 vs 개인)

## 2. 매수 기회 발굴 (Buy Opportunities)
**즉시 매수 (Immediate Buy)**
- 종목명과 구체적 근거
- 예상 수익률
- 리스크/보상 비율

**전략적 매수 대기 (Strategic Buy on Dip)**
- 매수 대기 가격대
- 트리거 조건

## 3. 리스크 관리 (Risk Management)
**매도 고려 종목**
- 약세 전환 징후 종목
- 리스크 요인

**포지션 축소 고려**
- 과열 구간 종목

## 4. 타이밍 전략 (Timing Strategy)
**단기 트레이딩 (1-4주)**
- 진입/청산 시그널
- 데이 트레이딩 vs 스윙 전략

**중기 투자 (1-3개월)**
- 포지션 빌딩 전략
- 분할 매수/매도 계획

## 5. 시나리오 분석
**Bull Case (강세 시나리오 60%)**
- 트리거 이벤트
- 수혜 종목

**Base Case (중립 시나리오 30%)**
- 예상 흐름

**Bear Case (약세 시나리오 10%)**
- 위험 신호
- 방어 전략

## 6. 장기 투자 전략 (6-12개월)
- 구조적 성장 스토리
- 핵심 보유 종목
- 포트폴리오 최적 구성비

한글로 작성하되, 월스트리트 리서치 리포트 수준의 깊이있는 분석을 제공하세요.
정량적 데이터와 정성적 판단을 조화롭게 제시하고, 실행 가능한 액션 플랜을 포함하세요.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ Gemini 분석 오류: {str(e)}\n\n힌트: 모델명이 'gemini-2.0-flash-exp' 인지 확인하세요."

def get_openai_stock_analysis(ticker, stock_data, fundamental_data):
    """OpenAI를 활용한 개별 종목 분석"""
    try:
        if not openai_client:
            return "⚠️ OpenAI API 키가 설정되지 않았습니다."
        
        prompt = f"""
당신은 세계 최고의 주식 애널리스트입니다.
다음 종목에 대한 심층 분석을 제공해주세요.

# 종목: {ticker}

## 기술적 분석 데이터
{json.dumps(stock_data, ensure_ascii=False, indent=2)}

## 펀더멘털 데이터
{json.dumps(fundamental_data, ensure_ascii=False, indent=2)}

# 분석 요구사항

## 1. 종목 개요 및 현재 상태
- 비즈니스 모델 핵심 요약
- 현재 주가 수준 평가
- VWAP 기준 기술적 위치

## 2. 투자 의견 (Buy/Hold/Sell)
- 명확한 투자 의견과 근거
- 신뢰도 수준 (High/Medium/Low)

## 3. 매수/매도 전략
**매수 시나리오**
- 적정 매수 가격대
- 분할 매수 전략
- 매수 후 홀딩 기간

**매도 시나리오**
- 목표 수익률
- 손절 라인
- 부분 익절 전략

## 4. 리스크 분석
- 주요 리스크 요인 3가지
- 리스크 완화 전략
- 최악의 시나리오 대응

## 5. 밸류에이션 분석
- 현재 밸류에이션 수준 (저평가/적정/고평가)
- 목표 주가 산출 근거
- 유사 기업 비교

## 6. 투자 타임라인
**단기 (1-3개월)**
- 주요 모니터링 지표
- 예상 주가 레인지

**장기 (6-12개월)**
- 성장 동력
- 구조적 강점

한글로 전문적이고 실용적인 분석을 제공하세요.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "당신은 세계 최고의 주식 애널리스트입니다. 정확하고 실행 가능한 투자 인사이트를 제공합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ OpenAI 분석 오류: {str(e)}"

def get_gemini_stock_analysis(ticker, stock_data, fundamental_data):
    """Gemini AI를 활용한 개별 종목 분석"""
    try:
        if not GEMINI_API_KEY:
            return "⚠️ Gemini API 키가 설정되지 않았습니다."
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
당신은 월스트리트 톱티어 투자은행의 수석 애널리스트입니다.
다음 종목에 대한 프로페셔널 리서치 리포트를 작성하세요.

# 종목: {ticker}

## 기술적 분석 데이터
{json.dumps(stock_data, ensure_ascii=False, indent=2)}

## 펀더멘털 데이터
{json.dumps(fundamental_data, ensure_ascii=False, indent=2)}

# 리서치 리포트 구성

## Executive Summary
- 투자 의견: BUY / HOLD / SELL
- 목표 주가
- 상승/하락 여력
- 핵심 투자 포인트 3가지

## 1. 비즈니스 & 산업 분석
- 핵심 사업 구조
- 경쟁 우위 요소
- 산업 내 포지셔닝

## 2. 재무 분석
**수익성**
- 마진 분석
- ROE/ROIC 평가

**성장성**
- 매출/이익 성장 트렌드
- 향후 성장 동력

**재무 건전성**
- 부채 수준
- 현금 흐름

## 3. 밸류에이션
- 멀티플 분석 (PER, PEG, PBR)
- 동종 업체 대비 비교
- DCF/목표주가 산출

## 4. 기술적 분석
**VWAP 분석**
- 현재 포지션
- 지지/저항 레벨

**모멘텀 지표**
- 추세 강도
- 거래량 패턴

## 5. 투자 전략
**매수 전략**
- 최적 진입 가격
- 포지션 사이징
- 분할 매수 플랜

**리스크 관리**
- 손절선 설정
- 헤지 전략
- 포트폴리오 비중

## 6. 시나리오 분석
**상승 시나리오 (Upside Case)**
- 트리거 이벤트
- 목표 수익률

**하락 시나리오 (Downside Case)**
- 리스크 요인
- 방어 전략

## 7. 투자 타임라인
**단기 (1-3개월)**
- 주요 이벤트/지표
- 전술적 트레이딩

**중장기 (6-12개월)**
- 구조적 테마
- 전략적 홀딩

## 8. 액션 플랜
- 즉시 실행 가능한 구체적 행동 지침
- 모니터링 체크리스트

한글로 작성하되, 골드만삭스/모건스탠리 수준의 리서치 퀄리티를 유지하세요.
정량적 근거와 정성적 판단을 균형있게 제시하고, 실무에 바로 적용 가능한 인사이트를 제공하세요.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ Gemini 분석 오류: {str(e)}"

# ==================== 메인 앱 ====================
st.title("📊 Anchored VWAP 분석 대시보드")
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
    st.markdown("**🤖 AI 분석 상태**")
    if openai_client:
        st.success("✅ OpenAI 연결됨")
    else:
        st.warning("⚠️ OpenAI 미연결")
    
    if GEMINI_API_KEY:
        st.success("✅ Gemini 연결됨")
    else:
        st.warning("⚠️ Gemini 미연결")

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎯 TOP 5 추천",
    "📊 전체 분석",
    "📈 차트",
    "💼 펀더멘털",
    "🤖 OpenAI 분석",
    "🧠 Gemini AI 분석",
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
    st.header("🤖 OpenAI 종합 시장 분석")
    
    if not openai_client:
        st.warning("⚠️ OpenAI API가 연결되지 않았습니다. secrets.toml에 OPENAI_API_KEY를 추가해주세요.")
    else:
        st.info("💡 GPT-4를 활용한 세계 최고 수준의 퀀트 분석을 제공합니다.")
    
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
    st.header("🧠 Gemini AI 종합 시장 분석")
    
    if not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API가 연결되지 않았습니다. secrets.toml에 GEMINI_API_KEY를 추가해주세요.")
    else:
        st.info("💡 Google Gemini Pro를 활용한 심층적인 투자 인사이트를 제공합니다.")
    
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
st.caption("데이터 출처: Yahoo Finance | 분석 기준: Anchored VWAP | AI: OpenAI GPT-4, Google Gemini)
