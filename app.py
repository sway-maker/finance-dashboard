import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import plotly.express as px
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time

# --- 1. 頁面配置 ---
st.set_page_config(
    page_title="全球財經儀表板",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 樣式調整 (CSS) & 自動更新腳本 (JS) ---
auto_refresh_script = """
    <script>
        var timeLeft = 300000;
        setTimeout(function(){
            window.location.reload(1);
        }, timeLeft);
    </script>
"""

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 1rem;} 
            .stRadio { margin-bottom: 2rem; }
            div[data-testid="column"] { padding: 0.5rem; }
            
            /* 讓右上角的更新時間漂亮一點 */
            .last-updated {
                text-align: right;
                font-size: 0.8rem;
                color: gray;
                margin-bottom: 1rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(auto_refresh_script, unsafe_allow_html=True)

# --- 3. 核心邏輯層 (Data Fetching) ---

class DataFetcher:
    @staticmethod
    def fetch_indices():
        """獲取全球主要大盤指數"""
        tickers = {
            '台灣加權 (TWSE)': '^TWII',
            'S&P 500': '^GSPC',
            '納斯達克 (Nasdaq)': '^IXIC',
            '道瓊工業 (DJI)': '^DJI',
            '費城半導體 (SOX)': '^SOX',
            '日經 225 (Nikkei)': '^N225'
        }
        return DataFetcher._fetch_yahoo_data(tickers, 'Index')

    @staticmethod
    def fetch_commodities():
        tickers = {
            '黃金 (Gold)': 'GC=F',
            '白銀 (Silver)': 'SI=F',
            '銅 (Copper)': 'HG=F',
            '紐約輕原油 (Oil)': 'CL=F',
            '天然氣 (Gas)': 'NG=F',
            '黃豆 (Soybean)': 'ZS=F',
            '玉米 (Corn)': 'ZC=F',
            '小麥 (Wheat)': 'ZW=F'
        }
        return DataFetcher._fetch_yahoo_data(tickers, 'Commodity')

    @staticmethod
    def _fetch_yahoo_data(tickers, type_label):
        data_list = []
        try:
            hist_data = yf.download(list(tickers.values()), period="1y", progress=False, auto_adjust=False, threads=False)
            
            if isinstance(hist_data.columns, pd.MultiIndex):
                closes = hist_data['Close']
            else:
                closes = hist_data['Close'] if 'Close' in hist_data else hist_data

            for name, ticker in tickers.items():
                if ticker in closes.columns:
                    series = closes[ticker].dropna()
                    for date, price in series.items():
                        data_list.append({
                            'Date': date,
                            'Currency': name,
                            'Code': ticker,
                            'Price': price,
                            'Type': type_label
                        })
        except Exception as e:
            print(f"Error: {e}")
            pass
            
        df = pd.DataFrame(data_list)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df

    @staticmethod
    def fetch_currency_rates():
        url_base = "https://rate.bot.com.tw"
        try:
            resp = requests.get(f"{url_base}/xrt", timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            rows = soup.find('table', title='牌告匯率').find('tbody').find_all('tr')
            currencies = []
            for row in rows:
                link = row.find('td', class_='currency').find('div', class_='visible-phone').text.strip()
                code = link.split(' ')[1].replace('(', '').replace(')', '')
                name = row.find('div', class_='visible-phone').text.strip()
                currencies.append((code, name))
        except Exception:
            return pd.DataFrame()

        def fetch_single(code, name):
            target_url = f"{url_base}/xrt/quote/l6m/{code}"
            try:
                r = requests.get(target_url, timeout=10)
                s = BeautifulSoup(r.text, 'html.parser')
                table = s.find('table', class_='table-striped')
                if not table: return []
                
                data = []
                for tr in table.find('tbody').find_all('tr'):
                    tds = tr.find_all('td')
                    date = tds[0].text.strip()
                    spot_buy = pd.to_numeric(tds[4].text.strip(), errors='coerce')
                    if pd.isna(spot_buy):
                        spot_buy = pd.to_numeric(tds[2].text.strip(), errors='coerce')

                    data.append({
                        'Date': date,
                        'Currency': name,
                        'Code': code,
                        'Price': spot_buy,
                        'Type': 'Forex'
                    })
                return data
            except:
                return []

        all_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_single, code, name) for code, name in currencies]
            for f in futures:
                res = f.result()
                if res: all_data.extend(res)
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
        return df

@st.cache_data(ttl=300, show_spinner="正在同步最新市場數據...")
def load_all_data():
    df_indices = DataFetcher.fetch_indices()
    df_forex = DataFetcher.fetch_currency_rates()
    df_comm = DataFetcher.fetch_commodities()
    
    df_all = pd.concat([df_indices, df_forex, df_comm], ignore_index=True)
    if not df_all.empty:
        df_all = df_all.sort_values('Date')
    return df_all

# --- 4. 繪圖組件 ---

def plot_small_chart(df, title, color_hex):
    """繪製市場總覽的小型圖表"""
    fig = px.line(df, x='Date', y='Price', render_mode='svg')
    fig.update_traces(line_color=color_hex, line_width=2)
    
    fig.update_layout(
        title_text="",  
        xaxis_title=None, 
        yaxis_title=None,
        margin=dict(l=0, r=0, t=0, b=20), 
        height=250,
        showlegend=False,
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(
            showgrid=False, 
            showticklabels=True, 
            tickformat='%m/%d',
            nticks=5
        ),
        yaxis=dict(showgrid=True, gridcolor='whitesmoke', showticklabels=True),
        dragmode='pan'
    )
    return fig

def render_interactive_page(df, page_title):
    """匯率與原物料的詳細頁面"""
    c1, c2 = st.columns([3, 1])
    with c1:
        st.header(page_title)
    with c2:
        st.markdown(f'<div class="last-updated">最後更新: {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
    
    available_items = df['Currency'].unique()
    default_items = [available_items[0]] if len(available_items) > 0 else None
    
    selected_items = st.multiselect("選擇比較項目", available_items, default=default_items)
    
    if not selected_items:
        st.info("請選擇至少一個項目。")
        return

    final_df = df[df['Currency'].isin(selected_items)]
    st.markdown("---")

    cols = st.columns(len(selected_items))
    for idx, item in enumerate(selected_items):
        item_data = final_df[final_df['Currency'] == item].sort_values('Date')
        if len(item_data) >= 1:
            curr = item_data.iloc[-1]['Price']
            delta_str = None
            if len(item_data) >= 2:
                prev = item_data.iloc[-2]['Price']
                delta = curr - prev
                pct = (delta / prev) * 100
                delta_str = f"{delta:+.2f} ({pct:+.2f}%)"
            with cols[idx % 4]:
                st.metric(item, f"{curr:,.2f}", delta_str)

    fig = px.line(final_df, x='Date', y='Price', color='Currency', 
                 color_discrete_sequence=px.colors.qualitative.Bold, render_mode='svg')
    fig.update_layout(
        height=550, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        dragmode='pan'
    )
    fig.update_traces(line_width=2.5)
    
    config = {
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'toImageButtonOptions': {'format': 'png', 'scale': 2}
    }
    st.plotly_chart(fig, use_container_width=True, config=config)

# --- 5. 主程式 ---

def main():
    df_all = load_all_data()
    if df_all.empty:
        st.error("無法連接數據源。")
        return

    # Sidebar
    st.sidebar.subheader("瀏覽模式")
    page_options = ["市場總覽", "各國匯率", "原物料"]
    selected_page = st.sidebar.radio("頁面選擇", page_options, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("時間維度")
    time_range = st.sidebar.slider("天數", 7, 365, 90)

    # Date Filter
    end_date = df_all['Date'].max()
    start_date = end_date - timedelta(days=time_range)
    date_mask = (df_all['Date'] >= start_date) & (df_all['Date'] <= end_date)

    if selected_page == "市場總覽":
        col_header, col_time = st.columns([3, 1])
        with col_header:
            st.header("全球市場總覽")
            st.caption("主要指數儀表板")
        with col_time:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(f'<div class="last-updated">最後更新: {current_time}<br><span style="font-size:0.7em">(每 5 分鐘自動刷新)</span></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        df_idx = df_all[(df_all['Type'] == 'Index') & date_mask]
        
        target_order = [
            '台灣加權 (TWSE)', 'S&P 500', '納斯達克 (Nasdaq)',
            '道瓊工業 (DJI)', '費城半導體 (SOX)', '日經 225 (Nikkei)'
        ]
        
        cols = st.columns(3)
        colors = ['#2980B9', '#27AE60', '#C0392B', '#8E44AD', '#D35400', '#2C3E50']
        
        for i, ticker_name in enumerate(target_order):
            sub_df = df_idx[df_idx['Currency'] == ticker_name].sort_values('Date')
            
            if not sub_df.empty:
                curr = sub_df.iloc[-1]['Price']
                prev = sub_df.iloc[-2]['Price'] if len(sub_df) > 1 else curr
                delta = curr - prev
                pct = (delta / prev) * 100
                
                with cols[i % 3]:
                    st.metric(
                        label=ticker_name, 
                        value=f"{curr:,.2f}", 
                        delta=f"{delta:+.2f} ({pct:+.2f}%)"
                    )
                    chart_fig = plot_small_chart(sub_df, ticker_name, colors[i % len(colors)])
                    st.plotly_chart(chart_fig, use_container_width=True, config={'displayModeBar': False})
            else:
                with cols[i % 3]:
                    st.warning(f"{ticker_name}: 無數據")

    elif selected_page == "各國匯率":
        df_forex = df_all[(df_all['Type'] == 'Forex') & date_mask]
        render_interactive_page(df_forex, "各國匯率走勢")

    elif selected_page == "原物料":
        df_comm = df_all[(df_all['Type'] == 'Commodity') & date_mask]
        render_interactive_page(df_comm, "國際原物料")

if __name__ == "__main__":
    main()
