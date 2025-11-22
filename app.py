import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta

st.set_page_config(page_title="Carteira V70 Final", layout="wide")

# --- GUIA DE USO ---
with st.expander("📘 Guia Rápido", expanded=False):
    st.markdown("""
    1.  **Ajuste de Preço:** Na barra lateral, digite o Ticker e o Preço. Ao dar **ENTER**, ele salva.
    2.  **Gráfico:** Clique na linha da tabela para ver o gráfico detalhado.
    3.  **Ajuste de Split:** Se o triângulo estiver fora do lugar, use o Fator Multiplicador na barra lateral.
    """)

st.title("📊 Dashboard Pro (Completo)")

# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================
def converter_numero_br(valor):
    try:
        if pd.isna(valor): return 0.0
        if isinstance(valor, (int, float)): return float(valor)
        valor = str(valor).strip().replace('R$', '').strip()
        if '.' in valor and ',' in valor: valor = valor.replace('.', '').replace(',', '.')
        elif ',' in valor: valor = valor.replace(',', '.')
        return float(valor)
    except: return 0.0

def classificar_arca_com_emoji(ticker):
    ticker = ticker.upper().strip()
    etfs_internacionais = ['IVVB11', 'WRLD11', 'EURP11', 'NASD11', 'SPXI11', 'XINA11', 'GOLD11', 'BBSD11', 'HASH11', 'COIN11', 'HODL11', 'QBTC11', 'ETH11', 'BITO11', 'DEFI11', 'WEB311']
    units_acoes = ['TAEE11', 'SAPR11', 'KLBN11', 'ALUP11', 'SANB11', 'BPAC11', 'ITUB11', 'BBDC11', 'ENGI11', 'TIET11', 'CPLE11']

    if ticker in etfs_internacionais: return 'Internacional', '🌎'
    if ticker.endswith('34') or ticker.endswith('33') or ticker.endswith('32'): return 'Internacional', '🌎'
    if ticker in units_acoes: return 'Ações Brasil', '🇧🇷'
    if ticker.endswith('11') or ticker.endswith('11B'): return 'Real Estate (FIIs)', '🏢'
    return 'Ações Brasil', '🇧🇷'

# ==========================================
# 2. LEITURA
# ==========================================
@st.cache_data(ttl=600)
def carregar_negociacao(arquivo):
    try:
        arquivo.seek(0)
        if arquivo.name.lower().endswith('.csv'):
            try: df_bruto = pd.read_csv(arquivo, sep=';', encoding='latin1', on_bad_lines='skip')
            except: 
                arquivo.seek(0)
                df_bruto = pd.read_csv(arquivo, sep=',', encoding='utf-8', on_bad_lines='skip')
        else: 
            df_bruto = pd.read_excel(arquivo, header=None)

        linha_inicio = -1
        for i, row in df_bruto.iterrows():
            texto = row.astype(str).str.cat().lower()
            if 'quantidade' in texto and ('preço' in texto or 'preco' in texto):
                linha_inicio = i
                break
        
        if linha_inicio != -1:
            if arquivo.name.lower().endswith('.csv'):
                arquivo.seek(0)
                df = pd.read_csv(arquivo, sep=';', encoding='latin1', skiprows=linha_inicio)
            else: 
                df = pd.read_excel(arquivo, header=linha_inicio)
        else: 
            df = df_bruto

        mapa = {}
        for col in df.columns:
            c = str(col).strip().lower()
            if 'data' in c: mapa[col] = 'Data'
            elif 'compra' in c or 'tipo' in c: mapa[col] = 'Tipo'
            elif 'papel' in c or 'código' in c or 'ativo' in c or 'ticker' in c: mapa[col] = 'Ticker'
            elif 'quantidade' in c: mapa[col] = 'Qtd'
            elif 'preço' in c or 'preco' in c: mapa[col] = 'Preco'

        df = df.rename(columns=mapa)
        colunas_essenciais = ['Data', 'Tipo', 'Ticker', 'Qtd', 'Preco']
        if not all(c in df.columns for c in colunas_essenciais): return pd.DataFrame()
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce').dt.normalize()
        return df.dropna(subset=['Data'])
    except: return pd.DataFrame()

# ==========================================
# 3. PROCESSAMENTO
# ==========================================
@st.cache_data(ttl=600)
def processar_dados(df):
    carteira = {}
    hist = []
    df = df.sort_values('Data')

    for idx, row in df.iterrows():
        try:
            raw_ticker = str(row['Ticker']).strip().upper()
            if raw_ticker.endswith('F') and len(raw_ticker) > 5: ticker = raw_ticker[:-1]
            else: ticker = raw_ticker
            if len(ticker) < 3: continue

            if ticker not in carteira:
                carteira[ticker] = {'qtd': 0, 'pm': 0, 'st': 0, 'data_ini': row['Data'], 'custo_venda_acumulado': 0}
            
            d = carteira[ticker]
            tipo = str(row['Tipo']).lower()
            qtd = converter_numero_br(row['Qtd'])
            preco = converter_numero_br(row['Preco'])
            total = qtd * preco
            
            if 'c' in tipo or 'compra' in tipo:
                custo_ant = d['qtd'] * d['pm']
                novo_custo = total
                nova_qtd = d['qtd'] + qtd
                if nova_qtd > 0: d['pm'] = (custo_ant + novo_custo) / nova_qtd
                d['qtd'] = nova_qtd
                
            elif 'v' in tipo or 'venda' in tipo:
                if qtd > 0:
                    lucro = (preco - d['pm']) * qtd
                    custo_venda = d['pm'] * qtd
                    d['custo_venda_acumulado'] += custo_venda
                    d['st'] += lucro
                    d['qtd'] -= qtd
                    if d['qtd'] < 0.001: d['qtd'] = 0; d['pm'] = 0
            
            hist.append({'Data': row['Data'], 'Ticker': ticker, 'Qtd_Pos': d['qtd']})
        except: continue
    return carteira, pd.DataFrame(hist)

@st.cache_data(ttl=3600)
def calcular_dividendos_detalhado(carteira_keys, df_hist):
    res_total_por_ativo = {}
    lista_detalhada = []
    progresso = st.progress(0, text="Buscando proventos...")
    total_tickers = len([t for t in carteira_keys if "TESOURO" not in t and "IPCA" not in t])
    contador = 0

    for t_br in carteira_keys:
        if "TESOURO" in t_br or "IPCA" in t_br: continue
        t_sa = f"{t_br}.SA"
        contador += 1
        progresso.progress(min(contador / max(total_tickers, 1), 1.0), text=f"Analisando: {t_br}")
        try:
            ticker_obj = yf.Ticker(t_sa)
            divs = ticker_obj.dividends
            if not divs.empty:
                divs.index = divs.index.tz_localize(None).normalize()
                data_min = df_hist[df_hist['Ticker'] == t_br]['Data'].min()
                divs_filtrados = divs[divs.index >= data_min]
                soma_ativo = 0
                for data_div, valor_div in divs_filtrados.items():
                    data_referencia = data_div - timedelta(days=1)
                    movs_ate_data = df_hist[(df_hist['Ticker'] == t_br) & (df_hist['Data'] <= data_referencia)]
                    if not movs_ate_data.empty:
                        qtd_na_mao = movs_ate_data.iloc[-1]['Qtd_Pos']
                        if qtd_na_mao > 0:
                            valor_recebido = qtd_na_mao * valor_div
                            soma_ativo += valor_recebido
                            lista_detalhada.append({'Data': data_div.strftime('%d/%m/%Y'), 'Ativo': t_br, 'Valor Unit.': valor_div, 'Qtd na Data': qtd_na_mao, 'Total Recebido': valor_recebido})
                res_total_por_ativo[t_br] = soma_ativo
        except: pass
    progresso.empty()
    return res_total_por_ativo, pd.DataFrame(lista_detalhada)

@st.cache_data(ttl=600)
def get_precos(carteira):
    t_list = [f"{t}.SA" for t, d in carteira.items() if d['qtd'] > 0 and "TESOURO" not in t]
    if not t_list: return {}
    try:
        d = yf.download(t_list, period="5d", auto_adjust=False, progress=False)['Close']
        v = d.ffill().iloc[-1]
        res = {}
        if len(t_list) == 1: res[t_list[0]] = float(v)
        else:
            for t in t_list:
                if t in v and not pd.isna(v[t]): res[t] = float(v[t])
        return res
    except: pass
    return {}

# --- BENCHMARKS ---
@st.cache_data(ttl=3600)
def get_benchmarks(data_inicio):
    indices = {'Ibovespa 🇧🇷': '^BVSP', 'S&P 500 🇺🇸': '^GSPC', 'IFIX (ETF) 🏢': 'XFIX11.SA', 'CDI (Via LFTS11) 🐢': 'LFTS11.SA', 'Dólar 💵': 'BRL=X'}
    resultados = {}
    try:
        tickers_lista = list(indices.values())
        dados = yf.download(tickers_lista, start=data_inicio, progress=False)['Close']
        if not dados.empty:
            if dados.index.tz is not None: dados.index = dados.index.tz_localize(None)
            for nome, ticker in indices.items():
                try:
                    serie = pd.Series()
                    if isinstance(dados.columns, pd.MultiIndex):
                         if ticker in dados.columns: serie = dados[ticker]
                    else:
                         if ticker in dados.columns: serie = dados[ticker]
                         elif len(tickers_lista) == 1: serie = dados
                    serie = serie.dropna()
                    if not serie.empty:
                        val_ini = float(serie.iloc[0])
                        val_fim = float(serie.iloc[-1])
                        if val_ini > 0: resultados[nome] = (val_fim - val_ini) / val_ini
                except: pass
    except: pass
    return resultados

# ==========================================
# 4. GRÁFICO POP-UP (CANDLESTICK + TRIÂNGULOS)
# ==========================================
@st.dialog("Análise de Execução", width="large")
def mostrar_grafico_popup(ticker, df_completo, fator_ajuste=1.0):
    ticker_clean = ticker.strip().upper()
    if ticker_clean.endswith('F') and len(ticker_clean) > 5:
        ticker_clean = ticker_clean[:-1]

    st.subheader(f"Histórico: {ticker_clean}")
    
    df_ativo = df_completo[df_completo['Ticker'].astype(str).str.contains(ticker_clean, case=False)].copy()
    if df_ativo.empty:
        st.warning("Sem dados de negociação.")
        return

    df_ativo['Data'] = pd.to_datetime(df_ativo['Data']).dt.normalize()
    df_ativo['Qtd'] = df_ativo['Qtd'].apply(converter_numero_br)
    df_ativo['Preco'] = df_ativo['Preco'].apply(converter_numero_br)
    df_ativo['Tipo Norm'] = df_ativo['Tipo'].apply(lambda x: 'Compra' if 'c' in str(x).lower() else 'Venda')
    
    hist_price = pd.DataFrame()
    try:
        ticker_sa = f"{ticker_clean}.SA" if not "TESOURO" in ticker_clean else ticker_clean
        dados_yahoo = yf.download(ticker_sa, period="10y", progress=False, auto_adjust=False)
        if not dados_yahoo.empty:
            if 'Close' in dados_yahoo:
                close_data = dados_yahoo['Close']
                if close_data.index.tz is not None:
                    close_data.index = close_data.index.tz_localize(None).normalize()
                else:
                    close_data.index = close_data.index.normalize()
                data_min = df_ativo['Data'].min() - timedelta(days=30)
                hist_price = dados_yahoo[dados_yahoo.index >= data_min]
                if isinstance(hist_price.columns, pd.MultiIndex):
                    hist_price.columns = hist_price.columns.get_level_values(0)
    except Exception as e: pass

    fig = go.Figure()
    if not hist_price.empty:
        fig.add_trace(go.Candlestick(
            x=hist_price.index,
            open=hist_price['Open'], high=hist_price['High'],
            low=hist_price['Low'], close=hist_price['Close'],
            name='Mercado',
            hovertext=[f"Data: {d.strftime('%d/%m/%Y')}<br>Abertura: R$ {o:.2f}<br>Fech: R$ {c:.2f}" for d,o,c in zip(hist_price.index, hist_price['Open'], hist_price['Close'])],
            hoverinfo="text"
        ))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    cores = {'Compra': '#00CC96', 'Venda': '#EF553B'}
    simbolos = {'Compra': 'triangle-up', 'Venda': 'triangle-down'}
    
    # Pega preço atual para calcular rentabilidade no tooltip
    try:
        preco_hoje = hist_price['Close'].iloc[-1]
    except: preco_hoje = 0

    for tipo in ['Compra', 'Venda']:
        df_t = df_ativo[df_ativo['Tipo Norm'] == tipo]
        if not df_t.empty:
            preco_plot = df_t['Preco'] * fator_ajuste 
            
            rent_texts = []
            for p in df_t['Preco']:
                if preco_hoje > 0:
                    diff = (preco_hoje - p) / p
                    sinal = "+" if diff > 0 else ""
                    cor_res = "#90EE90" if diff > 0 else "#FFB6C1"
                    rent_texts.append(f"<span style='color:{cor_res}'><b>{sinal}{diff:.2%}</b></span>")
                else: rent_texts.append("-")

            fig.add_trace(go.Scatter(
                x=df_t['Data'], y=preco_plot, mode='markers', name=f"Sua {tipo}",
                marker=dict(color='white', size=12, symbol=simbolos[tipo], line=dict(width=2, color=cores[tipo])),
                customdata=list(zip(df_t['Preco'], rent_texts, df_t['Qtd'])),
                hovertemplate=f"<b>{tipo} Real</b><br>Data: %{{x|%d/%m/%Y}}<br>Preço Pago: R$ %{{customdata[0]:.2f}}<br>Resultado Hoje: %{{customdata[1]}}<br>Qtd: %{{customdata[2]}}<extra></extra>",
            ))

    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=20, r=20, t=40, b=20), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. EXECUÇÃO
# ==========================================
if 'ajustes_preco' not in st.session_state: st.session_state['ajustes_preco'] = {}
if 'ajustes_pm' not in st.session_state: st.session_state['ajustes_pm'] = {}
if 'fator_split' not in st.session_state: st.session_state['fator_split'] = 1.0

with st.sidebar:
    st.header("⚙️ Ajustes")
    rf_manual = st.number_input("Renda Fixa (R$)", 0.0, step=100.0)
    st.divider()
    st.markdown("### 📉 Ajuste Gráfico (Splits)")
    st.session_state['fator_split'] = st.number_input("Fator Multiplicador", value=1.0, step=0.1, format="%.2f")

    st.divider()
    tab1, tab2 = st.tabs(["Cotação", "Custo (PM)"])
    with tab1:
        with st.form(key='form_cot'):
            ap = st.text_input("Ativo (Cot)").upper().strip()
            np = st.number_input("R$ Atual", 0.0, format="%.2f")
            if st.form_submit_button("Salvar"): st.session_state['ajustes_preco'][ap] = np; st.rerun()
        if st.session_state['ajustes_preco']:
            if st.button("Limpar Cot"): st.session_state['ajustes_preco'] = {}; st.rerun()
    with tab2:
        with st.form(key='form_pm'):
            apm = st.text_input("Ativo (PM)").upper().strip()
            npm = st.number_input("R$ Custo", 0.0, format="%.2f")
            if st.form_submit_button("Salvar"): st.session_state['ajustes_pm'][apm] = npm; st.rerun()
        if st.session_state['ajustes_pm']:
            if st.button("Limpar PM"): st.session_state['ajustes_pm'] = {}; st.rerun()

arquivo = st.file_uploader("Upload Negociação", type=['xlsx', 'csv'])

if arquivo:
    df_raw = carregar_negociacao(arquivo)

    if not df_raw.empty:
        carteira, df_hist = processar_dados(df_raw)
        divs_dict, df_extrato_divs = calcular_dividendos_detalhado(list(carteira.keys()), df_hist)
        with st.spinner("Buscando preços..."):
            precos_dict = get_precos(carteira)
        
        ajustes_p = st.session_state['ajustes_preco']
        ajustes_pm = st.session_state['ajustes_pm']
        final = []
        ativos_custo = []

        for t, d in carteira.items():
            qtd = d['qtd']
            pm = ajustes_pm.get(t, d['pm'])
            st_val = d['st']
            custo_venda = d.get('custo_venda_acumulado', 0)
            div = divs_dict.get(t, 0)
            
            curr = pm
            fonte = "Custo"
            if f"{t}.SA" in precos_dict: curr = precos_dict[f"{t}.SA"]; fonte="Online"
            if t in ajustes_p: curr = ajustes_p[t]; fonte="Manual"
            if fonte == "Custo" and qtd > 0 and "TESOURO" not in t and "IPCA" not in t: ativos_custo.append(t)
            
            val_hj = qtd * curr
            investido_hj = qtd * pm
            lucro_aberto_cotacao = val_hj - investido_hj
            
            rent_total_decimal = 0
            var_cotacao_decimal = 0
            yoc_decimal = 0
            rent_st_decimal = 0
            
            if investido_hj > 0:
                var_cotacao_decimal = (val_hj - investido_hj) / investido_hj
                rent_total_decimal = (lucro_aberto_cotacao + div) / investido_hj
                yoc_decimal = div / investido_hj
            if custo_venda > 0: rent_st_decimal = st_val / custo_venda
            
            classe, emoji = classificar_arca_com_emoji(t)

            if qtd > 0 or abs(st_val) > 0.01 or div > 0:
                final.append({
                    'Ativo': t, 'Nome Visual': f"{emoji} {t}", 'Emoji': emoji, 'Classe': classe,
                    'Qtd': qtd, 'PM': pm, 'Preço Atual': curr, 'Fonte Preço': fonte,
                    'Valor Hoje': val_hj, 'Investido': investido_hj,
                    'Dividendos': div, 'Lucro Swing Trade': st_val,
                    'Rent. (%)': rent_total_decimal, 'Var. Cotação (%)': var_cotacao_decimal,
                    'Rent. Swing (%)': rent_st_decimal, 'YoC (%)': yoc_decimal
                })
        
        df_final = pd.DataFrame(final)
        
        if not df_final.empty:
            df_ativos = df_final[df_final['Qtd'] > 0].copy()
            df_encerrados = df_final[ (df_final['Qtd'] == 0) & (abs(df_final['Lucro Swing Trade']) > 0.01) ].copy()
            
            patrimonio_total = df_ativos['Valor Hoje'].sum() + rf_manual
            investido_total = df_ativos['Investido'].sum()
            lucro_valorizacao = df_ativos['Valor Hoje'].sum() - investido_total
            divs_total = df_final['Dividendos'].sum()
            st_total = df_final['Lucro Swing Trade'].sum()
            resultado_geral = lucro_valorizacao + divs_total + st_total
            
            perc_geral = 0
            yoc_global = 0
            if investido_total > 0: 
                perc_geral = (resultado_geral / investido_total) * 100
                yoc_global = (divs_total / investido_total) * 100

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Patrimônio Total", f"R$ {patrimonio_total:,.2f}", help="Ações + FIIs + Renda Fixa")
            c2.metric("Lucro Total Geral", f"R$ {resultado_geral:,.2f}", delta=f"{perc_geral:.2f}% (Retorno)", help="Valorização + Dividendos + Vendas")
            perc_val = 0
            if investido_total > 0: perc_val = (lucro_valorizacao / investido_total) * 100
            c3.metric("Valorização (Cotação)", f"R$ {lucro_valorizacao:,.2f}", delta=f"{perc_val:.2f}%", delta_color="normal")
            c4.metric("Dividendos", f"R$ {divs_total:,.2f}", delta=f"{yoc_global:.2f}% (YoC)")
            c5.metric("Swing Trade", f"R$ {st_total:,.2f}")

            if ativos_custo: st.warning(f"⚠️ Usando preço de custo para: {', '.join(ativos_custo)}. Ajuste na barra lateral.")

            st.divider()
            
            # TREEMAP
            if not df_ativos.empty:
                st.subheader("🗺️ Mapa")
                df_ativos['Label'] = df_ativos['Valor Hoje'].apply(lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                df_ativos['Root'] = 'Carteira'
                fig_tree = px.treemap(df_ativos, path=['Root', 'Nome Visual'], values='Valor Hoje', color='Rent. (%)', 
                                      color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                                      custom_data=['Label', 'Rent. (%)', 'Classe'])
                fig_tree.update_traces(texttemplate="%{label}<br>%{customdata[0]}", hovertemplate="<b>%{label}</b><br>%{customdata[0]}<br>%{customdata[1]:.2%}")
                fig_tree.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=500)
                st.plotly_chart(fig_tree, use_container_width=True)

            st.divider()

            # BENCHMARKS
            st.subheader("🏆 Comparativo de Mercado")
            try:
                data_inicio = df_raw['Data'].min()
                benchmarks = get_benchmarks(data_inicio)
                if benchmarks:
                    cols_b = st.columns(len(benchmarks))
                    for i, (nome, rent) in enumerate(benchmarks.items()):
                        diff = perc_geral - (rent * 100)
                        cols_b[i].metric(nome, f"{rent:.2%}", delta=f"{diff:.2f}% vs {nome}")
            except: pass

            st.divider()

            # ARCA COM RENTABILIDADE (CORRIGIDA AQUI!)
            col_a1, col_a2 = st.columns([1, 2])
            
            # Agrupamento somando Valor e Investido para calcular rentabilidade
            dados_arca = df_ativos.groupby('Classe')[['Valor Hoje', 'Investido']].sum().reset_index()
            dados_arca['Rentabilidade'] = (dados_arca['Valor Hoje'] - dados_arca['Investido']) / dados_arca['Investido']

            if rf_manual > 0:
                nova = pd.DataFrame({
                    'Classe': ['Caixa/RF 💰'], 'Valor Hoje': [rf_manual], 
                    'Investido': [rf_manual], 'Rentabilidade': [0.0]
                })
                dados_arca = pd.concat([dados_arca, nova], ignore_index=True)
            
            with col_a1:
                fig_arca = px.pie(dados_arca, values='Valor Hoje', names='Classe', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_arca, use_container_width=True)
            
            with col_a2:
                dados_arca['%'] = dados_arca['Valor Hoje'] / patrimonio_total
                
                # Estilização da Tabela ARCA
                def style_rent(v):
                    color = '#2ca02c' if v >= 0 else '#d62728'
                    return f'color: {color}; font-weight: bold;'
                
                st.dataframe(
                    dados_arca[['Classe', 'Valor Hoje', '%', 'Rentabilidade']].style
                    .format({'Valor Hoje': 'R$ {:,.2f}', '%': '{:.1%}', 'Rentabilidade': '{:.2%}'})
                    .bar(subset=['%'], color='#5fba7d')
                    .map(style_rent, subset=['Rentabilidade']),
                    use_container_width=True, hide_index=True
                )

            st.divider()

            def style_rentability(v):
                color = '#2ca02c' if v >= 0 else '#d62728'
                return f'color: {color}; font-weight: bold;'

            st.subheader("📋 Carteira (Clique na Linha)")
            if not df_ativos.empty:
                cols = ['Nome Visual', 'Ativo', 'Qtd', 'PM', 'Preço Atual', 'Var. Cotação (%)', 'Rent. (%)', 'YoC (%)', 'Valor Hoje', 'Dividendos']
                
                event = st.dataframe(
                    df_ativos[cols].style.format({
                        'Qtd': '{:g}', 'PM': 'R$ {:.2f}', 'Preço Atual': 'R$ {:.2f}', 'Valor Hoje': 'R$ {:.2f}', 'Dividendos': 'R$ {:.2f}',
                        'Rent. (%)': '{:.2%}', 'Var. Cotação (%)': '{:.2%}', 'YoC (%)': '{:.2%}'
                    })
                    .map(style_rentability, subset=['Rent. (%)', 'Var. Cotação (%)', 'YoC (%)'])
                    .bar(subset=['Valor Hoje'], color='#d1e7dd'),
                    use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
                )
                
                if event.selection.rows:
                    indice_sel = event.selection.rows[0]
                    ativo_clicado = df_ativos.iloc[indice_sel]['Ativo']
                    mostrar_grafico_popup(ativo_clicado, df_raw, st.session_state.get('fator_split', 1.0))
            
            st.divider()
            
            with st.expander("📂 Histórico e Vendas (Swing Trade)"):
                if not df_encerrados.empty:
                    st.dataframe(df_encerrados[['Nome Visual', 'Lucro Swing Trade', 'Dividendos', 'Rent. Swing (%)']].style.format({
                        'Lucro Swing Trade': 'R$ {:.2f}', 'Dividendos': 'R$ {:.2f}', 'Rent. Swing (%)': '{:.2%}'
                    }).map(style_rentability, subset=['Rent. Swing (%)']), use_container_width=True, hide_index=True)
                else: st.info("Nenhuma venda.")
                
            with st.expander("🧾 Extrato Detalhado de Proventos"):
                if not df_extrato_divs.empty:
                    st.dataframe(df_extrato_divs.sort_values('Data', ascending=False).style.format({
                        'Valor Unit.': 'R$ {:.4f}', 'Total Recebido': 'R$ {:.2f}', 'Qtd': '{:g}'
                    }), use_container_width=True)
                else: st.info("Nenhum dividendo encontrado.")

        else: st.warning("Sem dados.")
    else: st.error("Arquivo inválido.")
