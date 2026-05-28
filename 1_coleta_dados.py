"""
GAIE - Generative AI for Engineering
Script 1: Coleta de Dados Reais do Vento Solar

Fontes primárias (dados reais):
  - NOAA SWPC Solar Wind MAG  : Bz, Bt — 7 dias × 1min → ~9750 linhas
  - NOAA SWPC Solar Wind Plasma: velocidade, densidade, temperatura
  - NOAA SWPC KP Index         : índice geomagnético 3-horário
  - NASA DONKI FLR             : eventos de flare solar (B/C/M/X)
  - NASA DONKI GST             : tempestades geomagnéticas (proxy de CME)

Dataset final: dados reais com suplemento sintético para garantir
               diversidade de tempestades e mínimo de 1000 linhas.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Chave NASA (via variável de ambiente ou valor padrão para teste)
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
NASA_BASE_URL = "https://api.nasa.gov/DONKI"

# ─────────────────────────────────────────────────────────────
# BLOCO 1: NOAA SWPC — Dados do Vento Solar em Tempo Real
# ─────────────────────────────────────────────────────────────

def coletar_mag_noaa() -> pd.DataFrame:
    """
    Coleta dados magnéticos do vento solar da NOAA SWPC (últimos 7 dias, resolução 1 minuto).
    Endpoint: https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json
    Retorna: timestamp, bz, campo_bt (e componentes bx, by auxiliares).
    """
    url = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
    try:
        print("  [NOAA] Coletando dados magnéticos (MAG 7-day)...")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Formato: [[header_list], [row1], [row2], ...]
        colunas = data[0]   # ['time_tag', 'bx_gsm', 'by_gsm', 'bz_gsm', 'lon_gsm', 'lat_gsm', 'bt']
        df = pd.DataFrame(data[1:], columns=colunas)
        df = df.rename(columns={"bz_gsm": "bz", "bt": "campo_bt"})
        df["timestamp"] = pd.to_datetime(df["time_tag"])
        for col in ["bz", "campo_bt", "bx_gsm", "by_gsm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["timestamp", "bz", "campo_bt"]].dropna()
        print(f"  ✓ MAG: {len(df):,} registros | {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        return df
    except Exception as e:
        print(f"  ✗ Erro MAG NOAA: {e}")
        return pd.DataFrame()


def coletar_plasma_noaa() -> pd.DataFrame:
    """
    Coleta dados de plasma do vento solar da NOAA SWPC (últimos 7 dias, resolução 1 minuto).
    Endpoint: https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json
    Retorna: timestamp, velocidade_vento, densidade_protons, temperatura.
    """
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
    try:
        print("  [NOAA] Coletando dados de plasma (Plasma 7-day)...")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        colunas = data[0]   # ['time_tag', 'density', 'speed', 'temperature']
        df = pd.DataFrame(data[1:], columns=colunas)
        df = df.rename(columns={
            "density":     "densidade_protons",
            "speed":       "velocidade_vento",
            "temperature": "temperatura",
        })
        df["timestamp"] = pd.to_datetime(df["time_tag"])
        for col in ["densidade_protons", "velocidade_vento", "temperatura"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["timestamp", "velocidade_vento", "densidade_protons", "temperatura"]].dropna()
        print(f"  ✓ Plasma: {len(df):,} registros")
        return df
    except Exception as e:
        print(f"  ✗ Erro Plasma NOAA: {e}")
        return pd.DataFrame()


def coletar_kp_noaa() -> pd.DataFrame:
    """
    Coleta KP Index 3-horário da NOAA SWPC (últimos 7 dias).
    Endpoint: https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
    Formato atual: lista de dicionários (não mais lista de listas).
    """
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    try:
        print("  [NOAA] Coletando KP Index...")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Formato atual: [{"time_tag": "...", "Kp": 2.0, ...}, ...]
        if isinstance(data[0], dict):
            df = pd.DataFrame(data)
            df = df.rename(columns={"time_tag": "timestamp_kp", "Kp": "kp_index"})
        else:
            # Fallback: formato antigo lista-de-listas
            df = pd.DataFrame(data[1:], columns=data[0])
            df = df.rename(columns={"time_tag": "timestamp_kp"})
            kp_col = next((c for c in df.columns if "kp" in c.lower()), df.columns[1])
            df = df.rename(columns={kp_col: "kp_index"})

        df["timestamp_kp"] = pd.to_datetime(df["timestamp_kp"])
        df["kp_index"] = pd.to_numeric(df["kp_index"], errors="coerce")
        df = df[["timestamp_kp", "kp_index"]].dropna()
        print(f"  ✓ KP Index: {len(df)} registros (3-horários) | KP máx={df['kp_index'].max():.1f}")
        return df
    except Exception as e:
        print(f"  ✗ Erro KP NOAA: {e}")
        return pd.DataFrame()


def mesclar_vento_solar(df_mag: pd.DataFrame, df_plasma: pd.DataFrame,
                         df_kp: pd.DataFrame) -> pd.DataFrame:
    """
    Mescla dados magnéticos, de plasma e KP Index num único DataFrame por timestamp.
    - MAG + Plasma: merge_asof por timestamp com tolerância de 2 minutos
    - KP (3-horário): forward-fill para resolução de 1 minuto
    """
    if df_mag.empty or df_plasma.empty:
        return pd.DataFrame()

    # Merge MAG + Plasma no tempo mais próximo (±2 min)
    df = pd.merge_asof(
        df_mag.sort_values("timestamp"),
        df_plasma.sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("2min"),
        direction="nearest",
    )

    # Injeta KP Index por forward-fill (3h → 1min)
    if not df_kp.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_kp.sort_values("timestamp_kp").rename(columns={"timestamp_kp": "timestamp"}),
            on="timestamp",
            direction="backward",
        )
        df["kp_index"] = df["kp_index"].ffill().bfill()
    else:
        # Sem KP disponível: estima a partir de Bz (relação empírica simplificada)
        print("  ⚠ KP não disponível — estimando a partir do Bz")
        df["kp_index"] = np.clip(np.maximum(-df["bz"], 0) * 0.3, 0, 9)

    df = df.dropna(subset=["bz", "velocidade_vento", "densidade_protons"])
    print(f"  ✓ Dataset mesclado: {len(df):,} linhas")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# BLOCO 2: NASA DONKI — Eventos Solares
# ─────────────────────────────────────────────────────────────

def coletar_flares_nasa(inicio: str, fim: str) -> list:
    """
    Coleta eventos de flare solar da API NASA DONKI.
    Retorna lista com beginTime, classType, etc.
    """
    url = f"{NASA_BASE_URL}/FLR"
    try:
        print(f"  [NASA] Coletando flares ({inicio} → {fim})...")
        r = requests.get(url, params={"startDate": inicio, "endDate": fim,
                                       "api_key": NASA_API_KEY}, timeout=20)
        r.raise_for_status()
        flares = r.json()
        print(f"  ✓ Flares: {len(flares)} eventos")
        return flares
    except Exception as e:
        print(f"  ✗ Erro FLR NASA: {e}")
        return []


def coletar_gst_nasa(inicio: str, fim: str) -> list:
    """
    Coleta tempestades geomagnéticas (GST) da API NASA DONKI.
    Usado como proxy para identificar períodos de CME ativo.
    """
    url = f"{NASA_BASE_URL}/GST"
    try:
        print(f"  [NASA] Coletando tempestades GST ({inicio} → {fim})...")
        r = requests.get(url, params={"startDate": inicio, "endDate": fim,
                                       "api_key": NASA_API_KEY}, timeout=20)
        r.raise_for_status()
        gsts = r.json()
        print(f"  ✓ GST: {len(gsts)} tempestades")
        return gsts
    except Exception as e:
        print(f"  ✗ Erro GST NASA: {e}")
        return []


def classe_flare_para_int(classe_str: str) -> int:
    """Converte string de classe de flare para inteiro (0=sem, 1=B, 2=C, 3=M, 4=X)."""
    if not classe_str:
        return 0
    return {"B": 1, "C": 2, "M": 3, "X": 4}.get(str(classe_str).upper()[:1], 0)


def adicionar_eventos_solares(df: pd.DataFrame, flares: list, gsts: list) -> pd.DataFrame:
    """
    Mapeia eventos de flare e tempestade GST para o dataset de séries temporais.
    - flare_classe: classe máxima de flare ativo ±4h de cada ponto
    - cme: 1 se dentro de uma janela de 48h após início de GST (proxy de CME)
    - velocidade_cme: 0 (não disponível sem endpoint CME)
    """
    df = df.copy()
    df["flare_classe"] = 0
    df["cme"] = 0
    df["velocidade_cme"] = 0.0

    ts = df["timestamp"]

    # Adiciona flares
    n_flares_mapeados = 0
    for ev in flares:
        try:
            t_ini = pd.to_datetime(ev.get("beginTime", ""))
            t_fim_raw = ev.get("endTime")
            t_fim = pd.to_datetime(t_fim_raw) if t_fim_raw else t_ini + pd.Timedelta("2h")
            classe = classe_flare_para_int(ev.get("classType", ""))
            if classe == 0:
                continue
            # Janela de influência: início até fim + 4h (partículas chegam atrasadas)
            mask = (ts >= t_ini) & (ts <= t_fim + pd.Timedelta("4h"))
            if mask.any():
                df.loc[mask, "flare_classe"] = df.loc[mask, "flare_classe"].clip(lower=classe)
                n_flares_mapeados += 1
        except Exception:
            continue

    # Adiciona tempestades GST como proxy de CME
    n_gst_mapeados = 0
    for ev in gsts:
        try:
            t_ini = pd.to_datetime(ev.get("startTime", ""))
            # Janela: 24h antes (driver chegando) até 48h depois (período de tempestade)
            mask = (ts >= t_ini - pd.Timedelta("24h")) & (ts <= t_ini + pd.Timedelta("48h"))
            if mask.any():
                df.loc[mask, "cme"] = 1
                n_gst_mapeados += 1
        except Exception:
            continue

    print(f"  ✓ Flares mapeados: {n_flares_mapeados}/{len(flares)} | GST mapeados: {n_gst_mapeados}/{len(gsts)}")
    return df


# ─────────────────────────────────────────────────────────────
# BLOCO 3: Suplemento Sintético para Diversidade de Tempestades
# ─────────────────────────────────────────────────────────────

def kp_para_nivel_g(kp_val: float) -> int:
    """Converte KP Index para nível de tempestade G (NOAA)."""
    if kp_val < 5:   return 0
    elif kp_val < 6: return 1
    elif kp_val < 7: return 2
    elif kp_val < 8: return 3
    elif kp_val < 9: return 4
    else:            return 5


def gerar_suplemento_sintetico(n_amostras: int, seed: int = 99,
                                 forcar_tempestades: bool = True) -> pd.DataFrame:
    """
    Gera amostras sintéticas para cobrir casos de tempestade (G1–G5)
    que podem estar sub-representados nos dados reais recentes.

    Usa distribuições físicas de Borovsky & Denton (2006) e Newell et al. (2008).
    """
    np.random.seed(seed)

    if forcar_tempestades:
        # 40% amostras de tempestade, 60% calmas — garante diversidade
        n_storm = int(n_amostras * 0.4)
        n_calm  = n_amostras - n_storm

        # Condições de tempestade: Bz muito negativo, velocidade alta
        bz_s   = np.random.normal(-18, 8,  n_storm)
        vel_s  = np.random.normal(650,  80, n_storm)
        den_s  = np.random.lognormal(2.8, 0.6, n_storm)
        cme_s  = np.random.binomial(1, 0.6, n_storm)
        flare_s = np.random.choice([2,3,4], n_storm, p=[0.5,0.35,0.15])

        # Condições calmas
        bz_c   = np.random.normal(1,  4, n_calm)
        vel_c  = np.random.normal(380, 50, n_calm)
        den_c  = np.random.lognormal(2.2, 0.7, n_calm)
        cme_c  = np.random.binomial(1, 0.03, n_calm)
        flare_c = np.random.choice([0,1,2], n_calm, p=[0.6,0.3,0.1])

        bz    = np.concatenate([bz_s, bz_c])
        vel   = np.concatenate([vel_s, vel_c])
        den   = np.concatenate([den_s, den_c])
        cme   = np.concatenate([cme_s, cme_c])
        flare = np.concatenate([flare_s, flare_c])
        idx_shuffle = np.random.permutation(n_amostras)
        bz, vel, den, cme, flare = bz[idx_shuffle], vel[idx_shuffle], den[idx_shuffle], cme[idx_shuffle], flare[idx_shuffle]
    else:
        bz    = np.random.normal(0, 5, n_amostras)
        vel   = np.random.normal(450, 100, n_amostras)
        den   = np.random.lognormal(2.3, 0.7, n_amostras)
        cme   = np.random.binomial(1, 0.11, n_amostras)
        flare = np.random.choice([0,1,2,3,4], n_amostras, p=[0.54,0.20,0.14,0.09,0.03])

    bz  = np.clip(bz, -65, 25)
    vel = np.clip(vel, 200, 1100)
    den = np.clip(den, 1, 120)

    # Campo total
    by  = np.random.normal(0, 4, n_amostras)
    bt  = np.sqrt(bz**2 + by**2 + np.random.normal(0,3,n_amostras)**2)
    bt  = np.clip(bt, 1, 85)

    # Pressão dinâmica
    mp  = 1.6726e-27
    pre = np.clip(0.5 * mp * den * 1e6 * (vel * 1e3)**2 * 1e9, 0.4, 45)

    # Temperatura
    tmp = np.clip(np.random.lognormal(4.1, 0.65, n_amostras), 5, 350)

    # KP Index (semi-empírico)
    bz_neg = np.maximum(-bz, 0)
    newell = np.where(bz_neg > 0, (vel**(4/3)) * (bz_neg**(2/3)), 0.0)
    newell_norm = newell / (newell.max() + 1e-6)
    kp = np.clip(
        0.0025*vel + 0.18*bz_neg + 0.012*den + 0.9*cme + 2.0*newell_norm + 0.45*flare
        + np.random.normal(0, 0.55, n_amostras),
        0, 9
    )

    # Timestamps: preenchidos com datas há 1–6 meses (passado recente)
    fim = datetime.now() - timedelta(days=8)  # não sobrepõe dados reais de 7 dias
    ini = fim - timedelta(hours=n_amostras)
    datas = pd.date_range(end=fim, periods=n_amostras, freq="h")

    df = pd.DataFrame({
        "timestamp":         datas,
        "bz":                bz.round(2),
        "campo_bt":          bt.round(2),
        "velocidade_vento":  vel.round(1),
        "densidade_protons": den.round(2),
        "temperatura":       tmp.round(1),
        "cme":               cme.astype(int),
        "velocidade_cme":    0.0,
        "flare_classe":      flare.astype(int),
        "kp_index":          kp.round(2),
        "nivel_g":           np.array([kp_para_nivel_g(k) for k in kp]),
        "pressao_dinamica":  pre.round(3),
        "fonte":             "sintetico",
    })
    return df


# ─────────────────────────────────────────────────────────────
# BLOCO 4: Pipeline Principal
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    print("\n" + "=" * 62)
    print("  GAIE — Script 1: Coleta de Dados Reais + Sintéticos")
    print("=" * 62)

    # ── 1. Dados reais NOAA SWPC ──────────────────────────────────
    print("\n[1/4] NOAA SWPC — Dados reais do vento solar...")
    df_mag    = coletar_mag_noaa()
    df_plasma = coletar_plasma_noaa()
    df_kp     = coletar_kp_noaa()

    df_real = mesclar_vento_solar(df_mag, df_plasma, df_kp)

    # ── 2. Eventos NASA DONKI ─────────────────────────────────────
    print("\n[2/4] NASA DONKI — Eventos solares...")
    fim   = datetime.now().strftime("%Y-%m-%d")
    ini7  = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")   # janela real

    # Usa a mesma janela de 8 dias para coincidir com os dados NOAA reais
    flares = coletar_flares_nasa(ini7, fim)
    gsts   = coletar_gst_nasa(ini7, fim)

    if not df_real.empty:
        # Calcula pressão dinâmica
        mp = 1.6726e-27
        df_real["pressao_dinamica"] = np.clip(
            0.5 * mp * df_real["densidade_protons"] * 1e6 * (df_real["velocidade_vento"] * 1e3)**2 * 1e9,
            0.4, 45
        )
        # Nível G
        df_real["nivel_g"] = df_real["kp_index"].apply(kp_para_nivel_g)
        # Adiciona eventos (flares, GST)
        df_real = adicionar_eventos_solares(df_real, flares, gsts)
        df_real["fonte"] = "real_noaa"

    # ── 3. Suplemento sintético ───────────────────────────────────
    print("\n[3/4] Avaliando necessidade de suplemento sintético...")

    n_real = len(df_real) if not df_real.empty else 0
    n_storms_real = (df_real["nivel_g"] >= 1).sum() if not df_real.empty else 0

    print(f"  Dados reais disponíveis : {n_real:,} linhas")
    print(f"  Eventos de tempestade   : {n_storms_real} linhas (nivel_g >= 1)")

    # Decide quantas linhas sintéticas adicionar
    # - Se dados reais < 2000 linhas → completa para 3000
    # - Sempre garante diversidade de tempestades (mínimo 150 eventos G1+)
    n_sintetico = max(0, 3000 - n_real)
    if n_storms_real < 150:
        n_sintetico = max(n_sintetico, 1500)  # força linhas com tempestades

    if n_sintetico > 0:
        print(f"  Gerando {n_sintetico} linhas sintéticas complementares...")
        df_sint = gerar_suplemento_sintetico(
            n_amostras=n_sintetico,
            forcar_tempestades=(n_storms_real < 150)
        )

        if not df_real.empty:
            # Alinha colunas antes de concatenar
            cols_comuns = [c for c in df_sint.columns if c in df_real.columns]
            df_final = pd.concat([df_real[cols_comuns], df_sint[cols_comuns]], ignore_index=True)
        else:
            df_final = df_sint
    else:
        df_final = df_real
        print("  Dados reais suficientes — sem suplemento necessário")

    # ── 4. Salvar dataset final ───────────────────────────────────
    print("\n[4/4] Salvando dataset final...")
    df_final = df_final.sort_values("timestamp").reset_index(drop=True)
    caminho = os.path.join("data", "solar_wind_dataset.csv")
    df_final.to_csv(caminho, index=False)

    # Sumário
    n_real_final  = (df_final["fonte"] == "real_noaa").sum() if "fonte" in df_final.columns else "?"
    n_sint_final  = (df_final["fonte"] == "sintetico").sum() if "fonte" in df_final.columns else "?"

    print("\n" + "=" * 62)
    print(f"  Dataset salvo: {caminho}")
    print(f"  Linhas totais : {len(df_final):,}")
    print(f"  Colunas       : {df_final.shape[1]}")
    print(f"  Período       : {df_final['timestamp'].min().date()} → {df_final['timestamp'].max().date()}")
    print(f"  Dados reais   : {n_real_final:,} linhas (NOAA SWPC)")
    print(f"  Dados sintétic: {n_sint_final:,} linhas (suplemento)")
    print("\n  Distribuição do Nível G:")
    labels = ["G0 (calmo)", "G1 (menor)", "G2 (moderado)",
              "G3 (forte)", "G4 (severo)", "G5 (extremo)"]
    for g, cnt in df_final["nivel_g"].value_counts().sort_index().items():
        print(f"    {labels[g]}: {cnt:5d} ({cnt/len(df_final)*100:.1f}%)")
    print("\n  Eventos NASA DONKI:")
    print(f"    Flares: {len(flares)} | Tempestades GST: {len(gsts)}")
    print("=" * 62)

    return df_final


if __name__ == "__main__":
    main()
