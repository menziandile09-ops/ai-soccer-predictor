import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Soccer Predictor", page_icon="⚽")
st.title("⚽ AI Soccer Prediction - All Leagues")

LEAGUES = {
    "Premier League": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "La Liga": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "Serie A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "Bundesliga": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "Ligue 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "Championship": "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
    "Serie B": "https://www.football-data.co.uk/mmz4281/2526/I2.csv",
    "2. Bundesliga": "https://www.football-data.co.uk/mmz4281/2526/D2.csv",
    "Ligue 2": "https://www.football-data.co.uk/mmz4281/2526/F2.csv",
    "Segunda": "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",
    "Turkey Super Lig": "https://www.football-data.co.uk/mmz4281/2526/T1.csv",
    "Scotland Prem": "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
}

@st.cache_data
def load_all():
    dfs = []
    for league, url in LEAGUES.items():
        try:
            d = pd.read_csv(url)
            d = d[['HomeTeam','AwayTeam','FTHG','FTAG','FTR']].dropna()
            d['League'] = league
            dfs.append(d)
        except:
            pass
    return pd.concat(dfs)

df = load_all()
st.success(f"Loaded {len(df)} matches from {df['League'].nunique()} leagues")

elo = {}
def get_elo(t): return elo.get(t, 1500)
for _, r in df.iterrows():
    eh, ea = get_elo(r['HomeTeam']), get_elo(r['AwayTeam'])
    exp_h = 1/(1+10**((ea-eh)/400))
    score = 1 if r['FTR']=='H' else 0.5 if r['FTR']=='D' else 0
    elo[r['HomeTeam']] = eh + 20*(score-exp_h)
    elo[r['AwayTeam']] = ea + 20*((1-score)-(1-exp_h))

df['elo_diff'] = df.apply(lambda r: get_elo(r['HomeTeam'])-get_elo(r['AwayTeam']), axis=1)
df['home_goals'] = df.groupby('HomeTeam')['FTHG'].transform('mean')
df['away_goals'] = df.groupby('AwayTeam')['FTAG'].transform('mean')

model = LogisticRegression(max_iter=1000)
model.fit(df[['elo_diff','home_goals','away_goals']], df['FTR'])

league = st.selectbox("Choose League", sorted(df['League'].unique()))
teams = sorted(df[df['League']==league]['HomeTeam'].unique())
home = st.selectbox("Home Team", teams)
away = st.selectbox("Away Team", teams, index=1)

if st.button("PREDICT", use_container_width=True):
    f = [[get_elo(home)-get_elo(away),
          df[df['HomeTeam']==home]['FTHG'].mean(),
          df[df['AwayTeam']==away]['FTAG'].mean()]]
    f = np.nan_to_num(f, nan=1.2)
    probs = model.predict_proba(f)[0]
    d = dict(zip(model.classes_, probs))
    h, dr, a = d.get('H',0)*100, d.get('D',0)*100, d.get('A',0)*100
    exp_g = f[0][1]+f[0][2]
    st.write(f"Home Win: {h:.1f}% | Draw: {dr:.1f}% | Away Win: {a:.1f}%")
    st.write(f"Expected Goals: {exp_g:.2f}")
    if h > 55: pick = f"HOME WIN - {home}"
    elif a > 55: pick = f"AWAY WIN - {away}"
    elif exp_g > 2.6: pick = "OVER 2.5 GOALS"
    elif exp_g < 2.0: pick = "UNDER 2.5 GOALS"
    else: pick = "DOUBLE CHANCE 1X / 12 - No clear edge"
    st.success(f"AI PICK: {pick}")
    # BTTS - Both Teams To Score
    home_xg = f[0][1]
    away_xg = f[0][2]
    p_home_scores = 1 - np.exp(-home_xg)
    p_away_scores = 1 - np.exp(-away_xg)
    btts_yes = p_home_scores * p_away_scores * 100
    btts_no = 100 - btts_yes
    st.write(f"BTTS YES: {btts_yes:.1f}% | BTTS NO: {btts_no:.1f}%")
    if btts_yes > 55:
        st.success("BTTS PICK: YES")
    elif btts_no > 55:
        st.success("BTTS PICK: NO")
    else:
        st.info("BTTS PICK: Not clear - skip")
from bookings import show_bookings
from highest_half import show_highest_half
