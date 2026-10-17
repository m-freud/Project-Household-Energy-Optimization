import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
ORIG_DB = ROOT_DIR / "sqlite" / "energy.db"
NEW_DB = ROOT_DIR / "sqlite" / "energy_v2.db"
EXT_DB = ROOT_DIR / "sqlite" / "staging" / "DE_pv_gen.db"

st.set_page_config(page_title="PV replacement browser", layout="wide")


@st.cache_data
def load_pv_gen(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query('SELECT * FROM "pv_gen" ORDER BY period', conn)
    finally:
        conn.close()
    return df.set_index("period")


@st.cache_data
def load_pv_player_ids(db_path: str) -> list[int]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT player_id FROM pv_gen_metadata ORDER BY player_id").fetchall()
    finally:
        conn.close()
    return [int(r[0]) for r in rows]


def to_long(orig: pd.DataFrame, new: pd.DataFrame, orig_ids, new_ids) -> pd.DataFrame:
    parts = []
    for source, df, ids in (("orig", orig, orig_ids), ("new", new, new_ids)):
        for pid in ids:
            parts.append(
                pd.DataFrame(
                    {
                        "period": df.index,
                        "generation": df[str(pid)].to_numpy(),
                        "series": f"{source} {pid}",
                        "source": source,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["period", "generation", "series", "source"]
    )


def line_chart(data: pd.DataFrame, height: int = 400) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("period:Q", title="Period"),
            y=alt.Y("generation:Q", title="PV generation (kW)"),
            color=alt.Color("series:N", title="Series"),
            strokeDash=alt.StrokeDash("source:N", scale=alt.Scale(domain=["orig", "new"], range=[[1, 0], [6, 3]])),
            tooltip=["series", "period", "generation"],
        )
        .properties(height=height)
        .interactive()
    )


orig_df = load_pv_gen(str(ORIG_DB))
new_df = load_pv_gen(str(NEW_DB))
player_ids = load_pv_player_ids(str(ORIG_DB))
n_players = len(player_ids)

st.title("PV generation: original vs. new profiles")

if "pv_idx" not in st.session_state:
    st.session_state.pv_idx = 0


def _step(delta: int) -> None:
    st.session_state.pv_idx = (st.session_state.pv_idx + delta) % n_players


left, mid, right = st.columns([1, 6, 1], vertical_alignment="center")
left.button("◀", on_click=_step, args=(-1,), use_container_width=True)
right.button("▶", on_click=_step, args=(1,), use_container_width=True)
mid.select_slider("PV player", options=list(range(n_players)), format_func=lambda i: str(player_ids[i]), key="pv_idx")

pid = player_ids[st.session_state.pv_idx]
st.altair_chart(line_chart(to_long(orig_df, new_df, [pid], [pid])), use_container_width=True)

st.divider()

st.subheader("Overlay")
c1, c2 = st.columns(2)
orig_sel = c1.multiselect("Original profiles", player_ids, default=[], key="pv_orig_sel")
new_sel = c2.multiselect("New profiles", player_ids, default=[], key="pv_new_sel")

if orig_sel or new_sel:
    st.altair_chart(line_chart(to_long(orig_df, new_df, orig_sel, new_sel), height=500), use_container_width=True)
else:
    st.info("Select profiles above.")

st.divider()

st.subheader("External data: browse days per player")


@st.cache_data
def load_ext_pv(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            'SELECT substr(timestamp_utc, 1, 10) AS date, period, residential1, residential3, '
            "residential4, residential6 FROM pv_gen ORDER BY timestamp_utc",
            conn,
        )
    finally:
        conn.close()
    return df


@st.cache_data
def complete_days(db_path: str, col: str, pool_only: bool) -> list[str]:
    df = load_ext_pv(db_path)[["date", col]].dropna()
    counts = df.groupby("date").size()
    days = counts[counts == 96].index.tolist()
    if pool_only:
        conn = sqlite3.connect(db_path)
        try:
            pool = {r[0] for r in conn.execute("SELECT date FROM pv_gen_pool WHERE player_id = ?", (col,))}
        finally:
            conn.close()
        days = [d for d in days if d in pool]
    return days


def _step_day(key: str, n: int, delta: int) -> None:
    st.session_state[key] = (st.session_state[key] + delta) % n


@st.fragment
def ext_player_view(col: str, pool_only: bool) -> None:
    days = complete_days(str(EXT_DB), col, pool_only)
    key = f"ext_idx_{col}"
    st.session_state.setdefault(key, 0)
    if st.session_state[key] >= len(days):
        st.session_state[key] = 0
    st.markdown(f"**{col}** ({len(days)} days)")
    b1, b2, b3 = st.columns([1, 1, 4], vertical_alignment="center")
    b1.button("◀", key=f"prev_{col}", on_click=_step_day, args=(key, len(days), -1), use_container_width=True)
    b2.button("▶", key=f"next_{col}", on_click=_step_day, args=(key, len(days), 1), use_container_width=True)
    b3.select_slider("Day", options=list(range(len(days))), format_func=lambda i: days[i], key=key, label_visibility="collapsed")
    day = days[st.session_state[key]]
    data = load_ext_pv(str(EXT_DB))
    data = data[data["date"] == day][["period", col]].rename(columns={col: "generation"})
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(x=alt.X("period:Q", title="Period"), y=alt.Y("generation:Q", title="PV generation (kW)"))
        .properties(height=300, title=day)
    )
    st.altair_chart(chart, use_container_width=True)


ext_cols = ["residential1", "residential3", "residential4", "residential6"]
pool_only = st.checkbox("Only days in pv_gen_pool", value=True)
row1 = st.columns(2)
row2 = st.columns(2)
for cell, col in zip(row1 + row2, ext_cols):
    with cell:
        ext_player_view(col, pool_only)

st.divider()
st.subheader("Data availability timeline")

avail_rows = []
for col in ext_cols:
    pool_days = set(complete_days(str(EXT_DB), col, True))
    for d in complete_days(str(EXT_DB), col, False):
        avail_rows.append({"player": col, "date": d, "status": "in pv_gen_pool" if d in pool_days else "available"})
avail = pd.DataFrame(avail_rows)
avail["date"] = pd.to_datetime(avail["date"])
avail["end"] = avail["date"] + pd.Timedelta(days=1)

st.altair_chart(
    alt.Chart(avail)
    .mark_bar()
    .encode(
        x=alt.X("date:T", title="Date"),
        x2="end:T",
        y=alt.Y("player:N", title=None, sort=ext_cols),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["available", "in pv_gen_pool"], range=["#c9ced6", "#e8590c"]),
            title=None,
        ),
        tooltip=["player", "date:T", "status"],
    )
    .properties(height=180),
    use_container_width=True,
)
