import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
ORIG_DB = ROOT_DIR / "sqlite" / "energy.db"
NEW_DB = ROOT_DIR / "sqlite" / "energy_v2.db"
N_PLAYERS = 250

st.set_page_config(page_title="Profile replacement browser", layout="wide")


@st.cache_data
def load_base_load(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query('SELECT * FROM "base_load" ORDER BY period', conn)
    finally:
        conn.close()
    return df.set_index("period")


def to_long(orig: pd.DataFrame, new: pd.DataFrame, orig_ids, new_ids) -> pd.DataFrame:
    parts = []
    for source, df, ids in (("orig", orig, orig_ids), ("new", new, new_ids)):
        for pid in ids:
            parts.append(
                pd.DataFrame(
                    {
                        "period": df.index,
                        "load": df[str(pid)].to_numpy(),
                        "series": f"{source} {pid}",
                        "source": source,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["period", "load", "series", "source"]
    )


def line_chart(data: pd.DataFrame, height: int = 400) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("period:Q", title="Period"),
            y=alt.Y("load:Q", title="Load (kW)"),
            color=alt.Color("series:N", title="Series"),
            strokeDash=alt.StrokeDash("source:N", scale=alt.Scale(domain=["orig", "new"], range=[[1, 0], [6, 3]])),
            tooltip=["series", "period", "load"],
        )
        .properties(height=height)
        .interactive()
    )


orig_df = load_base_load(str(ORIG_DB))
new_df = load_base_load(str(NEW_DB))
player_ids = list(range(1, N_PLAYERS + 1))

st.title("Base load: original vs. new profiles")

# Browser per player
if "player" not in st.session_state:
    st.session_state.player = 1


def _step(delta: int) -> None:
    st.session_state.player = (st.session_state.player - 1 + delta) % N_PLAYERS + 1


left, mid, right = st.columns([1, 6, 1], vertical_alignment="center")
left.button("◀", on_click=_step, args=(-1,), use_container_width=True)
right.button("▶", on_click=_step, args=(1,), use_container_width=True)
mid.select_slider("Player", options=player_ids, key="player")

pid = st.session_state.player
st.altair_chart(line_chart(to_long(orig_df, new_df, [pid], [pid])), use_container_width=True)

st.divider()

# Free overlay
st.subheader("Overlay")
c1, c2 = st.columns(2)
orig_sel = c1.multiselect("Original profiles", player_ids, default=[], key="orig_sel")
new_sel = c2.multiselect("New profiles", player_ids, default=[], key="new_sel")

if orig_sel or new_sel:
    st.altair_chart(line_chart(to_long(orig_df, new_df, orig_sel, new_sel), height=500), use_container_width=True)
else:
    st.info("Select profiles above.")
