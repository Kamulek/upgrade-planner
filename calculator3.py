import streamlit as st
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
EXCEL_FILE_NAME = "data.xlsx"
MATERIALS = ["Satin", "Gilded Threads", "Artisan's Vision"]
OPTIMIZATION_COLUMNS = ["Power", "KvK points"]
TARGET_COLUMNS = ["Tier", "Stars"]
UPGRADE_ITEMS = ["Hat", "Necklace", "Coat", "Pants", "Ring", "Cane"]

# --- UTILITIES ---
def format_number(value: int) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except Exception:
        return str(value)


def get_tier_style(tier: str):
    """Return border color and shadow for given tier."""
    color_map = {
        "Green": ("#15a80a", "0 0 10px #15a80a"),
        "Blue": ("#00b7ff", "0 0 10px #00b7ff"),
        "Purple": ("#9300ff", "0 0 10px #9300ff"),
        "Gold": ("#ffc400", "0 0 10px #ffc400"),
    }
    base = (tier.split(" ")[0] if isinstance(tier, str) else "")
    return color_map.get(base, ("#777777", "0 0 5px #777777"))


@st.cache_data
def load_data(file_name: str) -> pd.DataFrame:
    """Load and sanitize Excel data, return cumulative-materials DataFrame."""
    try:
        df = pd.read_excel(file_name, sheet_name=0)
    except FileNotFoundError:
        st.error(f"Error: Excel file '{file_name}' not found.")
        return pd.DataFrame()

    df = df.dropna(how="all").reset_index(drop=True)

    for c in TARGET_COLUMNS:
        if c not in df.columns:
            df[c] = ""
        else:
            df[c] = df[c].astype(str).fillna("")

    expected_numerical = MATERIALS + OPTIMIZATION_COLUMNS
    for col in expected_numerical:
        if col in df.columns:
            if df[col].dtype == object or df[col].dtype == str:
                df[col] = df[col].astype(str).replace(r"\s+", "", regex=True).replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    df["Tier"] = df["Tier"].astype(str)
    df["Stars"] = df["Stars"].astype(str)
    df["Upgrade_Level"] = (df["Tier"].fillna("") + " " + df["Stars"].fillna("")) .str.strip()
    df["Combined_Key"] = df["Upgrade_Level"].str.replace(" ", "_", regex=False)

    df_cumulative = df.copy()
    for mat in MATERIALS:
        df_cumulative[mat] = df_cumulative[mat].cumsum().astype(int)

    return df_cumulative


excel_data = load_data(EXCEL_FILE_NAME)

if excel_data.empty:
    st.set_page_config(layout="wide", page_title="Upgrade Planner")
    st.title("Upgrade Planner")
    st.stop()


@st.cache_data
def get_level_map(df: pd.DataFrame) -> dict:
    return {row["Upgrade_Level"]: idx for idx, row in df.iterrows()}


LEVEL_MAP = get_level_map(excel_data)
MAX_LEVEL_INDEX = len(excel_data) - 1


@st.cache_data
def get_delta_costs(df: pd.DataFrame) -> pd.DataFrame:
    """From cumulative costs compute per-level delta costs and delta optimization columns."""
    delta_df = df.copy()
    for i in range(1, len(df)):
        for mat in MATERIALS:
            delta_df.loc[i, mat] = int(df.loc[i, mat] - df.loc[i - 1, mat])
    for mat in MATERIALS:
        delta_df.loc[0, mat] = int(df.loc[0, mat])

    for col in OPTIMIZATION_COLUMNS:
        absolute = df[col].astype(int)
        delta_df[f"Delta_{col}"] = absolute.diff().fillna(absolute.iloc[0]).astype(int)
        delta_df[col] = absolute

    return delta_df


DELTA_COSTS = get_delta_costs(excel_data)


def calculate_single_target_cost(item: str, target_level: str, current_level: str):
    """Return required materials to raise item from current_level to target_level."""
    start_index = LEVEL_MAP.get(current_level, -1)
    target_index = LEVEL_MAP.get(target_level, -1)

    if target_index == -1 or start_index >= target_index:
        return None, f"Target {target_level} for {item} is invalid or is lower/equal to the current level {current_level}."

    required = DELTA_COSTS.iloc[start_index + 1: target_index + 1][MATERIALS].sum().to_dict()
    required = {k: int(v) for k, v in required.items()}
    return required, None


def render_material_stats(mat: str, required_qty: int, possessed_qty: int, col):
    """Render progress and metrics for a single material. Returns True when missing."""
    required_qty = int(required_qty)
    possessed_qty = int(possessed_qty)
    if required_qty <= 0:
        with col:
            st.markdown(f"**{mat}**")
            st.write("No requirement")
        return False

    missing = max(0, required_qty - possessed_qty)
    progress_percent = int(min(100, (possessed_qty / required_qty) * 100)) if required_qty > 0 else 100

    with col:
        st.markdown(f"**{mat}**")
        st.markdown(f"<div style='font-size:1.2em; font-weight:bold;'>{format_number(possessed_qty)} / {format_number(required_qty)}</div>", unsafe_allow_html=True)
        st.progress(progress_percent)
        delta_label = f"Surplus: +{format_number(possessed_qty - required_qty)}" if missing == 0 else f"Missing: -{format_number(missing)}"
        st.metric(label="Required Amount", value=format_number(required_qty), delta=delta_label, delta_color="inverse" if missing > 0 else "normal")
    return missing > 0


# --- NEW: SEQUENTIAL OPTIMIZATION LOGIC ---
# This function simulates repeatedly picking the single most profitable achievable upgrade,
# applying it (subtracting materials, updating that item's level), and repeating until no more
# achievable upgrades remain. It returns the path, total gain, and remaining materials.

def find_best_sequential_path(current_levels: dict, current_materials: dict, optimization_target: str):
    opt_key = f"Delta_{optimization_target}"
    weights = {"Satin": 1, "Gilded Threads": 10, "Artisan's Vision": 50}

    mats = current_materials.copy()
    levels = current_levels.copy()
    path = []
    total_gain = 0

    while True:
        candidates = []
        for item in UPGRADE_ITEMS:
            cur_level = levels.get(item, "None / Not Acquired")
            cur_index = LEVEL_MAP.get(cur_level, -1)
            next_index = cur_index + 1
            if not (0 <= next_index <= MAX_LEVEL_INDEX):
                continue

            row = DELTA_COSTS.iloc[next_index]
            delta_value = int(row.get(opt_key, 0))
            if delta_value <= 0:
                continue

            cost = {m: int(row[m]) for m in MATERIALS}
            # check affordability given current simulated mats
            if not all(mats.get(m, 0) >= cost[m] for m in MATERIALS):
                continue

            norm_cost = sum(cost[m] * weights.get(m, 1) for m in MATERIALS)
            ratio = delta_value / norm_cost if norm_cost > 0 else 0

            candidates.append({
                "item": item,
                "current_level": cur_level,
                "next_level": row["Upgrade_Level"],
                "delta_value": delta_value,
                "cost": cost,
                "ratio": ratio,
            })

        if not candidates:
            break

        # choose best candidate by ratio, tie-breaker by higher absolute gain
        candidates.sort(key=lambda x: (x["ratio"], x["delta_value"]), reverse=True)
        best = candidates[0]

        # apply best: subtract materials, update level, append to path
        for m in MATERIALS:
            mats[m] -= best["cost"][m]
        levels[best["item"]] = best["next_level"]
        total_gain += best["delta_value"]
        path.append(best)

    return path, total_gain, mats


# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Upgrade Planner")
st.title("Upgrade Level and Material Planner")
st.markdown("---")

# initialize session state
if "possessed_materials" not in st.session_state:
    st.session_state["possessed_materials"] = {m: 0 for m in MATERIALS}
if "current_item_levels" not in st.session_state:
    st.session_state["current_item_levels"] = {i: "None / Not Acquired" for i in UPGRADE_ITEMS}
if "target_upgrades" not in st.session_state:
    st.session_state["target_upgrades"] = []

# 1. Input current materials
st.subheader("1. Current warehouse (materials)")
mat_cols = st.columns(len(MATERIALS))
material_steps = {"Satin": 100, "Gilded Threads": 10, "Artisan's Vision": 10}
for i, material in enumerate(MATERIALS):
    with mat_cols[i]:
        init = st.session_state["possessed_materials"].get(material, 0)
        st.session_state["possessed_materials"][material] = st.number_input(
            f"Amount of {material}:",
            min_value=0,
            value=int(init),
            step=material_steps.get(material, 1),
            key=f"input_mat_{material}"
        )
st.success("Material data saved.")
st.markdown("---")

# 2. Input current item levels
st.subheader("2. Select current upgrade level for each item")
valid_levels = excel_data["Upgrade_Level"].unique().tolist()
sorted_levels = sorted(valid_levels, key=lambda x: LEVEL_MAP.get(x, len(excel_data)))
level_options = ["None / Not Acquired"] + sorted_levels

item_cols = st.columns(len(UPGRADE_ITEMS))
for i, item in enumerate(UPGRADE_ITEMS):
    with item_cols[i]:
        init_level = st.session_state["current_item_levels"].get(item, "None / Not Acquired")
        index = level_options.index(init_level) if init_level in level_options else 0
        sel = st.selectbox(f"{item}:", level_options, index=index, key=f"level_{item}")
        st.session_state["current_item_levels"][item] = sel

st.success("Current item levels saved.")
st.markdown("---")

# 3. Planner
st.header("3. Upgrade Planning and Optimization")
mode = st.radio(
    "Select planning mode:",
    [
        "1. Calculate maximum possible level for one item",
        "2. Check cost to reach specific levels (Multi-Target)",
        "3. Find optimal sequential upgrade path"
    ],
    index=0
)

# Mode 1 (fixed missing calculation and table)
if mode.startswith("1."):
    st.subheader("Maximum Possible Upgrade Level")
    item_to_check = st.selectbox("Select item to check:", UPGRADE_ITEMS, key='max_item')
    current_level = st.session_state["current_item_levels"][item_to_check]

    if st.button("Find Max Level", key='btn_find_max'):
        mats = st.session_state["possessed_materials"].copy()
        start_index = LEVEL_MAP.get(current_level, -1)
        next_index = start_index + 1
        final_level = current_level
        total_cost = {m: 0 for m in MATERIALS}
        path = []
        last_impossible = None
        last_impossible_remaining = None

        while 0 <= next_index <= MAX_LEVEL_INDEX:
            row = DELTA_COSTS.iloc[next_index]
            level_name = row["Upgrade_Level"]
            cost = row[MATERIALS].astype(int)
            can_afford = True
            for m in MATERIALS:
                if mats.get(m, 0) < int(cost[m]):
                    can_afford = False
                    last_impossible = (level_name, cost.to_dict())
                    last_impossible_remaining = mats.copy()
                    break
            if not can_afford:
                break
            path.append((level_name, {m: int(cost[m]) for m in MATERIALS}))
            for m in MATERIALS:
                mats[m] -= int(cost[m])
                total_cost[m] += int(cost[m])
            final_level = level_name
            next_index += 1

        if path:
            st.markdown("**Upgrade Path:**")
            html = "<div style='display:flex;gap:8px;overflow:auto;padding:8px;background:#111;border-radius:8px;'>"
            for idx, (lvl, costs) in enumerate(path):
                border_color, box_shadow = get_tier_style(excel_data.iloc[LEVEL_MAP[lvl]]["Tier"])
                cost_lines = "<br>".join([f"{format_number(v)} {k}" for k, v in costs.items() if v > 0]) or "No cost"
                html += f"<div style='min-width:150px;padding:10px;border-radius:8px;border:3px solid {border_color};box-shadow:{box_shadow};background:rgba(0,0,0,0.3);color:white;text-align:center;'><div style='font-weight:bold;'>{lvl}</div><div style='font-size:0.85em;color:gray;margin-top:6px;'>{cost_lines}</div></div>"
                if idx < len(path) - 1:
                    html += "<div style='align-self:center;color:#ccc;margin:0 6px;'>➡️</div>"

            if last_impossible:
                lvl_imp, costs_imp = last_impossible
                rem = last_impossible_remaining or st.session_state['possessed_materials']
                # build a small table: Required | Remaining | Missing
                rows = ""
                for m in MATERIALS:
                    req = int(costs_imp.get(m, 0))
                    rem_val = int(rem.get(m, 0))
                    missing = max(0, req - rem_val)
                    rows += f"<tr><td style='padding:6px;border-bottom:1px solid #333;'>{m}</td><td style='padding:6px;border-bottom:1px solid #333;text-align:right;'>{format_number(req)}</td><td style='padding:6px;border-bottom:1px solid #333;text-align:right;'>{format_number(rem_val)}</td><td style='padding:6px;border-bottom:1px solid #333;text-align:right;color:#ff5555;'>{format_number(missing)}</td></tr>"

                table_html = f"<table style='width:320px;border-collapse:collapse;margin-left:10px;'><thead><tr><th style='text-align:left;padding:6px;border-bottom:2px solid #444;'>Material</th><th style='text-align:right;padding:6px;border-bottom:2px solid #444;'>Required</th><th style='text-align:right;padding:6px;border-bottom:2px solid #444;'>Remaining</th><th style='text-align:right;padding:6px;border-bottom:2px solid #444;'>Missing</th></tr></thead><tbody>{rows}</tbody></table>"

                html += f"<div style='align-self:center;color:#ff5555;margin:0 6px;'>❌</div>"
                html += f"<div style='min-width:260px;padding:10px;border-radius:8px;border:3px dashed #444;background:rgba(100,0,0,0.12);color:white;text-align:center;'><div style='font-weight:bold;margin-bottom:8px;'>{lvl_imp} (Too Costly)</div>{table_html}</div>"

            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

            st.markdown("---")
            st.success(f"MAXIMUM ACHIEVABLE LEVEL for {item_to_check} is: **{final_level}**")
            st.markdown("---")
            col_cost, col_remaining = st.columns(2)
            with col_cost:
                st.markdown("**Total Path Cost:**")
                for m in MATERIALS:
                    if total_cost[m] > 0:
                        st.metric(m, format_number(total_cost[m]))
            with col_remaining:
                st.markdown("**Remaining Materials:**")
                for m in MATERIALS:
                    st.metric(m, format_number(mats[m]))
        else:
            if last_impossible:
                rem = last_impossible_remaining or st.session_state['possessed_materials']
                rows = ""
                for m in MATERIALS:
                    req = int(last_impossible[1].get(m, 0))
                    rem_val = int(rem.get(m, 0))
                    missing = max(0, req - rem_val)
                    rows += f"<tr><td style='padding:6px;border-bottom:1px solid #333;'>{m}</td><td style='padding:6px;border-bottom:1px solid #333;text-align:right;'>{format_number(req)}</td><td style='padding:6px;border-bottom:1px solid #333;text-align:right;'>{format_number(rem_val)}</td><td style='padding:6px;border-bottom:1px solid #333;text-align:right;color:#ff5555;'>{format_number(missing)}</td></tr>"
                table_html = f"<table style='width:320px;border-collapse:collapse;'><thead><tr><th style='text-align:left;padding:6px;border-bottom:2px solid #444;'>Material</th><th style='text-align:right;padding:6px;border-bottom:2px solid #444;'>Required</th><th style='text-align:right;padding:6px;border-bottom:2px solid #444;'>Remaining</th><th style='text-align:right;padding:6px;border-bottom:2px solid #444;'>Missing</th></tr></thead><tbody>{rows}</tbody></table>"
                st.warning("")
                st.markdown(f"<div style='padding:10px;border-radius:8px;border:3px dashed #444;background:rgba(100,0,0,0.12);color:white;'>{table_html}</div>", unsafe_allow_html=True)
            else:
                st.info("No upgrades available to perform.")

# Mode 2 (unchanged) - Multi-target costs
elif mode.startswith("2."):
    st.subheader("Cost to reach target levels")
    cost_mode = st.radio(
        "How should material costs be calculated?",
        ["Combined (Total cost for ALL targets)", "Individual (Check cost for each target)"],
        index=0,
        horizontal=True,
        key='cost_calculation_mode'
    )

    def add_target():
        if len(level_options) > 1:
            st.session_state['target_upgrades'].append({'item': UPGRADE_ITEMS[0], 'target_level': level_options[1]})
        else:
            st.session_state['target_upgrades'].append({'item': UPGRADE_ITEMS[0], 'target_level': 'None / Not Acquired'})

    def remove_target(index):
        st.session_state['target_upgrades'].pop(index)

    if not st.session_state['target_upgrades']:
        add_target()

    targets_to_remove = []
    for i, target in enumerate(st.session_state['target_upgrades']):
        item_in_session = target['item']
        current_level = st.session_state['current_item_levels'][item_in_session]
        current_index = LEVEL_MAP.get(current_level, -1)
        target_options_filtered = [lvl for lvl in sorted_levels if LEVEL_MAP.get(lvl, -1) > current_index]
        if not target_options_filtered:
            target_options_filtered = ["Max Level Reached"]

        target_level_index = 0
        if target['target_level'] in target_options_filtered:
            target_level_index = target_options_filtered.index(target['target_level'])

        col_item, col_target, col_remove = st.columns([0.3, 0.5, 0.2])

        with col_item:
            item_selected = st.selectbox(
                "Item:",
                UPGRADE_ITEMS,
                index=UPGRADE_ITEMS.index(target['item']),
                key=f'target_item_{i}'
            )
            st.session_state['target_upgrades'][i]['item'] = item_selected
            st.caption(f"Current: **{st.session_state['current_item_levels'][item_selected]}**")

        with col_target:
            level_selected = st.selectbox(
                "Target Level:",
                target_options_filtered,
                index=target_level_index,
                key=f'target_level_{i}'
            )
            st.session_state['target_upgrades'][i]['target_level'] = level_selected

        with col_remove:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Remove ➖", key=f'remove_btn_{i}', use_container_width=True):
                targets_to_remove.append(i)

        if cost_mode == "Individual (Check cost for each target)":
            item = st.session_state['target_upgrades'][i]['item']
            target_level = st.session_state['target_upgrades'][i]['target_level']
            current_level = st.session_state['current_item_levels'][item]

            if target_level != "Max Level Reached" and current_level != target_level:
                required_cost, error = calculate_single_target_cost(item, target_level, current_level)

                if error:
                    st.error(error)
                elif required_cost:
                    st.markdown(f"**Required for {item} to reach {target_level}**")
                    required_cols = st.columns(len(MATERIALS))
                    target_can_afford = True
                    for j, mat in enumerate(MATERIALS):
                        required_qty = required_cost[mat]
                        possessed_qty = st.session_state['possessed_materials'].get(mat, 0)
                        if render_material_stats(mat, required_qty, possessed_qty, required_cols[j]):
                            target_can_afford = False

                    if target_can_afford:
                        st.success(f"You can afford to upgrade {item} to {target_level} individually.")
                    else:
                        st.warning(f"You need more materials to upgrade {item} to {target_level}.")

        st.markdown("---")

    if targets_to_remove:
        for index in sorted(targets_to_remove, reverse=True):
            remove_target(index)
        st.rerun()

    st.button("Add another target ➕", on_click=add_target, key='add_target_btn')

    if cost_mode == "Combined (Total cost for ALL targets)":
        if st.button("Calculate Total Cost (Combined Mode)", key='btn_calc_total_cost', type="primary"):
            total_required_cost = {m: 0 for m in MATERIALS}
            valid_targets = True

            for target in st.session_state['target_upgrades']:
                item = target['item']
                target_level = target['target_level']
                current_level = st.session_state['current_item_levels'][item]

                if target_level == "Max Level Reached" or current_level == target_level:
                    continue

                required_cost, error = calculate_single_target_cost(item, target_level, current_level)

                if error:
                    valid_targets = False
                    st.error(f"Error for {item}: {error}")
                    break

                for mat in MATERIALS:
                    total_required_cost[mat] += required_cost[mat]

            if valid_targets:
                st.subheader("Summary of Total Required Materials (Combined)")
                can_afford = True

                required_cols = st.columns(len(MATERIALS))
                for i, mat in enumerate(MATERIALS):
                    required_qty = total_required_cost[mat]
                    possessed_qty = st.session_state['possessed_materials'].get(mat, 0)

                    if render_material_stats(mat, required_qty, possessed_qty, required_cols[i]):
                        can_afford = False

                st.markdown("---")
                if can_afford:
                    st.success(f"You have enough materials for all selected upgrades!")
                else:
                    st.warning(f"You need more materials for the combined target levels.")

# Mode 3: Optimization (fixed sequential logic)
elif mode.startswith("3."):
    st.subheader("Optimal Sequential Upgrade Path (Focus on KvK/Power points)")

    optimization_goal = st.selectbox(
        "Select Optimization Goal:",
        OPTIMIZATION_COLUMNS,
        index=OPTIMIZATION_COLUMNS.index('KvK points'),
        key='optimization_target_select'
    )

    st.markdown("---")

    if st.button(f"Find Best Sequential Path for {optimization_goal}", key='btn_find_optimal', type="primary"):
        path, total_gain, remaining = find_best_sequential_path(
            st.session_state['current_item_levels'],
            st.session_state['possessed_materials'],
            optimization_goal
        )

        if not path:
            st.info("No achievable upgrades found with your current materials or all items have reached maximum level.")
        else:
            st.success(f"Found {len(path)} sequential upgrades. Total {optimization_goal} gain: {format_number(total_gain)}")
            st.markdown("---")

            # show path steps
            for i, step in enumerate(path):
                item = step['item']
                cur = step['current_level']
                nxt = step['next_level']
                gain = step['delta_value']
                cost = step['cost']
                row = DELTA_COSTS[DELTA_COSTS['Upgrade_Level'] == nxt].iloc[0]
                tier = row['Tier']
                border_color, box_shadow = get_tier_style(tier)

                st.markdown(f"<div style='border:3px solid {border_color}; padding:10px; border-radius:8px; margin-bottom:8px;'>"
                            f"<b>Step {i+1}:</b> {item} — {cur} ➜ {nxt}<br>"
                            f"Gain: +{format_number(gain)} {optimization_goal}<br>"
                            f"Cost: {', '.join([f'{format_number(v)} {k}' for k,v in cost.items() if v>0])}"
                            f"</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Remaining Materials after sequential upgrades")
            cols_rem = st.columns(len(MATERIALS))
            for j, m in enumerate(MATERIALS):
                cols_rem[j].metric(m, format_number(remaining[m]))

# Data preview
with st.expander("Preview Loaded Excel Data and Delta Costs"):
    st.markdown("**Raw Data (Cumulative costs and statistics)**")
    st.dataframe(excel_data, use_container_width=True)
    st.markdown("**Delta Costs (Incremental costs and statistics per upgrade)**")
    st.dataframe(DELTA_COSTS, use_container_width=True)
