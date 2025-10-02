import random
import math
import pandas as pd

OUT_INTERACTIONS = "interactions.csv"

N_ADV = 1000
N_POTIONS = 100
PAIRS_PER_ADV = 10
SEED = 42

SIGMOID_K = 40.0
SIGMOID_MID = 0.7

CLASS_WEIGHTS = [0.4, 0.3, 0.3]  # Fighter, Wizard, Paladin distribution
LEVEL_MIN = 1
LEVEL_MAX = 20

# -----------------------------
# Dice helpers
# -----------------------------
def roll(ndice: int, die_size: int) -> int:
    return sum(random.randint(1, die_size) for _ in range(ndice))

def roll_stat(ndice: int, die_size: int, modifier: int = 0) -> int:
    return roll(ndice, die_size) + modifier

# -----------------------------
# Damage profiles
# -----------------------------
def fighter_damage(level: int):
    str_mod = 3
    if level <= 5:
        phys = roll_stat(1, 8, str_mod)
    elif level <= 10:
        phys = roll_stat(2, 8, 2 * str_mod)
    elif level <= 16:
        phys = roll_stat(3, 8, 3 * str_mod)
    else:
        phys = roll_stat(4, 8, 4 * str_mod)

    if level <= 5:
        magic = 0
    elif level <= 10:
        magic = roll(1, 4)
    elif level <= 16:
        magic = roll(2, 4)
    else:
        magic = roll(3, 4)
    return phys, magic

def wizard_damage(level: int):
    if level <= 4:
        magic = roll(1, 12)
    elif level <= 10:
        magic = roll(2, 12)
    elif level <= 16:
        magic = roll(3, 12)
    else:
        magic = roll(4, 12)
    phys = roll(1, 4)
    return phys, magic

def paladin_damage(level: int):
    str_mod = 3
    if level <= 4:
        phys = roll_stat(1, 8, str_mod)
        magic = roll(1, 6)
    elif level <= 10:
        phys = roll_stat(2, 8, 2 * str_mod)
        magic = roll(2, 6)
    elif level <= 16:
        phys = roll_stat(2, 8, 2 * str_mod)
        magic = roll(3, 6)
    else:
        phys = roll_stat(3, 8, 3 * str_mod)
        magic = roll(4, 6)
    return phys, magic

profiles = {
    "Fighter": fighter_damage,
    "Wizard":  wizard_damage,
    "Paladin": paladin_damage,
}

# -----------------------------
# Adventurer generator
# -----------------------------
def generate_adventurers():
    random.seed(SEED)
    rows = []
    classes = list(profiles.keys())
    for adv_id in range(N_ADV):
        cls = random.choices(classes, weights=CLASS_WEIGHTS, k=1)[0]
        lvl = random.randint(LEVEL_MIN, LEVEL_MAX)
        phys, magic = profiles[cls](lvl)
        rows.append({
            "adv_id": adv_id,
            "class": cls,
            "level": lvl,
            "avg_phys": phys,
            "avg_magic": magic
        })
    return pd.DataFrame(rows)

# -----------------------------
# Potion generator
# -----------------------------
def sample_rgb_sum_100():
    a = random.randint(0, 100)
    b = random.randint(0, 100)
    lo, hi = (a, b) if a <= b else (b, a)
    r = lo
    g = hi - lo
    bl = 100 - hi
    return r, g, bl

def generate_potions():
    random.seed(SEED + 1)
    rows = []
    for pid in range(N_POTIONS):
        r, g, b = sample_rgb_sum_100()
        rows.append({"potion_id": pid, "red": r, "green": g, "blue": b})
    return pd.DataFrame(rows)

# -----------------------------
# Preferences & enjoyment
# -----------------------------
def raw_preference(cls: str, r: int, g: int, b: int) -> float:
    R, G, B = float(r), float(g), float(b)
    total = R + G + B
    if total <= 0:
        return 0.0
    if cls == "Fighter":
        # Ideal: 100 red, 0 green, 0 blue
        return R / total
    if cls == "Wizard":
        # Ideal: 0 red, 0 green, 100 blue
        return B / total
    if cls == "Paladin":
        # Ideal: 50 red, 0 green, 50 blue (balanced red/blue, no green)
        rb = R + B
        if rb <= 0:
            return 0.0
        balance = 1.0 - (abs(R - B) / rb)  # 1.0 when R == B
        return (rb / total) * balance      # 1.0 when R == B and G == 0
    return (R + B) / (2.0 * total)

def sigmoid_enjoyment(raw_pref: float) -> float:
    z = max(-60.0, min(60.0, SIGMOID_K * (raw_pref - SIGMOID_MID)))
    return 1.0 / (1.0 + math.exp(-z))

# -----------------------------
# Interaction generator
# -----------------------------
def generate_interactions(adventurers_df, potions_df):
    rng = random.Random(SEED + 2)
    potion_ids = potions_df["potion_id"].tolist()
    n_potions = len(potion_ids)

    rows = []
    for _, adv in adventurers_df.iterrows():
        if PAIRS_PER_ADV <= n_potions:
            chosen = rng.sample(potion_ids, PAIRS_PER_ADV)
        else:
            chosen = [rng.choice(potion_ids) for _ in range(PAIRS_PER_ADV)]

        for pid in chosen:
            pot = potions_df.loc[potions_df["potion_id"] == pid].iloc[0]
            r, g, b = int(pot["red"]), int(pot["green"]), int(pot["blue"])
            rp = raw_preference(adv["class"], r, g, b)
            enjoyment = sigmoid_enjoyment(rp)

            rows.append({
                "adv_id": int(adv["adv_id"]),
                "potion_id": int(pid),
                "class": adv["class"],
                "level": int(adv["level"]),
                "avg_phys": int(adv["avg_phys"]),
                "avg_magic": int(adv["avg_magic"]),
                "red": r, "green": g, "blue": b,
                "raw_pref": round(rp, 6),
                "enjoyment": round(enjoyment, 6)
            })

    return pd.DataFrame(rows)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df_adv = generate_adventurers()
    df_pot = generate_potions()
    df_int = generate_interactions(df_adv, df_pot)

    df_int.to_csv(OUT_INTERACTIONS, index=False)

    print(f"Wrote {len(df_int)} interactions to {OUT_INTERACTIONS}")
    print("Adventurers head:\n", df_adv.head(), "\n")
    print("Potions head:\n", df_pot.head(), "\n")
    print("Interactions head:\n", df_int.head())
