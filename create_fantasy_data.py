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

CLASS_WEIGHTS = [0.4, 0.3, 0.3]

def roll(ndice: int, die_size: int, modifier: int = 0) -> int:
    return sum(random.randint(1, die_size) for _ in range(ndice)) + modifier
def fighter_damage(level: int):
    str_mod = 3
    if level <= 5:
        phys, magic = roll(1, 8, str_mod), 0
    elif level <= 10:
        phys, magic = roll(2, 8, 2 * str_mod), roll(1, 4)
    elif level <= 16:
        phys, magic = roll(3, 8, 3 * str_mod), roll(2, 4)
    else:
        phys, magic = roll(4, 8, 4 * str_mod), roll(3, 4)
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
        return roll(1, 8, str_mod), roll(1, 6)
    elif level <= 10:
        return roll(2, 8, 2 * str_mod), roll(2, 6)
    elif level <= 16:
        return roll(2, 8, 2 * str_mod), roll(3, 6)
    else:
        return roll(3, 8, 3 * str_mod), roll(4, 6)

profiles = {
    "Fighter": fighter_damage,
    "Wizard":  wizard_damage,
    "Paladin": paladin_damage,
}

def generate_adventurers():
    random.seed(SEED)
    rows = []
    classes = list(profiles.keys())
    for adv_id in range(N_ADV):
        cls = random.choices(classes, weights=CLASS_WEIGHTS, k=1)[0]
        lvl = random.randint(1, 20)
        phys, magic = profiles[cls](lvl)
        rows.append({
            "adv_id": adv_id,
            "class": cls,
            "level": lvl,
            "avg_phys": phys,
            "avg_magic": magic
        })
    return pd.DataFrame(rows)

def sample_rgb_sum_100():
    a, b = random.randint(0, 100), random.randint(0, 100)
    a, b = (a, b) if a <= b else (b, a)
    return a, b - a, 100 - b

def generate_potions():
    random.seed(SEED + 1)
    rows = []
    for pid in range(N_POTIONS):
        r, g, b = sample_rgb_sum_100()
        rows.append({"potion_id": pid, "red": r, "green": g, "blue": b})
    return pd.DataFrame(rows)

def raw_preference(cls: str, r: int, g: int, b: int) -> float:
    R, G, B = float(r), float(g), float(b)
    total = 100
    if total <= 0:
        return 0.0
    if cls == "Fighter":
        return R / total
    if cls == "Wizard":
        return B / total
    if cls == "Paladin":
        rb = R + B
        if rb <= 0:
            return 0.0
        balance = 1.0 - (abs(R - B) / rb)
        return (rb / total) * balance
    return (R + B) / (2.0 * total)

def sigmoid_enjoyment(raw_pref: float) -> float:
    z = max(-60.0, min(60.0, SIGMOID_K * (raw_pref - SIGMOID_MID)))
    return 1.0 / (1.0 + math.exp(-z))

def generate_interactions(adventurers_df, potions_df):
    rng = random.Random(SEED + 2)
    potion_ids = potions_df["potion_id"].tolist()
    potion_lookup = potions_df.set_index("potion_id")[["red", "green", "blue"]].to_dict("index")

    rows = []
    for _, adv in adventurers_df.iterrows():
        chosen = rng.sample(potion_ids, min(PAIRS_PER_ADV, len(potion_ids)))

        for pid in chosen:
            pot = potion_lookup[pid]
            r, g, b = pot["red"], pot["green"], pot["blue"]
            rp = raw_preference(adv["class"], r, g, b)
            enjoyment = sigmoid_enjoyment(rp)

            rows.append({
                "adv_id": adv["adv_id"],
                "potion_id": pid,
                "class": adv["class"],
                "level": adv["level"],
                "avg_phys": adv["avg_phys"],
                "avg_magic": adv["avg_magic"],
                "red": r, "green": g, "blue": b,
                "raw_pref": round(rp, 6),
                "enjoyment": round(enjoyment, 6)
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_adv = generate_adventurers()
    df_pot = generate_potions()
    df_int = generate_interactions(df_adv, df_pot)

    df_int.to_csv(OUT_INTERACTIONS, index=False)

    print(f"Wrote {len(df_int)} interactions to {OUT_INTERACTIONS}")
    print("Adventurers head:\n", df_adv.head(), "\n")
    print("Potions head:\n", df_pot.head(), "\n")
    print("Interactions head:\n", df_int.head())
