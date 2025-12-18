import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def set_seed(seed):
    np.random.seed(seed)


def extract(data, key):
    return [d[key] for d in data]


class Bee:
    def __init__(self, x, y, age=0, energy=0):
        self.x = x
        self.y = y
        self.age = age
        self.energy = energy
        self.pollen_collected = 0


def bee_forage(
    bee,
    flower_x,
    flower_y,
    flower_nectar,
    flower_pollinated,
    decay,
    harvest_rate
):
    if flower_nectar.sum() <= 0:
        return 0.0

    dx = flower_x - bee.x
    dy = flower_y - bee.y

    distances = np.sqrt(dx * dx + dy * dy)

    weights = flower_nectar * np.exp(-decay * distances)
    total = weights.sum()
    if total <= 0:
        return 0.0

    weights /= total
    idx = np.random.choice(len(flower_x), p=weights)

    bee.x = flower_x[idx]
    bee.y = flower_y[idx]

    if bee.pollen_collected > 0 and flower_pollinated[idx] == 0:
        flower_pollinated[idx] = 1
        bee.pollen_collected -= 1

    max_harvest = flower_nectar[idx] * harvest_rate
    harvest = min(flower_nectar[idx], max_harvest)
    flower_nectar[idx] -= harvest

    if harvest > 0:
        bee.pollen_collected += 1
    return harvest


def flower_reproduction(
    i,
    flower_x,
    flower_y,
    flower_capacity,
    flower_dev_time,
    spread,
    density,
):  

    if density > 1:
        density = 1
    p_setseed = 1-3*density**2+2*density**3
    if np.random.random() > p_setseed:
        return None

    dx = np.random.uniform(-spread, spread)
    dy = np.random.uniform(-spread, spread)

    L = 100.0

    new_x = flower_x[i] + dx
    if new_x < 0:
        new_x = -new_x
    elif new_x > L:
        new_x = 2*L - new_x

    new_y = flower_y[i] + dy
    if new_y < 0:
        new_y = -new_y
    elif new_y > L:
        new_y = 2*L - new_y

    return (
        new_x,
        new_y,
        flower_capacity[i],
        flower_dev_time[i],
    )


def simulate_day(
    bees,
    flower_x,
    flower_y,
    flower_capacity,
    flower_nectar,
    flower_pollinated,
    flower_timer,
    flower_dev_time,
    decay,
    visits_per_bee,
    bee_food,
    regen_rate,
    harvest_rate,
    carrying_capacity,
    spread,
    bee_reproduction_cost,
    bee_background_mortality,
    flower_survival_prob
):
    for bee in bees:
        for _ in range(visits_per_bee):
            harvest = bee_forage(
                bee,
                flower_x,
                flower_y,
                flower_nectar,
                flower_pollinated,
                decay,
                harvest_rate
            )
            bee.energy += harvest
            if flower_nectar.sum() <= 0:
                break

    for bee in bees:
        bee.energy -= bee_food
    bees = [b for b in bees if b.energy > 0]

    density = len(flower_x) / carrying_capacity

    flower_nectar[:] = np.minimum(
        flower_capacity,
        flower_nectar + flower_capacity * regen_rate
    )

    new_flowers = []
    for i in range(len(flower_x)):
        if flower_pollinated[i] == 1:
            flower_timer[i] += 1
            if flower_timer[i] >= flower_dev_time[i]:
                flower_pollinated[i] = 0
                flower_timer[i] = 0
                offspring = flower_reproduction(
                    i,
                    flower_x, flower_y,
                    flower_capacity, flower_dev_time,
                    spread, density
                )
                if offspring:
                    new_flowers.append(offspring)

    survive = np.random.random(len(flower_x)) < flower_survival_prob
    flower_x = flower_x[survive]
    flower_y = flower_y[survive]
    flower_capacity = flower_capacity[survive]
    flower_nectar = flower_nectar[survive]
    flower_pollinated = flower_pollinated[survive]
    flower_timer = flower_timer[survive]
    flower_dev_time = flower_dev_time[survive]

    if new_flowers:
        nx, ny, nc, nd = zip(*new_flowers)
        flower_x = np.concatenate([flower_x, np.array(nx)])
        flower_y = np.concatenate([flower_y, np.array(ny)])
        flower_capacity = np.concatenate([flower_capacity, np.array(nc)])
        flower_nectar = np.concatenate([flower_nectar, np.array(nc)])
        flower_pollinated = np.concatenate([flower_pollinated, np.zeros(len(nx))])
        flower_timer = np.concatenate([flower_timer, np.zeros(len(nx))])
        flower_dev_time = np.concatenate([flower_dev_time, np.array(nd)])

    for bee in bees:
        bee.pollen_collected = 0

    new_bees = []
    survivors = []
    for bee in bees:
        bee.age += 1
        if bee.energy >= bee_reproduction_cost and bee.age >= 10:
            n = int(bee.energy // bee_reproduction_cost)
            for _ in range(n):
                new_bees.append(Bee(bee.x, bee.y))
        elif bee.age >= 20:
            continue
        else:
            survivors.append(bee)

    bees = [b for b in survivors if np.random.random() > bee_background_mortality]
    bees.extend(new_bees)

    return (
        bees,
        flower_x, flower_y,
        flower_capacity, flower_nectar,
        flower_pollinated, flower_timer,
        flower_dev_time
    )


def run_simulation_once(
    NUM_DAYS=1000,
    HARVEST_RATE=0.0125,
    REGEN_RATE=0.16,
    INITIAL_BEES=10,
    INITIAL_FLOWERS=100,
    FLOWER_CAPACITY=10,
    CARRYING_CAPACITY=300,
    DECAY=0.015,
    VISITS_PER_BEE=10,
    BEE_FOOD=1,
    FLOWER_SPREAD=10,
    FLOWER_DEVELOPMENT_TIME=10,
    BEE_BACKGROUND_MORTALITY=0.01,
    FLOWER_SURVIVAL_PROB=0.98,
    SEED=None
):
    if SEED is not None:
        set_seed(SEED)

    flower_x = np.random.uniform(0, 100, INITIAL_FLOWERS)
    flower_y = np.random.uniform(0, 100, INITIAL_FLOWERS)
    flower_capacity = np.full(INITIAL_FLOWERS, FLOWER_CAPACITY)
    flower_nectar = flower_capacity.copy()
    flower_pollinated = np.zeros(INITIAL_FLOWERS)
    flower_timer = np.zeros(INITIAL_FLOWERS)
    flower_dev_time = np.maximum(
        1, FLOWER_DEVELOPMENT_TIME + np.random.randint(-3, 4, INITIAL_FLOWERS)
    )

    bees = [Bee(np.random.uniform(0, 100), np.random.uniform(0, 100), age=np.random.randint(0, 11))
            for _ in range(INITIAL_BEES)]

    bee_history = []
    flower_history = []
    regen_quality_history = []
    harvest_quality_history = []

    BEE_REPRODUCTION_COST = max(1.0, 10*(100*HARVEST_RATE - 1)/2)
    for _ in range(NUM_DAYS):
        regen_quality = np.random.uniform(0.8, 1.2)
        harvest_quality = np.random.uniform(0.8, 1.2)

        regen_quality_history.append(regen_quality)
        harvest_quality_history.append(harvest_quality)

        bees, flower_x, flower_y, flower_capacity, flower_nectar, \
        flower_pollinated, flower_timer, flower_dev_time = simulate_day(
            bees,
            flower_x, flower_y, flower_capacity, flower_nectar,
            flower_pollinated, flower_timer, flower_dev_time,
            DECAY,
            VISITS_PER_BEE,
            BEE_FOOD,
            REGEN_RATE*regen_quality,
            HARVEST_RATE*harvest_quality,
            CARRYING_CAPACITY,
            FLOWER_SPREAD,
            BEE_REPRODUCTION_COST,
            BEE_BACKGROUND_MORTALITY,
            FLOWER_SURVIVAL_PROB
        )
        bee_history.append(len(bees))
        flower_history.append(len(flower_x))
        if len(bees) == 0 or len(flower_x) == 0:
            print("Extinction occurred")
            break

    return bee_history, flower_history, np.array(regen_quality_history), np.array(harvest_quality_history)


def classify_outcome(bee_history, flower_history, threshold=1):
    bees_end = bee_history[-1]
    flowers_end = flower_history[-1]

    b_alive = bees_end >= threshold
    f_alive = flowers_end >= threshold

    if b_alive and f_alive:
        return "coexist"
    elif b_alive and not f_alive:
        return "flowers_extinct"
    elif not b_alive and f_alive:
        return "bees_extinct"
    else:
        return "both_extinct"


def longest_bad_streak(quality, threshold=1.0):
    bad = quality < threshold
    max_streak = streak = 0
    for b in bad:
        if b:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def plot_time_series(bee_history, flower_history, show=True):
    days = np.arange(len(bee_history))
    plt.plot(days, bee_history, label="Bees")
    plt.plot(days, flower_history, label="Flowers")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.title("Bee and Flower Populations Over Time")
    plt.legend()
    if show:
        plt.show()


def parameter_sweep_plot(NUM_DAYS=300, RUNS_PER_POINT=10):
    regen_values = np.linspace(0.08, 1, 20)
    p_values = np.linspace(0.02, 0.12, 20)

    coexist_prob = np.zeros((len(regen_values), len(p_values)))

    k = 0
    for i, regen in enumerate(regen_values):
        for j, p in enumerate(p_values):

            harvest = p * regen

            outcomes = []
            for r in range(RUNS_PER_POINT):
                print(
                    f"run {k} of total: "
                    f"{len(regen_values)*len(p_values)*RUNS_PER_POINT}"
                )
                k += 1
                bee_hist, flower_hist, _, _ = run_simulation_once(
                    NUM_DAYS,
                    HARVEST_RATE=harvest,
                    REGEN_RATE=regen,
                    SEED=None
                )

                outcome = classify_outcome(bee_hist, flower_hist)
                outcomes.append(outcome)

            counts = Counter(outcomes) 

            coexist_prob[i, j] = counts["coexist"] / RUNS_PER_POINT

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        coexist_prob,
        origin="lower",
        aspect="auto",
        extent=[
            p_values[0], p_values[-1],
            regen_values[0], regen_values[-1]
        ]
    )

    plt.colorbar(im, label="Probability of coexistence")
    plt.xlabel(r"$r = R_{\mathrm{harvest}} / R_{\mathrm{regen}}$")
    plt.ylabel(r"$R_{\mathrm{regen}}$")
    plt.title(f"Coexistence probability after {NUM_DAYS} days")
    plt.show()


def streak_analysis(
    runs=50,
    W=10,
    bad_threshold=0.9
):
    collapse = []
    persist = []
    bee_histories = []
    flower_histories = []
    outcomes = []

    for _ in range(runs):
        print(f"Simulation run {_+1} of {runs}")
        bee_hist, flower_hist, regen_q, harvest_q = run_simulation_once()
        extinct = (bee_hist[-1] == 0) or (flower_hist[-1] == 0)
        bee_histories.append(bee_hist)
        flower_histories.append(flower_hist)
        outcomes.append(extinct)

        rq = regen_q[-W:] if len(regen_q) >= W else regen_q
        hq = harvest_q[-W:] if len(harvest_q) >= W else harvest_q

        metrics = {
            "mean_regen_quality": rq.mean(),
            "mean_harvest_quality": hq.mean(),
            "regen_bad_streak": longest_bad_streak(rq, bad_threshold),
            "harvest_bad_streak": longest_bad_streak(hq, bad_threshold),
            "length": len(rq)
        }

        if extinct:
            collapse.append(metrics)
        else:
            persist.append(metrics)

    print(f"Collapsing runs: {len(collapse)}")
    print(f"Persistent runs: {len(persist)}")

    labels = ["Extinction", "Coexistence"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].boxplot(
        [extract(collapse, "regen_bad_streak"),
         extract(persist, "regen_bad_streak")],
        labels=labels
    )
    axes[0, 0].set_title("Regeneration bad-day streak")
    axes[0, 0].set_ylabel("Longest streak, last 10 days")

    axes[0, 1].boxplot(
        [extract(collapse, "harvest_bad_streak"),
         extract(persist, "harvest_bad_streak")],
        labels=labels
    )
    axes[0, 1].set_title("Harvest bad-day streak")

    axes[1, 0].boxplot(
        [extract(collapse, "mean_regen_quality"),
         extract(persist, "mean_regen_quality")],
        labels=labels
    )
    axes[1, 0].set_title("Mean regeneration quality")
    axes[1, 0].set_ylabel("Mean quality, last 10 days")

    axes[1, 1].boxplot(
        [extract(collapse, "mean_harvest_quality"),
         extract(persist, "mean_harvest_quality")],
        labels=labels
    )
    axes[1, 1].set_title("Mean harvest quality")

    plt.tight_layout()
    plt.show()
