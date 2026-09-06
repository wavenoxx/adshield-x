"""Rebuild outputs/sample_pool.csv.

The console's "generate a sample" draws from a held-out stream rather than
synthesising clicks on demand, because the entity-graph features only mean
anything when they were computed over a whole stream. This rebuilds that stream
with the training parameters and a seed that was never used for fitting.

    python scripts/build_sample_pool.py
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data as D
from features import add_entity_velocity_features

OUT = os.path.join(ROOT, "outputs", "sample_pool.csv")
SEED = 20260906          # never used to fit anything

t0 = time.time()
df = add_entity_velocity_features(D.generate_clickstream(n=24_000, seed=SEED))
pool = df.sample(n=8_000, random_state=1).reset_index(drop=True)
pool = pool.drop(columns=[c for c in ("bot_family",) if c in pool.columns])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pool.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(pool):,} clicks, "
      f"{os.path.getsize(OUT)/1e6:.1f} MB, "
      f"{(pool['Clicked']=='Bot').mean():.1%} fraudulent, "
      f"in {time.time()-t0:.1f}s")
