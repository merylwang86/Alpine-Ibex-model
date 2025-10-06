import os, numpy as np, pandas as pd

class Environment:
    """
    Generates synthetic terrain and salt locations and saves to /data.
    v2: default number of salt points = 15
    """
    def __init__(self, project_root, n=100, n_salt=15, seed=42):
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        self.terrain = self._create_terrain(n)
        self.salt_points = self._create_salt_points(n, n_salt)

    def _create_terrain(self, n):
        # smooth-ish random field via sums of gaussians
        base = self.rng.random((n,n))
        terrain = base * 1.2
        np.save(os.path.join(self.data_dir, "terrain.npy"), terrain)
        return terrain

    def _create_salt_points(self, n, count):
        pts = self.rng.integers(low=10, high=n-10, size=(count, 2))
        df = pd.DataFrame(pts, columns=["x","y"])
        df.to_csv(os.path.join(self.data_dir, "salt_points.csv"), index=False)
        return pts
