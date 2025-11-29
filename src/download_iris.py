import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

def main():
    iris = load_iris(as_frame=True)
    df = iris.frame  # признаки + target

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_path = raw_dir / "data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved Iris dataset to {out_path}")

if __name__ == "__main__":
    main()
