import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame  # готовый DataFrame со всеми признаками + целевой колонкой

df.to_csv("data/raw/data.csv", index=False)
print("Saved to data/raw/data.csv")
