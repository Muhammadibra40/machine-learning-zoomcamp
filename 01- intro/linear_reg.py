import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()

linear_reg_data_path = os.getenv("LINEAR_REG_DATA_PATH")

df = pd.read_csv(linear_reg_data_path)

plt.scatter(df["Study Time (hours)"], df["Score (%)"])
plt.show()