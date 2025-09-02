import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(input_file):
    df = pd.read_csv(input_file)

    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns)
    print("Missing Values:\n", df.isnull().sum())

    # Crop distribution
    top_crops = df["Item"].value_counts().head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_crops.values, y=top_crops.index)
    plt.title("Top 10 Crops by Frequency")
    plt.show()

    # Yearly production trend
    yearly_prod = df.groupby("Year")["Production"].sum()
    plt.figure(figsize=(10,5))
    plt.plot(yearly_prod.index, yearly_prod.values, marker="o")
    plt.title("Total Production Over Time")
    plt.xlabel("Year")
    plt.ylabel("Production (tons)")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(6,4))
    sns.heatmap(df[["Area_harvested","Yield","Production"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    input_file = "data/cleaned_agriculture_data.csv"
    run_eda(input_file)