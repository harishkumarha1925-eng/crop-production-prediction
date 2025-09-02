import pandas as pd

def preprocess_faostat(input_file, output_file):
    # Load Excel
    df = pd.read_excel(input_file, sheet_name=0)

    # Clean column names (remove spaces, standardize)
    df.columns = df.columns.str.strip()

    print("Available columns:", df.columns.tolist())  # 👈 debug check

    # Pivot table
    df_pivot = df.pivot_table(
        index=["Area", "Item", "Year"],
        columns="Element",
        values="Value",   # 👈 must match column exactly
        aggfunc="first"
    ).reset_index()

    # Rename columns
    df_pivot = df_pivot.rename(columns={
        "Area harvested": "Area_harvested",
        "Yield": "Yield",
        "Production": "Production"
    })

    keep_cols = ["Area", "Item", "Year", "Area_harvested", "Yield", "Production"]
    df_pivot = df_pivot[keep_cols]

    # Drop missing target values
    df_pivot = df_pivot.dropna(subset=["Production"])

    # Save cleaned CSV
    df_pivot.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")


if __name__ == "__main__":
    input_file = "data/FAOSTAT_data.xlsx"   # put your original Excel file here
    output_file = "data/cleaned_agriculture_data.csv"
    preprocess_faostat(input_file, output_file)