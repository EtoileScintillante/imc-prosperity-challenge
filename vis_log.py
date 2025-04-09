import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from time import sleep

""""
Once on the website, scroll down to logbox and download data from results
Put the csv file in the data folder
"""

folder_data = "pnl_visualization/data/"
folder_res = "pnl_visualization/res/"
filename = "bollinger"


def plot_pnl():
    df = pd.read_csv(folder_data + filename + ".csv", delimiter=";")

    # Convert 'timestamp' to numeric and sort the DataFrame
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    print("Columns in CSV:", df.columns.tolist())
    plt.figure(figsize=(12, 8))

    # Get unique products
    products = df["product"].unique()

    for prod in products:
        prod_df = df[df["product"] == prod]
        plt.plot(prod_df["timestamp"], prod_df["profit_and_loss"], label=f"{prod} PNL")

    total_df = df.groupby("timestamp")["profit_and_loss"].sum().reset_index()
    plt.plot(
        total_df["timestamp"],
        total_df["profit_and_loss"],
        color="black",
        label="Total PNL",
    )

    plt.xlabel("Timestamp")
    plt.ylabel("Profit & Loss")
    plt.title("PNL par Actif")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # savefig
    plt.savefig(folder_res + "pnl_plot_" + filename + ".png")
    plt.show()


if __name__ == "__main__":
    plot_pnl()
