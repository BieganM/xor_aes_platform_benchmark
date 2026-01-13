import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def generate_charts(csv_file, output_dir):
    try:
        df = pd.read_csv(csv_file, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Replace Platform names
    df["Platform"] = df["Platform"].replace(
        {"WSL": "RTX 3070", "RTX": "RTX 3070", "macOS": "M4 Pro"}
    )

    # In block_size_results.csv, FileSize_MB actually holds Block Size in MB.
    # Rename for clarity
    df.rename(columns={"FileSize_MB": "BlockSizeMB"}, inplace=True)

    # Calculate BlockSize in KB for labeling/plotting
    df["BlockSizeKB"] = df["BlockSizeMB"] * 1024

    # Create a unique Label for Platform + Engine
    # This ensures that macOS OpenMP and WSL OpenMP are distinct in legend
    df["Label"] = df["Platform"] + " - " + df["Engine"]

    # Set style
    sns.set_theme(style="whitegrid")

    # Process Algorithms separately
    algorithms = df["Algorithm"].unique()

    for algo in algorithms:
        plt.figure(figsize=(10, 6))

        # Filter data for this algorithm
        data = df[df["Algorithm"] == algo]

        # Plot using 'Label' for hue and style to get distinct colors/markers
        # We can also map styles: e.g. OpenMP is always dashed?
        # Let's rely on seaborn defaults for now, they are usually distinct enough
        ax = sns.lineplot(
            data=data,
            x="BlockSizeMB",
            y="Throughput_MBs",
            hue="Label",
            style="Label",
            markers=True,
            dashes=False,
            linewidth=2.5,
            markersize=8,
        )

        plt.title(f"{algo} Throughput vs Block Size", fontsize=14, fontweight="bold")
        plt.xlabel("Block Size", fontsize=12)
        plt.ylabel("Throughput (MB/s)", fontsize=12)
        plt.xscale("log")

        # Format X ticks
        # We tested: 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16
        ticks = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        tick_labels = [
            "64KB",
            "128KB",
            "256KB",
            "512KB",
            "1MB",
            "2MB",
            "4MB",
            "8MB",
            "16MB",
        ]
        plt.xticks(ticks, tick_labels, rotation=45)

        plt.grid(True, which="major", ls="-", alpha=0.5)
        plt.grid(True, which="minor", ls=":", alpha=0.2)
        plt.legend(title="Platform - Engine", loc="upper left")

        # Clean filename
        safe_algo = algo.replace(" ", "_").replace("-", "_").lower()
        output_file = os.path.join(output_dir, f"block_size_impact_{safe_algo}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Chart saved to {output_file}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Block Size Performance Charts"
    )
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument(
        "--output-dir", default="results/image", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_charts(args.csv, args.output_dir)
