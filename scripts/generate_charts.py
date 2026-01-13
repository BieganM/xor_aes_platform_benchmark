#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

COLORS = {
    "macOS": "#FF6B6B",
    "M4 Pro": "#FF6B6B",
    "WSL": "#4ECDC4",
    "RTX": "#4ECDC4",
    "RTX 3070": "#4ECDC4",
    "Linux": "#45B7D1",
    "XOR": "#FF6B6B",
    "AES-256-CTR": "#4ECDC4",
    "Sequential": "#636EFA",
    "OpenMP": "#EF553B",
    "OpenCL": "#00CC96",
    "Metal": "#AB63FA",
    "CUDA": "#FFA15A",
}


def load_data(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    # Replace Platform names
    if "Platform" in df.columns:
        df["Platform"] = df["Platform"].replace(
            {"WSL": "RTX 3070", "RTX": "RTX 3070", "macOS": "M4 Pro"}
        )
    return df


def save_single_plot(fig, output_dir, filename):
    plt.figure(fig.number)  # Set current figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")


def plot_throughput_vs_filesize(df, output_dir):
    # Combined plot
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # for idx, algo in enumerate(["XOR", "AES-256-CTR"]):
    #     ax = axes[idx]
    #     algo_data = df[df["Algorithm"] == algo]

    #     for platform in algo_data["Platform"].unique():
    #         for engine in algo_data["Engine"].unique():
    #             subset = algo_data[
    #                 (algo_data["Platform"] == platform)
    #                 & (algo_data["Engine"] == engine)
    #             ]
    #             if len(subset) > 0:
    #                 best_per_size = subset.groupby("FileSize_MB")[
    #                     "Throughput_MBs"
    #                 ].max()
    #                 label = f"{platform} - {engine}"
    #                 ax.plot(
    #                     best_per_size.index,
    #                     best_per_size.values,
    #                     "o-",
    #                     label=label,
    #                     linewidth=2,
    #                     markersize=8,
    #                 )

    #     ax.set_xlabel("File Size [MB]")
    #     ax.set_ylabel("Throughput [MB/s]")
    #     ax.set_title(f"{algo} - Throughput vs File Size")
    #     ax.legend(loc="best", fontsize=9)
    #     ax.grid(True, alpha=0.3)
    #     ax.set_xscale("log")

    # plt.tight_layout()
    # plt.savefig(
    #     os.path.join(output_dir, "throughput_vs_filesize.png"),
    #     dpi=150,
    #     bbox_inches="tight",
    # )
    # plt.close()
    # print("Saved: throughput_vs_filesize.png")

    # plt.close(fig) # Close the scratch figure used above if we act differently

    # Redo cleanly:
    # 1. Individual XOR
    # 2. Individual AES
    # 3. Combined

    fig_combined, axes_combined = plt.subplots(1, 2, figsize=(14, 6))

    for idx, algo in enumerate(["XOR", "AES-256-CTR"]):
        safe_algo = algo.replace(" ", "_").replace("-", "_").lower()

        # Individual Figure
        fig_single, ax_single = plt.subplots(figsize=(8, 6))

        # We need to plot on both ax_single and axes_combined[idx]
        targets = [ax_single, axes_combined[idx]]

        algo_data = df[df["Algorithm"] == algo]
        for platform in algo_data["Platform"].unique():
            for engine in algo_data["Engine"].unique():
                subset = algo_data[
                    (algo_data["Platform"] == platform)
                    & (algo_data["Engine"] == engine)
                ]
                if len(subset) > 0:
                    best_per_size = subset.groupby("FileSize_MB")[
                        "Throughput_MBs"
                    ].max()
                    label = f"{platform} - {engine}"

                    for ax in targets:
                        ax.plot(
                            best_per_size.index,
                            best_per_size.values,
                            "o-",
                            label=label,
                            linewidth=2,
                            markersize=8,
                        )

        for ax in targets:
            ax.set_xlabel("File Size [MB]")
            ax.set_ylabel("Throughput [MB/s]")
            ax.set_title(f"{algo} - Throughput vs File Size")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

        save_single_plot(
            fig_single, output_dir, f"throughput_vs_filesize_{safe_algo}.png"
        )
        plt.close(fig_single)

    plt.figure(fig_combined.number)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "throughput_vs_filesize.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_combined)
    print("Saved: throughput_vs_filesize.png")


def plot_execution_time_comparison(df, output_dir):
    platforms = df["Platform"].unique()
    engines = df["Engine"].unique()
    algorithms = df["Algorithm"].unique()

    fig_combined, axes_combined = plt.subplots(1, 2, figsize=(14, 6))

    for idx, algo in enumerate(algorithms):
        safe_algo = algo.replace(" ", "_").replace("-", "_").lower()
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        targets = [ax_single, axes_combined[idx]]

        algo_data = df[df["Algorithm"] == algo]
        x = np.arange(len(engines))
        width = 0.35

        for ax in targets:
            for i, platform in enumerate(platforms):
                platform_data = algo_data[algo_data["Platform"] == platform]
                times = []
                for engine in engines:
                    engine_data = platform_data[platform_data["Engine"] == engine]
                    if len(engine_data) > 0:
                        times.append(engine_data["Time_Sec"].min() * 1000)
                    else:
                        times.append(0)

                color = COLORS.get(platform, "#888888")
                ax.bar(
                    x + i * width - width / 2, times, width, label=platform, color=color
                )

            ax.set_xlabel("Engine")
            ax.set_ylabel("Execution Time [ms]")
            ax.set_title(f"{algo} - Execution Time per Method")
            ax.set_xticks(x)
            ax.set_xticklabels(engines, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        save_single_plot(fig_single, output_dir, f"execution_time_{safe_algo}.png")
        plt.close(fig_single)

    plt.figure(fig_combined.number)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "execution_time_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_combined)
    print("Saved: execution_time_comparison.png")


def plot_max_throughput_per_method(df, output_dir):
    fig_combined, axes_combined = plt.subplots(1, 2, figsize=(14, 6))

    for idx, algo in enumerate(["XOR", "AES-256-CTR"]):
        safe_algo = algo.replace(" ", "_").replace("-", "_").lower()
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        targets = [ax_single, axes_combined[idx]]

        algo_data = df[df["Algorithm"] == algo]
        max_throughput = (
            algo_data.groupby(["Platform", "Engine"])["Throughput_MBs"]
            .max()
            .unstack(fill_value=0)
        )

        for ax in targets:
            max_throughput.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
            ax.set_xlabel("Platform")
            ax.set_ylabel("Maximum Throughput [MB/s]")
            ax.set_title(f"{algo} - Maximum Throughput per Method")
            ax.legend(title="Engine", loc="best")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            for container in ax.containers:
                ax.bar_label(container, fmt="%.0f", fontsize=8, rotation=90, padding=3)

        save_single_plot(fig_single, output_dir, f"max_throughput_{safe_algo}.png")
        plt.close(fig_single)

    plt.figure(fig_combined.number)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "max_throughput_per_method.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_combined)
    print("Saved: max_throughput_per_method.png")


def plot_energy_consumption(df, output_dir):
    # Convert Joules to mWh: E(mWh) = E(J) / 3.6
    df["Energy_per_MB_mWh"] = (df["Energy_Joules"] / df["FileSize_MB"]) / 3.6

    fig_combined, axes_combined = plt.subplots(1, 2, figsize=(14, 6))

    for idx, algo in enumerate(["XOR", "AES-256-CTR"]):
        safe_algo = algo.replace(" ", "_").replace("-", "_").lower()
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        targets = [ax_single, axes_combined[idx]]

        algo_data = df[df["Algorithm"] == algo]
        energy_data = (
            algo_data.groupby(["Platform", "Engine"])["Energy_per_MB_mWh"]
            .mean()
            .unstack(fill_value=0)
        )

        for ax in targets:
            energy_data.plot(kind="bar", ax=ax, colormap="plasma", width=0.8)
            ax.set_xlabel("Platform")
            ax.set_ylabel("Energy per MB [mWh/MB]")
            ax.set_title(f"{algo} - Energy Consumption per Method")
            ax.legend(title="Engine", loc="best")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        save_single_plot(fig_single, output_dir, f"energy_consumption_{safe_algo}.png")
        plt.close(fig_single)

    plt.figure(fig_combined.number)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "energy_consumption.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig_combined)
    print("Saved: energy_consumption.png")


def plot_xor_vs_aes(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    best_results = df.loc[
        df.groupby(["Platform", "Algorithm", "Engine"])["Throughput_MBs"].idxmax()
    ]

    platforms = best_results["Platform"].unique()
    algorithms = ["XOR", "AES-256-CTR"]

    x = np.arange(len(platforms))
    width = 0.35

    for i, algo in enumerate(algorithms):
        algo_data = best_results[best_results["Algorithm"] == algo]
        throughputs = []
        for platform in platforms:
            platform_data = algo_data[algo_data["Platform"] == platform]
            if len(platform_data) > 0:
                throughputs.append(platform_data["Throughput_MBs"].max())
            else:
                throughputs.append(0)

        color = COLORS.get(algo, "#888888")
        bars = ax.bar(
            x + i * width - width / 2, throughputs, width, label=algo, color=color
        )

        for bar, val in zip(bars, throughputs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 500,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Platform")
    ax.set_ylabel("Maximum Throughput [MB/s]")
    ax.set_title("XOR vs AES-256-CTR - Best Performance by Platform")
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "xor_vs_aes_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: xor_vs_aes_comparison.png")


def plot_speedup_efficiency(df, output_dir):
    openmp_data = df[df["Engine"] == "OpenMP"]

    if len(openmp_data) == 0 or openmp_data["NumThreads"].nunique() <= 1:
        print("Skipping speedup/efficiency charts - no thread scaling data")
        return

    fig_combined, axes_combined = plt.subplots(2, 2, figsize=(14, 12))

    for idx, algo in enumerate(["XOR", "AES-256-CTR"]):
        safe_algo = algo.replace(" ", "_").replace("-", "_").lower()

        # We need 2 single plots per algo: Speedup and Efficiency
        fig_speedup, ax_speedup = plt.subplots(figsize=(8, 6))
        fig_efficiency, ax_efficiency = plt.subplots(figsize=(8, 6))

        # Mapping:
        # Combined [0, idx] -> Speedup
        # Combined [1, idx] -> Efficiency

        algo_data = openmp_data[openmp_data["Algorithm"] == algo]

        for platform in algo_data["Platform"].unique():
            platform_data = algo_data[algo_data["Platform"] == platform]

            for file_size in platform_data["FileSize_MB"].unique():
                size_data = platform_data[
                    platform_data["FileSize_MB"] == file_size
                ].sort_values("NumThreads")
                if len(size_data) > 1:
                    label = f"{platform} {file_size}MB"

                    # Plot Speedup
                    for ax in [ax_speedup, axes_combined[0, idx]]:
                        ax.plot(
                            size_data["NumThreads"],
                            size_data["Speedup"],
                            "o-",
                            label=label,
                            linewidth=2,
                            markersize=6,
                        )

                    # Plot Efficiency
                    for ax in [ax_efficiency, axes_combined[1, idx]]:
                        ax.plot(
                            size_data["NumThreads"],
                            size_data["Efficiency"] * 100,
                            "o-",
                            label=label,
                            linewidth=2,
                            markersize=6,
                        )

        max_threads = openmp_data["NumThreads"].max()

        # Configure Speedup Axes
        for ax in [ax_speedup, axes_combined[0, idx]]:
            ax.plot([1, max_threads], [1, max_threads], "k--", alpha=0.5, label="Ideal")
            ax.set_xlabel("Number of Threads")
            ax.set_ylabel("Speedup")
            ax.set_title(f"{algo} - Speedup vs Threads (OpenMP)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        save_single_plot(fig_speedup, output_dir, f"speedup_{safe_algo}.png")
        plt.close(fig_speedup)

        # Configure Efficiency Axes
        for ax in [ax_efficiency, axes_combined[1, idx]]:
            ax.axhline(
                y=100, color="k", linestyle="--", alpha=0.5, label="Ideal (100%)"
            )
            ax.set_xlabel("Number of Threads")
            ax.set_ylabel("Efficiency [%]")
            ax.set_title(f"{algo} - Parallel Efficiency (OpenMP)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 120)

        save_single_plot(fig_efficiency, output_dir, f"efficiency_{safe_algo}.png")
        plt.close(fig_efficiency)

    plt.figure(fig_combined.number)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "speedup_efficiency.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig_combined)
    print("Saved: speedup_efficiency.png")


def print_summary(df):
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\nPlatforms: {', '.join(df['Platform'].unique())}")
    print(f"Algorithms: {', '.join(df['Algorithm'].unique())}")
    print(f"Engines: {', '.join(df['Engine'].unique())}")
    print(f"File Sizes: {', '.join(map(str, sorted(df['FileSize_MB'].unique())))} MB")

    print("\n" + "-" * 70)
    print("VERIFICATION STATUS")
    print("-" * 70)
    total = len(df)
    passed = len(df[df["Verified"] == "PASS"])
    print(f"Total: {total} | Passed: {passed} ({100*passed/total:.1f}%)")

    print("\n" + "-" * 70)
    print("BEST RESULTS")
    print("-" * 70)

    for algo in df["Algorithm"].unique():
        algo_data = df[df["Algorithm"] == algo]
        best = algo_data.loc[algo_data["Throughput_MBs"].idxmax()]
        print(f"\n{algo}:")
        print(f"  Platform: {best['Platform']}, Engine: {best['Engine']}")
        print(f"  Throughput: {best['Throughput_MBs']:.2f} MB/s")
        print(f"  Time: {best['Time_Sec']*1000:.2f} ms")

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        csv_path = "results/combined_results.csv"
    else:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Force output to results/image unless specified?
    # The script uses output_dir = dirname(csv_path).
    # Let's override it to reference 'results/image' if we are in project root.
    output_dir = "results/image"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)

    print_summary(df)

    print("\nGenerating charts...")
    plot_throughput_vs_filesize(df, output_dir)
    plot_execution_time_comparison(df, output_dir)
    plot_max_throughput_per_method(df, output_dir)
    plot_energy_consumption(df, output_dir)
    plot_xor_vs_aes(df, output_dir)
    plot_speedup_efficiency(df, output_dir)

    print(f"\nAll charts saved to: {output_dir}/")


if __name__ == "__main__":
    main()
