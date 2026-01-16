"""
Analyze NPU profiling results and generate insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


class ProfilingAnalyzer:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path

    def analyze_tiling_impact(self):
        """Analyze how tiling affects latency"""
        print("="*80)
        print("Tiling Impact Analysis")
        print("="*80)

        # Group by total tiles
        tile_analysis = self.df.groupby('total_tiles').agg({
            'latency_ms': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)

        print("\nLatency by Total Tiles:")
        print(tile_analysis)

        # Analyze efficiency (latency per tile)
        self.df['latency_per_tile'] = self.df['latency_ms'] / self.df['total_tiles']

        print("\nLatency per Tile Statistics:")
        print(self.df['latency_per_tile'].describe())

        return tile_analysis

    def analyze_dimension_impact(self):
        """Analyze how each dimension affects latency"""
        print("\n" + "="*80)
        print("Dimension Impact Analysis")
        print("="*80)

        # Correlation analysis
        dims = ['M', 'K', 'N', 'num_tiles_M', 'num_tiles_K', 'num_tiles_N', 'total_tiles']
        correlation = self.df[dims + ['latency_ms']].corr()['latency_ms'].sort_values(ascending=False)

        print("\nCorrelation with Latency:")
        print(correlation)

        return correlation

    def find_tiling_boundaries(self):
        """Find performance cliffs at tiling boundaries"""
        print("\n" + "="*80)
        print("Tiling Boundary Analysis")
        print("="*80)

        # Find cases where tiles change by 1
        boundaries = []

        for tiles in ['num_tiles_M', 'num_tiles_K', 'num_tiles_N']:
            df_sorted = self.df.sort_values(tiles)

            for i in range(len(df_sorted) - 1):
                curr = df_sorted.iloc[i]
                next_row = df_sorted.iloc[i + 1]

                if next_row[tiles] == curr[tiles] + 1:
                    latency_increase = next_row['latency_ms'] - curr['latency_ms']
                    pct_increase = (latency_increase / curr['latency_ms']) * 100

                    if abs(pct_increase) > 10:  # Significant change
                        boundaries.append({
                            'dimension': tiles,
                            'tiles_before': curr[tiles],
                            'tiles_after': next_row[tiles],
                            'latency_before': curr['latency_ms'],
                            'latency_after': next_row['latency_ms'],
                            'increase_pct': pct_increase
                        })

        if boundaries:
            boundary_df = pd.DataFrame(boundaries)
            print("\nSignificant Performance Changes at Boundaries:")
            print(boundary_df.to_string())
        else:
            print("\nNo significant performance cliffs found")

        return boundaries

    def recommend_buckets(self, target_efficiency: float = 0.9):
        """Recommend optimal bucket sizes for LLM serving"""
        print("\n" + "="*80)
        print(f"Bucket Recommendations (target efficiency: {target_efficiency*100}%)")
        print("="*80)

        # Calculate efficiency: actual_ops / (tiles * tile_size_ops)
        self.df['compute_efficiency'] = (self.df['M'] * self.df['K'] * self.df['N']) / \
                                        (self.df['total_tiles'] * 128 * 128 * 512)

        # Find configurations with high efficiency
        efficient_configs = self.df[self.df['compute_efficiency'] >= target_efficiency]

        print(f"\nFound {len(efficient_configs)} configurations with >= {target_efficiency*100}% efficiency")

        if len(efficient_configs) > 0:
            # Recommend buckets for M (batch_size * seq_len)
            M_buckets = sorted(efficient_configs['M'].unique())
            print(f"\nRecommended M buckets: {M_buckets[:20]}")  # First 20

            # Recommend buckets for K/N (hidden dimensions)
            K_buckets = sorted(efficient_configs['K'].unique())
            N_buckets = sorted(efficient_configs['N'].unique())
            print(f"\nRecommended K buckets: {K_buckets[:20]}")
            print(f"\nRecommended N buckets: {N_buckets[:20]}")

        return efficient_configs

    def plot_results(self, output_dir: str = None):
        """Generate visualization plots"""
        if output_dir is None:
            output_dir = str(Path(self.csv_path).parent / "plots")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 1. Latency vs Total Tiles
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['total_tiles'], self.df['latency_ms'], alpha=0.5)
        plt.xlabel('Total Tiles')
        plt.ylabel('Latency (ms)')
        plt.title('Latency vs Total Tiles')
        plt.savefig(f"{output_dir}/latency_vs_tiles.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Latency per Tile Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['latency_per_tile'], bins=50, edgecolor='black')
        plt.xlabel('Latency per Tile (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency per Tile Distribution')
        plt.savefig(f"{output_dir}/latency_per_tile_dist.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Heatmap: M vs K (fixing N to median)
        median_N = self.df['N'].median()
        subset = self.df[self.df['N'] == median_N]
        if len(subset) > 0:
            pivot = subset.pivot_table(values='latency_ms', index='M', columns='K', aggfunc='mean')
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Latency (ms)'})
            plt.title(f'Latency Heatmap (N={median_N})')
            plt.savefig(f"{output_dir}/latency_heatmap_MxK.png", dpi=150, bbox_inches='tight')
            plt.close()

        print(f"\nPlots saved to {output_dir}")

    def generate_report(self, output_path: str = None):
        """Generate comprehensive analysis report"""
        if output_path is None:
            output_path = str(Path(self.csv_path).parent / "analysis_report.txt")

        with open(output_path, 'w') as f:
            f.write("NPU Matrix Multiplication Profiling Analysis Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Dataset: {self.csv_path}\n")
            f.write(f"Total configurations tested: {len(self.df)}\n\n")

            # Basic statistics
            f.write("Latency Statistics (ms):\n")
            f.write(str(self.df['latency_ms'].describe()) + "\n\n")

            # Tiling statistics
            f.write("Tiling Statistics:\n")
            f.write(f"  Total tiles range: {self.df['total_tiles'].min()} - {self.df['total_tiles'].max()}\n")
            f.write(f"  Mean tiles: {self.df['total_tiles'].mean():.2f}\n\n")

            # Top 10 fastest configurations
            f.write("Top 10 Fastest Configurations:\n")
            top_10 = self.df.nsmallest(10, 'latency_ms')[['M', 'K', 'N', 'latency_ms', 'total_tiles']]
            f.write(top_10.to_string() + "\n\n")

            # Top 10 slowest configurations
            f.write("Top 10 Slowest Configurations:\n")
            bottom_10 = self.df.nlargest(10, 'latency_ms')[['M', 'K', 'N', 'latency_ms', 'total_tiles']]
            f.write(bottom_10.to_string() + "\n\n")

        print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze NPU profiling results')
    parser.add_argument('csv_file', type=str, help='Path to profiling results CSV file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--report', action='store_true', help='Generate text report')

    args = parser.parse_args()

    analyzer = ProfilingAnalyzer(args.csv_file)

    # Run analyses
    analyzer.analyze_tiling_impact()
    analyzer.analyze_dimension_impact()
    analyzer.find_tiling_boundaries()
    analyzer.recommend_buckets()

    if args.plot:
        analyzer.plot_results()

    if args.report:
        analyzer.generate_report()


if __name__ == "__main__":
    main()
