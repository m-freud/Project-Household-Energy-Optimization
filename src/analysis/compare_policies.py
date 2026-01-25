"""
Policy Performance Analysis & Comparison

This module provides comprehensive analysis and visualization of household energy
optimization policies. Generates executive-ready reports comparing cost, consumption,
and efficiency metrics across different control strategies.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


class PolicyAnalyzer:
    """Analyze and compare energy optimization policy performance."""
    
    def __init__(self, db_path='sqlite/energy.db'):
        """Initialize analyzer with database connection."""
        self.db_path = db_path
        self.conn = None
        self.results_df = None
        
    def load_results(self) -> pd.DataFrame:
        """Load simulation results from SQLite database."""
        query = """
            SELECT 
                policy_name,
                player_id,
                CAST(total_cost as REAL) as total_cost,
                CAST(total_grid_import as REAL) as total_grid_import,
                CAST(total_grid_export as REAL) as total_grid_export,
                CAST(total_pv_generation as REAL) as total_pv_generation,
                CAST(total_self_consumption as REAL) as total_self_consumption,
                CAST(bess_cycles as REAL) as bess_cycles,
                CAST(ev1_final_soc as REAL) as ev1_final_soc,
                CAST(ev2_final_soc as REAL) as ev2_final_soc,
                has_pv,
                has_bess,
                has_ev1,
                has_ev2
            FROM simulation_results
        """
        df = pd.read_sql_query(query, self.db_path)
        
        # Convert boolean columns
        for col in ['has_pv', 'has_bess', 'has_ev1', 'has_ev2']:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        return df
    
    def compute_summary_stats(self):
        """Compute comprehensive summary statistics by policy."""
        if self.results_df is None:
            self.load_results()
        
        assert self.results_df is not None, "Failed to load results"
        summary = self.results_df.groupby('policy').agg({
            'total_cost': ['mean', 'std', 'median', 'min', 'max'],
            'total_consumption': ['mean', 'std', 'median', 'min', 'max'],
            'player_id': 'count'
        }).round(2)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={'player_id_count': 'num_households'})
        
        return summary
    
    def compute_savings(self, baseline_policy='no_control'):
        """Compute cost and consumption savings vs baseline."""
        if self.results_df is None:
            self.load_results()
        
        baseline = self.results_df[self.results_df['policy'] == baseline_policy]
        
        if baseline.empty:
            print(f"Warning: Baseline policy '{baseline_policy}' not found.")
            return None
        
        baseline_avg_cost = baseline['total_cost'].mean()
        baseline_avg_consumption = baseline['total_consumption'].mean()
        
        savings = []
        for policy in self.results_df['policy'].unique():
            if policy == baseline_policy:
                continue
            
            policy_data = self.results_df[self.results_df['policy'] == policy]
            avg_cost = policy_data['total_cost'].mean()
            avg_consumption = policy_data['total_consumption'].mean()
            
            cost_savings = baseline_avg_cost - avg_cost
            cost_savings_pct = (cost_savings / baseline_avg_cost * 100) if baseline_avg_cost > 0 else 0
            
            consumption_savings = baseline_avg_consumption - avg_consumption
            consumption_savings_pct = (consumption_savings / baseline_avg_consumption * 100) if baseline_avg_consumption > 0 else 0
            
            savings.append({
                'policy': policy,
                'cost_savings_abs': cost_savings,
                'cost_savings_pct': cost_savings_pct,
                'consumption_savings_abs': consumption_savings,
                'consumption_savings_pct': consumption_savings_pct
            })
        
        return pd.DataFrame(savings)
    
    def analyze_by_configuration(self):
        """Analyze performance by household configuration (PV/BESS)."""
        if self.results_df is None:
            self.load_results()
        
        # Create configuration groups
        self.results_df['config'] = self.results_df.apply(
            lambda row: f"PV={row['has_pv']}, BESS={row['has_bess']}", axis=1
        )
        
        config_analysis = self.results_df.groupby(['policy', 'config']).agg({
            'total_cost': 'mean',
            'total_consumption': 'mean',
            'player_id': 'count'
        }).round(2)
        
        config_analysis.columns = ['avg_cost', 'avg_consumption', 'count']
        
        return config_analysis
    
    def print_executive_summary(self, baseline_policy='no_control'):
        """Print executive summary report."""
        print("=" * 80)
        print(" " * 20 + "POLICY PERFORMANCE EXECUTIVE SUMMARY")
        print("=" * 80)
        print()
        
        # Overall statistics
        print("📊 OVERALL STATISTICS")
        print("-" * 80)
        summary = self.compute_summary_stats()
        print(summary.to_string())
        print()
        
        # Savings analysis
        print("💰 SAVINGS vs BASELINE")
        print("-" * 80)
        savings = self.compute_savings(baseline_policy)
        if savings is not None:
            for _, row in savings.iterrows():
                print(f"\n{row['policy'].upper()}")
                print(f"  Cost Savings:        ${row['cost_savings_abs']:.2f} ({row['cost_savings_pct']:.1f}%)")
                print(f"  Consumption Savings: {row['consumption_savings_abs']:.2f} kWh ({row['consumption_savings_pct']:.1f}%)")
        print()
        
        # Configuration breakdown
        print("🏠 PERFORMANCE BY HOUSEHOLD CONFIGURATION")
        print("-" * 80)
        config_analysis = self.analyze_by_configuration()
        print(config_analysis.to_string())
        print()
        
        print("=" * 80)
    
    def plot_cost_comparison(self, save_path='results/cost_comparison.png'):
        """Create violin plot comparing cost distributions."""
        if self.results_df is None:
            self.load_results()
        
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=self.results_df, x='policy', y='total_cost', palette='Set2')
        plt.title('Total Cost Distribution by Policy', fontsize=16, fontweight='bold')
        plt.xlabel('Policy', fontsize=12)
        plt.ylabel('Total Cost ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Create directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_consumption_comparison(self, save_path='results/consumption_comparison.png'):
        """Create violin plot comparing consumption distributions."""
        if self.results_df is None:
            self.load_results()
        
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=self.results_df, x='policy', y='total_consumption', palette='Set3')
        plt.title('Total Consumption Distribution by Policy', fontsize=16, fontweight='bold')
        plt.xlabel('Policy', fontsize=12)
        plt.ylabel('Total Consumption (kWh)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_savings_bar(self, baseline_policy='no_control', save_path='results/savings_comparison.png'):
        """Create bar chart showing savings vs baseline."""
        savings = self.compute_savings(baseline_policy)
        
        if savings is None or savings.empty:
            print("No savings data to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cost savings
        colors = ['green' if x > 0 else 'red' for x in savings['cost_savings_pct']]
        ax1.barh(savings['policy'], savings['cost_savings_pct'], color=colors, alpha=0.7)
        ax1.axvline(0, color='black', linewidth=0.8)
        ax1.set_xlabel('Cost Savings (%)', fontsize=12)
        ax1.set_title('Cost Savings vs Baseline', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Consumption savings
        colors = ['green' if x > 0 else 'red' for x in savings['consumption_savings_pct']]
        ax2.barh(savings['policy'], savings['consumption_savings_pct'], color=colors, alpha=0.7)
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Consumption Savings (%)', fontsize=12)
        ax2.set_title('Consumption Savings vs Baseline', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_configuration_heatmap(self, save_path='results/config_heatmap.png'):
        """Create heatmap showing performance by household configuration."""
        if self.results_df is None:
            self.load_results()
        
        # Pivot for cost heatmap
        pivot_cost = self.results_df.pivot_table(
            values='total_cost',
            index='policy',
            columns='config',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_cost, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                    cbar_kws={'label': 'Average Cost ($)'})
        plt.title('Average Cost by Policy & Configuration', fontsize=16, fontweight='bold')
        plt.xlabel('Household Configuration', fontsize=12)
        plt.ylabel('Policy', fontsize=12)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_comprehensive_dashboard(self, baseline_policy='no_control', save_path='results/dashboard.png'):
        """Create comprehensive 4-panel dashboard."""
        if self.results_df is None:
            self.load_results()
        
        savings = self.compute_savings(baseline_policy)
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Cost Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=self.results_df, x='policy', y='total_cost', ax=ax1, palette='Set2')
        ax1.set_title('Cost Distribution by Policy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Policy', fontsize=10)
        ax1.set_ylabel('Total Cost ($)', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Panel 2: Consumption Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=self.results_df, x='policy', y='total_consumption', ax=ax2, palette='Set3')
        ax2.set_title('Consumption Distribution by Policy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Policy', fontsize=10)
        ax2.set_ylabel('Total Consumption (kWh)', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Panel 3: Savings Bar Chart
        ax3 = fig.add_subplot(gs[1, 0])
        if savings is not None and not savings.empty:
            colors = ['green' if x > 0 else 'red' for x in savings['cost_savings_pct']]
            ax3.barh(savings['policy'], savings['cost_savings_pct'], color=colors, alpha=0.7)
            ax3.axvline(0, color='black', linewidth=0.8)
            ax3.set_xlabel('Cost Savings (%)', fontsize=10)
            ax3.set_title(f'Savings vs {baseline_policy}', fontsize=14, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
        
        # Panel 4: Summary Statistics Table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        summary = self.compute_summary_stats()[['total_cost_mean', 'total_consumption_mean', 'num_households']]
        summary = summary.round(2)
        
        table_data = []
        table_data.append(['Policy', 'Avg Cost ($)', 'Avg Consumption (kWh)', 'N'])
        for policy, row in summary.iterrows():
            table_data.append([
                policy,
                f"{row['total_cost_mean']:.2f}",
                f"{row['total_consumption_mean']:.2f}",
                f"{int(row['num_households'])}"
            ])
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.3, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Energy Optimization Policy Performance Dashboard', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def generate_full_report(self, baseline_policy='no_control', output_dir='results'):
        """Generate complete analysis report with all visualizations."""
        print("\n" + "=" * 80)
        print(" " * 25 + "GENERATING ANALYSIS REPORT")
        print("=" * 80 + "\n")
        
        # Load data
        self.load_results()
        print(f"✓ Loaded {len(self.results_df)} simulation results")
        print(f"✓ Policies analyzed: {', '.join(self.results_df['policy'].unique())}\n")
        
        # Print summary
        self.print_executive_summary(baseline_policy)
        
        # Generate visualizations
        print("📈 GENERATING VISUALIZATIONS")
        print("-" * 80)
        self.plot_comprehensive_dashboard(baseline_policy, f'{output_dir}/dashboard.png')
        self.plot_cost_comparison(f'{output_dir}/cost_comparison.png')
        self.plot_consumption_comparison(f'{output_dir}/consumption_comparison.png')
        self.plot_savings_bar(baseline_policy, f'{output_dir}/savings_comparison.png')
        self.plot_configuration_heatmap(f'{output_dir}/config_heatmap.png')
        
        print("\n" + "=" * 80)
        print(" " * 28 + "REPORT COMPLETE")
        print("=" * 80 + "\n")
        
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Run analysis
    analyzer = PolicyAnalyzer()
    analyzer.generate_full_report(baseline_policy='no_control', output_dir='results')
