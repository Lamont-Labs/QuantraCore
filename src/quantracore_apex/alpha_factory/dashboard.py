"""
Equity Curve Dashboard and Plotting.

Generates visualizations of portfolio performance for the alpha factory.
"""

import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class EquityCurvePlotter:
    """
    Equity curve visualization and dashboard generation.
    
    Creates PNG charts and HTML dashboards for portfolio tracking.
    """
    
    def __init__(self, output_dir: str = "static"):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_equity_curve(
        self,
        csv_path: str = "equity_curve.csv",
        output_path: Optional[str] = None,
        title: str = "QuantraCore Live NAV"
    ) -> Optional[str]:
        """
        Generate equity curve PNG.
        
        Args:
            csv_path: Path to equity curve CSV
            output_path: Output PNG path
            title: Chart title
            
        Returns:
            Output path if successful, None otherwise
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return None
        
        if not os.path.exists(csv_path):
            logger.warning(f"Equity curve file not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty or 'nav' not in df.columns:
                logger.warning("Empty or invalid equity curve data")
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            
            output_path = output_path or os.path.join(self.output_dir, "equity.png")
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            nav_millions = df['nav'] / 1_000_000
            ax.plot(df['date'], nav_millions, linewidth=2, color='#00ff88', label='NAV')
            
            ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.3, label='Starting $1M')
            
            ax.fill_between(df['date'], nav_millions, alpha=0.2, color='#00ff88')
            
            ax.set_title(title, fontsize=16, fontweight='bold', color='white')
            ax.set_ylabel('NAV ($ Millions)', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper left')
            
            if len(df) > 0:
                start_nav = df['nav'].iloc[0]
                end_nav = df['nav'].iloc[-1]
                total_return = (end_nav - start_nav) / start_nav * 100
                
                stats_text = f"Return: {total_return:+.2f}%  |  NAV: ${end_nav/1e6:.3f}M"
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                       fontsize=10, color='#00ff88', alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, facecolor='#1a1a2e')
            plt.close()
            
            logger.info(f"Equity curve saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return None
    
    def generate_dashboard_html(
        self,
        output_path: Optional[str] = None,
        refresh_seconds: int = 300
    ) -> str:
        """
        Generate simple HTML dashboard.
        
        Args:
            output_path: Output HTML path
            refresh_seconds: Auto-refresh interval
            
        Returns:
            Output path
        """
        output_path = output_path or os.path.join(self.output_dir, "index.html")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>QuantraCore Alpha Factory</title>
    <meta http-equiv="refresh" content="{refresh_seconds}">
    <style>
        body {{
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            text-align: center;
        }}
        h1 {{
            color: #00ff88;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #888;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .chart {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.2);
        }}
        .footer {{
            margin-top: 20px;
            color: #666;
            font-size: 12px;
        }}
        .live {{
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
    </style>
</head>
<body>
    <h1><span class="live"></span>QuantraCore Alpha Factory</h1>
    <p class="subtitle">Live Paper Trading Research | Auto-refresh every {refresh_seconds // 60} minutes</p>
    <img src="equity.png?t={int(datetime.now().timestamp())}" alt="Equity Curve" class="chart">
    <p class="footer">
        Research Mode Only | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    </p>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Dashboard generated: {output_path}")
        return output_path
    
    def refresh(self, csv_path: str = "equity_curve.csv"):
        """
        Refresh equity plot and dashboard.
        
        Args:
            csv_path: Path to equity curve CSV
        """
        self.plot_equity_curve(csv_path)
        self.generate_dashboard_html()


def refresh_dashboard(csv_path: str = "equity_curve.csv", output_dir: str = "static"):
    """
    Convenience function to refresh dashboard.
    
    Args:
        csv_path: Path to equity curve CSV
        output_dir: Output directory
    """
    plotter = EquityCurvePlotter(output_dir)
    plotter.refresh(csv_path)
