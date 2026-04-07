import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def _create_report_figure(face_id, durations, change_count, total_time, is_aggregate=False):
    """Internal helper to create the matplotlib figure for a report."""
    # Calculate Metrics
    attention_score = (durations['Center'] / total_time) * 100 if total_time > 0 else 0
    minutes = total_time / 60.0
    movement_freq = change_count / minutes if minutes > 0 else 0

    # Create Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    title_suffix = " (Aggregate)" if is_aggregate else f" (Face ID: {face_id})"
    fig.suptitle('Movement Analytics Dashboard' + title_suffix, fontsize=20, fontweight='bold')

    # --- Plot 1: Direction Distribution (Pie Chart) ---
    labels = list(durations.keys())
    sizes = [durations[l] for l in labels]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    
    # Filter out zero durations for the pie chart
    plot_labels = [l for l, s in zip(labels, sizes) if s > 0]
    plot_sizes = [s for s in sizes if s > 0]
    
    if plot_sizes:
        ax1.pie(plot_sizes, labels=plot_labels, autopct='%1.1f%%', startangle=140, colors=colors)
    else:
        ax1.text(0.5, 0.5, "No Data", ha='center')
    ax1.set_title('Directional Attention Distribution', fontsize=15)

    # --- Plot 2: Metrics Summary (Text Table) ---
    ax2.axis('off')
    summary_text = (
        f"Session Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Target: {'All Faces' if is_aggregate else f'Face {face_id}'}\n"
        f"--------------------------------------------------\n"
        f"Total Observation Time: {total_time:.1f} seconds\n"
        f"--------------------------------------------------\n"
        f"ATTENTION SCORE: {attention_score:.1f}%\n"
        f"(Time spent looking directly at screen)\n"
        f"--------------------------------------------------\n"
    )
    
    ax2.text(0.1, 0.5, summary_text, fontsize=14, family='monospace', verticalalignment='center')
    ax2.set_title('Performance Metrics', fontsize=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def generate_session_report(face_analytics_list):
    """
    Generates a single visual report (PNG) aggregating data from all tracked faces.
    Returns the generated filename.
    """
    if not face_analytics_list:
        print("No analytics data to report.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Aggregate Statistics
    agg_durations = {d: 0.0 for d in ["Center", "Left", "Right", "Up", "Down"]}
    agg_changes = 0
    agg_time = 0.0
    
    for entry in face_analytics_list:
        for d, dur in entry['durations'].items():
            agg_durations[d] += dur
        agg_changes += entry['change_count']
        agg_time += sum(entry['durations'].values())

    if agg_time <= 0:
        print("Insufficient data for report.")
        return None

    # 2. Create Visualization (Aggregate)
    fig = _create_report_figure(None, agg_durations, agg_changes, agg_time, is_aggregate=True)
    report_filename = f"session_report_{timestamp}.png"
    fig.savefig(report_filename)
    plt.close(fig)

    return report_filename

