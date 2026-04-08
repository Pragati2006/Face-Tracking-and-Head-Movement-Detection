import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def _create_report_figure(face_analytics_list, agg_durations, agg_time):
    """Creates a comprehensive visual report for all faces and aggregate data."""
    num_faces = len(face_analytics_list)
    # 1 chart for aggregate + 1 row per face (max 5 for layout stability)
    display_faces = face_analytics_list[:5] # Limit display to first 5 faces to avoid overflow
    num_rows = 1 + len(display_faces)
    
    fig = plt.figure(figsize=(12, 4 * num_rows))
    fig.suptitle('Face Tracking Cumulative Analytics Report', fontsize=22, fontweight='bold', y=0.98)
    
    # --- Section 1: Aggregate Summary ---
    ax_agg_pie = plt.subplot2grid((num_rows, 2), (0, 0))
    labels = list(agg_durations.keys())
    sizes = [agg_durations[l] for l in labels]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    
    plot_labels = [l for l, s in zip(labels, sizes) if s > 0]
    plot_sizes = [s for s in sizes if s > 0]
    
    if plot_sizes:
        ax_agg_pie.pie(plot_sizes, labels=plot_labels, autopct='%1.1f%%', startangle=140, colors=colors)
    else:
        ax_agg_pie.text(0.5, 0.5, "No Data", ha='center')
    ax_agg_pie.set_title('Overall Directional Distribution', fontsize=14, fontweight='bold')

    ax_agg_text = plt.subplot2grid((num_rows, 2), (0, 1))
    ax_agg_text.axis('off')
    
    agg_attention = (agg_durations['Center'] / agg_time * 100) if agg_time > 0 else 0
    summary_text = (
        f"SESSION SUMMARY\n"
        f"----------------------------------\n"
        f"Total Unique Faces: {num_faces}\n"
        f"Total Session Time: {agg_time:.1f}s\n"
        f"Overall Attention: {agg_attention:.1f}%\n"
        f"----------------------------------"
    )
    ax_agg_text.text(0.1, 0.5, summary_text, fontsize=14, family='monospace', verticalalignment='center')

    # --- Section 2: Individual Face Breakdowns ---
    for i, face in enumerate(display_faces):
        row = i + 1
        fid = face['face_id']
        f_durations = face['durations']
        f_total = face['total_time']
        f_attention = (f_durations['Center'] / f_total * 100) if f_total > 0 else 0
        
        # Pie chart for this face
        ax_f_pie = plt.subplot2grid((num_rows, 2), (row, 0))
        f_plot_labels = [l for l, s in f_durations.items() if s > 0]
        f_plot_sizes = [s for s in f_durations.values() if s > 0]
        
        if f_plot_sizes:
            ax_f_pie.pie(f_plot_sizes, labels=f_plot_labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax_f_pie.set_title(f"Face ID: {fid} Distribution", fontsize=12)
        
        # Stats for this face
        ax_f_text = plt.subplot2grid((num_rows, 2), (row, 1))
        ax_f_text.axis('off')
        f_text = (
            f"FACE ID: {fid}\n"
            f"Total Visible: {f_total:.1f}s\n"
            f"Attention Score: {f_attention:.1f}%\n"
            f"Direction Changes: {face['change_count']}"
        )
        ax_f_text.text(0.1, 0.5, f_text, fontsize=12, family='monospace', verticalalignment='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def generate_session_report(face_analytics_list):
    """
    Generates a cumulative visual report (PNG) showing individual and aggregate data.
    """
    if not face_analytics_list:
        print("No analytics data to report.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Aggregate Statistics
    agg_durations = {d: 0.0 for d in ["Center", "Left", "Right", "Up", "Down"]}
    agg_time = 0.0
    
    for entry in face_analytics_list:
        for d, dur in entry['durations'].items():
            agg_durations[d] += dur
        agg_time += entry['total_time']

    if agg_time <= 0:
        print("Insufficient data for report.")
        return None

    # 2. Create Visualization
    fig = _create_report_figure(face_analytics_list, agg_durations, agg_time)
    report_filename = f"session_report_{timestamp}.png"
    fig.savefig(report_filename)
    plt.close(fig)

    return report_filename

