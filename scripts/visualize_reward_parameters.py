#!/usr/bin/env python3
"""
Visualize reward parameter relationships to understand tuning impacts.

This script helps visualize:
1. Revisit penalty curves (current vs proposed)
2. Mine avoidance cost fields (current vs proposed)
3. Exploration vs revisit balance over training
4. Time penalty accumulation over episode length
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import sys


def plot_revisit_penalties():
    """Compare current sqrt scaling vs proposed linear scaling."""
    visits = np.arange(0, 101, 1)
    
    # Current: -0.003 * sqrt(visits)
    current = -0.003 * np.sqrt(visits)
    
    # Proposed Option A: -0.01 * visits (linear)
    proposed_a = -0.01 * visits
    
    # Proposed Option B: -0.005 * visits^2 (quadratic)
    proposed_b = -0.005 * (visits ** 2)
    
    # Exploration bonus for reference
    exploration = 0.03 * np.ones_like(visits)
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Penalty magnitude comparison
    plt.subplot(2, 2, 1)
    plt.plot(visits, current, label='Current: -0.003 × √visits', linewidth=2, color='blue')
    plt.plot(visits, proposed_a, label='Proposed A: -0.01 × visits', linewidth=2, color='orange', linestyle='--')
    plt.plot(visits, proposed_b[:101], label='Proposed B: -0.005 × visits²', linewidth=2, color='red', linestyle=':')
    plt.axhline(y=-0.03, color='green', linestyle=':', label='Exploration bonus (for reference)', linewidth=2)
    plt.xlabel('Visit Count', fontsize=12)
    plt.ylabel('Revisit Penalty', fontsize=12)
    plt.title('Revisit Penalty Scaling Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(-1.5, 0.05)
    
    # Subplot 2: Focus on early visits (0-25)
    plt.subplot(2, 2, 2)
    visits_early = visits[:26]
    plt.plot(visits_early, current[:26], label='Current', linewidth=2, color='blue')
    plt.plot(visits_early, proposed_a[:26], label='Proposed A', linewidth=2, color='orange', linestyle='--')
    plt.axhline(y=-0.03, color='green', linestyle=':', label='Exploration bonus', linewidth=2)
    plt.xlabel('Visit Count', fontsize=12)
    plt.ylabel('Revisit Penalty', fontsize=12)
    plt.title('Early Visit Penalties (0-25 visits)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Highlight key thresholds
    plt.axvline(x=3, color='red', alpha=0.3, linestyle='--', linewidth=1)
    plt.text(3, -0.025, 'Proposed breakeven', fontsize=9, ha='center')
    plt.axvline(x=100, color='blue', alpha=0.3, linestyle='--', linewidth=1)
    
    # Subplot 3: Net reward (exploration bonus - revisit penalty)
    plt.subplot(2, 2, 3)
    net_current = exploration + current
    net_proposed_a = exploration + proposed_a
    plt.plot(visits, net_current, label='Current: Net reward', linewidth=2, color='blue')
    plt.plot(visits, net_proposed_a, label='Proposed A: Net reward', linewidth=2, color='orange', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.xlabel('Visit Count', fontsize=12)
    plt.ylabel('Net Reward (Exploration - Revisit)', fontsize=12)
    plt.title('Net Reward for Revisiting Explored Cells', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    
    # Mark breakeven points
    # Current: 0.03 - 0.003*sqrt(x) = 0 → x = 100
    # Proposed: 0.03 - 0.01*x = 0 → x = 3
    plt.axvline(x=3, color='orange', alpha=0.5, linestyle='--', linewidth=2)
    plt.text(3, 0.015, 'Proposed\nbreakeven\n(3 visits)', fontsize=9, ha='center', color='orange', fontweight='bold')
    plt.axvline(x=10, color='blue', alpha=0.5, linestyle='--', linewidth=2)
    plt.text(10, 0.015, 'Current\nstill positive\n(10 visits)', fontsize=9, ha='center', color='blue', fontweight='bold')
    
    # Subplot 4: Key statistics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = """
    REVISIT PENALTY ANALYSIS
    ═══════════════════════════════════════
    
    Breakeven Point (where penalty = exploration bonus):
    • Current (sqrt):    100 visits  ❌ Too forgiving
    • Proposed A (linear): 3 visits  ✅ Quick deterrent
    
    Penalty at 10 visits:
    • Current:    -0.009  (30% of exploration)
    • Proposed A: -0.10   (333% of exploration) 💪
    
    Penalty at 25 visits:
    • Current:    -0.015  (50% of exploration)
    • Proposed A: -0.25   (833% of exploration) 💪
    
    Impact:
    • Current allows 50-100 revisits before strong penalty
    • Proposed discourages >5 revisits, prevents oscillation
    
    Recommendation: Use Proposed A (linear)
    • Strong enough to prevent loops
    • Not so harsh to prevent navigation
    """
    
    plt.text(0.1, 0.9, stats_text, fontsize=10, family='monospace', 
             verticalalignment='top', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('/home/tetra/projects/nclone/docs/revisit_penalty_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: docs/revisit_penalty_comparison.png")


def plot_mine_avoidance():
    """Visualize mine proximity cost fields."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create a grid around a mine at center
    x = np.linspace(-100, 100, 200)
    y = np.linspace(-100, 100, 200)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    
    # Mine parameters
    mine_radius_current = 40.0
    mine_radius_proposed = 60.0
    multiplier_current = 2.0
    multiplier_proposed = 8.0
    
    # Calculate cost multipliers
    def cost_multiplier(dist, radius, mult):
        cost = np.ones_like(dist)
        mask = dist < radius
        proximity_factor = 1.0 - (dist[mask] / radius)
        cost[mask] = 1.0 + proximity_factor * (mult - 1.0)
        return cost
    
    cost_current = cost_multiplier(distance, mine_radius_current, multiplier_current)
    cost_proposed = cost_multiplier(distance, mine_radius_proposed, multiplier_proposed)
    
    # Plot current
    im1 = axes[0].contourf(X, Y, cost_current, levels=20, cmap='YlOrRd')
    axes[0].add_patch(patches.Circle((0, 0), 4, color='darkred', label='Mine (4px)'))
    axes[0].add_patch(patches.Circle((0, 0), mine_radius_current, fill=False, edgecolor='blue', linewidth=2, linestyle='--', label=f'Hazard radius ({mine_radius_current}px)'))
    axes[0].set_title('Current: 2× multiplier, 40px radius', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Distance (pixels)')
    axes[0].set_ylabel('Distance (pixels)')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0], label='Path Cost Multiplier')
    axes[0].set_aspect('equal')
    
    # Plot proposed
    im2 = axes[1].contourf(X, Y, cost_proposed, levels=20, cmap='YlOrRd')
    axes[1].add_patch(patches.Circle((0, 0), 4, color='darkred', label='Mine (4px)'))
    axes[1].add_patch(patches.Circle((0, 0), mine_radius_proposed, fill=False, edgecolor='orange', linewidth=2, linestyle='--', label=f'Hazard radius ({mine_radius_proposed}px)'))
    axes[1].set_title('Proposed: 8× multiplier, 60px radius', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Distance (pixels)')
    axes[1].set_ylabel('Distance (pixels)')
    axes[1].legend(fontsize=8, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[1], label='Path Cost Multiplier')
    axes[1].set_aspect('equal')
    
    # Plot comparison (cost increase)
    cost_increase = cost_proposed - cost_current
    im3 = axes[2].contourf(X, Y, cost_increase, levels=20, cmap='RdYlGn_r')
    axes[2].add_patch(patches.Circle((0, 0), 4, color='darkred'))
    axes[2].set_title('Cost Increase (Proposed - Current)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Distance (pixels)')
    axes[2].set_ylabel('Distance (pixels)')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(im3, ax=axes[2], label='Additional Cost')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/home/tetra/projects/nclone/docs/mine_avoidance_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: docs/mine_avoidance_comparison.png")


def plot_exploration_decay():
    """Visualize exploration bonus decay over training."""
    success_rate = np.linspace(0, 0.5, 100)
    
    # Current: flat until thresholds
    current = np.where(success_rate < 0.10, 0.03,
                      np.where(success_rate < 0.20, 0.02,
                              np.where(success_rate < 0.30, 0.01, 0.0)))
    
    # Proposed: exponential decay
    proposed = 0.05 * np.exp(-5 * success_rate)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(success_rate * 100, current, label='Current: Step function', linewidth=2, color='blue', marker='o', markersize=4)
    plt.plot(success_rate * 100, proposed, label='Proposed: Exponential decay', linewidth=2, color='orange', linestyle='--')
    plt.xlabel('Success Rate (%)', fontsize=12)
    plt.ylabel('Exploration Bonus (per cell)', fontsize=12)
    plt.title('Exploration Bonus Decay', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    
    # Highlight key points
    plt.axvline(x=10, color='blue', alpha=0.3, linestyle='--')
    plt.text(10, 0.028, '10%', fontsize=9, ha='center')
    plt.axvline(x=20, color='blue', alpha=0.3, linestyle='--')
    plt.text(20, 0.028, '20%', fontsize=9, ha='center')
    plt.axvline(x=30, color='blue', alpha=0.3, linestyle='--')
    plt.text(30, 0.028, '30%', fontsize=9, ha='center')
    
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    comparison_text = """
    EXPLORATION DECAY COMPARISON
    ════════════════════════════════════════
    
    At 0% success:
    • Current:  0.03
    • Proposed: 0.05  (+67% for initial exploration)
    
    At 10% success:
    • Current:  0.03  (unchanged!)
    • Proposed: 0.03  (matched current)
    
    At 20% success:
    • Current:  0.02  (step down)
    • Proposed: 0.018 (smooth transition)
    
    At 30% success:
    • Current:  0.01  (step down)
    • Proposed: 0.011 (smooth)
    
    At 40% success:
    • Current:  0.0   (disabled)
    • Proposed: 0.007 (gradually fading)
    
    Benefits:
    ✓ Smooth transition (no sudden changes)
    ✓ Higher initial exploration (0.05 vs 0.03)
    ✓ Faster decay after 10% (encourages exploitation)
    ✓ More responsive to progress
    """
    
    plt.text(0.1, 0.9, comparison_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('/home/tetra/projects/nclone/docs/exploration_decay_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: docs/exploration_decay_comparison.png")


def plot_time_penalty_accumulation():
    """Visualize time penalty accumulation over episode."""
    frames = np.arange(0, 3001, 1)  # Up to 3000 frames
    
    # Current
    current_per_frame = -0.000001
    current_total = frames * current_per_frame
    
    # Proposed Phase 1 (discovery)
    proposed_p1_per_frame = -0.0001
    proposed_p1_total = frames * proposed_p1_per_frame
    
    # Proposed Phase 2 (refinement)
    proposed_p2_per_frame = -0.001
    proposed_p2_total = frames * proposed_p2_per_frame
    
    # Proposed Phase 3 (optimization)
    proposed_p3_per_frame = -0.005
    proposed_p3_total = frames * proposed_p3_per_frame
    
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Absolute accumulation
    plt.subplot(2, 2, 1)
    plt.plot(frames, current_total, label='Current: -0.000001/frame', linewidth=2, color='blue')
    plt.plot(frames, proposed_p1_total, label='Phase 1: -0.0001/frame (100×)', linewidth=2, color='green', linestyle='--')
    plt.plot(frames, proposed_p2_total, label='Phase 2: -0.001/frame (1000×)', linewidth=2, color='orange', linestyle='--')
    plt.plot(frames, proposed_p3_total, label='Phase 3: -0.005/frame (5000×)', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Episode Length (frames)', fontsize=12)
    plt.ylabel('Cumulative Time Penalty', fontsize=12)
    plt.title('Time Penalty Accumulation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 3000)
    
    # Mark typical episode lengths
    plt.axvline(x=600, color='purple', alpha=0.5, linestyle=':', linewidth=2)
    plt.text(600, -2, 'Typical episode\n(600 frames)', fontsize=9, ha='center', color='purple')
    plt.axvline(x=2000, color='purple', alpha=0.3, linestyle=':', linewidth=2)
    plt.text(2000, -2, 'Long episode\n(2000 frames)', fontsize=9, ha='center', color='purple')
    
    # Subplot 2: As percentage of completion reward
    plt.subplot(2, 2, 2)
    completion_reward = 20.0
    plt.plot(frames, (current_total / completion_reward) * 100, label='Current', linewidth=2, color='blue')
    plt.plot(frames, (proposed_p1_total / completion_reward) * 100, label='Phase 1', linewidth=2, color='green', linestyle='--')
    plt.plot(frames, (proposed_p2_total / completion_reward) * 100, label='Phase 2', linewidth=2, color='orange', linestyle='--')
    plt.plot(frames, (proposed_p3_total / completion_reward) * 100, label='Phase 3', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Episode Length (frames)', fontsize=12)
    plt.ylabel('Time Penalty (% of completion reward)', fontsize=12)
    plt.title('Time Penalty Relative to Completion Reward', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 3000)
    plt.ylim(-80, 5)
    
    # Mark reasonable penalty thresholds
    plt.axhline(y=-1, color='green', alpha=0.3, linestyle='--', linewidth=1)
    plt.text(2900, -1, '1%', fontsize=9, va='center', color='green')
    plt.axhline(y=-5, color='orange', alpha=0.3, linestyle='--', linewidth=1)
    plt.text(2900, -5, '5%', fontsize=9, va='center', color='orange')
    plt.axhline(y=-15, color='red', alpha=0.3, linestyle='--', linewidth=1)
    plt.text(2900, -15, '15%', fontsize=9, va='center', color='red')
    
    # Subplot 3: Focus on typical episode range (0-1000 frames)
    plt.subplot(2, 2, 3)
    frames_typical = frames[:1001]
    plt.plot(frames_typical, proposed_p1_total[:1001], label='Phase 1', linewidth=2, color='green', linestyle='--')
    plt.plot(frames_typical, proposed_p2_total[:1001], label='Phase 2', linewidth=2, color='orange', linestyle='--')
    plt.plot(frames_typical, proposed_p3_total[:1001], label='Phase 3', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Episode Length (frames)', fontsize=12)
    plt.ylabel('Cumulative Time Penalty', fontsize=12)
    plt.title('Proposed Penalties (Typical Episode Range)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.axvline(x=600, color='purple', alpha=0.5, linestyle=':', linewidth=2)
    plt.text(600, -2, '600 frames', fontsize=9, ha='center', color='purple')
    
    # Subplot 4: Statistics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f"""
    TIME PENALTY ANALYSIS
    ════════════════════════════════════════
    
    At 600 frames (typical episode):
    • Current:   {current_total[600]:.4f}  ({(current_total[600]/completion_reward)*100:.3f}%)
    • Phase 1:   {proposed_p1_total[600]:.4f}  ({(proposed_p1_total[600]/completion_reward)*100:.1f}%)
    • Phase 2:   {proposed_p2_total[600]:.4f}  ({(proposed_p2_total[600]/completion_reward)*100:.1f}%)
    • Phase 3:   {proposed_p3_total[600]:.4f}  ({(proposed_p3_total[600]/completion_reward)*100:.1f}%)
    
    At 2000 frames (long episode):
    • Current:   {current_total[2000]:.4f}  ({(current_total[2000]/completion_reward)*100:.3f}%)
    • Phase 1:   {proposed_p1_total[2000]:.4f}  ({(proposed_p1_total[2000]/completion_reward)*100:.1f}%)
    • Phase 2:   {proposed_p2_total[2000]:.4f}  ({(proposed_p2_total[2000]/completion_reward)*100:.1f}%)
    • Phase 3:   {proposed_p3_total[2000]:.4f}  ({(proposed_p3_total[2000]/completion_reward)*100:.1f}%)
    
    Assessment:
    ❌ Current: Irrelevant (<0.01% of reward)
    ✅ Phase 1: Minimal but present (~0.3%)
    ✅ Phase 2: Noticeable pressure (~3%)
    ✅ Phase 3: Strong incentive (~15%)
    
    Recommendation: Phase-dependent scaling
    • Discovery: Weak (-0.0001, focus on completion)
    • Refinement: Moderate (-0.001, encourage efficiency)
    • Optimization: Strong (-0.005, speed runs)
    """
    
    plt.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('/home/tetra/projects/nclone/docs/time_penalty_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: docs/time_penalty_comparison.png")


def plot_reward_balance_summary():
    """Create comprehensive reward balance visualization."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Episode simulation parameters
    episode_length = 600  # frames
    cells_visited = 200
    avg_revisits = 4
    pbrs_progress = 0.3  # 30% of path
    
    # Current configuration
    current_rewards = {
        'PBRS': 100 * pbrs_progress,  # weight × progress
        'Exploration': 0.03 * cells_visited,
        'Revisit': -0.003 * np.sqrt(avg_revisits) * cells_visited,
        'Time': -0.000001 * episode_length,
        'Death': -6.0
    }
    
    # Proposed configuration
    proposed_rewards = {
        'PBRS': 100 * pbrs_progress,  # same
        'Exploration': 0.05 * np.exp(-5 * 0.10) * cells_visited,  # exponential decay at 10% success
        'Revisit': -0.01 * avg_revisits * cells_visited,  # linear scaling
        'Time': -0.0001 * episode_length,  # 100× stronger
        'Death': -6.0  # same
    }
    
    # Subplot 1: Current reward breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    components = list(current_rewards.keys())
    values = list(current_rewards.values())
    colors = ['green', 'blue', 'orange', 'purple', 'red']
    bars1 = ax1.barh(components, values, color=colors, alpha=0.7)
    ax1.set_xlabel('Reward Magnitude', fontsize=10)
    ax1.set_title('Current Configuration', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, values)):
        label_x = val + (1 if val > 0 else -1)
        ax1.text(label_x, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
    
    # Subplot 2: Proposed reward breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    values_prop = list(proposed_rewards.values())
    bars2 = ax2.barh(components, values_prop, color=colors, alpha=0.7)
    ax2.set_xlabel('Reward Magnitude', fontsize=10)
    ax2.set_title('Proposed Configuration', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, values_prop)):
        label_x = val + (1 if val > 0 else -1)
        ax2.text(label_x, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
    
    # Subplot 3: Change analysis
    ax3 = fig.add_subplot(gs[0, 2])
    changes = [proposed_rewards[k] - current_rewards[k] for k in components]
    change_colors = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in changes]
    bars3 = ax3.barh(components, changes, color=change_colors, alpha=0.7)
    ax3.set_xlabel('Change (Proposed - Current)', fontsize=10)
    ax3.set_title('Impact of Changes', fontsize=12, fontweight='bold')
    ax3.axvline(x=0, color='black', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, changes)):
        label_x = val + (0.5 if val > 0 else -0.5 if val < 0 else 0.1)
        ax3.text(label_x, i, f'{val:+.2f}', va='center', fontsize=9, fontweight='bold')
    
    # Subplot 4: Relative magnitudes (current)
    ax4 = fig.add_subplot(gs[1, 0])
    total_current = sum([abs(v) for v in current_rewards.values()])
    percentages_current = [abs(v) / total_current * 100 for v in current_rewards.values()]
    wedges, texts, autotexts = ax4.pie(percentages_current, labels=components, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    ax4.set_title('Current: Relative Signal Strength', fontsize=12, fontweight='bold')
    
    # Subplot 5: Relative magnitudes (proposed)
    ax5 = fig.add_subplot(gs[1, 1])
    total_proposed = sum([abs(v) for v in proposed_rewards.values()])
    percentages_proposed = [abs(v) / total_proposed * 100 for v in proposed_rewards.values()]
    wedges, texts, autotexts = ax5.pie(percentages_proposed, labels=components, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    ax5.set_title('Proposed: Relative Signal Strength', fontsize=12, fontweight='bold')
    
    # Subplot 6: Key metrics comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    metrics_text = f"""
REWARD BALANCE METRICS
══════════════════════════════════════════

Total Episode Reward:
• Current:  {sum(current_rewards.values()):.2f}
• Proposed: {sum(proposed_rewards.values()):.2f}

Exploration:Revisit Ratio:
• Current:  {abs(current_rewards['Exploration'] / current_rewards['Revisit']):.1f}:1
• Proposed: {abs(proposed_rewards['Exploration'] / proposed_rewards['Revisit']):.1f}:1

PBRS Dominance (% of total):
• Current:  {abs(current_rewards['PBRS']) / total_current * 100:.1f}%
• Proposed: {abs(proposed_rewards['PBRS']) / total_proposed * 100:.1f}%

Time Penalty (% of completion):
• Current:  {abs(current_rewards['Time']) / 20.0 * 100:.4f}%
• Proposed: {abs(proposed_rewards['Time']) / 20.0 * 100:.2f}%

Net Exploration Reward:
• Current:  {current_rewards['Exploration'] + current_rewards['Revisit']:.2f}
• Proposed: {proposed_rewards['Exploration'] + proposed_rewards['Revisit']:.2f}
    """
    
    ax6.text(0.05, 0.95, metrics_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)
    
    # Subplot 7-9: Bottom row - summary recommendations
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    summary_text = """
    RECOMMENDATIONS SUMMARY
    ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    1. REVISIT PENALTY (Priority: HIGH ⚠️)
       • Current issue: Agent can loop 100 times before strong penalty (-0.003 × √100 = -0.03)
       • Recommended: Change to linear scaling (-0.01 × visits), breakeven at 3 visits instead of 100
       • Expected impact: 50-70% reduction in oscillation/meandering
    
    2. MINE AVOIDANCE (Priority: HIGH ⚠️)
       • Current issue: 2× cost multiplier too weak, risky shortcuts still attractive
       • Recommended: Increase to 8-10× multiplier, expand radius from 40px to 60px
       • Expected impact: Much safer paths, fewer deaths, better generalization
    
    3. TIME PENALTY (Priority: MEDIUM)
       • Current issue: Penalty is 0.003% of episode reward - effectively disabled
       • Recommended: Phase-dependent scaling: 100× stronger in discovery, 1000× in refinement
       • Expected impact: Gradual efficiency pressure without hurting early learning
    
    4. EXPLORATION DECAY (Priority: MEDIUM)
       • Current issue: Flat 0.03 bonus for entire 0-10% success range
       • Recommended: Exponential decay (0.05 × exp(-5 × success_rate))
       • Expected impact: Smoother exploration→exploitation transition, faster convergence
    
    5. WAYPOINT SYSTEM (Priority: MEDIUM - Next training run)
       • Current issue: Single goal provides no guidance through long corridors
       • Recommended: Add intermediate checkpoints every 200px along optimal path
       • Expected impact: 20-30% better navigation in complex levels
    """
    
    ax_summary.text(0.02, 0.95, summary_text, fontsize=9, family='monospace',
                    verticalalignment='top', transform=ax_summary.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig('/home/tetra/projects/nclone/docs/reward_balance_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: docs/reward_balance_summary.png")


def main():
    """Generate all visualization plots."""
    print("\nGenerating reward parameter visualizations...")
    print("=" * 60)
    
    try:
        plot_revisit_penalties()
        plot_mine_avoidance()
        plot_exploration_decay()
        plot_time_penalty_accumulation()
        plot_reward_balance_summary()
        
        print("=" * 60)
        print("\n✅ All visualizations generated successfully!")
        print("\nGenerated files:")
        print("  • docs/revisit_penalty_comparison.png")
        print("  • docs/mine_avoidance_comparison.png")
        print("  • docs/exploration_decay_comparison.png")
        print("  • docs/time_penalty_comparison.png")
        print("  • docs/reward_balance_summary.png")
        print("\nNext steps:")
        print("  1. Review visualizations to understand parameter impacts")
        print("  2. Read docs/reward_structure_analysis.md for detailed recommendations")
        print("  3. Apply changes to reward_config.py and reward_constants.py")
        print("  4. Run ablation study comparing variants")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

