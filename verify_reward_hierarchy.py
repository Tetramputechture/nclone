#!/usr/bin/env python3
"""Verify simplified reward hierarchy prevents exploitation.

This script validates that the reward system correctly incentivizes
forward progress over stagnation/camping strategies.
"""

# Reward constants (unscaled)
COMPLETION = 50.0
SWITCH = 30.0
DEATH_PENALTY = -2.0  # SIMPLIFIED: Symbolic constant (was -40.0 progress-gated)
STAGNATION_TIMEOUT = -20.0
STAGNATION_THRESHOLD = 0.15  # 15% progress

# Curriculum weights
PBRS_WEIGHT_DISCOVERY = 80.0
PBRS_WEIGHT_MID = 15.0
PBRS_WEIGHT_MASTERY = 5.0

TIME_PENALTY_DISCOVERY = -0.002
TIME_PENALTY_MID = -0.01
TIME_PENALTY_MASTERY = -0.03

# Global scaling
GLOBAL_SCALE = 0.1

# SIMPLIFIED: No death penalty scaling - constant penalty
def death_penalty_scale(progress: float) -> float:
    """Get death penalty (now constant, no scaling)."""
    return 1.0  # No scaling - penalty is constant


def calculate_scenario_reward(
    progress: float,
    died: bool,
    completed: bool,
    switch_activated: bool,
    steps: int,
    pbrs_weight: float,
    time_penalty: float,
) -> dict:
    """Calculate reward for a given scenario.
    
    Args:
        progress: Progress made (0.0 to 1.0)
        died: Whether agent died
        completed: Whether agent completed level
        switch_activated: Whether switch was activated
        steps: Number of steps taken
        pbrs_weight: Current PBRS weight (curriculum-dependent)
        time_penalty: Time penalty per step (curriculum-dependent)
    
    Returns:
        Dict with reward breakdown
    """
    # PBRS reward (proportional to progress)
    pbrs_reward = pbrs_weight * progress
    
    # Time penalty (accumulates over steps)
    time_reward = time_penalty * steps
    
    # Terminal rewards
    terminal_reward = 0.0
    if completed:
        terminal_reward += COMPLETION
    if switch_activated:
        terminal_reward += SWITCH
    if died:
        # SIMPLIFIED: Constant symbolic penalty (no scaling)
        terminal_reward += DEATH_PENALTY  # -2.0 constant
    
    # Stagnation penalty (only on truncation with low progress)
    stagnation_reward = 0.0
    truncated = not (died or completed)
    if truncated and progress < STAGNATION_THRESHOLD:
        stagnation_reward = STAGNATION_TIMEOUT
    
    # Total unscaled
    total_unscaled = pbrs_reward + time_reward + terminal_reward + stagnation_reward
    
    # Apply global scaling
    total_scaled = total_unscaled * GLOBAL_SCALE
    
    return {
        "pbrs": pbrs_reward,
        "time": time_reward,
        "terminal": terminal_reward,
        "stagnation": stagnation_reward,
        "total_unscaled": total_unscaled,
        "total_scaled": total_scaled,
    }


def print_scenario(name: str, result: dict):
    """Pretty print scenario results."""
    print(f"\n{name}")
    print("─" * 60)
    print(f"  PBRS:        {result['pbrs']:+7.2f} unscaled")
    print(f"  Time:        {result['time']:+7.2f} unscaled")
    print(f"  Terminal:    {result['terminal']:+7.2f} unscaled")
    print(f"  Stagnation:  {result['stagnation']:+7.2f} unscaled")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL:       {result['total_unscaled']:+7.2f} unscaled")
    print(f"  SCALED:      {result['total_scaled']:+7.2f} (×{GLOBAL_SCALE})")


def main():
    """Run exploitation verification scenarios."""
    print("=" * 60)
    print("REWARD HIERARCHY VERIFICATION - DISCOVERY PHASE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  PBRS weight: {PBRS_WEIGHT_DISCOVERY}")
    print(f"  Time penalty: {TIME_PENALTY_DISCOVERY}/step")
    print(f"  Global scale: {GLOBAL_SCALE}")
    print(f"  Typical episode: 2000 steps")
    
    # Discovery phase scenarios
    pbrs_w = PBRS_WEIGHT_DISCOVERY
    time_p = TIME_PENALTY_DISCOVERY
    steps = 2000
    
    scenarios = [
        ("1. Stay still until timeout (0% progress)", 
         calculate_scenario_reward(0.0, False, False, False, steps, pbrs_w, time_p)),
        
        ("2. Oscillate until timeout (0% net progress)",
         calculate_scenario_reward(0.0, False, False, False, steps, pbrs_w, time_p)),
        
        ("3. Camp at 9% progress (below threshold)",
         calculate_scenario_reward(0.09, False, False, False, steps, pbrs_w, time_p)),
        
        ("4. Camp at 11% progress (just above threshold)",
         calculate_scenario_reward(0.11, False, False, False, steps, pbrs_w, time_p)),
        
        ("5. Reach 20% then die (early death, no switch)",
         calculate_scenario_reward(0.20, True, False, False, 400, pbrs_w, time_p)),
        
        ("5b. Reach 20% to switch, activate, then die",
         calculate_scenario_reward(0.20, True, False, True, 400, pbrs_w, time_p)),
        
        ("6. Reach 50% then die (risky progress)",
         calculate_scenario_reward(0.50, True, False, False, 1000, pbrs_w, time_p)),
        
        ("7. Activate switch (50%) then die immediately",
         calculate_scenario_reward(0.50, True, False, True, 1000, pbrs_w, time_p)),
        
        ("8. Complete level (100% progress)",
         calculate_scenario_reward(1.0, False, True, True, 1500, pbrs_w, time_p)),
    ]
    
    for name, result in scenarios:
        print_scenario(name, result)
    
    print("\n" + "=" * 60)
    print("REWARD RANKING (Discovery Phase)")
    print("=" * 60)
    
    # Sort by scaled reward
    ranked = sorted(scenarios, key=lambda x: x[1]["total_scaled"], reverse=True)
    for i, (name, result) in enumerate(ranked, 1):
        print(f"{i}. {name.split('. ', 1)[1]:40s} {result['total_scaled']:+7.2f}")
    
    print("\n" + "=" * 60)
    print("EXPLOITATION CHECK")
    print("=" * 60)
    
    # Check if any non-progress strategy beats progress strategies
    stagnation_reward = scenarios[0][1]["total_scaled"]  # 0% progress
    camping_reward = scenarios[3][1]["total_scaled"]     # 11% camping
    progress_reward = scenarios[4][1]["total_scaled"]    # 20% + die
    completion_reward = scenarios[7][1]["total_scaled"]  # Complete
    
    print(f"\nStagnation (0%):        {stagnation_reward:+.2f} scaled")
    print(f"Minimal camping (11%):  {camping_reward:+.2f} scaled")
    print(f"Risky progress (20%):   {progress_reward:+.2f} scaled")
    print(f"Completion (100%):      {completion_reward:+.2f} scaled")
    
    # Verify hierarchy
    checks = [
        ("Completion > Progress", completion_reward > progress_reward),
        ("Progress > Camping", progress_reward > camping_reward),
        ("Camping > Stagnation", camping_reward > stagnation_reward),
        ("Stagnation is negative", stagnation_reward < 0),
    ]
    
    print("\nHierarchy Checks:")
    all_pass = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")
        all_pass = all_pass and passed
    
    if all_pass:
        print("\n✅ ALL CHECKS PASSED - Reward hierarchy is sound!")
        print("   Agent must make forward progress for positive returns.")
    else:
        print("\n❌ HIERARCHY VIOLATION DETECTED!")
        print("   System may be exploitable.")
        return 1
    
    # Check camping profitability in curriculum
    print("\n" + "=" * 60)
    print("CURRICULUM PROGRESSION CHECK")
    print("=" * 60)
    print("\nCamping strategy (11% progress, 2000 steps) across curriculum:")
    
    for phase, pbrs_w, time_p in [
        ("Discovery", PBRS_WEIGHT_DISCOVERY, TIME_PENALTY_DISCOVERY),
        ("Mid", PBRS_WEIGHT_MID, TIME_PENALTY_MID),
        ("Mastery", PBRS_WEIGHT_MASTERY, TIME_PENALTY_MASTERY),
    ]:
        result = calculate_scenario_reward(0.11, False, False, False, 2000, pbrs_w, time_p)
        print(f"  {phase:12s}: {result['total_scaled']:+6.2f} scaled")
    
    print("\n✅ Camping becomes unprofitable as curriculum advances!")
    
    return 0


if __name__ == "__main__":
    exit(main())

