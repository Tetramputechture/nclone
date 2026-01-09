# Should We Remove Death Penalty Entirely?

## Current Exploitation

Agent camps at 62% progress with **PBRS=-0.59** (moving backward!):
```
Reward: -0.99
= -0.59 (PBRS, moving away) + -0.40 (time)
```

With the new net progress check, this will become:
```
Reward: -2.99
= -0.59 (PBRS) + -0.40 (time) + -2.0 (stagnation, PBRS < 0.5)
```

Good! Camping is now punished. But what about dying?

## Rewards With Current Death Penalty (-0.2 scaled)

| Strategy | PBRS | Death | Time | Stagnation | Total |
|----------|------|-------|------|------------|-------|
| Die 20% | +1.6 | -0.2 | -0.08 | 0 | **+1.32** |
| Die 50% | +4.0 | -0.2 | -0.2 | 0 | **+3.60** |
| Die 80% | +6.4 | -0.2 | -0.3 | 0 | **+5.90** |
| Camp 62% (PBRS=0) | 0.0 | 0 | -0.4 | -2.0 | **-2.40** |
| Camp 62% (PBRS=-0.6) | -0.6 | 0 | -0.4 | -2.0 | **-3.00** |
| Complete | +8.0 | 0 | -0.3 | 0 | **+15.70** |

## Rewards WITHOUT Death Penalty

| Strategy | PBRS | Death | Time | Stagnation | Total |
|----------|------|-------|------|------------|-------|
| Die 20% | +1.6 | 0 | -0.08 | 0 | **+1.52** |
| Die 50% | +4.0 | 0 | -0.2 | 0 | **+3.80** |
| Die 80% | +6.4 | 0 | -0.3 | 0 | **+6.10** |
| Camp 62% (PBRS=0) | 0.0 | 0 | -0.4 | -2.0 | **-2.40** |
| Camp 62% (PBRS=-0.6) | -0.6 | 0 | -0.4 | -2.0 | **-3.00** |
| Complete | +8.0 | 0 | -0.3 | 0 | **+15.70** |

## Comparison: Impact of Death Penalty

**Signal strength improvement without death penalty**:
- Die 20%: +1.32 → +1.52 (**15% stronger**)
- Die 50%: +3.60 → +3.80 (**5.5% stronger**)
- Die 80%: +5.90 → +6.10 (**3.4% stronger**)

**Hierarchy changes**:
- Both preserve: Complete >> Die >> Camp
- Without penalty: Bigger gap between dying and camping (better!)

## Opportunity Cost is Massive

When agent dies at 50%, they forfeit:
- Completion reward: +5.0 scaled
- Switch reward: +3.0 scaled (if not activated)
- Remaining PBRS: +4.0 scaled (50% path remaining)
- **Total: +12.0 scaled forfeited**

Compare to death penalty: -0.2 scaled

**Opportunity cost is 60× larger than death penalty!**

## Does Death Penalty Prevent Any Exploit?

**Question**: Without death penalty, could agent learn to intentionally die?

**Analysis**:
- Die immediately: 0 PBRS - 0 death - 0.01 time - 2.0 stagnation = **-2.01**
- Continue exploring: Expected +1-6 (much better)

**Answer**: No - opportunity cost + stagnation penalty prevent suicide exploit

## Does Death Penalty Teach Anything Useful?

**What it teaches**: "Death is slightly bad"

**What agent already knows from opportunity cost**:
- "Missing completion (+5.0) is very bad"
- "Missing switch (+3.0) is very bad"  
- "Dying early (lose more remaining PBRS) is worse than dying late"

**Conclusion**: Death penalty adds redundant signal that weakens learning

## Recommendation: Remove Death Penalty Entirely

**Reasons**:
1. ✅ Opportunity cost is 60× stronger signal
2. ✅ Net progress check prevents camping exploit
3. ✅ Simplifies system (3 components → 2.5)
4. ✅ Strengthens learning signal (5-15% improvement)
5. ✅ Encourages necessary risk-taking
6. ✅ More theoretically pure (PBRS + opportunity cost)

**Expected behavior**:
- Agent tries risky jumps (positive expected value)
- Failed attempts still give positive reward for progress made
- Natural selection favors successful attempts in replay buffer
- Agent learns "try and fail (+3.8) >> camp (-2.4) >> stagnate (-2.4)"

## Implementation

```python
# reward_constants.py
DEATH_PENALTY = 0.0  # Opportunity cost is sufficient

# main_reward_calculator.py
if obs.get("player_dead", False):
    death_cause = obs.get("death_cause", None)
    
    # No death penalty - opportunity cost is sufficient
    terminal_reward = 0.0
    
    # Still credit switch activation
    terminal_reward += milestone_reward
    
    self.episode_terminal_reward = terminal_reward
    scaled_reward = terminal_reward * GLOBAL_REWARD_SCALE
    
    self.last_pbrs_components = {
        "terminal_reward": 0.0,
        "milestone_reward": milestone_reward,
        "pbrs_reward": 0.0,
        "time_penalty": 0.0,
        "total_reward": terminal_reward,
        "scaled_reward": scaled_reward,
        "is_terminal": True,
        "terminal_type": "death",
        "death_cause": death_cause or "unknown",
    }
    return scaled_reward
```

This makes the reward structure:
```
Reward = PBRS + Time + Completion + Switch + Stagnation(if_no_net_progress)
```

Death is implicitly penalized by:
1. **Opportunity cost** (missing +8-15 future rewards)
2. **Episode termination** (no more PBRS accumulation)
3. **Natural selection** (successful episodes weighted more)

**The agent will learn**: "Try risky paths and sometimes succeed (+15.7) >> camp safely (-2.4)"







