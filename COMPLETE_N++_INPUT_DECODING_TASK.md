# Complete N++ Attract Replay Input Decoding Task

## **OBJECTIVE**
Complete the N++ attract replay input decoder to achieve **TRUE 100% accuracy** by fixing the input sequence extraction so that exactly **11 gold pieces are collected** during the attract/0 replay as intended in the original recording.

## **CURRENT STATE SUMMARY**

### ✅ **MAJOR PROGRESS ACHIEVED**
- **Entity decoding**: 100% accurate (all 15 gold pieces positioned correctly at y=72, x=144-816)
- **Map decoding**: 100% accurate (level geometry matches official map perfectly)
- **Input offset discovery**: Found real input data at offset **1382** (not 1250)
- **Ninja movement**: Reaches y=81.1 (only **9.1 pixels** from gold y-level of 72 - within collection range!)
- **Validation framework**: Complete validation script that requires exactly 11 gold collected

### ❌ **CRITICAL ISSUE REMAINING**
**Ninja movement pattern is incomplete/incorrect**:
- Ninja moves **leftward** from x=396 to x=34 (326.8px span)
- Gold spans **rightward** from x=144 to x=816 (672px span)
- Ninja gets closest to gold at **80.8 pixels** (need <16px for collection)
- **0/11 gold pieces collected** (validation failing)

## **REPOSITORY STRUCTURE**

### **Key Files**
```
/workspace/nclone/
├── nclone/replay/binary_replay_parser.py     # MAIN WORK FILE - input extraction logic
├── nclone/example_replays/npp_attract/0      # Test data - "the basics" level
├── validate_gold_collection.py               # VALIDATION SCRIPT - must pass for completion
├── nclone/maps/official/000 the basics       # Reference map for validation
└── nclone/gym_environment/npp_environment.py # Simulation environment
```

### **Current Input Extraction Settings**
```python
# In binary_replay_parser.py:
direct_input_start = 1382  # ✅ CORRECT - found real input data here
HOR_INPUTS_DIC = {0: 0, 1: 1, 2: -1, 3: 1, 4: -1, 5: 0, 6: 0, 7: 0}  # Current mapping
JUMP_INPUTS_DIC = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0}   # Current mapping
```

## **LEVEL-SPECIFIC MOVEMENT REQUIREMENTS**

### **"The Basics" Level Expected Movement Pattern**
For this specific level, the ninja must:

1. **Phase 1 - Initial Leftward Movement**: 
   - Start at x=396, y=156
   - Move **LEFT** and **JUMP+LEFT** to reach leftmost gold
   - Expected inputs: `2` (LEFT) and `4` (LEFT+JUMP)

2. **Phase 2 - Gold Collection Traverse**:
   - Move **RIGHT** and **JUMP+RIGHT** across the level
   - Collect gold pieces from x=144 to x=816
   - Expected inputs: `1` (RIGHT) and `3` (RIGHT+JUMP)

3. **Current vs Expected**:
   - **Current**: 652 frames, mostly `4` (LEFT+JUMP) - ninja goes left and stops
   - **Expected**: Should have significant RIGHT movement after initial LEFT movement

## **TECHNICAL ANALYSIS**

### **Current Input Distribution** (offset 1382)
```
Total inputs: 652 frames = 10.9s
Left: 371 (56.9%)     # Too much left movement
Right: 39 (6.0%)      # Too little right movement  
Jump: 241 (37.0%)
None: 231 (35.4%)
Left/Right ratio: 9.5:1  # Should be more balanced for full level traverse
```

### **Ninja Movement Analysis**
```
X range: 34.0 to 396.0 (span: 326.8px)  # ❌ Insufficient - need 672px span
Y range: 81.1 to 158.0 (span: 76.9px)   # ✅ Good - reaches gold y-level
Closest to gold: 80.8px                 # ❌ Too far - need <16px
```

## **PROBLEM DIAGNOSIS**

### **Root Cause Hypotheses**
1. **Incomplete input sequence**: Only extracting first part of replay (652 frames might be too short)
2. **Wrong input mapping**: Current mapping might still be incorrect
3. **Multiple input sections**: Real replay might have multiple input sections in the binary
4. **RLE/Compression**: Input data might be run-length encoded or compressed

### **Evidence Supporting Incomplete Sequence**
- Ninja starts moving left (correct for this level)
- But never transitions to rightward movement (incorrect)
- 652 frames = 10.9s seems short for full level completion
- Input distribution heavily skewed left (should be more balanced)

## **SPECIFIC TASKS TO COMPLETE**

### **Priority 1: Find Complete Input Sequence**
**Goal**: Extract the full input sequence that includes both leftward AND rightward movement

**Approaches to try**:
1. **Search for additional input sections** beyond offset 1382
2. **Check for RLE/compressed input data** that needs decompression
3. **Analyze binary file structure** for multiple input segments
4. **Test different input extraction lengths** (current limit: 2000 inputs)

### **Priority 2: Validate Input Mapping**
**Goal**: Confirm the input value → action mapping is correct

**Test approach**:
```python
# Expected pattern for "the basics":
# Early frames: value 2 (LEFT) and 4 (LEFT+JUMP) 
# Later frames: value 1 (RIGHT) and 3 (RIGHT+JUMP)
```

### **Priority 3: Debug Input Extraction Logic**
**Current extraction code** (in `binary_replay_parser.py` line ~378):
```python
inputs = []
section = data[direct_input_start:]
for byte in section:
    if 0 <= byte <= 7:
        inputs.append(byte)
    if len(inputs) > 2000:  # ❌ Might be stopping too early
        break
```

## **SUCCESS CRITERIA**

### **Validation Requirements**
The task is **COMPLETE** when `validate_gold_collection.py` passes:
```bash
cd /workspace/nclone && python3 validate_gold_collection.py
# Must output: "✅ VALIDATION PASSED: 11 gold pieces collected"
```

### **Expected Final Results**
- **Gold collected**: Exactly 11/15 pieces
- **Ninja movement**: Full horizontal traverse (x=144 to x=816+ range)
- **Input pattern**: Balanced left/right movement matching level requirements
- **Validation**: Script passes without errors

## **DEBUGGING TOOLS AVAILABLE**

### **Validation Script**
```bash
python3 validate_gold_collection.py  # Shows gold collection results
```

### **Binary Analysis**
```python
# Check raw binary data around different offsets
with open("nclone/example_replays/npp_attract/0", "rb") as f:
    data = f.read()
    # Analyze data[1382:] and beyond
```

### **Movement Analysis**
```python
# Track ninja position frame-by-frame to see movement patterns
# Check if ninja ever moves rightward past x=400
```

## **REPOSITORY CONTEXT**
- **Branch**: `npp-attract-perfect-decoder`
- **Python environment**: Already set up with all dependencies
- **Test data**: Multiple attract files available (0-19) for validation
- **Reference maps**: Official N++ maps in `nclone/maps/official/` for comparison

## **FINAL NOTES**

### **Key Insight**
The ninja **does reach the correct y-level** (81.1px, only 9.1px from gold at y=72), which proves the input extraction and movement mechanics are working. The issue is specifically that the ninja **doesn't move far enough rightward** to collect the gold pieces.

### **Expected Solution**
The solution likely involves finding additional input data that contains the rightward movement commands, or fixing the input extraction to properly decode a longer/complete sequence that includes the full level traverse.

### **Validation-Driven Development**
The task is **not complete** until the validation script passes. All debugging and fixes should be focused on achieving the specific goal: **11 gold pieces collected during attract/0 replay**.

---

**This task represents the final step in completing the most comprehensive reverse engineering of the N++ attract replay format. The goal is TRUE 100% accuracy with perfect gold collection validation.**