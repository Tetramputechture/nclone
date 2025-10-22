# NClone Replay Validation Scripts

This directory contains tools for validating N++ replay datasets.

## validate_replays.py

Comprehensive validation script for replay datasets used in BC training and evaluation.

### What It Does

This script validates that replay files:

1. **Load correctly** - Parse without errors
2. **Execute successfully** - Simulate without physics errors
3. **Result in player_won=True** - Player reaches the exit
4. **Match metadata** - Success flag matches actual outcome
5. **Generate observations** - Produce valid observation data

### Usage

#### Basic Validation

Validate all replays in a directory:

```bash
python scripts/validate_replays.py bc_replays/
```

#### Save Results to JSON

```bash
python scripts/validate_replays.py bc_replays/ --output validation_results.json
```

#### Validate Specific Pattern

```bash
python scripts/validate_replays.py bc_replays/ --pattern "train_*.replay"
```

#### Limit Number of Replays (for Testing)

```bash
python scripts/validate_replays.py bc_replays/ --max-replays 5
```

#### Enable Verbose Output

```bash
python scripts/validate_replays.py bc_replays/ --verbose
```

### Output

#### Console Output

The script provides real-time progress with clear indicators:

```
[  1/31] 20251022_142001_train_very_simple_100001.replay... ✅ VALID
[  2/31] 20251022_142003_train_mine_heavy_100002.replay... ✅ VALID
[  3/31] 20251022_142009_train_very_simple_100003.replay... ✅ VALID
...
```

#### Summary Statistics

```
================================================================================
VALIDATION SUMMARY
================================================================================
Total replays:          31
✅ Valid replays:        31 (100.0%)
❌ Invalid replays:      0 (0.0%)

================================================================================
✅ SUCCESS: All replays are valid!
All replays execute correctly and result in player_won=True

This dataset is ready for BC training.
================================================================================
```

#### JSON Output

When using `--output`, results are saved in structured JSON format:

```json
{
  "validation_results": [
    {
      "filename": "replay1.replay",
      "valid": true,
      "error_type": null,
      "error_message": null,
      "player_won": true,
      "player_dead": false,
      "success_flag": true,
      "frame_count": 123,
      "input_count": 123
    }
  ],
  "summary": {
    "total": 31,
    "valid": 31,
    "invalid": 0
  }
}
```

### Error Detection

The script detects and categorizes various issues:

#### EXECUTION_ERROR
- Physics simulation errors
- Corrupted replay data
- Missing map data

#### PLAYER_DIED
- Replay inputs lead to player death
- Not a successful completion

#### INCOMPLETE
- Player doesn't reach the exit
- Run ends without completion

#### MISSING_PLAYER_WON
- Observation system missing player_won field
- Requires nclone#43 fix

#### NO_OBSERVATIONS
- Replay execution produced no observations
- Critical simulator issue

### Exit Codes

- `0` - All replays valid
- `1` - One or more replays invalid or error occurred

This makes the script suitable for CI/CD pipelines:

```bash
python scripts/validate_replays.py bc_replays/ || exit 1
```

### When to Use This Script

#### Before BC Training
Always validate replay datasets before using them for behavioral cloning:

```bash
python scripts/validate_replays.py bc_replays/
# If all valid, proceed with BC training
python npp-rl/scripts/train_and_compare.py \
    --replay-data-dir bc_replays/ \
    --bc-epochs 10
```

#### After Recording New Replays
After adding new replays to your dataset:

```bash
python scripts/validate_replays.py bc_replays/ --output validation_report.json
```

#### In CI/CD Pipelines
Add to your continuous integration:

```yaml
- name: Validate replays
  run: python scripts/validate_replays.py bc_replays/
```

#### Debugging Replay Issues
Use verbose mode to debug specific replays:

```bash
python scripts/validate_replays.py bc_replays/ --verbose --max-replays 5
```

### Requirements

- `nclone` package installed
- `player_won` and `player_dead` fields in observations (nclone PR #43)

### Related Documentation

- See `/workspace/BC_PRETRAINING_FIXES.md` for full context on BC pipeline fixes
- See nclone PR #43 for player_won/player_dead implementation
- See npp-rl PR #53 for BC weight loading fixes

### Example: Validating a New Dataset

```bash
# Step 1: Validate the dataset
python scripts/validate_replays.py my_new_replays/ --output validation.json

# Step 2: Check the results
cat validation.json | jq '.summary'

# Step 3: If all valid, use for training
python npp-rl/scripts/train_and_compare.py \
    --replay-data-dir my_new_replays/ \
    --bc-epochs 10 \
    --experiment-name my_experiment
```

### Troubleshooting

**Issue**: `MISSING_PLAYER_WON` errors

**Solution**: Update nclone to include PR #43 (player_won field in observations)

---

**Issue**: `PLAYER_DIED` or `INCOMPLETE` errors

**Solution**: These replays don't represent successful completions. Remove them or re-record with correct inputs.

---

**Issue**: `EXECUTION_ERROR` errors

**Solution**: Check replay file integrity and simulator version compatibility. May indicate corrupted replay data.

---

**Issue**: All replays valid but BC training fails

**Solution**: Check that npp-rl includes PR #53 (BC weight loading fix) and uses the correct replay directory path.
