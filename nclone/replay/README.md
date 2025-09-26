# N++ Replay Processing Module

This module contains tools for processing N++ replay data including binary replay parsing, format conversion, data validation, and video generation.

## Components

### Binary Replay Parser (`binary_replay_parser.py`)

Converts N++ binary replay files ("trace" mode) to JSONL format compatible with the npp-rl training pipeline.

#### Overview

The N++ Binary Replay Parser processes original N++ replay files and converts them to the JSONL format expected by the `replay_ingest.py` tool. It simulates the game frame-by-frame to extract complete state information.

#### Input Formats

The parser supports multiple N++ replay formats:

##### Trace Mode Format

Standard N++ "trace" mode replay files with the following structure:

```
replay_directory/
├── inputs_0        # Binary file: zlib-compressed input sequence for replay 1
├── inputs_1        # Binary file: zlib-compressed input sequence for replay 2
├── inputs_2        # Binary file: zlib-compressed input sequence for replay 3
├── inputs_3        # Binary file: zlib-compressed input sequence for replay 4
└── map_data        # Binary file: Raw map geometry data
```

##### npp_attract Format

Single-file N++ attract mode replay files (e.g., `npp_attract/0`, `npp_attract/1`):

```
npp_attract/
├── 0               # Binary file: Type 1 format with embedded input sequence
├── 1               # Binary file: Type 1 format with embedded input sequence
├── 2               # Binary file: Type 1 format with embedded input sequence
└── ...             # Additional attract mode replays (typically 0-10)
```

**npp_attract Format Structure:**

The npp_attract format contains N++ intro/tutorial level replays with embedded level references and input sequences. These files have been reverse-engineered to support two distinct binary formats:

###### Binary Format Specification

**Common Header (First 20 bytes):**
```
Offset  Size  Type     Description
0-3     4     uint32   Level ID (primary map reference, e.g., 1292-1687)
4-7     4     uint32   Size/Checksum value
8-11    4     uint32   Unknown field (often 0xFFFFFFFF)
12-15   4     uint32   Entity count or flags
16      1     uint8    Format type flag (determines Type 1 vs Type 2)
17-19   3     bytes    Additional header data
```

**Type 1 Format (Most Common):**
- **Detection**: Format flag and size patterns indicate Type 1
- **Input Data Location**: Starts at offset 184 bytes
- **Structure**: `[Header][Padding/Metadata][Input Sequence]`
- **Input Length**: Determined by file size minus header offset
- **Map Data**: Level ID references external map definitions

**Type 2 Format (Less Common):**
- **Detection**: Different format flag and size patterns
- **Input Data Location**: Variable offset (auto-detected)
- **Structure**: `[Header][Map Data Section][Input Sequence]`
- **Input Length**: Calculated based on embedded map data size
- **Map Data**: May contain embedded map information

###### Level ID Mapping

The Level ID field (bytes 0-3) serves as the primary map reference:

| Level ID Range | Description | Example Names |
|----------------|-------------|---------------|
| 1292-1327 | Basic Tutorial Levels | "he basics", "alljumptroduction" |
| 1328-1400 | Movement Tutorials | "wall-to-wall", "accepting your limitations" |
| 1401-1500 | Advanced Mechanics | "jump mechanics", "wall jumping" |
| 1501-1687 | Complex Scenarios | "complex navigation", "precision timing" |

**Level Name Extraction:**
Level names are extracted from the binary data using pattern matching and are often truncated or contain artifacts (e.g., "he basics" instead of "the basics").

###### Map Data Correlation

Since npp_attract files contain level references rather than complete map data, the system supports multiple correlation strategies:

**1. Official Levels Directory Support:**
```bash
# Use official_levels/ directory containing multiple .txt files
python -m nclone.replay.video_generator \
  --input npp_attract/0 \
  --output video.mp4 \
  --binary-replay \
  --official-levels official_levels/
```

**2. Single Map File Support:**
```bash
# Use single map definition file
python -m nclone.replay.video_generator \
  --input npp_attract/0 \
  --output video.mp4 \
  --binary-replay \
  --map-file levels.txt
```

**3. Correlation Strategies:**
- **Exact Name Match**: Direct match between attract level name and map file level name
- **Fuzzy Matching**: Similarity-based matching using n-gram analysis
- **Level ID Correlation**: Match by extracted Level ID if available in map files
- **Cleaned Name Matching**: Remove common prefixes/suffixes for better matching
- **Fallback to Empty Maps**: Generate minimal playable maps when no match found

###### Map Definition File Format

Official level files use the N++ standard format:
```
$level_name#map_data#
$another_level#map_data#
...
```

**Example:**
```
$the basics#00000000101010101010101010101010101010101010101010101010101010101010101010101010101010101010600000000000000000000000000000000000000000000000000000000000000000000000000000000070000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000101010101010101010101010101010101010101010101010101010101010101010101010101010000000000000000000000000000000701010101010101010101010101010101010101010101010101010900000000000000000000000000000001010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010001024a1003001210501cae000f082c003c083c004c084c005c081c002c085c006c086c007c087c008c088c0001069e02ae0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#
$walljumptroduction#000000001010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101060000000000000000000000000000000000000000070101010101010101010101010101010101010101000000000000000000000000000000000000000000000101010101010101010101010101010101010101000000000000000000000000000000000000000000000101010101010101010101010101010101010101000000000000000000000000000000000000000000000101010101010101010101010101010101010101010101010900000000000000000000000008010101010101010101010101010101010101010101010101010101010101010109000000000801010101010101010101010101010101010101010101010101010101010101010101010101000000000101010101010101010101010101010101010101010101010101010101010101010101010101000000000101010101010101010101010101010101010101010101010101010101010101010101010101000000000101010101010101010101010101010101010101010101010101010101010101010101010101000000000101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010100010432300000001e5e3a5e365e325e3e524a52465242524e564a56465642564e5a4a5a465a425a40010c72385c2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000#
```

### Map Format Compatibility

The nclone replay system supports two distinct map formats that are fully compatible with the simulation engine:

#### N++ Official Levels Format (.txt files)

**Source**: `official_levels/` directory containing N++ level definitions
**Structure**: Text-based format with character encoding
**Usage**: Real N++ levels for authentic gameplay reproduction

**Format Details:**
- **File Extension**: `.txt`
- **Encoding**: Character-based representation
- **Size**: ~2000-2500 characters per level
- **Header**: `$level_name#`
- **Data**: Character string representing tiles and entities
- **Terminator**: `#`

**Character Mapping:**
- `0` = Empty space
- `1` = Wall/solid tile  
- `6` = Special tiles (spawn points)
- `7` = Special tiles (exit areas)
- `8`, `9` = Other special elements
- `a-f` = Hexadecimal values for entities and advanced elements

#### nclone Binary Format (.dat files and test_maps/)

**Source**: `test_maps/` directory and converted N++ maps
**Structure**: Binary format optimized for nclone simulation
**Usage**: Native nclone maps and converted N++ levels

**Format Details:**
- **File Extension**: No extension (binary files) or `.dat`
- **Size**: Exactly 1245 bytes
- **Structure**: Fixed binary layout

**Binary Layout:**
```
Bytes 0-183:    Header (184 bytes)
  0-3:          Map type [6, 0, 0, 0]
  4-7:          Size/checksum [221, 4, 0, 0]  
  8-11:         Unknown field [255, 255, 255, 255]
  12-15:        Entity count [4, 0, 0, 0]
  16-19:        Additional header [37, 0, 0, 0]
  20-183:       Header padding/metadata

Bytes 184-1149: Tile Data (966 bytes)
  42×23 grid of tile values
  Row-major order storage
  Values: 0=empty, 1=wall, 6=special, etc.

Bytes 1150-1244: Entity Data (95 bytes)
  5-byte entity records: [type, x_low, x_high, y_low, y_high]
  Multiple entities supported
  Remaining bytes zero-padded
```

#### Format Conversion Process

The replay system automatically converts between formats:

1. **N++ → nclone Conversion:**
   - Parse character-based N++ data
   - Map characters to nclone tile values
   - Generate 1245-byte binary structure
   - Extract entity information
   - Create compatible header

2. **Loading Process:**
   - Binary files: Direct loading as integer list
   - Text files: Parse and convert to binary format
   - Both formats: Identical simulation behavior

#### Compatibility Verification

Both formats produce identical simulation results:

```python
# Test format compatibility
from nclone.nplay_headless import NPlayHeadless

# Load native nclone binary format
nplay1 = NPlayHeadless()
nplay1.load_map('nclone/test_maps/simple-walk')

# Load converted N++ format  
nplay2 = NPlayHeadless()
nplay2.load_map_from_map_data(converted_npp_map)

# Both produce identical tile_dic and entity_dic structures
assert nplay1.sim.tile_dic == nplay2.sim.tile_dic
assert nplay1.sim.entity_dic.keys() == nplay2.sim.entity_dic.keys()
```

#### Usage Examples

```bash
# Use native nclone binary format
python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4 --custom-map nclone/test_maps/simple-walk

# Use N++ text format (single file)
python -m nclone.replay.video_generator --input npp_attract/0 --output video.mp4 --binary-replay --map-file levels.txt

# Use N++ text format (directory)
python -m nclone.replay.video_generator --input npp_attract/0 --output video.mp4 --binary-replay --official-levels official_levels/
```

This dual-format support ensures that both native nclone maps (test_maps/) and official N++ levels (official_levels/) work seamlessly with the replay system, providing maximum flexibility for video generation and analysis.

###### Automatic Format Detection

The parser automatically detects the format type using multiple heuristics:

```python
def detect_format_type(data: bytes) -> str:
    """
    Detect npp_attract format type based on binary patterns.
    
    Type 1 Indicators:
    - Format flag at offset 16
    - Specific size/checksum patterns
    - Input data starts at offset 184
    
    Type 2 Indicators:
    - Different format flag patterns
    - Variable input data offset
    - Embedded map data sections
    """
```

###### Input Sequence Processing

Input sequences use the same encoding as trace mode files:

| Byte Value | Horizontal Movement | Jump | Combined Action |
|------------|-------------------|------|-----------------|
| 0 | 0 (none) | 0 (no) | No input |
| 1 | 0 (none) | 1 (yes) | Jump only |
| 2 | 1 (right) | 0 (no) | Right only |
| 3 | 1 (right) | 1 (yes) | Right + Jump |
| 4 | -1 (left) | 0 (no) | Left only |
| 5 | -1 (left) | 1 (yes) | Left + Jump |
| 6 | -1 (left) | 0 (no) | Left only (alternate) |
| 7 | -1 (left) | 1 (yes) | Left + Jump (alternate) |

**Processing Features:**
- **Automatic Format Detection**: Parser identifies format type and processes accordingly
- **Map Data Generation**: Creates valid map structure when insufficient data is available
- **Enhanced Correlation**: Smart matching between attract files and official level definitions
- **Fallback Support**: Generates empty maps when no official level match is found

#### Input Encoding

Each byte in the input files represents a combined input state (0-7):

| Value | Horizontal | Jump | Description |
|-------|------------|------|-------------|
| 0     | 0          | 0    | No input |
| 1     | 0          | 1    | Jump only |
| 2     | 1          | 0    | Right only |
| 3     | 1          | 1    | Right + Jump |
| 4     | -1         | 0    | Left only |
| 5     | -1         | 1    | Left + Jump |
| 6     | -1         | 0    | Left only (alternate) |
| 7     | -1         | 1    | Left + Jump (alternate) |

#### Output Format

The parser generates JSONL files compatible with the existing replay ingestion pipeline. Each line represents a single frame:

```json
{
  "timestamp": 1692345678.123,
  "level_id": "level_001",
  "frame_number": 42,
  "player_state": {
    "position": {"x": 150.5, "y": 200.3},
    "velocity": {"x": 2.1, "y": -0.5},
    "on_ground": true,
    "wall_sliding": false,
    "jump_time_remaining": 0.0
  },
  "player_inputs": {
    "left": false,
    "right": true,
    "jump": false,
    "restart": false
  },
  "entities": [
    {
      "type": "mine",
      "position": {"x": 180.0, "y": 220.0},
      "active": true
    }
  ],
  "level_bounds": {
    "width": 1056,
    "height": 600
  },
  "meta": {
    "session_id": "level_001_session_000",
    "player_id": "binary_replay",
    "quality_score": 0.8,
    "completion_status": "in_progress"
  }
}
```

#### Usage

```bash
# Process a single replay directory
python -m nclone.replay.binary_replay_parser --input replays/level_001 --output datasets/raw/

# Process multiple replay directories
python -m nclone.replay.binary_replay_parser --input replays/ --output datasets/raw/

# Enable verbose logging
python -m nclone.replay.binary_replay_parser --input replays/ --output datasets/raw/ --verbose
```

### Action Converter (`convert_actions.py`)

Utility for converting between different N++ action representations.

#### Supported Formats

- **Text**: `"NOOP"`, `"Jump"`, `"Right"`, `"Jump + Right"`, `"Left"`, `"Jump + Left"`
- **Symbol**: `"-"`, `"^"`, `">"`, `"/"`, `"<"`, `"\\"`
- **Index**: `0`, `1`, `2`, `3`, `4`, `5` (matching NppEnvironment action space)

#### Usage

```bash
# Convert text actions to symbols
python -m nclone.replay.convert_actions --input actions.txt --output symbols.txt --input-format text --output-format symbol

# Convert symbols to discrete indices
python -m nclone.replay.convert_actions --input symbols.txt --output indices.txt --input-format symbol --output-format index

# Convert comma-separated actions
python -m nclone.replay.convert_actions --input "NOOP,Jump,Right" --output-format symbol --separator ","
```

### Video Generator (`video_generator.py`)

Generates MP4 videos from N++ replay data using the nclone simulation environment for accurate visual representation.

#### Features

- **JSONL Support**: Generate videos from JSONL replay files
- **Binary Replay Support**: Generate videos directly from binary replay files
- **Automatic Map Detection**: Uses map_data_path from JSONL files when available
- **Custom Map Support**: Override with custom map files
- **High Quality Output**: H.264 encoded MP4 with configurable framerate
- **Frame Validation**: Validates replay data before processing

#### Usage

```bash
# Generate video from JSONL replay file
python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4

# Generate video from binary replay file
python -m nclone.replay.video_generator --input npp_attract/1 --output video.mp4 --binary-replay

# Generate video with custom framerate
python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4 --fps 30

# Generate video with custom map data
python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4 --custom-map map.dat

# Generate video from binary replay with official levels directory
python -m nclone.replay.video_generator --input npp_attract/0 --output video.mp4 --binary-replay --official-levels official_levels/

# Generate video from binary replay with single map file
python -m nclone.replay.video_generator --input npp_attract/0 --output video.mp4 --binary-replay --map-file levels.txt

# Enable verbose logging
python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4 --verbose
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input JSONL or binary replay file | Required |
| `--output` | Output MP4 file path | Required |
| `--binary-replay` | Input is binary replay file (not JSONL) | False |
| `--fps` | Video framerate | 60 |
| `--custom-map` | Custom map file for rendering | Auto-detect |
| `--map-file` | Single N++ map definition file | None |
| `--official-levels` | Directory containing official N++ level .txt files | None |
| `--verbose` | Enable detailed logging | False |

#### Video Output Specifications

- **Format**: MP4 (H.264 codec)
- **Resolution**: 1056x600 pixels (N++ native resolution)
- **Quality**: High quality (CRF 18)
- **Pixel Format**: YUV420P (compatible with most players)

## Video Export from Replays

The replay system supports exporting videos from both binary replay files and JSONL files using two methods:

1. **nclone Video Generator** (Recommended): Built-in video generation using `video_generator.py`
2. **npp-rl Integration**: External video generation using npp-rl tools

### Method 1: nclone Video Generator (Recommended)

Use the built-in video generator for the simplest workflow:

```bash
# Generate video from JSONL file (with automatic map detection)
python -m nclone.replay.video_generator --input replay.jsonl --output video.mp4

# Generate video from binary replay file
python -m nclone.replay.video_generator --input npp_attract/1 --output video.mp4 --binary-replay
```

### Method 2: npp-rl Integration

For advanced features or integration with the npp-rl training pipeline:

#### Prerequisites

1. **npp-rl Repository**: Clone the npp-rl repository alongside nclone:
   ```bash
   git clone https://github.com/Tetramputechture/npp-rl.git
   ```

2. **Environment Setup**: Set up the environment for headless video generation:
   ```bash
   export XDG_RUNTIME_DIR=/tmp
   ```

#### Video Export Methods

##### Method 1: Direct from Binary Replays

Generate videos directly from N++ binary replay files (npp_attract format):

```bash
# Convert binary replay to JSONL first
python -m nclone.replay.binary_replay_parser --input nclone/example_replays/npp_attract/1 --output replay_1.jsonl

# Generate video from JSONL using npp-rl tools
cd npp-rl
python tools/replay_ingest.py \
  --input ../nclone/replay_1.jsonl \
  --output-video ../nclone/replay_video_1.mp4 \
  --generate-video \
  --custom-map ../nclone/nclone/test_maps/simple-walk
```

##### Method 2: Batch Processing Multiple Files

Process multiple npp_attract files and generate videos:

```bash
# Process multiple binary replays
for i in {0..4}; do
  python -m nclone.replay.binary_replay_parser \
    --input nclone/example_replays/npp_attract/$i \
    --output replay_${i}.jsonl
done

# Generate videos for each replay
cd npp-rl
for i in {0..4}; do
  python tools/replay_ingest.py \
    --input ../nclone/replay_${i}.jsonl \
    --output-video ../nclone/replay_video_${i}.mp4 \
    --generate-video \
    --custom-map ../nclone/nclone/test_maps/simple-walk
done
```

##### Method 3: From Existing JSONL Files

If you already have JSONL replay files:

```bash
cd npp-rl
python tools/replay_ingest.py \
  --input path/to/replay.jsonl \
  --output-video output_video.mp4 \
  --generate-video \
  --custom-map ../nclone/nclone/test_maps/simple-walk \
  --fps 60
```

### Video Generation Options

The `replay_ingest.py` tool supports several options for video generation:

| Option | Description | Default |
|--------|-------------|---------|
| `--generate-video` | Enable video generation mode | Required |
| `--output-video` | Output MP4 file path | Required |
| `--custom-map` | Map file to use for rendering | Auto-detect |
| `--fps` | Video framerate | 60 |
| `--verbose` | Enable detailed logging | False |

### Available Test Maps

The following test maps are available for video generation:

- `simple-walk` - Basic walking level
- `complex-path-switch-required` - Complex navigation
- `jump-then-fall` - Jump mechanics
- `wall-jump-required` - Wall jumping
- `long-jump-reachable` - Long jump mechanics
- `minefield` - Mine navigation
- `drone-reachable` - Drone interactions

### Video Output Format

Generated videos have the following characteristics:

- **Format**: MP4 (H.264 codec)
- **Resolution**: 1056x600 pixels (N++ native resolution)
- **Framerate**: 60 FPS (configurable)
- **Quality**: High quality (CRF 18)
- **Duration**: Matches replay length

### Example Output

A successful video generation will show:

```
2025-09-26 16:27:01,267 - INFO - Generating 518 frames...
2025-09-26 16:27:13,122 - INFO - Running ffmpeg: ffmpeg -y -framerate 60 -i /tmp/tmp3i923uyj/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -crf 18 replay_video_1.mp4
2025-09-26 16:27:14,092 - INFO - Video generated successfully: replay_video_1.mp4
```

### Troubleshooting Video Export

#### Common Issues

1. **XDG_RUNTIME_DIR Error**:
   ```bash
   export XDG_RUNTIME_DIR=/tmp
   ```

2. **Map File Not Found**:
   - Use `--custom-map` with full path to a test map
   - Available maps are in `nclone/test_maps/`

3. **FFmpeg Not Found**:
   ```bash
   # Install FFmpeg (Ubuntu/Debian)
   sudo apt-get install ffmpeg
   
   # Install FFmpeg (macOS)
   brew install ffmpeg
   ```

4. **Memory Issues with Large Replays**:
   - Process replays in smaller batches
   - Use lower FPS settings for very long replays

#### Validation

Verify video generation success:

```bash
# Check video file exists and has content
ls -la replay_video_1.mp4

# Check video properties with ffprobe
ffprobe -v quiet -print_format json -show_format -show_streams replay_video_1.mp4
```

## Integration with ML Pipeline

After generating JSONL files with the binary replay parser, process them with the ML pipeline tools:

```bash
# Convert binary replays to JSONL
python -m nclone.replay.binary_replay_parser --input replays/ --output datasets/raw/

# Process JSONL files for training (from npp-rl project)
python tools/replay_ingest.py --input datasets/raw/ --output datasets/processed/ --profile rich

# Generate videos from replays (from npp-rl project)
python tools/replay_ingest.py --input datasets/raw/replay.jsonl --output-video replay.mp4 --generate-video
```

## Entity Type Mapping

The binary parser maps numeric entity types to names:

| Type ID | Name | Description |
|---------|------|-------------|
| 1 | mine | Toggle mine |
| 2 | gold | Gold collectible |
| 3 | exit_door | Level exit |
| 4 | exit_switch | Switch to activate exit |
| 5 | door | Regular door |
| 6 | locked_door | Locked door |
| 8 | trap_door | Trap door |
| 10 | launch_pad | Launch pad |
| 11 | one_way_platform | One-way platform |
| 14 | drone | Regular drone |
| 17 | bounce_block | Bounce block |
| 20 | thwump | Thwump enemy |
| 21 | toggle_mine | Toggle mine |
| 25 | death_ball | Death ball |
| 26 | mini_drone | Mini drone |

## Dependencies

- nclone package (simulator and physics)
- Python 3.7+
- Standard library modules: json, zlib, pathlib, etc.

## Processing Pipeline

1. **Detection**: Scan for trace mode files (`inputs_*` and `map_data`)
2. **Loading**: Load and decompress input sequences and map data
3. **Simulation**: Run nclone simulator frame-by-frame with decoded inputs
4. **Extraction**: Extract player state, inputs, and entity data at each frame
5. **Output**: Generate JSONL files with timestamped frame data

## Assumptions

- Level dimensions are fixed at 1056x600 pixels
- Frame rate is 60 FPS for timestamp calculation
- Default quality score is 0.8 (0.9 for completed levels)
- Player ID is set to "binary_replay" for all converted replays

## Error Handling

- Invalid or corrupted replay files are skipped with error logging
- Missing input files are handled gracefully
- Simulation failures are logged and tracked in statistics

## Statistics

The parser outputs processing statistics including:
- Directories processed
- Replays processed/failed
- Frames generated
- Success rate
- Average frames per replay

## Limitations

- Currently only supports "trace" mode (not "splits" mode)
- Fixed level dimensions assumption
- Limited entity state extraction (could be enhanced)
- No support for custom quality scoring based on actual gameplay
