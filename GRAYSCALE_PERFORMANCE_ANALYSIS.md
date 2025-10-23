# Grayscale Rendering Performance Analysis

## Problem Identification

The `_blit_grayscale()` method introduced in commit b76689d is causing significant performance issues during replay ingestion.

### Root Cause

The method performs expensive numpy operations on FULL SCREEN surfaces (600x600 = 360,000 pixels) for EVERY blit operation:

```python
def _blit_grayscale(self, source_surface, offset):
    # 1. Copy entire RGB surface to numpy array (360K * 3 bytes = 1MB)
    rgb_array = pygame.surfarray.array3d(source_surface)  # EXPENSIVE
    
    # 2. Copy entire alpha surface to numpy array (360K bytes)
    alpha_array = pygame.surfarray.array_alpha(source_surface)  # EXPENSIVE
    
    # 3. Floating point grayscale conversion on ALL pixels
    gray_wh = (0.2989 * rgb_array[:,:,0] + 0.5870 * rgb_array[:,:,1] + 0.1140 * rgb_array[:,:,2]).astype(np.uint8)
    
    # 4. Alpha blending on ALL pixels
    blended = (src_gray_region * alpha_normalized + dst_region * (1 - alpha_normalized)).astype(np.uint8)
```

### Performance Impact

**Per Frame:**
- Called 2x per frame (entities + tiles)
- Processes 720,000 pixels per frame
- 2MB+ of memory copied per frame
- Extensive floating-point math

**During Replay Ingestion:**
- For a 1000-frame replay: 2000 calls to `_blit_grayscale()`
- For 100 replays: 200,000 calls
- **This explains why replay ingestion is hanging/slow!**

## Optimization Options

### Option 1: Disable Visual Observations (IMMEDIATE FIX)

The `UnifiedObservationExtractor` already has a flag to disable visual rendering:

```python
# In unified_observation_extractor.py line 37
def __init__(self, enable_visual_observations: bool = False):
```

And line 305 only renders when enabled:
```python
if self.enable_visual_observations:
    obs["screen"] = nplay_wrapper.render()
```

**Action:** Ensure replay ingestion disables visual observations if not needed.

### Option 2: Optimize _blit_grayscale() (RECOMMENDED)

Several optimization strategies:

#### 2a. Process Only Non-Transparent Regions
Instead of processing the entire surface, detect and process only the bounding box of visible content:

```python
def _blit_grayscale_optimized(self, source_surface, offset):
    # Get bounding rect of non-transparent pixels
    bounding_rect = source_surface.get_bounding_rect()
    if bounding_rect.width == 0 or bounding_rect.height == 0:
        return  # Nothing to draw
    
    # Only process the bounding rect region
    clipped_surface = source_surface.subsurface(bounding_rect)
    # ... process only this smaller region
```

#### 2b. Use Integer Math Instead of Floating Point
```python
# Instead of: gray = 0.2989*R + 0.5870*G + 0.1140*B
# Use:       gray = (77*R + 150*G + 29*B) >> 8  # Integer approximation
gray_wh = ((77 * rgb_array[:,:,0] + 150 * rgb_array[:,:,1] + 29 * rgb_array[:,:,2]) >> 8).astype(np.uint8)
```

#### 2c. Batch Surface Copies
Lock surfaces once and process multiple blits:

```python
def begin_blit_batch(self):
    self.screen_pixels = pygame.surfarray.pixels2d(self.screen)
    
def blit_grayscale_batched(self, source_surface, offset):
    # Reuse locked screen_pixels instead of locking/unlocking each time
    
def end_blit_batch(self):
    del self.screen_pixels
```

### Option 3: Pre-convert Entity/Tile Surfaces to Grayscale

Instead of converting during blit, convert the source surfaces once:

```python
# In entity_renderer.draw_entities():
if self.grayscale:
    entities_surface = self._convert_to_grayscale(entities_surface)
    # Now just normal blit, no conversion needed
```

### Option 4: Use Pygame's Built-in Capabilities

Instead of manual numpy operations, leverage pygame's color conversion:

```python
def _blit_grayscale_fast(self, source_surface, offset):
    # Convert source to grayscale first using pygame
    gray_surface = pygame.Surface(source_surface.get_size(), depth=8)
    gray_surface.set_palette([(i, i, i) for i in range(256)])
    
    # Use pygame's built-in pixel format conversion
    pygame.transform.threshold(
        dest_surface=gray_surface,
        surface=source_surface,
        search_color=(0,0,0),
        threshold=(0,0,0,0),
        set_color=(0,0,0),
        set_behavior=1
    )
    
    self.screen.blit(gray_surface, offset)
```

## Recommended Solution

**Immediate (for user):**
1. Check if visual observations are actually needed during replay ingestion
2. If not, ensure `enable_visual_observations=False` in UnifiedObservationExtractor

**Short-term (optimization):**
1. Implement Option 2a: Only process non-transparent bounding regions
2. Implement Option 2b: Use integer math for grayscale conversion
3. This should reduce processing by 80-90% for sparse entity/tile drawings

**Long-term (architecture):**
1. Consider rendering entities/tiles directly as grayscale when grayscale mode is enabled
2. This eliminates the need for expensive post-processing entirely

## Benchmarking

Before optimization:
- ~X ms per frame (TBD - need to measure)
- Replay ingestion: ~X minutes for 100 replays (TBD)

After optimization:
- Target: <1ms per frame
- Replay ingestion: <1 minute for 100 replays

## Implementation Priority

1. **HIGH**: Verify if visual observations are needed for replay ingestion
2. **HIGH**: If yes, implement bounding-rect optimization (Option 2a)
3. **MEDIUM**: Add integer math optimization (Option 2b)
4. **LOW**: Consider pre-conversion approach (Option 3)

## Related Files

- `/workspace/nclone/nclone/nsim_renderer.py` - Contains `_blit_grayscale()` method
- `/workspace/nclone/nclone/replay/unified_observation_extractor.py` - Replay ingestion rendering
- `/workspace/nclone/nclone/entity_renderer.py` - Entity surface creation
- `/workspace/npp-rl/tools/replay_ingest.py` - Replay ingestion entry point
