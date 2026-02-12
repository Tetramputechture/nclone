//! Binary map data parser.
//! Direct port of nclone/map_loader.py + TileSegmentFactory.

use crate::constants::{MAP_HEIGHT, MAP_WIDTH, TILE_SIZE};
use crate::entities::exit_door::ExitDoor;
use crate::entities::exit_switch::ExitSwitch;
use crate::entities::toggle_mine::ToggleMine;
use crate::entities::Entity;
use crate::map::tiles::{TILE_SEGMENT_CIRCULAR_MAP, TILE_SEGMENT_DIAG_MAP, TILE_SEGMENT_ORTHO_MAP};
use crate::segments::{CircularSegment, LinearSegment, Segment};
use crate::types::EntityType;
use std::collections::HashMap;

/// Parsed map data ready for simulation.
pub struct MapData {
    pub tile_grid: [[u8; MAP_HEIGHT]; MAP_WIDTH],
    pub segments: Vec<Segment>,
    pub segment_grid: [[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH],
    pub entities: Vec<Entity>,
    pub ninja_spawn_x: f64,
    pub ninja_spawn_y: f64,
}

/// Parse raw map data bytes into simulation-ready structures.
pub fn load_map(map_data: &[i32]) -> MapData {
    let tile_grid = [[0u8; MAP_HEIGHT]; MAP_WIDTH];
    let segment_grid: [[Vec<usize>; MAP_HEIGHT]; MAP_WIDTH] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    let mut data = MapData {
        tile_grid,
        segments: Vec::new(),
        segment_grid,
        entities: Vec::new(),
        ninja_spawn_x: 0.0,
        ninja_spawn_y: 0.0,
    };

    // Load tiles
    load_tiles(map_data, &mut data);

    // Create segments using the two-phase approach matching Python exactly
    create_segments_two_phase(&mut data);

    // Load entities
    load_entities(map_data, &mut data);

    // Load ninja spawn
    data.ninja_spawn_x = (map_data[1231] * 6) as f64;
    data.ninja_spawn_y = (map_data[1232] * 6) as f64;

    data
}

fn load_tiles(map_data: &[i32], data: &mut MapData) {
    // Extract tile data from map data (indices 184..1150)
    for x in 0..42 {
        for y in 0..23 {
            let tile_type = map_data[184 + x + y * 42] as u8;
            data.tile_grid[x + 1][y + 1] = tile_type;
        }
    }

    // Set outer edges to full tile (type 1)
    for x in 0..MAP_WIDTH {
        data.tile_grid[x][0] = 1;
        data.tile_grid[x][MAP_HEIGHT - 1] = 1;
    }
    for y in 0..MAP_HEIGHT {
        data.tile_grid[0][y] = 1;
        data.tile_grid[MAP_WIDTH - 1][y] = 1;
    }
}

/// Two-phase segment creation matching Python's TileSegmentFactory exactly.
///
/// Phase 1: Accumulate ortho values into half-cell grids.
///   Adjacent tiles' shared edges cancel out (e.g., two full tiles sharing
///   an edge produce state=0, so no segment is created between them).
///
/// Phase 2: Create LinearSegment objects from non-zero accumulated values.
fn create_segments_two_phase(data: &mut MapData) {
    // Half-cell accumulator grids (same as Python's hor_segment_dic / ver_segment_dic)
    let mut hor_segment_dic: HashMap<(i32, i32), i32> = HashMap::new();
    let mut ver_segment_dic: HashMap<(i32, i32), i32> = HashMap::new();

    // Phase 1: Process all tiles, accumulate ortho values and create non-ortho segments
    for x in 0..MAP_WIDTH {
        for y in 0..MAP_HEIGHT {
            let tile_type = data.tile_grid[x][y] as usize;
            if tile_type == 0 || tile_type >= 34 {
                continue;
            }

            let xcoord = x as i32;
            let ycoord = y as i32;

            // Accumulate orthogonal segment values (matching Python exactly)
            if tile_type < TILE_SEGMENT_ORTHO_MAP.len() {
                let ortho = &TILE_SEGMENT_ORTHO_MAP[tile_type];

                // Process horizontal segments
                for y_loop_idx in 0..3i32 {
                    for x_loop_idx in 0..2i32 {
                        let key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx);
                        let val = ortho[(2 * y_loop_idx + x_loop_idx) as usize] as i32;
                        *hor_segment_dic.entry(key).or_insert(0) += val;
                    }
                }

                // Process vertical segments
                for x_loop_idx in 0..3i32 {
                    for y_loop_idx in 0..2i32 {
                        let key = (2 * xcoord + x_loop_idx, 2 * ycoord + y_loop_idx);
                        let val = ortho[(2 * x_loop_idx + y_loop_idx + 6) as usize] as i32;
                        *ver_segment_dic.entry(key).or_insert(0) += val;
                    }
                }
            }

            // Create non-orthogonal segments (diagonal, circular)
            let tx = (x as f64) * TILE_SIZE;
            let ty = (y as f64) * TILE_SIZE;

            if tile_type < TILE_SEGMENT_DIAG_MAP.len() {
                let diag = &TILE_SEGMENT_DIAG_MAP[tile_type];
                if diag.0 {
                    let seg = Segment::Linear(LinearSegment::new(
                        tx + diag.1,
                        ty + diag.2,
                        tx + diag.3,
                        ty + diag.4,
                        true,
                    ));
                    let idx = data.segments.len();
                    data.segments.push(seg);
                    data.segment_grid[x][y].push(idx);
                }
            }

            if tile_type < TILE_SEGMENT_CIRCULAR_MAP.len() {
                let circ = &TILE_SEGMENT_CIRCULAR_MAP[tile_type];
                if circ.0 {
                    let seg = Segment::Circular(CircularSegment::new(
                        tx + circ.1,
                        ty + circ.2,
                        circ.3,
                        circ.4,
                        circ.5,
                        TILE_SIZE,
                    ));
                    let idx = data.segments.len();
                    data.segments.push(seg);
                    data.segment_grid[x][y].push(idx);
                }
            }
        }
    }

    // Phase 2: Convert accumulated ortho values into actual segments
    // This matches Python's _process_orthogonal_segments exactly.

    // Process horizontal segments
    for (&(hx, hy), &state) in &hor_segment_dic {
        if state == 0 {
            continue;
        }

        // Calculate which cell this segment belongs to (matching Python exactly)
        let cell_x = (hx as f64 / 2.0).floor() as usize;
        let cell_y = ((hy as f64 - 0.1 * state as f64) / 2.0).floor() as usize;

        // Create segment endpoints using half-cell coordinates (* 12 pixels)
        let mut p1 = (12.0 * hx as f64, 12.0 * hy as f64);
        let mut p2 = (12.0 * hx as f64 + 12.0, 12.0 * hy as f64);

        // Reverse direction for negative state
        if state == -1 {
            std::mem::swap(&mut p1, &mut p2);
        }

        let seg = Segment::Linear(LinearSegment::new(p1.0, p1.1, p2.0, p2.1, true));
        let idx = data.segments.len();
        data.segments.push(seg);

        if cell_x < MAP_WIDTH && cell_y < MAP_HEIGHT {
            data.segment_grid[cell_x][cell_y].push(idx);
        }
    }

    // Process vertical segments
    for (&(vx, vy), &state) in &ver_segment_dic {
        if state == 0 {
            continue;
        }

        // Calculate which cell this segment belongs to (matching Python exactly)
        let cell_x = ((vx as f64 - 0.1 * state as f64) / 2.0).floor() as usize;
        let cell_y = (vy as f64 / 2.0).floor() as usize;

        // Create segment endpoints
        // Note: vertical segments go from bottom to top (p1.y > p2.y)
        let mut p1 = (12.0 * vx as f64, 12.0 * vy as f64 + 12.0);
        let mut p2 = (12.0 * vx as f64, 12.0 * vy as f64);

        // Reverse direction for negative state
        if state == -1 {
            std::mem::swap(&mut p1, &mut p2);
        }

        let seg = Segment::Linear(LinearSegment::new(p1.0, p1.1, p2.0, p2.1, true));
        let idx = data.segments.len();
        data.segments.push(seg);

        if cell_x < MAP_WIDTH && cell_y < MAP_HEIGHT {
            data.segment_grid[cell_x][cell_y].push(idx);
        }
    }
}

fn load_entities(map_data: &[i32], data: &mut MapData) {
    if map_data.len() <= 1233 {
        return;
    }

    // First pass: find exit doors (type 3) so switches can reference them
    let mut exit_doors: Vec<usize> = Vec::new(); // indices of exit doors in data.entities

    // First pass: create exit doors
    let mut temp_index = 1230;
    while temp_index + 4 < map_data.len() {
        let entity_type = map_data[temp_index];
        let xcoord = map_data[temp_index + 1] as i32;
        let ycoord = map_data[temp_index + 2] as i32;

        if entity_type == 3 {
            // Exit door
            let door = ExitDoor::new(xcoord, ycoord);
            data.entities.push(Entity::ExitDoor(door));
            exit_doors.push(data.entities.len() - 1);
        }

        // Advance index
        if entity_type == 6 || entity_type == 8 {
            if temp_index + 9 < map_data.len()
                && map_data[temp_index + 7] != 0
                && map_data[temp_index + 8] == 0
                && map_data[temp_index + 9] == 0
            {
                temp_index += 10;
            } else {
                temp_index += 9;
            }
        } else {
            temp_index += 5;
        }
    }

    // Second pass: create all other in-scope entities
    let mut door_idx = 0; // which exit door we're pairing switches with
    let mut index = 1230;
    while index + 4 < map_data.len() {
        let entity_type = map_data[index];
        let xcoord = map_data[index + 1] as i32;
        let ycoord = map_data[index + 2] as i32;

        match EntityType::from_u8(entity_type as u8) {
            Some(EntityType::ToggleMine) | Some(EntityType::ToggleMineToggled) => {
                let mine = ToggleMine::new(entity_type as u8, xcoord, ycoord);
                data.entities.push(Entity::ToggleMine(mine));
            }
            Some(EntityType::ExitSwitch) => {
                // Associate with the corresponding exit door
                let parent_door = if door_idx < exit_doors.len() {
                    exit_doors[door_idx]
                } else {
                    0
                };
                let switch = ExitSwitch::new(xcoord, ycoord, parent_door);
                data.entities.push(Entity::ExitSwitch(switch));
                door_idx += 1;
            }
            Some(EntityType::ExitDoor) => {
                // Already created in first pass, skip
            }
            None => {
                // Unsupported entity type, skip
            }
        }

        // Advance index
        if entity_type == 6 || entity_type == 8 {
            if index + 9 < map_data.len()
                && map_data[index + 7] != 0
                && map_data[index + 8] == 0
                && map_data[index + 9] == 0
            {
                index += 10;
            } else {
                index += 9;
            }
        } else {
            index += 5;
        }
    }
}
