# N++ Query & Data Format Documentation

This document provides comprehensive documentation of the N++ (N Plus Plus) game's API query formats and binary data structures.

## Table of Contents

1. [Query Format](#query-format)
2. [Query Parameters](#query-parameters)
3. [Categories](#categories)
4. [Data Format Specifications](#data-format-specifications)
   - [Query Levels Header](#query-levels-header)
   - [Map File Format](#map-file-format)
   - [Replay File Format](#replay-file-format)
   - [Attract File Format](#attract-file-format)
5. [Entity Types and IDs](#entity-types-and-ids)
6. [Technical Notes](#technical-notes)

---

## Query Format

**Base URL:** `https://dojo.nplusplus.ninja/prod/steam/`

**Format:** `query + ? + parameters`

**Sample Query:** Retrieve the first page of results from "Newest" in Solo mode:
```
https://dojo.nplusplus.ninja/prod/steam/query_levels?steam_id=765611980412720628&steam_auth=&qt=10&mode=0&page=0
```

---

## Query Parameters

### Available Queries

| Query Name | Description |
|------------|-------------|
| `login` | Logs the user in his account (POST request) |
| `get_scores` | Returns leaderboard data for a specific level |
| `set_replay` | Returns replay data for a specific run |
| `query_levels` | Returns a list of userlevels from a specific tab and page |
| `search/levels` | Returns a list of userlevel results for a specific search |

### Parameter Specifications

| Parameter | Obligatory | Description | Values | Default |
|-----------|------------|-------------|---------|---------|
| `app_id` | No | ? | Integer | None |
| `steam_id` | Yes | Steam64ID of an active N++ player | 17-digit integer | None |
| `steam_auth` | Yes | Steam token, can be left empty | Base 16 integer | None |
| `user_id` | No | ID of the player (N+'s server) | Integer (0-120K currently) | None |
| `level_id` | get_scores | Self-explanatory | Integer (0-120K currently) | None |
| `replay_id` | get_scores | Self-explanatory | Integer (0-120K currently) | None |
| `player_id` | No | ? | Integer | None |
| `qt` | No | On get_scores, leaderboard tab<br>On query_levels, map tab | 0 (global), 1 (around), 2 (friends)<br>Integer, 7 to 36 (see right) | 0 |
| `mode` | search | Playing mode | 0 (solo), 1 (coop), 2 (race) | 0 |
| `page` | No | Page of results from query_levels | Integer (0-500) | 0 |
| `search` | search | Text query to search | String of text | None |

---

## Categories

The following categories are available for the `qt` parameter:

| ID | Category |
|----|----------|
| 7 | Best |
| 8 | Featured |
| 9 | Top Weekly |
| 10 | Newest |
| 11 | Hardest |
| 12 | Made by me, sorted by +++'s |
| 13 | Made by me, sorted by date |
| 14 | Favourited, sorted by date |
| 15 | Favourited, sorted by +++'s |
| 18 | Made by friends, sorted by date |
| 19 | Made by friends, sorted by +++'s |
| 21 | Favourited by friends |
| 22 | Tracked by friends, sorted by date |
| 24 | Tracked by friends, sorted by rank, scored |
| 23 | Tracked by friends, sorted by rank |
| 25 | Tracked by friends, sorted by rank, not scored |
| 26 | Tracked by me |
| 30 | Following, sorted by date |
| 31 | Following, sorted by +++'s |
| 36 | Search |

---

## Data Format Specifications

### Query Levels Header

**Total Size:** 48 bytes

| Bytes | Description |
|-------|-------------|
| 16 | Date of db update |
| 04 | Number of maps |
| 04 | Page |
| 04 | Type (?) |
| 04 | Category (7-36) |
| 04 | Game mode (0-2) |
| 04 | Cache duration (1200, 5) |
| 04 | Max page size (50, 25) |
| 04 | ? (0, 5) |

### Map Headers (44 bytes each)

| Bytes | Description |
|-------|-------------|
| 04 | Map ID |
| 04 | User ID |
| 16 | Author name (padded) |
| 04 | Number of +++s |
| 16 | Date of publishing |

### Map Data Blocks

| Bytes | Description |
|-------|-------------|
| 04 | Size of block in bytes |
| 02 | Entity count |
| ## | z-lib compressed map data |

**Notes:**
- All integers are little endian
- Position is in parentheses
- A query is capped at 500 results

---

### Map File Format

#### Header (7 bytes)

| Bytes | Description |
|-------|-------------|
| 04 | File length |
| 04 | ? |
| 04 | Game mode |
| 22 | ? |

#### Map Data

| Bytes | Description |
|-------|-------------|
| 128 | Level name (padded) |
| 18 | ? |
| 960 | Tile data |
| 80 | Entity counts |
| ## | Entity data |

#### Each Tile

| Bytes | Description |
|-------|-------------|
| 01 | Tile ID |

#### Each Entity

| Bytes | Description |
|-------|-------------|
| 01 | Entity ID |
| 01 | X coordinate |
| 01 | Y coordinate |
| 01 | Orientation |
| 01 | Mode |

**Notes:**
- Tiles are stored left to right, up to Row 23 (22-966 B)
- Each entity count is 2 bytes
- Tile counts are 11 unused tiles
- Entitys are sorted by ID

---

### Replay File Format

#### Header

| Bytes | Description |
|-------|-------------|
| 04 | ? (0) |
| 04 | Replay ID |
| 04 | Level ID |
| 04 | User ID |
| ## | z-lib compressed demo |

#### Demo Data

| Bytes | Description |
|-------|-------------|
| 01 | ? (0) |
| 04 | Length of data |
| 04 | ? (1) |
| 04 | Frame count |
| 04 | Level ID |
| 04 | Game mode |
| 01 | ? (0) |
| 01 | ? (4, 3) |
| 04 | ? (23-1) |
| ## | Demo (1 Byte/Frame) |

#### Frame Values

| Bit | Description |
|-----|-------------|
| 0 | Jump |
| 1 | Right |
| 2 | Left |
| 3 | Reset |

**Example frame:** `right + jump = 2^1 + 2^0 = 3`

---

### Attract File Format

#### Header

| Bytes | Description |
|-------|-------------|
| 04 | Length of map data |
| 04 | Length of demo data |

#### Map Data

| Bytes | Description |
|-------|-------------|
| 04 | Level ID |
| 04 | Game mode |
| 04 | ? (1) |
| 18 | ? (0) |
| 128 | Level name (padded) |
| 01 | ? |
| ## | Map data |

#### Demo Data

| Bytes | Description |
|-------|-------------|
| 04 | ? (0) |
| 04 | Length of data |
| 04 | ? (1) |
| 04 | Frame count |
| 04 | Level ID |
| 04 | Game mode |
| 01 | ? (0) |
| 04 | ? (1, 3) |
| 04 | ? (2^22 - 1) |
| ## | Demo (1 Byte/Frame) |

---

## Technical Notes

### Entity Orientation
- Value 0 indicates looking East, then rotates clockwise until Northeast (7)
- For each group of 4 tiles, the first one is the pictured one, and each successive one is obtained by rotating it clockwise (for the first 4 groups), or by reflecting it horizontally and vertically (for the last 4 groups)

### Data Compression
- Map data and demo data use z-lib compression
- All integers are stored in little-endian format
- Queries are capped at 500 results maximum
