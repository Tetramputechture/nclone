--[[
=============================================================================
N++ Ground-Truth Recorder / Playback  —  Cheat Engine Lua Script
=============================================================================

PURPOSE
  1. Inject scripted inputs (from inputs.json) into N++ frame-by-frame.
  2. Record the ninja's position, velocity, and state each frame.
  3. Write the recorded trace to game_trace.csv for offline comparison.

HOW TO USE
  1. Run export_attract_for_game.py to produce inputs.json and attract_info.json.
  2. Copy the exported level file (.lvl) to:
         C:\Users\Nick\Documents\Metanet\N++\levels\
  3. Launch N++, attach Cheat Engine (32-bit) to N++.exe.
  4. Open this script in CE's Lua engine (Table → Show Cheat Table Lua Script,
     or via the Lua engine window).
  5. Edit the CONFIG section below to set the correct file paths.
  6. Load the exported level in N++ and navigate to the level select screen
     (do NOT start the level yet).
  7. Execute this script.  It will print "Ready – start the level now."
  8. Start the level in N++.  The script hooks the physics tick automatically
     and begins injecting inputs + recording positions.
  9. When all inputs are consumed (or the ninja reaches a terminal state),
     the script saves game_trace.csv and prints "Done."

KNOWN ADDRESSES  (N++ v1.x / npp.dll)
  These were determined from community Cheat Engine research.
  They are relative to the npp.dll module base.  Run the ADDRESS DISCOVERY
  section once if the AOB scan below fails.

  Physics tick injection point:
      npp.dll + 0x2F010C   instruction: F2 0F 11 41 E8  (movsd [ecx-18], xmm0)
      At this point:
          ECX       = ninja struct pointer
          [ECX-18]  = xpos  (f64 double, new value just computed, in XMM0)
          [ECX-10]  = yspeed candidate / [ECX-08] = xspeed candidate
          [ECX+00]  = ypos (double)
          [ECX+98]  = ninja state (int32)

  Input state (best-guess offsets from ninja struct; verify via ADDRESS DISCOVERY):
      [ECX + INPUT_HOR_OFFSET]  = horizontal input  (-1 / 0 / 1 as int32 or f32)
      [ECX + INPUT_JUMP_OFFSET] = jump input         (0 / 1 as int32)

  If these offsets are wrong the inputs won't be injected correctly.
  The script will still RECORD positions correctly even without injection.

FALLBACK – RECORD ONLY (no input injection)
  Set INJECT_INPUTS = false.  Use the game's built-in attract / replay to play
  the level, and just record positions.  This is the safest starting point.
=============================================================================
]]

-- ============================================================ --
--  CONFIG  –  edit these before running                        --
-- ============================================================ --

-- Path to inputs.json produced by export_attract_for_game.py
-- (Windows-style path is fine here; use double backslash)
local INPUTS_JSON_PATH   = "D:\\nppdata\\inputs.json"

-- Path to attract_info.json (used for the initial-position scan)
local INFO_JSON_PATH     = "D:\\nppdata\\attract_info.json"

-- Where to write game_trace.csv
local OUTPUT_CSV_PATH    = "D:\\nppdata\\game_trace.csv"

-- Set to true to inject inputs each frame.
-- Set to false to record only (safe mode).
local INJECT_INPUTS      = true

-- AOB pattern for the physics tick injection point.
-- "F2 0F 11 41 E8" = movsd [ecx-18], xmm0
-- followed by "E9" (jmp) so the sequence is 6 bytes and unique in npp.dll.
local PHYSICS_AOB        = "F2 0F 11 41 E8 E9"

-- Offsets from ninja struct ECX to the game's input variables.
-- THESE NEED VERIFICATION.  Start with INJECT_INPUTS=false and use
-- ADDRESS DISCOVERY below to find the real offsets for your game version.
local INPUT_HOR_OFFSET   = 0x120   -- placeholder; verify before enabling injection
local INPUT_JUMP_OFFSET  = 0x124   -- placeholder; verify before enabling injection

-- Maximum frames to run before stopping (safety limit)
local MAX_FRAMES         = 25000

-- Print verbose per-frame info for the first N frames (0 = off)
local VERBOSE_FRAMES     = 30

-- ============================================================ --
--  END CONFIG                                                   --
-- ============================================================ --


-- ============================================================ --
--  SIMPLE JSON PARSER  (subset: arrays of objects with numbers)--
-- ============================================================ --

local function jsonParseInputs(text)
    -- Parses a JSON array like:
    -- [{"frame":1,"input_byte":2,"hor":1,"jump":0}, ...]
    -- Returns an array of tables indexed from 1.
    local result = {}
    for chunk in text:gmatch("%b{}") do
        local frame       = tonumber(chunk:match('"frame"%s*:%s*(%d+)'))
        local input_byte  = tonumber(chunk:match('"input_byte"%s*:%s*(%d+)'))
        local hor         = tonumber(chunk:match('"hor"%s*:%s*(%-?%d+)'))
        local jump        = tonumber(chunk:match('"jump"%s*:%s*(%d+)'))
        if frame then
            result[#result + 1] = {
                frame      = frame,
                input_byte = input_byte,
                hor        = hor,
                jump       = jump,
            }
        end
    end
    return result
end

local function jsonParseInfo(text)
    local info = {}
    info.initial_xpos = tonumber(text:match('"initial_xpos"%s*:%s*([%d%.]+)'))
    info.initial_ypos = tonumber(text:match('"initial_ypos"%s*:%s*([%d%.]+)'))
    info.attract_id   = text:match('"attract_id"%s*:%s*"([^"]+)"')
    return info
end


-- ============================================================ --
--  FILE HELPERS                                                 --
-- ============================================================ --

local function readFile(path)
    local f = io.open(path, "r")
    if not f then
        error("Cannot open file: " .. path)
    end
    local content = f:read("*all")
    f:close()
    return content
end

local function writeCSV(path, rows)
    local f = io.open(path, "w")
    if not f then
        error("Cannot write to: " .. path)
    end
    -- Header
    f:write("frame,xpos,ypos,xspeed,yspeed,state,airborn,walled,jump_duration\n")
    for _, r in ipairs(rows) do
        f:write(string.format(
            "%d,%.10f,%.10f,%.10f,%.10f,%d,%s,%s,%d\n",
            r.frame,
            r.xpos, r.ypos,
            r.xspeed, r.yspeed,
            r.state,
            tostring(r.airborn),
            tostring(r.walled),
            r.jump_duration
        ))
    end
    f:close()
end


-- ============================================================ --
--  STATE                                                        --
-- ============================================================ --

local gInputs     = {}   -- array of {frame, hor, jump}
local gRecorded   = {}   -- array of per-frame observations
local gFrameIdx   = 0    -- current frame (0-based index into gInputs)
local gHookAddr   = nil  -- address of the injected jump
local gNewmem     = nil  -- allocated code cave
local gNinjaECX   = nil  -- last known ninja struct address
local gDone       = false
local gInfo       = {}


-- ============================================================ --
--  ADDRESS DISCOVERY                                            --
-- ============================================================ --
--[[
Run these manually in CE's Lua window if the AOB scan fails or you need
to find input offsets.

Step 1: Find xpos address
    1. Start a level in N++.
    2. In CE Lua console:
           local xpos_val = <ninja_spawn_x> * 6   -- from attract_info.json
           -- Search all memory for this double value
           -- Use CE's UI: New Scan → Value Type = Double → Value = <xpos_val>
    3. Move the ninja, rescan for changed double values.
    4. Freeze the remaining addresses one at a time to find xpos.

Step 2: Find what writes to xpos
    1. Right-click the xpos address → "Find out what writes to this address"
    2. Play the level for a few frames.
    3. The instruction movsd [ecx-18],xmm0 at npp.dll+2F010C should appear.
    4. Note the ECX value at that moment.

Step 3: Find input variables
    1. With the level running and ninja idle, scan for value 0 (type int32).
    2. Press right arrow, rescan for value 1.
    3. Release right arrow, rescan for value 0.
    4. The few remaining addresses include hor_input.
    5. Repeat for jump_input (value 0/1 when Z/Shift is pressed).
    6. Compute offset = input_address - ECX from step 2.
    7. Update INPUT_HOR_OFFSET and INPUT_JUMP_OFFSET above.
--]]


-- ============================================================ --
--  PHYSICS HOOK                                                 --
-- ============================================================ --

local function findPhysicsAddr()
    -- AOB scan for the physics tick instruction in npp.dll
    local modBase = getAddressSafe("npp.dll")
    if not modBase then
        error("npp.dll not found.  Is N++ running and attached?")
    end

    print(string.format("[CE] npp.dll base: %X", modBase))

    -- AOB scan
    local scanResult = AOBScan(PHYSICS_AOB)
    if scanResult == nil or scanResult.Count == 0 then
        -- Try without the E9 trailer (less unique but more compatible)
        scanResult = AOBScan("F2 0F 11 41 E8")
        if scanResult == nil or scanResult.Count == 0 then
            error("Physics AOB not found.  Check PHYSICS_AOB or game version.")
        end
    end

    -- Use the first result
    local addr = scanResult[0]
    if type(addr) == "string" then
        addr = tonumber(addr, 16)
    end
    print(string.format("[CE] Physics tick found at: %X", addr))
    scanResult.destroy()
    return addr
end


local function installHook(physicsAddr)
    --[[
    We inject a 5-byte JMP at physicsAddr (replacing "F2 0F 11 41 E8").
    The original 5-byte instruction + the E9 jmp (1 byte) after it
    give us 6 bytes of room; we only need 5 for the JMP.

    Code cave logic (x86 32-bit):
        pushad                     ; save all registers
        mov  [ninjaECXStore], ecx  ; save ECX (ninja struct ptr)
        call luaCallback           ; call our Lua-registered callback
        popad                      ; restore all registers
        movsd [ecx-18], xmm0       ; original instruction
        jmp   returnAddr           ; back to caller
    ]]

    -- Allocate a code cave (256 bytes is plenty)
    local newmem = allocateMemory(256)
    gNewmem = newmem

    -- We need a place to stash ECX between the asm hook and the Lua callback
    local ecxStore    = allocateMemory(4)
    local callbackPtr = allocateMemory(4)

    -- Register a Lua callback that CE's engine will call from the hook
    -- We use createRef to get a stable function pointer
    local function onPhysicsTick()
        -- Read ECX that was stashed by the asm code
        local ecx = readInteger(ecxStore)
        if not ecx or ecx == 0 then return end
        gNinjaECX = ecx

        if gDone then return end

        gFrameIdx = gFrameIdx + 1
        local frameNum = gFrameIdx

        -- ---- Read ninja state ----
        -- [ECX - 0x18] = xpos (double)  -- just written by the instruction we hooked
        -- [ECX + 0x00] = ypos (double)  -- see disassembly near 2F00EB
        -- [ECX - 0x08] = xspeed?
        -- [ECX - 0x10] = yspeed?
        -- [ECX + 0x98] = state (int32)
        --
        -- NOTE: these offsets are derived from the public CE disassembly thread.
        -- They may need adjustment if the game version differs.
        local xpos   = readDouble(ecx - 0x18)  or 0.0
        local ypos   = readDouble(ecx + 0x00)  or 0.0
        local xspeed = readDouble(ecx - 0x08)  or 0.0
        local yspeed = readDouble(ecx - 0x10)  or 0.0
        local state  = readInteger(ecx + 0x98) or -1
        -- airborn / walled are booleans (byte/int at unknown offset; default false)
        local airborn      = false
        local walled       = false
        local jump_duration = 0
        -- TODO: find airborn/walled/jump_duration offsets via ADDRESS DISCOVERY

        -- ---- Record ----
        gRecorded[#gRecorded + 1] = {
            frame        = frameNum,
            xpos         = xpos,
            ypos         = ypos,
            xspeed       = xspeed,
            yspeed       = yspeed,
            state        = state,
            airborn      = airborn,
            walled       = walled,
            jump_duration = jump_duration,
        }

        if frameNum <= VERBOSE_FRAMES then
            print(string.format(
                "[CE] f%4d  pos=(%.4f,%.4f) vel=(%.4f,%.4f) state=%d",
                frameNum, xpos, ypos, xspeed, yspeed, state
            ))
        end

        -- ---- Inject input for the NEXT frame ----
        -- (We inject here so the game processes them before the next tick.)
        if INJECT_INPUTS and frameNum < #gInputs then
            local nextInput = gInputs[frameNum + 1]
            if nextInput then
                writeInteger(ecx + INPUT_HOR_OFFSET,  nextInput.hor)
                writeInteger(ecx + INPUT_JUMP_OFFSET, nextInput.jump)
            end
        end

        -- ---- Terminal state check ----
        if state == 8 or state == 6 or state == 7 then
            -- 8 = celebrating (win), 6/7 = dead/awaiting_death
            gDone = true
            finishRecording()
        elseif frameNum >= MAX_FRAMES then
            gDone = true
            finishRecording()
        end
    end

    -- Store pointer to Lua callback via createRef
    writeInteger(callbackPtr, createRef(onPhysicsTick))

    -- Build the x86 shellcode for the code cave
    local returnAddr = physicsAddr + 5  -- after the 5-byte instruction

    -- Helper to encode a 32-bit relative call/jmp offset
    local function rel32(from, to)
        local diff = to - from - 5
        -- return as 4 little-endian bytes
        diff = diff & 0xFFFFFFFF
        return {
            diff & 0xFF,
            (diff >> 8)  & 0xFF,
            (diff >> 16) & 0xFF,
            (diff >> 24) & 0xFF,
        }
    end

    -- Build shellcode bytes
    -- We use CE's auto assembler via a string instead of raw bytes for clarity
    local asmCode = string.format([[
        pushad
        mov [%X], ecx
        call %X
        popad
        movsd [ecx-18], xmm0
        jmp %X
    ]], ecxStore, readInteger(callbackPtr), returnAddr)

    -- Use CE's auto assembler to write the code cave
    local assembled = autoAssemble(string.format([[
        alloc(newmem_hook, 64)
        registersymbol(newmem_hook)

        newmem_hook:
            pushad
            mov [%d], ecx
            call %d
            popad
            movsd [ecx-18],xmm0
            jmp %d

        %d:
            jmp newmem_hook
    ]], ecxStore, readInteger(callbackPtr), returnAddr, physicsAddr))

    if not assembled then
        error("Auto-assembler failed.  Inspect the code cave template.")
    end

    print("[CE] Physics hook installed.")
    gHookAddr = physicsAddr
end


-- ============================================================ --
--  FINISH & SAVE                                                --
-- ============================================================ --

function finishRecording()
    print(string.format("[CE] Recording complete.  %d frames captured.", #gRecorded))

    -- Write CSV
    local ok, err = pcall(writeCSV, OUTPUT_CSV_PATH, gRecorded)
    if ok then
        print("[CE] game_trace.csv written to: " .. OUTPUT_CSV_PATH)
    else
        print("[CE] ERROR writing CSV: " .. tostring(err))
    end

    -- Uninstall hook (optional; comment out to keep recording after terminal)
    -- uninstallHook()

    print("[CE] Done.  Run compare_traces.py to analyse divergence.")
end


-- ============================================================ --
--  MAIN                                                         --
-- ============================================================ --

local function main()
    print("=== N++ Ground-Truth Recorder ===")

    -- Load inputs
    print("[CE] Loading inputs from: " .. INPUTS_JSON_PATH)
    local inputsText = readFile(INPUTS_JSON_PATH)
    gInputs = jsonParseInputs(inputsText)
    print(string.format("[CE] Loaded %d input frames.", #gInputs))

    -- Load attract info
    local ok, infoText = pcall(readFile, INFO_JSON_PATH)
    if ok then
        gInfo = jsonParseInfo(infoText)
        print(string.format("[CE] Attract: %s  spawn=(%.1f, %.1f)",
            gInfo.attract_id or "?",
            gInfo.initial_xpos or 0,
            gInfo.initial_ypos or 0))
        print("[CE] Use initial_xpos / initial_ypos to find xpos in CE memory scanner.")
    end

    -- Find and hook the physics tick
    local physicsAddr = findPhysicsAddr()
    installHook(physicsAddr)

    print("[CE] Ready – start the level in N++ now.")
    print("     The script will begin recording on the first physics tick.")
    print("     Waiting for frame 1...")

    -- If INJECT_INPUTS is enabled, pre-load frame 1 input (no hook yet so
    -- the game's natural input applies; the hook loads frame N+1 each tick)
    -- Input for frame 1 is injected on the frame-0 tick (before frame 1 executes).
end

-- Run
local ok, err = pcall(main)
if not ok then
    print("[CE] FATAL: " .. tostring(err))
    print("")
    print("=== ADDRESS DISCOVERY MODE ===")
    print("The automatic hook failed.  Follow these manual steps:")
    print("")
    print("Step 1: Scan for xpos")
    print("  - In N++, load the attract level and stand still at spawn.")
    print("  - CE → New Scan → Value Type=Double → value=" ..
          tostring(gInfo and gInfo.initial_xpos or "<initial_xpos from attract_info.json>"))
    print("  - Move ninja, rescan for changed double.")
    print("  - Repeat until 1-4 addresses remain.")
    print("")
    print("Step 2: Right-click xpos → 'Find what writes to this address'")
    print("  - Run the game a few frames.")
    print("  - The instruction  movsd [ecx-18],xmm0  should appear.")
    print("  - Note the module offset (npp.dll + XXXX).")
    print("  - Update PHYSICS_AOB in this script with the bytes at that address.")
    print("")
    print("Step 3: Find input variables")
    print("  - New scan → int32 → value 0 (ninja idle, no keys).")
    print("  - Press Right arrow → rescan → value 1.")
    print("  - Release → rescan → value 0.")
    print("  - Compute offset from ECX (from Step 2) to the found address.")
    print("  - Set INPUT_HOR_OFFSET and INPUT_JUMP_OFFSET above.")
end
