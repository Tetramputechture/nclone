// Per-frame ninja ground truth out of the running game.
//
//   FUN_1051D800  Simulator::tick   RVA 0x51D800   __thiscall, ECX = Simulator*
//   FUN_1051A160  Simulator::reset  RVA 0x51A160   -> demo boundary
//   ninja[0]      sim + 0x335BF8    inline, stride 0xE30
//   tiles         sim + 0x5C8       44*25 bytes
//
// State is sampled in onLeave, i.e. AFTER the tick, so record N is the state at
// the end of frame N. That matches nclone's sim_trace.json convention.
'use strict';

var RVA_TICK = 0x51D800;
var RVA_RESET = 0x51A160;
var NINJA0 = 0x335BF8;
var TILES = 0x5C8;
var TILE_N = 1100;

var SLAB = 0x200;              // bytes of ninja copied per frame
var REC = 4 + SLAB;            // u32 frame + slab
var BATCH = 60;                // frames per send

var buf = new Uint8Array(REC * BATCH);
var nbuf = 0;
var frame = 0;
var demo = 0;
var needHeader = true;
var installed = false;

function flush() {
  if (nbuf === 0) return;
  var out = buf.slice(0, nbuf * REC);
  send({ t: 'frames', demo: demo, n: nbuf }, out.buffer);
  nbuf = 0;
}

function install() {
  var m = Process.findModuleByName('npp.dll');
  if (m === null) return false;
  send({ t: 'module', base: m.base.toString(), size: m.size });

  Interceptor.attach(m.base.add(RVA_TICK), {
    onEnter: function () { this.sim = this.context.ecx; },
    onLeave: function () {
      var sim = this.sim;
      if (sim.isNull()) return;

      if (needHeader) {
        needHeader = false;
        var tiles = sim.add(TILES).readByteArray(TILE_N);
        send({ t: 'demo', demo: demo, sim: sim.toString() }, tiles);
      }

      frame++;
      var rec = new Uint8Array(sim.add(NINJA0).readByteArray(SLAB));
      var o = nbuf * REC;
      buf[o] = frame & 0xff;
      buf[o + 1] = (frame >>> 8) & 0xff;
      buf[o + 2] = (frame >>> 16) & 0xff;
      buf[o + 3] = (frame >>> 24) & 0xff;
      buf.set(rec, o + 4);
      nbuf++;
      if (nbuf === BATCH) flush();
    }
  });

  Interceptor.attach(m.base.add(RVA_RESET), {
    onEnter: function () {
      flush();
      send({ t: 'end', demo: demo, frames: frame });
      demo++;
      frame = 0;
      needHeader = true;
    }
  });

  send({ t: 'installed' });
  return true;
}

var iv = setInterval(function () {
  if (installed) { clearInterval(iv); return; }
  try { installed = install(); } catch (e) { send({ t: 'err', msg: '' + e }); }
}, 100);

rpc.exports = { flush: function () { flush(); } };
