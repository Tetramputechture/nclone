// Round 9: per-frame ninja AND entity state out of the running game.
//
// Simulator entity storage (`full/ninja_1051A400.c:1408-1453`):
//     count[i] = u16 at sim + 0x10 + 2*i
//     base[i]  = ptr at sim + 0x64 + 4*i
// The pools are one contiguous block in index order, so
//     stride[i] = (next_non_empty_base - base[i]) / count[i]
// which recovers the stride even for a pool holding a single entity. The last
// pool has no successor, so it falls back to measuring the distance between
// repeats of the vtable pointer at offset 0.
'use strict';

var RVA_TICK = 0x51D800;
var RVA_RESET = 0x51A160;
var NINJA0 = 0x335BF8;
var NPOOL = 32;          // 38+ reads garbage; entity types run 0..28
var NINJA_SLAB = 0xD40;   // through the 13 bones at 0xC60..0xD30
var SCAN = 0x4000;
var BATCH = 20;

var layout = null;       // {pools: [...], bytes: n}
var frame = 0;
var demo = 0;
var buf = null;
var nbuf = 0;
var recBytes = 0;
var installed = false;

function computeLayout(sim) {
  var live = [];
  for (var i = 0; i < NPOOL; i++) {
    var cnt, ptr;
    try {
      cnt = sim.add(0x10 + 2 * i).readU16();
      ptr = sim.add(0x64 + 4 * i).readPointer();
    } catch (e) { continue; }
    if (cnt > 0 && cnt < 4096 && !ptr.isNull()) {
      live.push({ i: i, count: cnt, base: ptr });
    }
  }
  live.sort(function (a, b) { return a.base.compare(b.base); });
  for (var k = 0; k < live.length; k++) {
    var e = live[k];
    var stride = null;
    if (k + 1 < live.length) {
      var span = live[k + 1].base.sub(e.base).toInt32();
      if (span > 0 && span % e.count === 0) stride = span / e.count;
    }
    if (stride === null) {
      try {
        var vt = e.base.readPointer();
        if (!vt.isNull()) {
          for (var off = 4; off < SCAN; off += 4) {
            if (e.base.add(off).readPointer().equals(vt)) { stride = off; break; }
          }
        }
      } catch (err) { /* leave null */ }
    }
    e.stride = stride;
    try { e.vtableRva = e.base.readPointer().sub(Process.findModuleByName('npp.dll').base).toInt32(); }
    catch (err) { e.vtableRva = null; }
  }
  var total = 0;
  var pools = [];
  for (var j = 0; j < live.length; j++) {
    var p = live[j];
    if (!p.stride) continue;
    pools.push({ i: p.i, count: p.count, stride: p.stride,
                 base: p.base.toString(), vtableRva: p.vtableRva,
                 offset: total });
    total += p.count * p.stride;
  }
  return { pools: pools, bytes: total, live: live.length };
}

function flush() {
  if (nbuf === 0) return;
  send({ t: 'frames', demo: demo, n: nbuf, rec: recBytes },
       buf.slice(0, nbuf * recBytes).buffer);
  nbuf = 0;
}

function install() {
  var m = Process.findModuleByName('npp.dll');
  if (m === null) return false;
  send({ t: 'module', base: m.base.toString() });

  Interceptor.attach(m.base.add(RVA_TICK), {
    onEnter: function () { this.sim = this.context.ecx; },
    onLeave: function () {
      var sim = this.sim;
      if (sim.isNull()) return;

      if (layout === null) {
        layout = computeLayout(sim);
        recBytes = 4 + NINJA_SLAB + layout.bytes;
        buf = new Uint8Array(recBytes * BATCH);
        nbuf = 0;
        send({ t: 'layout', demo: demo, sim: sim.toString(),
               pools: layout.pools, entityBytes: layout.bytes,
               recBytes: recBytes });
      }

      frame++;
      var o = nbuf * recBytes;
      buf[o] = frame & 0xff;
      buf[o + 1] = (frame >>> 8) & 0xff;
      buf[o + 2] = (frame >>> 16) & 0xff;
      buf[o + 3] = (frame >>> 24) & 0xff;
      buf.set(new Uint8Array(sim.add(NINJA0).readByteArray(NINJA_SLAB)), o + 4);
      var eo = o + 4 + NINJA_SLAB;
      for (var k = 0; k < layout.pools.length; k++) {
        var p = layout.pools[k];
        var n = p.count * p.stride;
        try {
          buf.set(new Uint8Array(new NativePointer(p.base).readByteArray(n)), eo + p.offset);
        } catch (e) { /* leave zeros */ }
      }
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
      layout = null;
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
