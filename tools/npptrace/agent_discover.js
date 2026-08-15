// Round 9 discovery: recover the Simulator's entity pool layout at runtime.
//
// From `full/ninja_1051A400.c:1408-1453` the Simulator holds parallel arrays:
//     count[i] = u16 at sim + 0x10 + 2*i
//     base[i]  = ptr at sim + 0x64 + 4*i
// with a per-type stride (0x30 and 0x40 appear inline; Ghidra lost the rest to
// registers). Each entity starts with a vtable pointer, so the stride is the
// distance between repeats of that pointer -- which we can just measure.
'use strict';

var RVA_TICK = 0x51D800;
var NPOOL = 40;          // generous; empty pools are reported and ignored
var SCAN = 0x4000;       // bytes of each pool to scan for the vtable repeat

var done = false;
var installed = false;

function install() {
  var m = Process.findModuleByName('npp.dll');
  if (m === null) return false;
  var base = m.base;

  Interceptor.attach(base.add(RVA_TICK), {
    onEnter: function () {
      if (done) return;
      var sim = this.context.ecx;
      if (sim.isNull()) return;
      done = true;

      var pools = [];
      for (var i = 0; i < NPOOL; i++) {
        var cnt, ptr;
        try {
          cnt = sim.add(0x10 + 2 * i).readU16();
          ptr = sim.add(0x64 + 4 * i).readPointer();
        } catch (e) { continue; }
        if (cnt === 0 || ptr.isNull()) {
          pools.push({ i: i, count: cnt, ptr: ptr.toString(), stride: null });
          continue;
        }
        // measure the stride: find the next repeat of the head dword (vtable)
        var stride = null, vt = null;
        try {
          vt = ptr.readPointer();
          if (!vt.isNull()) {
            for (var off = 4; off < SCAN; off += 4) {
              if (ptr.add(off).readPointer().equals(vt)) { stride = off; break; }
            }
          }
        } catch (e) { /* short pool */ }
        pools.push({
          i: i, count: cnt, ptr: ptr.toString(),
          vtable: vt ? vt.toString() : null,
          vtableRva: vt ? vt.sub(base).toInt32() : null,
          stride: stride
        });
      }
      send({ t: 'pools', sim: sim.toString(), base: base.toString(), pools: pools });

      // raw head of the Simulator, so the count/ptr arrays can be eyeballed
      send({ t: 'simhead', sim: sim.toString() },
           sim.readByteArray(0x140));

      // and a slab of the first non-empty pool for field hunting
      for (var k = 0; k < pools.length; k++) {
        if (pools[k].stride && pools[k].count) {
          var n = Math.min(pools[k].count, 8);
          send({ t: 'poolslab', i: pools[k].i, stride: pools[k].stride, n: n },
               ptrFrom(pools[k].ptr).readByteArray(pools[k].stride * n));
        }
      }
    }
  });
  send({ t: 'installed' });
  return true;
}

function ptrFrom(s) { return new NativePointer(s); }

var iv = setInterval(function () {
  if (installed) { clearInterval(iv); return; }
  try { installed = install(); } catch (e) { send({ t: 'err', msg: '' + e }); }
}, 100);
