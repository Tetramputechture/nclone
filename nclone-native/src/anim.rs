//! The ninja's animation skeleton.
//!
//! This exists for exactly one reason: the game's beam-vs-ninja test
//! (`FUN_1050B070`) intersects a hitscan beam against the ninja's **11 limbs**,
//! not against a circle. A beam that passes within `NINJA_RADIUS` of the centre
//! but threads *between* two limbs does not kill. Without a skeleton both ports
//! fall back to the circle prefilter alone, which is strictly more lethal than
//! the game -- and `docs/ENTITY_RE_ROUND8_FINDINGS.md` §6.3 measured that
//! difference directly against captured ground truth: with limbs on, `10133234`
//! goes from dying on frame 249 to winning on 507, and `10134679` from frame 10
//! to 142.
//!
//! The port is of `nclone/ninja.py`'s `update_graphics` / `calc_ninja_position`,
//! which is itself the port of `FUN_10516280` (`updateAnimation`) and
//! `FUN_105114B0` (the pose evaluator). Keeping the two engines frame-exact is
//! the hard constraint here, so this mirrors the Python arithmetic **operation
//! for operation** rather than the tidier form -- see the notes in
//! `Ninja::update_graphics`.

use std::sync::OnceLock;

/// `anim_data_line_new.txt.bin`, shipped both in this repo and in the game
/// install at `NPP/anim_data_line_new.txt.bin`. Layout: a `u32` pose count, then
/// that many 208-byte poses of 13 bones x (x, y) little-endian doubles.
///
/// Embedded rather than read at runtime because the Rust engine has no notion of
/// a project root, and a missing asset would silently change physics once the
/// limb test is the default.
const ANIM_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../anim_data_line_new.txt.bin"));

/// One pose: 13 bones, each a normalised `(x, y)` offset from the ninja centre.
pub type Pose = [[f64; 2]; 13];

/// The 11 limbs `FUN_1050B070` tests the beam against, as bone-index pairs
/// (round-7 §3.1). 0 and 1 are the two hubs, `2-8-0` and `3-9-0` the legs,
/// `4-10-1` and `5-11-1` the arms, 12 the head, and `6-7` one extra limb.
///
/// **This is not the renderer's limb list.** `render_utils.LIMBS` draws `(1, 12)`
/// where the collision skeleton has `(0, 1)`; only ten of the eleven are shared.
pub const NINJA_LIMB_PAIRS: [(usize, usize); 11] = [
    (12, 0),
    (0, 1),
    (2, 8),
    (8, 0),
    (3, 9),
    (9, 0),
    (4, 10),
    (10, 1),
    (5, 11),
    (11, 1),
    (6, 7),
];

/// Bones are stored normalised; the game places each at `pos + 24.0 * bone`
/// (`DAT_1056C360`), which is also how the renderer draws them.
pub const BONE_SCALE: f64 = 24.0;

/// Victory-dance clip ranges, `DANCE_DIC` in `physics_constants.py`. Only entry 0
/// is reachable here: the Python sim picks a random dance when `DANCE_RANDOM` is
/// set, and a random choice cannot be frame-exact across engines, so this port
/// always uses `DANCE_ID_DEFAULT`. The dance only plays in state 8, after the
/// demo has already been won, so nothing observable depends on the choice.
pub const DANCE_DEFAULT: (usize, usize) = (104, 104);

static POSES: OnceLock<Vec<Pose>> = OnceLock::new();

/// The decoded pose table. Panics if the embedded asset is malformed, which can
/// only happen if the file in the repo root is replaced with something else --
/// silently degrading to no skeleton would change physics without a signal.
pub fn poses() -> &'static [Pose] {
    POSES.get_or_init(|| {
        let count = u32::from_le_bytes(ANIM_BYTES[0..4].try_into().unwrap()) as usize;
        let need = 4 + count * 13 * 16;
        assert!(
            ANIM_BYTES.len() >= need,
            "anim_data_line_new.txt.bin: {} poses need {} bytes, file has {}",
            count,
            need,
            ANIM_BYTES.len()
        );
        let mut out = Vec::with_capacity(count);
        let mut off = 4;
        for _ in 0..count {
            let mut pose: Pose = [[0.0; 2]; 13];
            for bone in pose.iter_mut() {
                for c in bone.iter_mut() {
                    *c = f64::from_le_bytes(ANIM_BYTES[off..off + 8].try_into().unwrap());
                    off += 8;
                }
            }
            out.push(pose);
        }
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_table_decodes() {
        let p = poses();
        assert_eq!(p.len(), 2296, "pose count from the embedded asset");
        // Every bone offset is normalised, so nothing should sit far from the
        // origin. A misread stride shows up here immediately.
        for pose in p {
            for bone in pose {
                assert!(bone[0].abs() < 4.0 && bone[1].abs() < 4.0, "bone {:?}", bone);
            }
        }
    }
}
