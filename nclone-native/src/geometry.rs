//! Orientation <-> vector helpers shared by directional entities.
//!
//! Direct port of the orientation section of nclone/physics.py. These used to
//! be duplicated verbatim in `entities::one_way_platform` and
//! `entities::locked_door`; the drone / thwump / launch-pad ports need them too,
//! so they live here now.

/// Return a normalized vector pointing in the direction of an 8-way
/// orientation. Port of `map_orientation_to_vector` (nclone/physics.py:269).
///
/// `diag = sqrt(2)/2` is computed the same way Python's `math.sqrt(2)/2` is, so
/// the f64 is bit-identical. Python raises `KeyError` on an orientation outside
/// 0..7; map data is a byte and never does, so we mask instead of panicking.
#[inline]
pub fn map_orientation_to_vector(orientation: u8) -> (f64, f64) {
    let diag = (2.0_f64).sqrt() / 2.0;
    match orientation & 7 {
        0 => (1.0, 0.0),
        1 => (diag, diag),
        2 => (0.0, 1.0),
        3 => (-diag, diag),
        4 => (-1.0, 0.0),
        5 => (-diag, -diag),
        6 => (0.0, -1.0),
        7 => (diag, -diag),
        _ => unreachable!(),
    }
}

/// Round half to EVEN, matching Python's built-in `round()`.
///
/// Rust's `f64::round` rounds half away from zero, so `0.5` becomes `1` where
/// Python yields `0`. Only `map_vector_to_orientation` needs this, and only at
/// exact half-octant angles, but getting it wrong would be a silent one-octant
/// divergence rather than a crash -- so do it properly.
#[inline]
fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// Return an orientation value (0..7) from an arbitrary vector, rounding to the
/// nearest of the 8 directions. Port of `map_vector_to_orientation`
/// (nclone/physics.py:190).
#[inline]
pub fn map_vector_to_orientation(xdir: f64, ydir: f64) -> u8 {
    let mut angle = ydir.atan2(xdir);
    if angle < 0.0 {
        angle += 2.0 * std::f64::consts::PI;
    }
    let idx = round_half_even(8.0 * angle / (2.0 * std::f64::consts::PI));
    (idx as i64).rem_euclid(8) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_roundtrips_through_vector() {
        for o in 0u8..8 {
            let (x, y) = map_orientation_to_vector(o);
            assert_eq!(map_vector_to_orientation(x, y), o, "orientation {o}");
        }
    }

    #[test]
    fn axis_aligned_depen_normals_map_to_even_orientations() {
        // The only real caller (shove thwump) feeds axis-aligned depen normals.
        assert_eq!(map_vector_to_orientation(1.0, 0.0), 0);
        assert_eq!(map_vector_to_orientation(0.0, 1.0), 2);
        assert_eq!(map_vector_to_orientation(-1.0, 0.0), 4);
        assert_eq!(map_vector_to_orientation(0.0, -1.0), 6);
    }

    #[test]
    fn rounds_half_to_even_like_python() {
        // Python's round() is half-to-even; Rust's f64::round() is half-away-
        // from-zero. These are the cases where they disagree.
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        // Non-half values are unaffected.
        assert_eq!(round_half_even(0.4), 0.0);
        assert_eq!(round_half_even(0.6), 1.0);
    }

    #[test]
    fn half_octant_angles_match_python() {
        // Values cross-checked against the Python reference. Note pi/8 does NOT
        // land on an exact half: atan2(sin,cos) gives 0.5000000000000001, so
        // both sides round to 1. 3pi/8 *is* exactly 1.5, and half-to-even makes
        // it 2 -- which happens to agree with round-half-up here, so the real
        // guard for the rounding rule is `rounds_half_to_even_like_python`.
        let a = std::f64::consts::PI / 8.0;
        assert_eq!(map_vector_to_orientation(a.cos(), a.sin()), 1);
        let b = 3.0 * std::f64::consts::PI / 8.0;
        assert_eq!(map_vector_to_orientation(b.cos(), b.sin()), 2);
    }
}
