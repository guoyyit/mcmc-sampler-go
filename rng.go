package mcmc

import "math"

const (
	splitMixGolden = 0x9e3779b97f4a7c15
	twoPow53       = 1.0 / 9007199254740992.0 // 2^-53
)

// rng64 is the per-chain deterministic random source. The full internal
// state is serializable in an RNGSnapshot so checkpoints reproduce draws
// bit-for-bit.
type rng64 interface {
	// Float64 returns a pseudo-random number in [0,1).
	Float64() float64
	// NormalFloat64 returns a standard-normal pseudo-random number.
	NormalFloat64() float64
	// Snapshot captures the complete state for checkpointing.
	Snapshot() RNGSnapshot
	// Restore overwrites the state from a snapshot.
	Restore(s RNGSnapshot) error
}

// RNGSnapshot is a versioned, serializable RNG state.
type RNGSnapshot struct {
	Kind string `json:"kind"`
	Hi   uint64 `json:"hi"`
	Lo   uint64 `json:"lo"`
}

// splitMixRNG is a deterministic, full-period (2^64) SplitMix64 generator
// (Sebastiano Vigna). Its complete state is one uint64, which makes exact
// checkpoint reproduction straightforward.
type splitMixRNG struct {
	state uint64
}

func newSplitMix(seed uint64) *splitMixRNG { return &splitMixRNG{state: seed} }

func (r *splitMixRNG) next() uint64 {
	r.state += splitMixGolden
	return splitmix64(r.state)
}

func (r *splitMixRNG) Float64() float64 {
	return float64(r.next()>>11) * twoPow53
}

func (r *splitMixRNG) NormalFloat64() float64 {
	// Box-Muller, consuming two uniforms per call (no cached spare, so
	// no hidden state needs checkpointing).
	u1 := r.Float64()
	if u1 == 0 {
		u1 = twoPow53 // keep log(u1) finite
	}
	u2 := r.Float64()
	const twoPi = 6.283185307179586476925286766559005768
	z := math.Sqrt(-2*math.Log(u1)) * math.Cos(twoPi*u2)
	if math.IsNaN(z) || math.IsInf(z, 0) {
		// Defensive: the guard above makes this unreachable.
		return 0
	}
	return z
}

func (r *splitMixRNG) Snapshot() RNGSnapshot {
	return RNGSnapshot{Kind: "splitmix64", Hi: r.state}
}

func (r *splitMixRNG) Restore(s RNGSnapshot) error {
	if s.Kind != "splitmix64" {
		return ErrCheckpointCorrupt
	}
	r.state = s.Hi
	return nil
}

// chainSeed derives an independent, deterministic 64-bit seed for chain i
// from the master seed. It never uses time, PID, goroutine ID or map order.
func chainSeed(master uint64, chainIndex, chainCount int) uint64 {
	// Distinct, well-mixed seeds; inputs other than the master seed and
	// chain index deliberately do not enter the derivation.
	x := master ^ splitMixGolden
	x = splitmix64(x)
	x ^= uint64(chainIndex+1) * 0xbf58476d1ce4e5b9
	x ^= uint64(chainCount+1) * 0x94d049bb133111eb
	return splitmix64(x)
}

// splitmix64 is the SplitMix64 mixing step.
func splitmix64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}
