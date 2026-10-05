package rng

import (
	"encoding/binary"
	"fmt"
	"io"
	"math/rand/v2"
)

// streamFormatV1 is prefixed to every serialized stream so future internal
// changes can be detected instead of silently misinterpreted.
const streamFormatV1 byte = 1

// Source is a deterministic random stream whose full state can be serialized
// and restored, enabling crash recovery with bit-identical continuation.
type Source struct {
	chacha *rand.ChaCha8
	inner  *rand.Rand
}

// splitmix64 is a deterministic integer hash used only to expand the single
// user-supplied seed into the ChaCha8 32-byte key.
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// NewRoot creates the root stream of a job from a single seed. The derivation
// is a pure function of seed and never uses time, OS entropy or scheduling.
func NewRoot(seed uint64) *Source {
	var key [32]byte
	// Domain-separation constants keep distinct uses of the seed from
	// producing identical keys.
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(key[i*8:], splitmix64(seed+uint64(i)*0x9e3779b97f4a7c15+0xA17))
	}
	return newFromKey(key)
}

func newFromKey(key [32]byte) *Source {
	c := rand.NewChaCha8(key)
	return &Source{chacha: c, inner: rand.New(c)}
}

// Child derives a deterministic, statistically independent-looking stream
// for one chain.
// The same (parent state, index) pair always yields the same child stream and
// never depends on goroutine scheduling.
//
// Note that deriving a child advances the parent stream, so callers must
// derive children in a fixed order (for example chain 0..N-1).
func Child(parent *Source, index uint64) *Source {
	// Mix the index into the parent stream so two children of the same
	// parent never share a key even if the parent emits repeated blocks.
	scratch := make([]byte, 8)
	binary.LittleEndian.PutUint64(scratch, splitmix64(index))
	if _, err := parent.chacha.Read(scratch); err != nil {
		panic(fmt.Sprintf("rng: parent stream read failed: %v", err))
	}
	var key [32]byte
	if _, err := io.ReadFull(parent.chacha, key[:]); err != nil {
		panic(fmt.Sprintf("rng: child key generation failed: %v", err))
	}
	// Fold the index block into the first 8 key bytes.
	for i := 0; i < 8; i++ {
		key[i] ^= scratch[i]
	}
	return newFromKey(key)
}

func (s *Source) Float64() float64 { return s.inner.Float64() }

// Uint64 returns a raw 64-bit random word.
func (s *Source) Uint64() uint64 { return s.inner.Uint64() }

// Gaussian returns a standard normal variate.
func (s *Source) Gaussian() float64 { return s.inner.NormFloat64() }

// Marshal serializes the complete stream state. Restoring it produces exactly
// the same future variates as an uninterrupted stream.
func (s *Source) Marshal() ([]byte, error) {
	state, err := s.chacha.MarshalBinary()
	if err != nil {
		return nil, fmt.Errorf("rng: marshal stream state: %w", err)
	}
	return append([]byte{streamFormatV1}, state...), nil
}

// Unmarshal restores a stream previously produced by Marshal.
func Unmarshal(data []byte) (*Source, error) {
	if len(data) < 1 || data[0] != streamFormatV1 {
		return nil, fmt.Errorf("rng: unsupported stream format")
	}
	c := new(rand.ChaCha8)
	if err := c.UnmarshalBinary(data[1:]); err != nil {
		return nil, fmt.Errorf("rng: unmarshal stream state: %w", err)
	}
	return &Source{chacha: c, inner: rand.New(c)}, nil
}

// Restore resets an existing stream to a marshaled state in place. It allows
// holders that already reference the stream (for example a sampler) to see
// the restored state without being handed a new pointer, and unlike
// reconstructing a sampler it never advances the stream with jitter draws.
func (s *Source) Restore(data []byte) error {
	if len(data) < 1 || data[0] != streamFormatV1 {
		return fmt.Errorf("rng: unsupported stream format")
	}
	c := new(rand.ChaCha8)
	if err := c.UnmarshalBinary(data[1:]); err != nil {
		return fmt.Errorf("rng: unmarshal stream state: %w", err)
	}
	s.chacha = c
	s.inner = rand.New(c)
	return nil
}
