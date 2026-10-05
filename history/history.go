package history

import (
	"bytes"
	"encoding/gob"
	"fmt"
)

// Draw is one retained post-warmup sample.
type Draw struct {
	Iteration int
	Position  []float64
	LogProb   float64
}

// Policy bounds retained memory. Thinning keeps 1 of Thin retained draws;
// MaxSamples is a hard cap retaining only the newest MaxSamples draws
// (0 = unbounded).
type Policy struct {
	Thin       int
	MaxSamples int
}

type bufferState struct {
	Policy Policy
	Seen   int // number of post-warmup draws presented to Add
	Draws  []Draw
}

// Buffer stores retained draws. Every retained position is copied, so later
// mutation of the sampler's scratch arrays can never corrupt history.
type Buffer struct {
	policy Policy
	seen   int
	draws  []Draw
}

// New constructs a buffer, normalizing an unset thinning factor to 1.
func New(policy Policy) *Buffer {
	if policy.Thin < 1 {
		policy.Thin = 1
	}
	return &Buffer{policy: policy}
}

// Add presents a post-warmup draw. It is retained if it passes the thinning
// rule, after which the MaxSamples cap is enforced by dropping the oldest
// retained draw.
func (b *Buffer) Add(d Draw) {
	b.seen++
	if (b.seen-1)%b.policy.Thin != 0 {
		return
	}
	stored := Draw{Iteration: d.Iteration, LogProb: d.LogProb, Position: append([]float64(nil), d.Position...)}
	b.draws = append(b.draws, stored)
	if b.policy.MaxSamples > 0 && len(b.draws) > b.policy.MaxSamples {
		// Drop oldest; copy remaining into a fresh backing array so the
		// dropped capacity cannot retain references.
		trimmed := make([]Draw, b.policy.MaxSamples)
		copy(trimmed, b.draws[len(b.draws)-b.policy.MaxSamples:])
		b.draws = trimmed
	}
}

// Draws returns a deep copy of the retained draws.
func (b *Buffer) Draws() []Draw {
	out := make([]Draw, len(b.draws))
	for i, d := range b.draws {
		out[i] = Draw{Iteration: d.Iteration, LogProb: d.LogProb, Position: append([]float64(nil), d.Position...)}
	}
	return out
}

// Len returns the number of retained draws.
func (b *Buffer) Len() int { return len(b.draws) }

// Seen returns the number of post-warmup draws presented to Add.
func (b *Buffer) Seen() int { return b.seen }

// Marshal encodes the full buffer, including its thinning counter.
func (b *Buffer) Marshal() ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(bufferState{
		Policy: b.policy,
		Seen:   b.seen,
		Draws:  b.draws,
	}); err != nil {
		return nil, fmt.Errorf("history: marshal buffer: %w", err)
	}
	return buf.Bytes(), nil
}

// Unmarshal restores a buffer previously produced by Marshal.
func (b *Buffer) Unmarshal(data []byte) error {
	var st bufferState
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&st); err != nil {
		return fmt.Errorf("history: unmarshal buffer: %w", err)
	}
	if st.Policy.Thin < 1 {
		st.Policy.Thin = 1
	}
	b.policy = st.Policy
	b.seen = st.Seen
	b.draws = st.Draws
	return nil
}
