package mcmc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Checkpoint is the complete persisted state required to resume a run and
// reproduce the uninterrupted sample sequence bit-for-bit.
type Checkpoint struct {
	Version int             `json:"version"`
	Seed    uint64          `json:"seed"`
	Config  persistedConfig `json:"config"`
	Chains  []chainSnapshot `json:"chains"`
}

// checkpointVersion is the on-disk format version.
const checkpointVersion = 1

// persistedConfig is the JSON-safe subset of Config. The Target (which
// carries Go functions) is never serialized; callers construct the resume
// Sampler with the same Target, and only scalar parameters are verified.
type persistedConfig struct {
	Algo             Algorithm        `json:"algo"`
	Chains           int              `json:"chains"`
	Iterations       int              `json:"iterations"`
	Warmup           int              `json:"warmup"`
	MHStepSize       float64          `json:"mh_step_size"`
	HMCStepSize      float64          `json:"hmc_step_size"`
	TrajectorySteps  int              `json:"trajectory_steps"`
	TrajectoryMin    int              `json:"trajectory_min"`
	TrajectoryMax    int              `json:"trajectory_max"`
	FiniteDiffEps    float64          `json:"finite_diff_eps"`
	Adapt            AdaptMode        `json:"adapt"`
	TargetAcceptLow  float64          `json:"target_accept_low"`
	TargetAcceptHigh float64          `json:"target_accept_high"`
	InitScale        []float64        `json:"init_scale"`
	RHatThreshold    float64          `json:"rhat_threshold"`
	ESSTarget        float64          `json:"ess_target"`
	Checkpoint       CheckpointConfig `json:"checkpoint"`
	History          HistoryConfig    `json:"history"`
	Dim              int              `json:"dim"`
}

func persistConfig(c *Config) persistedConfig {
	return persistedConfig{
		Algo: c.Algo, Chains: c.Chains, Iterations: c.Iterations,
		Warmup: c.Warmup, MHStepSize: c.MHStepSize, HMCStepSize: c.HMCStepSize,
		TrajectorySteps: c.TrajectorySteps, TrajectoryMin: c.TrajectoryMin,
		TrajectoryMax: c.TrajectoryMax, FiniteDiffEps: c.FiniteDiffEps,
		Adapt: c.Adapt, TargetAcceptLow: c.TargetAcceptLow,
		TargetAcceptHigh: c.TargetAcceptHigh, InitScale: c.InitScale,
		RHatThreshold: c.RHatThreshold, ESSTarget: c.ESSTarget,
		Checkpoint: c.Checkpoint, History: c.History, Dim: c.Target.Dim,
	}
}

func (p persistedConfig) matches(c *Config) error {
	type pair struct {
		name string
		a, b interface{}
	}
	checks := []pair{
		{"algo", p.Algo, c.Algo}, {"chains", p.Chains, c.Chains},
		{"iterations", p.Iterations, c.Iterations}, {"warmup", p.Warmup, c.Warmup},
		{"adapt", p.Adapt, c.Adapt}, {"dim", p.Dim, c.Target.Dim},
		{"mh_step_size", p.MHStepSize, c.MHStepSize},
		{"hmc_step_size", p.HMCStepSize, c.HMCStepSize},
	}
	for _, x := range checks {
		if x.a != x.b {
			return fmt.Errorf("%w: %s mismatch %v vs %v",
				ErrCheckpointCorrupt, x.name, x.a, x.b)
		}
	}
	return nil
}

// saveCheckpoint atomically writes cp to path (temp file + rename).
func saveCheckpoint(path string, cp *Checkpoint) error {
	if path == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidConfig)
	}
	data, err := encodeCheckpoint(cp)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".mcmc-checkpoint-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

// loadCheckpoint reads and validates a checkpoint file.
func loadCheckpoint(path string) (*Checkpoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return decodeCheckpoint(data)
}

// encodeCheckpoint serializes a checkpoint.
func encodeCheckpoint(cp *Checkpoint) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetIndent("", "  ")
	if err := enc.Encode(cp); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// decodeCheckpoint parses a checkpoint.
func decodeCheckpoint(data []byte) (*Checkpoint, error) {
	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCheckpointCorrupt, err)
	}
	if cp.Version != checkpointVersion {
		return nil, fmt.Errorf("%w: version %d, want %d",
			ErrCheckpointCorrupt, cp.Version, checkpointVersion)
	}
	if len(cp.Chains) == 0 {
		return nil, fmt.Errorf("%w: no chains", ErrCheckpointCorrupt)
	}
	return &cp, nil
}
