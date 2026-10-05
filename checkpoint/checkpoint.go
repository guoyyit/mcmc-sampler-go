package checkpoint

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Version is the checkpoint format version.
const Version = 1

// FileName is the single checkpoint slot inside a checkpoint directory.
const FileName = "checkpoint.gob"

// ErrNoCheckpoint is returned by Load when no checkpoint exists yet.
var ErrNoCheckpoint = errors.New("checkpoint: no checkpoint found")

// ChainSnapshot is the complete resumable state of one chain.
type ChainSnapshot struct {
	ID        int
	Iteration int
	RNG       []byte // rng.Source state
	Sampler   []byte // sampler.Sampler state
	Adapt     []byte // adapt.Engine state (nil if unused)
	History   []byte // history.Buffer state
}

// Checkpoint is the complete resumable state of a job.
type Checkpoint struct {
	Version   int
	JobID     string
	UpdatedAt time.Time
	Seed      uint64
	Draws     int
	Warmup    int
	NumChains int
	Chains    []ChainSnapshot
}

// Store persists a single checkpoint slot atomically.
type Store interface {
	Save(cp *Checkpoint) error
	Load() (*Checkpoint, error)
}

// FileStore writes checkpoints to a directory using temp-file + rename, so a
// crash during Save can never leave a torn checkpoint behind.
type FileStore struct {
	dir string
}

// NewFileStore creates the directory if needed.
func NewFileStore(dir string) (*FileStore, error) {
	if dir == "" {
		return nil, errors.New("checkpoint: directory must not be empty")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("checkpoint: create directory: %w", err)
	}
	return &FileStore{dir: dir}, nil
}

func (s *FileStore) path() string { return filepath.Join(s.dir, FileName) }

// Save encodes the checkpoint gob-serialized and atomically replaces the slot.
func (s *FileStore) Save(cp *Checkpoint) error {
	if cp == nil {
		return errors.New("checkpoint: cannot save nil checkpoint")
	}
	cp.Version = Version
	cp.UpdatedAt = time.Now()

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(cp); err != nil {
		return fmt.Errorf("checkpoint: encode: %w", err)
	}

	final := s.path()
	tmp, err := os.CreateTemp(s.dir, ".checkpoint-*")
	if err != nil {
		return fmt.Errorf("checkpoint: create temp file: %w", err)
	}
	tmpName := tmp.Name()
	cleanup := func() { _ = os.Remove(tmpName) }

	if _, err := tmp.Write(buf.Bytes()); err != nil {
		_ = tmp.Close()
		cleanup()
		return fmt.Errorf("checkpoint: write: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		cleanup()
		return fmt.Errorf("checkpoint: sync: %w", err)
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return fmt.Errorf("checkpoint: close: %w", err)
	}
	if err := os.Rename(tmpName, final); err != nil {
		cleanup()
		return fmt.Errorf("checkpoint: rename: %w", err)
	}
	return nil
}

// Load reads and decodes the current checkpoint slot.
func (s *FileStore) Load() (*Checkpoint, error) {
	f, err := os.Open(s.path())
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrNoCheckpoint
		}
		return nil, fmt.Errorf("checkpoint: open: %w", err)
	}
	defer f.Close()

	var cp Checkpoint
	if err := gob.NewDecoder(f).Decode(&cp); err != nil {
		return nil, fmt.Errorf("checkpoint: decode: %w", err)
	}
	if cp.Version != Version {
		return nil, fmt.Errorf("checkpoint: unsupported version %d (want %d)", cp.Version, Version)
	}
	if len(cp.Chains) != cp.NumChains {
		return nil, fmt.Errorf("checkpoint: corrupt: chain count mismatch")
	}
	return &cp, nil
}
