package rng

import "testing"

func TestRootDeterminism(t *testing.T) {
	a := NewRoot(42)
	b := NewRoot(42)
	for i := 0; i < 1000; i++ {
		x, y := a.Float64(), b.Float64()
		if x != y {
			t.Fatalf("root streams diverged at %d: %v != %v", i, x, y)
		}
	}
	if NewRoot(1).Float64() == NewRoot(2).Float64() {
		t.Fatal("different seeds produced identical first draws")
	}
}

func TestChildDerivationOrderIndependentOfSchedule(t *testing.T) {
	// Derive children in order 0,1,2 twice: the streams must be identical
	// and depend only on (seed, index), never on when goroutines consume them.
	mk := func() []*Source {
		root := NewRoot(123)
		children := make([]*Source, 3)
		for i := 0; i < 3; i++ {
			children[i] = Child(root, uint64(i))
		}
		return children
	}
	first := mk()
	second := mk()
	// Simulate out-of-order consumption on the second run.
	order := []int{2, 0, 1}
	for _, i := range order {
		for k := 0; k < 100; k++ {
			if first[i].Gaussian() != second[i].Gaussian() {
				t.Fatalf("chain %d stream depends on consumption schedule", i)
			}
		}
	}
}

func TestMarshalRoundTripContinuation(t *testing.T) {
	s := NewRoot(99)
	var recorded []float64
	for i := 0; i < 500; i++ {
		recorded = append(recorded, s.Float64(), s.Gaussian())
	}

	// Replay first 200 variates, snapshot, then continue on a restored copy.
	a := NewRoot(99)
	var prefix []float64
	for i := 0; i < 200; i++ {
		prefix = append(prefix, a.Float64(), a.Gaussian())
	}
	data, err := a.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	b, err := Unmarshal(data)
	if err != nil {
		t.Fatal(err)
	}
	var suffix []float64
	for i := 0; i < 300; i++ {
		suffix = append(suffix, b.Float64(), b.Gaussian())
	}
	got := append(prefix, suffix...)
	for i := range recorded {
		if recorded[i] != got[i] {
			t.Fatalf("restored stream differs at variate %d: %v != %v", i, recorded[i], got[i])
		}
	}
}
