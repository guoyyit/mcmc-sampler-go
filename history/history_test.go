package history

import "testing"

func TestThinning(t *testing.T) {
	b := New(Policy{Thin: 3})
	for i := 1; i <= 10; i++ {
		b.Add(Draw{Iteration: i, Position: []float64{float64(i)}})
	}
	ds := b.Draws()
	want := []int{1, 4, 7, 10}
	if len(ds) != len(want) {
		t.Fatalf("got %d draws, want %d", len(ds), len(want))
	}
	for i, w := range want {
		if int(ds[i].Position[0]) != w {
			t.Fatalf("draw %d = %v, want iteration %d", i, ds[i].Position, w)
		}
	}
}

func TestMaxSamplesBounded(t *testing.T) {
	b := New(Policy{MaxSamples: 5})
	for i := 1; i <= 100; i++ {
		b.Add(Draw{Iteration: i, Position: []float64{float64(i)}})
		if b.Len() > 5 {
			t.Fatalf("buffer exceeded cap at add %d: %d", i, b.Len())
		}
	}
	ds := b.Draws()
	if len(ds) != 5 {
		t.Fatalf("len = %d, want 5", len(ds))
	}
	// Newest 5 retained.
	for i := 0; i < 5; i++ {
		if int(ds[i].Position[0]) != 96+i {
			t.Fatalf("position %d = %v, want %d", i, ds[i].Position, 96+i)
		}
	}
}

func TestCopyOnWrite(t *testing.T) {
	b := New(Policy{})
	x := []float64{1, 2}
	b.Add(Draw{Iteration: 1, Position: x})
	x[0] = 999
	got := b.Draws()
	if got[0].Position[0] != 1 {
		t.Fatalf("stored draw aliased caller slice: %v", got[0].Position)
	}
	got[0].Position[0] = -5
	again := b.Draws()
	if again[0].Position[0] != 1 {
		t.Fatal("Draws() returned aliased internal storage")
	}
}

func TestMarshalRoundTrip(t *testing.T) {
	b := New(Policy{Thin: 2, MaxSamples: 4})
	for i := 1; i <= 20; i++ {
		b.Add(Draw{Iteration: i, Position: []float64{float64(i)}})
	}
	data, err := b.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	c := New(Policy{})
	if err := c.Unmarshal(data); err != nil {
		t.Fatal(err)
	}
	if c.Len() != b.Len() {
		t.Fatalf("length after restore %d != %d", c.Len(), b.Len())
	}
	// Continue adding: thinning counter must resume consistently.
	before := c.Seen()
	c.Add(Draw{Iteration: 21, Position: []float64{21}})
	if c.Seen() != before+1 {
		t.Fatal("seen counter not restored")
	}
}
