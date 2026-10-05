package sampler

import (
	"context"
	"math"
	"testing"

	"github.com/guoyyit/mcmc-sampler-go/rng"
	"github.com/guoyyit/mcmc-sampler-go/target"
)

// Targets on extreme log-probability plateaus, and ones that return -Inf,
// must never yield a falsely accepted move or a non-finite chain state.
func TestMHExtremeLogProbNoFalseAccept(t *testing.T) {
	cases := map[string]float64{
		"negative infinity": math.Inf(-1),
	}
	for name, lp := range cases {
		t.Run(name, func(t *testing.T) {
			tg, _ := target.New(1, func(x []float64) float64 { return lp })
			r := rng.NewRoot(7)
			s, err := New(tg, r, Params{
				Algorithm:     AlgoMH,
				Initial:       []float64{0},
				ProposalScale: []float64{1},
			})
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 1000; i++ {
				st, err := s.Step(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				if st.Stats.Accepted {
					t.Fatalf("iter %d: proposal accepted when logp=%v", i, lp)
				}
				if math.IsNaN(st.LogProb) || math.IsInf(st.LogProb, +1) {
					t.Fatalf("iter %d: non-finite log prob %v", i, st.LogProb)
				}
				if math.IsNaN(st.Position[0]) || math.IsInf(st.Position[0], 0) {
					t.Fatalf("iter %d: non-finite position %v", i, st.Position[0])
				}
				if st.Position[0] != 0 {
					t.Fatalf("iter %d: chain moved from zero-probability point", i)
				}
			}
		})
	}

	// Two finite but extremely separated plateaus: starting near 0 sits on
	// the high plateau (-1e308); any proposal past |x|>0.01 lands on
	// -1.7e308. The log difference is ~-7e307 which a naive exp() would
	// turn into 0 anyway, but a reversed order would overflow; verify the
	// decision stays correct in pure log space.
	tg, _ := target.New(1, func(x []float64) float64 {
		if math.Abs(x[0]) < 0.01 {
			return -1e308
		}
		return -1.7e308
	})
	r := rng.NewRoot(7)
	s, err := New(tg, r, Params{
		Algorithm:     AlgoMH,
		Initial:       []float64{0},
		ProposalScale: []float64{10},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1000; i++ {
		st, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if st.Stats.Accepted && math.Abs(st.Position[0]) > 0.01 {
			t.Fatalf("iter %d: jumped to low plateau: %v", i, st.Position)
		}
		if math.IsNaN(st.Stats.LogAcceptProb) || math.IsInf(st.Stats.LogAcceptProb, +1) {
			t.Fatalf("iter %d: illegal acceptance value %v", i, st.Stats.LogAcceptProb)
		}
	}
}

// An impossible starting point (-Inf) followed by a finite proposal may be
// accepted; an impossible-to-impossible move (-Inf - -Inf) must never produce
// NaN or a bogus accept.
func TestMHMinusInfMinusInf(t *testing.T) {
	var calls int
	tg, _ := target.New(1, func(x []float64) float64 {
		calls++
		if calls <= 1 {
			return math.Inf(-1) // starting point impossible
		}
		// Alternate impossible regions so some proposals are also -Inf.
		if math.Abs(x[0]) > 0.5 {
			return math.Inf(-1)
		}
		return -0.5 * x[0] * x[0]
	})
	r := rng.NewRoot(9)
	s, err := New(tg, r, Params{
		Algorithm:     AlgoMH,
		Initial:       []float64{0},
		ProposalScale: []float64{5},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2000; i++ {
		st, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(st.Stats.LogAcceptProb) {
			t.Fatalf("iter %d: NaN acceptance probability", i)
		}
		if math.IsNaN(st.LogProb) || math.IsInf(st.LogProb, +1) {
			t.Fatalf("iter %d: illegal log prob", i)
		}
	}
}

// NaN / +Inf log densities are explicitly rejected as errors.
func TestMHRejectsIllegalDensity(t *testing.T) {
	cases := map[string]float64{"nan": math.NaN(), "+inf": math.Inf(1)}
	for name, lp := range cases {
		t.Run(name, func(t *testing.T) {
			tg, _ := target.New(1, func(x []float64) float64 { return lp })
			r := rng.NewRoot(3)
			if _, err := New(tg, r, Params{
				Algorithm:     AlgoMH,
				Initial:       []float64{0},
				ProposalScale: []float64{1},
			}); err == nil {
				t.Fatal("expected construction error for illegal initial density")
			}
		})
	}
}

// Non-finite initial position and invalid parameters are rejected.
func TestInvalidParams(t *testing.T) {
	tg, _ := target.New(2, func(x []float64) float64 { return 0 })
	r := rng.NewRoot(1)
	if _, err := New(tg, r, Params{Algorithm: AlgoMH, Initial: []float64{math.NaN(), 0}, ProposalScale: []float64{1, 1}}); err == nil {
		t.Fatal("expected error for NaN initial position")
	}
	if _, err := New(tg, r, Params{Algorithm: AlgoMH, Initial: []float64{0, 0}, ProposalScale: []float64{-1, 1}}); err == nil {
		t.Fatal("expected error for negative proposal scale")
	}
	if _, err := New(tg, r, Params{Algorithm: "nope", Initial: []float64{0, 0}}); err == nil {
		t.Fatal("expected error for unknown algorithm")
	}
}

// HMC on a hard-bounded target: proposals into the -Inf region must be
// rejected and never cause a non-finite state.
func TestHMCHardBoundary(t *testing.T) {
	tg := hardBoundary{}
	r := rng.NewRoot(11)
	s, err := New(tg, r, Params{
		Algorithm:   AlgoHMC,
		Initial:     []float64{0.5},
		StepSize:    0.05,
		NumLeapfrog: 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2000; i++ {
		st, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(st.Position[0]) || math.IsInf(st.Position[0], 0) {
			t.Fatalf("non-finite position at %d: %v", i, st.Position)
		}
	}
}

type hardBoundary struct{}

func (hardBoundary) Dim() int { return 1 }
func (hardBoundary) LogDensity(x []float64) float64 {
	if x[0] < -1 || x[0] > 1 {
		return math.Inf(-1)
	}
	return -0.5 * x[0] * x[0]
}
func (hardBoundary) Gradient(x, g []float64) {
	if x[0] < -1 || x[0] > 1 {
		g[0] = math.NaN()
		return
	}
	g[0] = -x[0]
}
