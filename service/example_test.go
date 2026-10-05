package service_test

import (
	"context"
	"fmt"

	"github.com/guoyyit/mcmc-sampler-go/adapt"
	"github.com/guoyyit/mcmc-sampler-go/diagnostics"
	"github.com/guoyyit/mcmc-sampler-go/history"
	"github.com/guoyyit/mcmc-sampler-go/sampler"
	"github.com/guoyyit/mcmc-sampler-go/service"
)

type unit2 struct{}

func (unit2) Dim() int { return 2 }
func (unit2) LogDensity(x []float64) float64 {
	return -0.5 * (x[0]*x[0] + x[1]*x[1])
}
func (unit2) Gradient(x, g []float64) { g[0], g[1] = -x[0], -x[1] }

func ExampleRun() {
	cfg := service.Config{
		JobID:     "doc-example",
		NumChains: 4,
		Draws:     1200,
		Warmup:    400,
		Seed:      42,
		Sampler: sampler.Params{
			Algorithm:   sampler.AlgoHMC,
			Initial:     []float64{0, 0},
			InitJitter:  []float64{1, 1},
			StepSize:    0.1,
			NumLeapfrog: 20,
		},
		Adapt: adapt.Config{
			Warmup:          400,
			TargetAcceptMin: 0.7,
			TargetAcceptMax: 0.9,
			AdaptMass:       true,
		},
		History:    history.Policy{},
		Thresholds: diagnostics.Thresholds{MaxRhat: 1.1, MinESS: 10},
	}
	result, err := service.Run(context.Background(), cfg, unit2{})
	if err != nil {
		fmt.Println("not converged:", err)
		return
	}
	fmt.Println("converged:", result.Converged)
	fmt.Println("chains:", len(result.Draws))
	// Output:
	// converged: true
	// chains: 4
}
