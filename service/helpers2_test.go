package service

import (
	"github.com/guoyyit/mcmc-sampler-go/adapt"
	"github.com/guoyyit/mcmc-sampler-go/diagnostics"
	"github.com/guoyyit/mcmc-sampler-go/history"
	"github.com/guoyyit/mcmc-sampler-go/sampler"
)

func sampler2D() sampler.Params {
	return sampler.Params{
		Algorithm:   sampler.AlgoHMC,
		Initial:     []float64{1, 1},
		InitJitter:  []float64{1, 3},
		StepSize:    0.05,
		NumLeapfrog: 20,
	}
}

func adaptCfg(warmup int) adapt.Config {
	return adapt.Config{
		Warmup:          warmup,
		TargetAcceptMin: 0.7,
		TargetAcceptMax: 0.9,
		InitStepSize:    0.05,
		AdaptMass:       true,
	}
}

func historyPol() history.Policy { return history.Policy{} }

func thr() diagnostics.Thresholds {
	return diagnostics.Thresholds{MaxRhat: 1.2, MinESS: 1}
}
