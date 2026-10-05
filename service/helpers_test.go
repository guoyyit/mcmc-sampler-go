package service

import "github.com/guoyyit/mcmc-sampler-go/checkpoint"

func checkpointStore(dir string) (*checkpoint.FileStore, error) {
	return checkpoint.NewFileStore(dir)
}
