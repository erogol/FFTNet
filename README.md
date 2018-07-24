## Unofficial Implementation of [FFTNet vocode](http://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/fftnet-jin2018.pdf) paper.

- [x] implement the model.
- [x] implement tests.
- [x] overfit on a single batch (sanity check).
- [x] linearize weights for eval time.
- [x] measure the run-time on GPU and CPU. (1 sec audio takes ~47 secs) If anyone knows additional tricks from the paper, let me know. So far I asked the authors but nobody returned. 
- [ ] train on LJSpeech spectrograms.
- [ ] distill model as in Parallel WaveNet paper.
