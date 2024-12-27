# nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_30_40_50.log 'python probe_experiments.py --coefficients 30_40_50'
# nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_40_40_40.log 'python probe_experiments.py --coefficients 40_40_40'
# nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_50_40_30.log 'python probe_experiments.py --coefficients 50_40_30'
# nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_36_40_44.log 'python probe_experiments.py --coefficients 36_40_44'
# nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_44_40_36.log 'python probe_experiments.py --coefficients 44_40_36'
# untrained models
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_00_00_00_untrained.log 'python probe_experiments.py --coefficients 0_0_0 --untrained_models'