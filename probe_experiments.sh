nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_30_40_50.log 'python probe_experiments.py --coefficients 30_40_50'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_40_40_40.log 'python probe_experiments.py --coefficients 40_40_40'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/probe_50_40_30.log 'python probe_experiments.py --coefficients 50_40_30'