# 30 40 50
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_30_40_50_i4_untrained.log 'python das_experiments.py --coefficients 30_40_50 --intervention_size 4 --untrained_models'
# 40 40 40
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_40_40_40_i4_untrained.log 'python das_experiments.py --coefficients 40_40_40 --intervention_size 4 --untrained_models'
# 50 40 30
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_50_40_30_i4_untrained.log 'python das_experiments.py --coefficients 50_40_30 --intervention_size 4 --untrained_models'
# 36 40 44
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_36_40_44_i4_untrained.log 'python das_experiments.py --coefficients 36_40_44 --intervention_size 4 --untrained_models'
# 44 40 36
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_44_40_36_i4_untrained.log 'python das_experiments.py --coefficients 44_40_36 --intervention_size 4 --untrained_models'