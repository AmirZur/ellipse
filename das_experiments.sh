# 30 40 50
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_30_40_50_i4.log 'python das_experiments.py --coefficients 30_40_50 --intervention_size 4'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_30_40_50_i8.log 'python das_experiments.py --coefficients 30_40_50 --intervention_size 8'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_30_40_50_i16.log 'python das_experiments.py --coefficients 30_40_50 --intervention_size 16'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_30_40_50_i32.log 'python das_experiments.py --coefficients 30_40_50 --intervention_size 32'
# 40 40 40
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_40_40_40_i4.log 'python das_experiments.py --coefficients 40_40_40 --intervention_size 4'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_40_40_40_i8.log 'python das_experiments.py --coefficients 40_40_40 --intervention_size 8'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_40_40_40_i16.log 'python das_experiments.py --coefficients 40_40_40 --intervention_size 16'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_40_40_40_i32.log 'python das_experiments.py --coefficients 40_40_40 --intervention_size 32'
# 50 40 30
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_50_40_30_i4.log 'python das_experiments.py --coefficients 50_40_30 --intervention_size 4'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_50_40_30_i8.log 'python das_experiments.py --coefficients 50_40_30 --intervention_size 8'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_50_40_30_i16.log 'python das_experiments.py --coefficients 50_40_30 --intervention_size 16'
nlprun -a ellipse -g 1 -r 40G -q jag -o logs/das_50_40_30_i32.log 'python das_experiments.py --coefficients 50_40_30 --intervention_size 32'