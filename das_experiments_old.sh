nlprun -a pyvene -g 1 -r 40G -q jag -o logs/120_150_i8.log 'python das_experiments.py --start_index=120 --end_index=150 --intervention_size 8'
nlprun -a pyvene -g 1 -r 40G -q jag -o logs/150_180_i8.log 'python das_experiments.py --start_index=150 --end_index=180 --intervention_size 8'
nlprun -a pyvene -g 1 -r 40G -q jag -o logs/120_150_i16.log 'python das_experiments.py --start_index=120 --end_index=150 --intervention_size 16'
nlprun -a pyvene -g 1 -r 40G -q jag -o logs/150_180_i16.log 'python das_experiments.py --start_index=150 --end_index=180 --intervention_size 16'
nlprun -a pyvene -g 1 -r 40G -q jag -o logs/60_90_i64.log 'python das_experiments.py --start_index=60 --end_index=90 --intervention_size 64'
nlprun -a pyvene -g 1 -r 40G -q jag -o logs/90_120_i64.log 'python das_experiments.py --start_index=90 --end_index=120 --intervention_size 64'
nlprun -a pyvene -g 1 -r 40G -q jag -o logs/270_300_i64.log 'python das_experiments.py --start_index=270 --end_index=300 --intervention_size 64'