nlprun -a pyvene -g 1 -r 40G -q jag -o logs/0_300_fc1_i2.log 'python das_experiments.py --start_index=0 --end_index=300 --intervention_size 2'
# nlprun -a pyvene -g 1 -r 40G -q jag -o logs/30_60_i8.log 'python das_experiments.py --start_index=30 --end_index=60 --intervention_size 8'
# nlprun -a pyvene -g 1 -r 40G -q jag -o logs/0_30_i16.log 'python das_experiments.py --start_index=0 --end_index=30 --intervention_size 16'
# nlprun -a pyvene -g 1 -r 40G -q jag -o logs/30_60_i16.log 'python das_experiments.py --start_index=30 --end_index=60 --intervention_size 16'
# nlprun -a pyvene -g 1 -r 40G -q jag -o logs/0_30_i32.log 'python das_experiments.py --start_index=0 --end_index=30 --intervention_size 32'
# nlprun -a pyvene -g 1 -r 40G -q jag -o logs/30_60_i32.log 'python das_experiments.py --start_index=30 --end_index=60 --intervention_size 32'
# nlprun -a pyvene -g 1 -r 40G -q jag -o logs/0_30_i64.log 'python das_experiments.py --start_index=0 --end_index=30 --intervention_size 64'
# nlprun -a pyvene -g 1 -r 40G -q jag -o logs/30_60_i64.log 'python das_experiments.py --start_index=30 --end_index=60 --intervention_size 64'