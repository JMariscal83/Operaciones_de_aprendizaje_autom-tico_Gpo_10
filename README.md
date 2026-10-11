# Operaciones_de_aprendizaje_autom-tico_Gpo_10
Operaciones de aprendizaje automático (Gpo 10)

# Corrida 1: Configuración Por Defecto (Baseline)
python train.py

# Corrida 2: Alta Regularización (C menor)
python train.py --C 0.01 --solver lbfgs --max_iter 100 --seed 42

# Corrida 3: Baja Regularización (C mayor)
python train.py --C 10.0 --solver lbfgs --max_iter 200 --seed 42

# Corrida 4: Solver Liblinear
python train.py --C 0.5 --solver liblinear --max_iter 100 --seed 42

# Corrida 5: Solver SAGA con regularización media
python train.py --C 2.0 --solver saga --max_iter 300 --seed 42
