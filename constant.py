CHECKPOINT_DIR = "_experiments"
DISEASE_CATEGORIES = ["obesity", "diabetes", "hypertriglyceridemia", "dyslipidemia", "liver dysfunction", "hypertension"]
TRAIN_RATIO = 0.8

#Hyperparameters of NF
RATIO_CAT = 1.0
RATIO_FC = 1.0
FLOW_STEPS = 2 

#Training parameters
LR = 5e-3
WEIGHT_DECAY = 1e-5
ALTUB_EPOCH = 5
ALTUB_LR = 5
SCHEDULER_T_0 = 20
MAX_EPOCH = 200

EVAL_EPOCH = 1
SAVE_EPOCH = 25