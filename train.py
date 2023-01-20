import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import torch

from torch.utils.tensorboard import SummaryWriter
from ignite.utils import setup_logger

from utils import CustomCosineAnnealingWarmUpRestarts
import constant as const

import os
import itertools
import random
import math

from model_flow import CD_Flow

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(model, optimizer, scheduler, epoch, checkpoint_dir):
    torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            },
            os.path.join(checkpoint_dir, "%d.pt" % epoch),
    )

#Custom scheduler
def build_scheduler(optimizer):
    return CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=const.SCHEDULER_T_0, eta_max = const.LR, T_up=0, gamma=1, T_mult=1)

#Adam optimizer, and SGD for AltUB
def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr = 0, weight_decay=const.WEIGHT_DECAY), torch.optim.SGD(model.base.parameters(),lr=0)

def build_model(dim_features, flow_steps, cond_dims=2):
    return CD_Flow(
        dim_features = dim_features, 
        flow_steps = flow_steps,
        cond_dims = cond_dims
    ).to('cuda')

def load_data(args): #Loading data and split them into x, c (condition), and y
    df = pd.read_csv(args.data)

    init_seeds(args.seed)
    category = args.category

    #Biomarkers for each disease
    biomarkers = {
        "obesity": ['bmi', 'height', 'weight', 'waist'], #Waist was strongly correlated with height and weight
        "diabetes": ['blood_sugar'],
        "hypertriglyceridemia": ['neutral_fat'],
        "dyslipidemia": ['hdl', 'ldl'],
        "liver dysfunction": ['got', 'gpt'],
        "hypertension": ['max_bp', 'min_bp']
    }

    df = df.dropna(subset = biomarkers[category]) #Dropping people who have missing values on these biomarkers
    df = df.drop(df.columns[:63], axis=1) #Dropping additional unnecessary features

    #Criteria of diseases in Table 2
    criteria = {
        "obesity": df['bmi']>=25,
        "diabetes": df['blood_sugar']>=126,
        "hypertriglyceridemia": df['neutral_fat']>=np.log1p(200), 
        "dyslipidemia": (df['hdl']<40)|(df['ldl']>=160),
        "liver dysfunction": (df['got']>np.log1p(40))|(df['gpt']>np.log1p(40)),
        "hypertension": (df['max_bp']>=130)|(df['min_bp']>=80),
    }

    #Conditions: age and gender
    con_vec = ["age", "gender"]

    #Split of controls and cases
    diseases = np.where(criteria[category], 1, 0)
    normal = df[diseases == 0].copy()
    diseases = df[diseases == 1].copy()

    #Split of train/test
    #Controls: 80% train, 20% test
    #Cases will be processed later
    x_diseases = diseases.drop(columns=biomarkers[category], axis=1)
    y_diseases = pd.DataFrame(np.ones(len(x_diseases)), columns=['diseases'], index=x_diseases.index)
    x_normal = normal.drop(columns=biomarkers[category], axis=1)
    y_normal = pd.DataFrame(np.zeros(len(x_normal)), columns=['diseases'], index=x_normal.index)

    x_train, x_test, y_train, y_test = train_test_split(x_normal, y_normal, test_size=0.2, stratify=normal['gender'])
    
    x_test = pd.concat([x_test, x_diseases])
    y_test = pd.concat([y_test, y_diseases])
    
    #Age and gender are stored in c_train/c_test
    c_train = torch.FloatTensor(x_train[con_vec].values).to('cuda')
    c_test = torch.FloatTensor(x_test[con_vec].values).to('cuda')

    #Removing age and gender from x_train, x_test
    x_train.drop(con_vec, axis=1, inplace=True)
    x_test.drop(con_vec, axis=1, inplace=True)

    return x_train, x_test, c_train, c_test, y_train, y_test

def preprocessing(x_train, x_test): #preprocissing steps

    imputer = IterativeImputer()
    tmp = x_train.copy()
    tmp = imputer.fit_transform(tmp)
    x_train[:] = tmp

    tmp = x_test.copy()
    tmp = imputer.transform(tmp)
    x_test[:] = tmp

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = torch.FloatTensor(x_train).to('cuda')
    x_test = torch.FloatTensor(x_test).to('cuda')

    return x_train, x_test


def train(args): #Whole training process
    #Loading data and split them
    x_train, x_test, c_train, c_test, y_train, y_test = load_data(args)

    #Preprocssing of x
    x_train, x_test = preprocessing(x_train, x_test)

    #Our model
    model = build_model(dim_features=x_train.shape[1], flow_steps = const.FLOW_STEPS, cond_dims = 2)

    #Experimental results will be stored here. (ex: ~/CHECKPOINT_DIR/exp3)
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    #This code will help you to record performances (Option)
    tb_logger = SummaryWriter(log_dir=checkpoint_dir)
    eval_logger = setup_logger(name='eval', filepath=os.path.join(checkpoint_dir, 'eval.txt'))

    optimizer, optimizer_altub = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    #If you don't use AltUB, parameters of base distribution shouldn't be trained.
    if args.use_altub is False:
        for param in model.base.parameters():
            param.requires_grad = False

    #Training process
    for epoch in range(const.MAX_EPOCH):
        train_one_epoch(args, model, x_train, c_train, epoch, optimizer, optimizer_altub, scheduler, tb_logger)
        if (epoch+1) % const.EVAL_EPOCH == 0:
            eval_one_epoch(args, model, x_test, y_test, c_test, epoch, tb_logger, eval_logger)
        if (epoch+1) % const.SAVE_EPOCH == 0:
            save_model(model, optimizer, scheduler, epoch, checkpoint_dir)

def train_one_epoch(args, model, x_train, c_train, epoch, optimizer, optimizer_altub, scheduler, tb_logger): 
    #Function for training of one epoch
    #Note that we didn't utilize a dataloader, because the training process with whole data required under VRAM 3GB.
    #Yet data were shuffled in train_test_split function.

    #General training code
    model.train()
    ret = model(x_train, c_train)
    loss = ret["loss"]
    model.zero_grad()
    loss.backward()

    #If you use AltUB, this code will update the parameters of base distribution periodically.
    #Otherwise, models are trained with only Adam optimizer.
    if args.use_altub and (epoch+1)%const.ALTUB_EPOCH == 0 and (epoch+1)>const.SCHEDULER_T_0:
        lr = scheduler.get_lr()[0]
        for param_group in optimizer_altub.param_groups:
            torch.nn.utils.clip_grad_value_(model.base.base_mean, 100)
            torch.nn.utils.clip_grad_value_(model.base.base_cov, 100)
            param_group['lr'] = lr * const.ALTUB_LR
        optimizer_altub.step()
    else:
        optimizer.step()
    scheduler.step()

    #Saving loss
    tb_logger.add_scalar('loss', torch.mean(loss), epoch)

#Function for evaluation
def eval_one_epoch(args, model, x_test, y_test, c_test, epoch, tb_logger, eval_logger):

    #Outputs from test dataset
    model.eval()
    with torch.no_grad():
        pred = model(x_test, c_test)
    score = np.array(pred['score'].detach().cpu())

    #Assessment of performance
    auroc, auprc = performance(y_test, score, args.eval_all)

    #Saving AUROC and AUPRC
    eval_logger.info(f'[epoch {epoch}] [AUROC {auroc}] [AUPRC {auprc}]')
    tb_logger.add_scalar('AUROC', auroc, epoch)
    tb_logger.add_scalar('AUPRC', auprc, epoch)

#Function for assessment of performance
#eval_all = True: Test for every case that conserves a base rate.
#eval_all = False: Test for a few case so that most of positive data are evalauted.


def performance(y_test, score, eval_all = False):
    #Filtering positive samples
    score = pd.DataFrame(score, index=y_test.index)
    positive_index = y_test[y_test['diseases']==1].index

    #Number of positive samples should be dropped for making same base rate with original dataset.
    #This process is necessary for fair comparison with LightGBM.
    #Without the process, the ratio of positive samples in test data are 1/(1-train_ratio) higher than that of original dataset.

    #Pseudo-data for understanding: train_ratio = 0.8, number of controls: 91, Number of cases = 9. (Base rate 9%)
    #Training data: 72 controls
    #Test data before these codes: 19 controls, 9 cases (Base rate 32%!!)   
    
    #(ex) num_drop: 7 
    #(ex) num_p_eval: 2
    #Therefore, test data will be composed of 19 controls, 2 cases (Base rate: 9.5% => Similar as the original dataset)
    num_drop = int(round(len(positive_index) * const.TRAIN_RATIO))
    num_p_eval = len(positive_index) - num_drop

    auroc_list = []
    auprc_list = []

    if eval_all is True:
        #Tests every possible combination of (19 controls, 2 cases) by sampling of 2 cases among 9 cases.
        #Highly time-consuming, not generally recommended 
        for drop_index in itertools.combinations(positive_index, num_drop): #every case
            drop_index = list(drop_index)
            score_eval = score.drop(drop_index)
            y_test_eval = y_test.drop(drop_index)
            auroc_list.append(roc_auc_score(y_test_eval, score_eval))
            auprc_list.append(average_precision_score(y_test_eval, score_eval))
    else:
        if num_p_eval > 1: #General case
            #9 cases are splitted into 2/2/2/2/1. 
            #Then four combinations using the sets with two positive samples are tested.
            eval_list = random.sample(list(positive_index), (len(positive_index)//num_p_eval) * num_p_eval)
            eval_list = [eval_list[i:i+num_p_eval] for i in range(0, len(eval_list), num_p_eval)]
        else: #Rare case
            eval_list = [list(positive_index)]

        for eval_index in eval_list:
            drop_index = list(set(positive_index)-set(eval_index))
            score_eval = score.drop(drop_index)
            y_test_eval = y_test.drop(drop_index)

            auroc_list.append(roc_auc_score(y_test_eval, score_eval))
            auprc_list.append(average_precision_score(y_test_eval, score_eval))
        
    #Average of AUROC and AUPRC are recorded.
    return sum(auroc_list)/len(auroc_list), sum(auprc_list)/len(auprc_list)