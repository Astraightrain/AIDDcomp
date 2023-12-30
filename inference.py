import torch
import os

from gps.featurizer import smiles2graph
from gps.transforms import RandomWalkGenerator
from gps.model import GPSEncoder
from gps.taskhead import GraphHead

from torch_geometric.data.batch import Batch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import pandas as pd

def parsing_argument():
    parser = argparse.ArgumentParser(description="GPS multitask model inference")
    parser.add_argument("--dataset", type=str, default="./dataset/test.csv")
    parser.add_argument("--output", type=str, default="inference_output.csv")

    return parser.parse_args()


def load_checkpoint(encoder, task_head, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_state_dict = {}
    task_head_dict = {}
    task = None
    if 'mlm' in checkpoint_path:
        task = 'MLM'
    elif 'hlm' in checkpoint_path:
        task = 'HLM'
    for key in checkpoint['state_dict'].keys():
        if 'encoder.' in key:
            encoder_state_dict.update({'.'.join(key.split('.')[1:]): checkpoint['state_dict'][key]})
        else:
            task_head_dict.update({'.'.join(key.split('.')[2:]):checkpoint['state_dict'][key]})
            
    encoder.eval()
    taskhead.eval()
    encoder.load_state_dict(encoder_state_dict)
    task_head.load_state_dict(task_head_dict, strict=True)
    return encoder, task_head, task

if __name__=="__main__":
    args = parsing_argument()
    df = pd.read_csv(args.dataset)

    rw = RandomWalkGenerator()
    graphs = [rw(smiles2graph(smi)) for smi in df['SMILES']]
    
    encoder = GPSEncoder()
    taskhead = GraphHead()

    checkpoint_5 = [
        #hlm
        './saved_model/ensemble5_hlm_1.ckpt',
        './saved_model/ensemble5_hlm_2.ckpt',
        './saved_model/ensemble5_hlm_3.ckpt',
        './saved_model/ensemble5_hlm_4.ckpt',
        './saved_model/ensemble5_hlm_5.ckpt',

        #mlm
        './saved_model/ensemble5_mlm_1.ckpt',
        './saved_model/ensemble5_mlm_2.ckpt',
        './saved_model/ensemble5_mlm_3.ckpt',
        './saved_model/ensemble5_mlm_4.ckpt',
        './saved_model/ensemble5_mlm_5.ckpt',
    ]

    mlms_5 = []
    hlms_5 = []
    for check in tqdm(checkpoint_5, desc = 'Ensemble 5 models'):
        e, t, task = load_checkpoint(encoder, taskhead, check)
        with torch.no_grad():
            data = Batch.from_data_list([data for data in graphs], [],[])
            x, batch = e(data)
            
            inferences = [tout.item()*100 for tout in t(x, batch)]
        
        if task=='MLM':
            mlms_5.append(inferences)
            
        elif task=='HLM':
            hlms_5.append(inferences)
    
    ensemble_mlms_5 = pd.DataFrame(mlms_5).mean().values
    ensemble_hlms_5 = pd.DataFrame(hlms_5).mean().values
    
    df['HLM_5'] = ensemble_hlms_5
    df['MLM_5'] = ensemble_mlms_5
    
    checkpoint_10 = [
        './saved_model/ensemble10_hlm_1.ckpt',
        './saved_model/ensemble10_hlm_2.ckpt',
        './saved_model/ensemble10_hlm_3.ckpt',
        './saved_model/ensemble10_hlm_4.ckpt',
        './saved_model/ensemble10_hlm_5.ckpt',
        './saved_model/ensemble10_hlm_6.ckpt',
        './saved_model/ensemble10_hlm_7.ckpt',
        './saved_model/ensemble10_hlm_8.ckpt',
        './saved_model/ensemble10_hlm_9.ckpt',
        './saved_model/ensemble10_hlm_10.ckpt',

        './saved_model/ensemble10_mlm_1.ckpt',
        './saved_model/ensemble10_mlm_2.ckpt',
        './saved_model/ensemble10_mlm_3.ckpt',
        './saved_model/ensemble10_mlm_4.ckpt',
        './saved_model/ensemble10_mlm_5.ckpt',
        './saved_model/ensemble10_mlm_6.ckpt',
        './saved_model/ensemble10_mlm_7.ckpt',
        './saved_model/ensemble10_mlm_8.ckpt',
        './saved_model/ensemble10_mlm_9.ckpt',
        './saved_model/ensemble10_mlm_10.ckpt',   
    ]

    mlms_10 = []
    hlms_10 = []
    for check in tqdm(checkpoint_10, desc = 'Ensemble 10 models'):
        e, t, task = load_checkpoint(encoder, taskhead, check)
        with torch.no_grad():
            data = Batch.from_data_list([data for data in graphs], [],[])
            x, batch = e(data)
            
            inferences = [tout.item()*100 for tout in t(x, batch)]
        
        if task=='MLM':
            mlms_10.append(inferences)
            
        elif task=='HLM':
            hlms_10.append(inferences)

    ensemble_mlms_10 = pd.DataFrame(mlms_10).mean().values
    ensemble_hlms_10 = pd.DataFrame(hlms_10).mean().values
    
    df['HLM_10'] = ensemble_hlms_10
    df['MLM_10'] = ensemble_mlms_10


    regression_tasks = [
    './saved_model/external_human.ckpt',
    './saved_model/external_mouse.ckpt',
    './saved_model/ncats_human.ckpt',
    './saved_model/ncats_rat.ckpt',
    
]

    classification_tasks = [
    
        './saved_model/shanghai_human.ckpt',
        './saved_model/shanghai_market.ckpt',
        './saved_model/shanghai_rat.ckpt',
        './saved_model/usarmy_human.ckpt',
    ]
    other_task_results = []
    
    for check in tqdm(regression_tasks, desc = 'External regression tasks'):
        e, t, task = load_checkpoint(encoder, taskhead, check)
        with torch.no_grad():
            data = Batch.from_data_list([data for data in graphs], [],[])
            x, batch = e(data)
            
            inferences = [tout.item()*100 for tout in t(x, batch)]
        
        other_task_results.append(inferences)

    for check in tqdm(classification_tasks, desc = 'External classification tasks'):
        e, t, task = load_checkpoint(encoder, taskhead, check)
        with torch.no_grad():
            data = Batch.from_data_list([data for data in graphs], [],[])
            x, batch = e(data)
            
            inferences = [F.sigmoid(tout).item() for tout in t(x, batch)]
        
        other_task_results.append(inferences)
    cols = ['external_hlm','external_mlm','ncats_hlm','ncats_rlm','shanghai2_hlm','shanghai2_market','shanghai2_rlm','usarmy_hlm']
    res = pd.DataFrame(other_task_results).T
    res.columns = cols
    df = pd.concat([df, res], axis = 1)
    
    try:
        os.makedirs('./results/')
    except:
        pass
    if '.csv' in args.output:
        df.to_csv(f'./results/{args.output}', index = False)
        print(f'save inference output at ./results/{args.output}')
    else:
        df.to_csv(f'./results/{args.output}.csv', index = False)
        print(f'save inference output at ./results/{args.output}.csv')