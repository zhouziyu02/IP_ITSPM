import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
from pathlib import Path
import gc
from tqdm import tqdm
import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from lib.utils import *
from lib.Dataset_MM import get_PAM_data, get_P12_data, get_P19_data, get_P12_data_zeroshot
from models.ITSPM import ITSPM


eps=1e-7
ITSPM_MODEL_NAMES = ['itspm']


class ITSPMClassifier(nn.Module):
    def __init__(self, dim, cls_dim, dropout=0.1):
        super().__init__()
        hidden = max(dim, 64)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, cls_dim),
        )

    def forward(self, enc_output):
        return self.net(enc_output)

def build_class_weights(dataloader, n_classes, device):
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for _, labels, _ in dataloader:
        labels = labels.detach().cpu().long().view(-1)
        valid = (labels >= 0) & (labels < n_classes)
        if valid.any():
            counts += torch.bincount(labels[valid], minlength=n_classes).float()

    if torch.any(counts == 0):
        return None

    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return weights.to(device)

def classification_selection_metric(opt, auroc, auprc):
    metric = getattr(opt, 'select_metric', 'auprc').lower()
    if metric == 'auroc':
        return auroc
    if metric == 'sum':
        return auroc + auprc
    return auprc

def apply_focal_weight(loss, logits, labels, gamma):
    if gamma <= 0:
        return loss
    probs = torch.softmax(logits, dim=-1)
    pt = probs.gather(1, labels.long().view(-1, 1)).squeeze(1).clamp_min(1e-6)
    return loss * ((1.0 - pt) ** gamma)

def binary_pairwise_rank_loss(logits, labels, margin=0.0):
    labels = labels.long().view(-1)
    if logits.size(-1) < 2:
        return logits.new_tensor(0.0)
    scores = logits[:, 1] - logits[:, 0]
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    return torch.nn.functional.softplus(neg.unsqueeze(0) - pos.unsqueeze(1) + margin).mean()

def train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier, scaler, epoch_i=0):
    """ Epoch operation in training phase. """

    model.train()
    classifier.train()
    losses = []
    sup_preds, sup_labels = [], []
    acc, auroc, auprc = 0,0,0

    training_data_list = list(training_data)
    num_total_batches = len(training_data_list)
    if not num_total_batches:
        raise ValueError('The training loader is empty.')
    num_sampled_batches = max(1, int(num_total_batches * opt.sample_rate))

    sampled_indices = np.random.choice(num_total_batches, size=num_sampled_batches, replace=False)

    sampled_training_data = [training_data_list[i] for i in sampled_indices]

    iter_count = 0
    time_now = time.time()

    for batch_idx, train_batch in enumerate(sampled_training_data):
        """ prepare data """
        train_batch, labels, seq_lens = map(lambda x: x.to(opt.device), train_batch)
        max_len = int(seq_lens.max().item())
        observed_data, observed_mask, observed_tp = \
            train_batch[:, :max_len, :opt.num_types], train_batch[:, :max_len, opt.num_types:2*opt.num_types],\
                  train_batch[:, :max_len, 2*opt.num_types:3*opt.num_types]
        del train_batch

        """ forward """
        optimizer.zero_grad()

        out = model(observed_tp, observed_data, observed_mask, opt) # [B,D]
        sup_pred = classifier(out)

        if sup_pred.dim() == 1:
            sup_pred = sup_pred.unsqueeze(0)

        loss_vec = pred_loss_func((sup_pred), labels)
        loss_vec = apply_focal_weight(loss_vec, sup_pred, labels, opt.focal_gamma)
        loss = torch.sum(loss_vec)
        if opt.rank_loss_weight > 0 and opt.n_classes == 2:
            loss = loss + opt.rank_loss_weight * binary_pairwise_rank_loss(
                sup_pred, labels, opt.rank_loss_margin)
        # sup_pred = torch.softmax(sup_pred, dim=-1)

        if not torch.isfinite(loss):
            raise FloatingPointError('Classification loss is not finite.')

        losses.append(loss.item())
        loss.backward()

        sup_preds.append(sup_pred.detach().cpu().numpy())
        sup_labels.append(labels.detach().cpu().numpy())

        del out, loss, sup_pred, labels

        B, L = observed_mask.size(0), observed_mask.size(1)

        optimizer.step()

        del observed_data, observed_mask, observed_tp
        gc.collect()
        torch.cuda.empty_cache()

        iter_count += 1
        if (batch_idx + 1) % 10 == 0:
            speed = (time.time() - time_now) / iter_count
            left_time = speed * (num_sampled_batches - (batch_idx + 1))
            print(f"\titers: {batch_idx + 1}, epoch: {epoch_i + 1} | loss: {losses[-1]:.7f}")
            print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
            iter_count = 0
            time_now = time.time()


    train_loss = np.average(losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels)
        sup_preds = np.concatenate(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)

        acc, auroc, auprc, _, _, _ = evaluate_mc(sup_labels, sup_preds, opt.n_classes)

    return acc, auroc, auprc, train_loss, num_sampled_batches

def eval_epoch(model, validation_data, pred_loss_func, opt, classifier, save_res=False):
    """ Epoch operation in evaluation phase. """

    model.eval()
    classifier.eval()

    valid_losses = []
    sup_preds = []
    sup_labels = []
    acc, auroc, auprc = 0,0,0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            """ prepare data """
            train_batch, labels, seq_lens = map(lambda x: x.to(opt.device), batch)
            max_len = int(seq_lens.max().item())

            observed_data, observed_mask, observed_tp = \
                train_batch[:, :max_len, :opt.num_types], train_batch[:, :max_len, opt.num_types:2*opt.num_types],\
                        train_batch[:, :max_len, 2*opt.num_types:3*opt.num_types]
            del train_batch

            out = model(observed_tp, observed_data, observed_mask, opt) # [B,L,K,D]
            sup_pred = classifier(out)

            if sup_pred.dim() == 1:
                sup_pred = sup_pred.unsqueeze(0)

            valid_loss_vec = pred_loss_func((sup_pred + eps), labels)
            valid_loss_vec = apply_focal_weight(valid_loss_vec, sup_pred, labels, opt.focal_gamma)
            valid_loss = torch.sum(valid_loss_vec)
            if not torch.isfinite(valid_loss):
                raise FloatingPointError('Classification evaluation loss is not finite.')
            # sup_pred = torch.softmax(sup_pred, dim=-1)

            sup_preds.append(sup_pred.detach().cpu().numpy())
            sup_labels.append(labels.detach().cpu().numpy())

            if valid_loss != 0:
                valid_losses.append(valid_loss.item())

            del out, observed_data, observed_mask, observed_tp, valid_loss

            gc.collect()
            torch.cuda.empty_cache()

    if not sup_preds:
        raise ValueError('The evaluation loader is empty.')
    valid_loss = np.average(valid_losses) if valid_losses else 0.0

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels, axis=0)
        sup_preds = np.concatenate(sup_preds, axis=0)
        sup_preds = np.nan_to_num(sup_preds)

        # save prediction results
        if save_res and opt.save_res is not None and hasattr(opt, 'model'):
            np.save(opt.save_res + '_prediction.npy', sup_preds)

    acc, auroc, auprc, precision, recall, F1 = evaluate_mc(sup_labels, sup_preds, opt.n_classes)
    return acc, auroc, auprc, precision, recall, F1, valid_loss

def run_experiment(model, training_data, validation_data, testing_data, optimizer, scheduler, pred_loss_func, opt, \
                        early_stopping=None, classifier=None, save_path=None):

    epoch = 0
    best_valid_metric = -np.inf
    scaler = None
    is_itspm = opt.model.lower() in ITSPM_MODEL_NAMES
    save_flag = True

    if not opt.test_only:
        """ Start training. """
        for epoch_i in range(opt.epoch):

            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            epoch_start = time.time()
            train_acc, train_auroc, train_auprc, train_loss, num_steps = train_epoch(model, training_data, optimizer, pred_loss_func, opt, classifier, scaler, epoch_i=epoch_i)
            epoch_time = time.time() - epoch_start
            print(f"Epoch: {epoch} cost time: {epoch_time:.2f}s")
            log_info(opt, 'Train', epoch, train_acc, start=epoch_start, auroc=train_auroc, auprc=train_auprc, loss=train_loss, save=save_flag)

            if not opt.retrain:
                start = time.time()
                valid_acc, valid_auroc, valid_auprc, valid_precision, valid_recall, valid_F1, valid_loss = eval_epoch(model, validation_data, pred_loss_func, opt, classifier)
                log_info(opt, 'Valid', epoch, valid_acc, auroc=valid_auroc, auprc=valid_auprc, start=start, precision=valid_precision, recall=valid_recall, F1=valid_F1, loss=valid_loss, save=save_flag)

                start = time.time()
                valid_metric = classification_selection_metric(opt, valid_auroc, valid_auprc)
                if not np.isfinite(valid_metric):
                    raise ValueError('The validation selection metric is undefined; use a split containing each class.')
                if(best_valid_metric < valid_metric):
                    best_valid_metric = valid_metric
                    test_acc, test_auroc, test_auprc, test_precision, test_recall, test_F1, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)
                    log_info(opt, 'Testing', epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, \
                             precision=test_precision, recall=test_recall, F1=test_F1, save=save_flag)

                    best_epoch = epoch
                    best_test_acc = test_acc
                    best_test_auroc = test_auroc
                    best_test_auprc = test_auprc
                    best_test_precision = test_precision
                    best_test_recall = test_recall
                    best_test_F1 = test_F1

                log_info(opt, '* Best Testing *', best_epoch, best_test_acc, start=start, auroc=best_test_auroc, auprc=best_test_auprc,\
                        precision=best_test_precision, recall=best_test_recall, F1=best_test_F1, save=save_flag)

                print(f"Epoch: {epoch}, Steps: {num_steps} "
                      f"| Train Loss: {train_loss:.7f} "
                      f"Vali Loss: {valid_loss:.7f} Vali AUROC: {valid_auroc:.5f} "
                      f"Test AUROC: {best_test_auroc:.5f} Test ACC: {best_test_acc:.5f}")

                if early_stopping is not None:
                    early_stopping(-valid_metric, model, classifier, epoch=epoch)

                    if early_stopping.early_stop: #and not opt.pretrain:
                        print(f"Early stopping triggered. No improvement for {early_stopping.counter} epoch(s).")
                        break
            else:
                start = time.time()
                test_acc, test_auroc, test_auprc, _, _, _, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)

                log_info(opt, 'Testing', epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, save=save_flag)

            scheduler.step()

    if not opt.retrain and not opt.test_only and save_path is not None and os.path.isfile(save_path):
        print("Testing...")
        model, classifier, _, _ = load_checkpoints(save_path, model, classifier=classifier, dp_flag=opt.dp_flag)

        start = time.time()
        test_acc, auroc, auprc, test_precision, test_recall, test_F1, _ = eval_epoch(model, testing_data, pred_loss_func, opt, classifier, save_res=True)

        if early_stopping is not None and early_stopping.best_epoch > 0:
            best_epoch = early_stopping.best_epoch
        else:
            best_epoch = epoch

        log_info(opt, 'Testing', best_epoch, test_acc, start=start, auroc=auroc, auprc=auprc, \
                  precision=test_precision, recall=test_recall, F1=test_F1, save=save_flag)

def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('--state', type=str, default='def')
    parser.add_argument('--model', type=str, default='ITSPM', help='model name (ITSPM)')

    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./checkpoints/classification/')
    parser.add_argument('--data_path', type=str, default=str(Path(__file__).resolve().parent / 'data'))

    parser.add_argument('-n',  type=int, default=12000, help="Size of the dataset")
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_types', type=int, default=23)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log', type=str, default='./logs/classification/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--task', type=str, required=True, choices=['PAM', 'P12', 'P19'])

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fillmiss', action='store_true')
    parser.add_argument('--dp_flag', action='store_true')
    parser.add_argument('--load_in_batch', action='store_true')

    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--n_classes',  type=int, default=2)

    parser.add_argument('--d_model', type=int, default=128, help="ITSPM hidden dimension")
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--sample-tp', type=float, default=1.0)
    parser.add_argument('--collate', type=str, default='indseq')
    parser.add_argument('--balanced_loss', action='store_true')
    parser.add_argument('--focal_gamma', type=float, default=0.0)
    parser.add_argument('--use_static', action='store_true')
    parser.add_argument('--no_auto_class_weights', action='store_true')
    parser.add_argument('--positive_weight', type=float, default=0.0)
    parser.add_argument('--rank_loss_weight', type=float, default=0.0)
    parser.add_argument('--rank_loss_margin', type=float, default=0.0)
    parser.add_argument('--disable_cls_fusion', action='store_true')
    parser.add_argument('--select_metric', type=str, default='auprc',
                        choices=['auprc', 'auroc', 'sum'])
    parser.add_argument('--zero_shot_age', action='store_true')
    parser.add_argument('--zero_shot_ICU', action='store_true')

    # dataset
    parser.add_argument('--split', type=str, default='1')

    # ITSPM arguments
    parser.add_argument('--n_ref_points', type=int, default=32)
    parser.add_argument('--n_scales', type=int, default=3)
    parser.add_argument('--n_mixer_layers', type=int, default=2)
    parser.add_argument('--max_event_tokens', type=int, default=None)
    parser.add_argument('--max_gap_tokens', type=int, default=None)

    opt = parser.parse_args()
    if not 0 < opt.sample_rate <= 1:
        parser.error('--sample_rate must be in (0, 1].')
    if opt.epoch <= 0 or opt.batch_size <= 0:
        parser.error('--epoch and --batch_size must be positive.')
    if opt.test_only and opt.load_path is None:
        parser.error('--test_only requires --load_path.')
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    seed = opt.seed

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # opt.device = torch.device('cpu')

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)
    print(input_command)

    setup_seed(seed)

    """ prepare dataloader """
    if opt.task == 'PAM':
        trainloader, validloader, testloader, opt.num_types, max_len = get_PAM_data(opt, opt.device)
        print("max_len:", max_len)
        if(opt.max_len == -1):
            opt.max_len = max_len
        opt.n_classes = 8
        opt.input_dim = opt.num_types

    elif opt.task == 'P12':
        if opt.zero_shot_age or opt.zero_shot_ICU:
            trainloader, validloader, testloader, opt.num_types, max_len = get_P12_data_zeroshot(opt, opt.device)
        else:
            trainloader, validloader, testloader, opt.num_types, max_len = get_P12_data(opt, opt.device)
        if(opt.max_len == -1):
            opt.max_len = max_len
        opt.n_classes = 2
        opt.input_dim = opt.num_types

    elif opt.task == 'P19':
        trainloader, validloader, testloader, opt.num_types, max_len = get_P19_data(opt, opt.device)
        if(opt.max_len == -1):
            opt.max_len = max_len
        opt.n_classes = 2
        opt.input_dim = opt.num_types

    opt.log = opt.root_path + opt.log

    if opt.save_path is not None:
        opt.save_path = opt.root_path + opt.save_path

    if opt.load_path is not None:
        opt.load_path = opt.root_path + opt.load_path

    """ prepare model """
    if opt.model.lower() in ['itspm']:
        model = ITSPM(opt)
    else:
        raise ValueError(f"Unsupported model '{opt.model}'. This cleaned project keeps ITSPM only.")

    print("! The backbone model is:", opt.model)

    para_list = list(model.parameters())

    if opt.model.lower() in ITSPM_MODEL_NAMES:
        mort_classifier = ITSPMClassifier(opt.d_model, opt.n_classes, dropout=opt.dropout)

    para_list += list(mort_classifier.parameters())

    # load model
    if opt.load_path is not None:
        print("Loading checkpoints...")
        model, mort_classifier, _, _ = load_checkpoints(opt.load_path, model, classifier=mort_classifier, dp_flag=False)


    model = model.to(opt.device)

    for mod in [model, mort_classifier]:
        if mod is not None:
            mod = mod.to(opt.device)

    if opt.dp_flag:
        model = nn.DataParallel(model)

    if opt.debug:
        opt.state='debug'
        exp_desc = f"{opt.task}_{opt.model}_{opt.state}"
    else:
        exp_desc = (
            f"{opt.task}_{opt.model}_{opt.state}_d{opt.d_model}_"
            f"ref{opt.n_ref_points}_scale{opt.n_scales}_lr{opt.lr}"
        )

    opt.log = f"{opt.log}{exp_desc}.log"
    print("! Log path:", opt.log)

    if opt.save_path is not None:
        opt.save_res = opt.save_path + exp_desc
        os.makedirs(opt.save_path, exist_ok=True)
        save_path = opt.save_path + exp_desc + '.h5'
    else:
        save_path = None

    """ optimizer and scheduler """

    params = (para_list)
    optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function """
    class_weights = None
    if opt.positive_weight > 0 and opt.n_classes == 2:
        class_weights = torch.tensor([1.0, opt.positive_weight], device=opt.device)
        print("[Info] Class weights:", class_weights.detach().cpu().numpy())
    elif opt.balanced_loss or (opt.model.lower() in ITSPM_MODEL_NAMES and not opt.no_auto_class_weights):
        class_weights = build_class_weights(trainloader, opt.n_classes, opt.device)
        if class_weights is not None:
            print("[Info] Class weights:", class_weights.detach().cpu().numpy())

    pred_loss_func = nn.CrossEntropyLoss(
        ignore_index=-1,
        reduction='none',
        weight=class_weights,
    ).to(opt.device)


    # setup the log file
    is_itspm = opt.model.lower() in ITSPM_MODEL_NAMES
    if not is_itspm:
        with open(opt.log, 'a') as f:
            f.write('[Info] parameters: {}\n'.format(opt))

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    print('[Info] parameters: {}'.format(opt))

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, save_path=save_path, dp_flag=opt.dp_flag)

    """ train the model """
    start_time = time.time()
    run_experiment(model, trainloader, validloader, testloader, optimizer, scheduler, pred_loss_func, opt, early_stopping, mort_classifier, save_path=save_path)
    training_duration = time.time() - start_time

    # Evaluate the best saved checkpoint
    if not opt.retrain and not opt.test_only:
        if save_path is not None and os.path.exists(save_path):
            model, mort_classifier, _, _ = load_checkpoints(save_path, model, classifier=mort_classifier, dp_flag=opt.dp_flag)
        elif early_stopping is not None and getattr(early_stopping, 'best_model_state', None) is not None:
            model.load_state_dict(early_stopping.best_model_state)
            if mort_classifier is not None and early_stopping.best_classifier_state is not None:
                mort_classifier.load_state_dict(early_stopping.best_classifier_state)

    test_acc, test_auroc, test_auprc, test_precision, test_recall, test_F1, _ = eval_epoch(
        model, testloader, pred_loss_func, opt, mort_classifier
    )

    best_epoch = early_stopping.best_epoch if (early_stopping is not None and early_stopping.best_epoch > 0) else opt.epoch

    # Construct hyperparameter string
    hparams = []
    if opt.model.lower() in ITSPM_MODEL_NAMES:
        token_hparams = ""
        if opt.max_event_tokens is not None or opt.max_gap_tokens is not None:
            token_hparams = f", max_event_tokens: {opt.max_event_tokens}, max_gap_tokens: {opt.max_gap_tokens}"
        hparams.append(f"d_model: {opt.d_model}, n_ref_points: {opt.n_ref_points}, n_scales: {opt.n_scales}, n_mixer_layers: {opt.n_mixer_layers}{token_hparams}")

    hparams_str = ", ".join(hparams)

    # Construct metrics string
    if opt.task in ['P12', 'P19']:
        metrics_str = f"AUROC: {test_auroc:.5f}, AUPRC: {test_auprc:.5f}"
    elif opt.task == 'PAM':
        metrics_str = f"Accuracy: {test_acc:.5f}, Precision: {test_precision:.5f}, Recall: {test_recall:.5f}, F1 score: {test_F1:.5f}"
    else:
        metrics_str = f"Accuracy: {test_acc:.5f}, AUROC: {test_auroc:.5f}, AUPRC: {test_auprc:.5f}"

    import datetime
    from datetime import timezone, timedelta
    tz_utc_8 = timezone(timedelta(hours=8))
    time_now_str = datetime.datetime.now(tz_utc_8).strftime("%Y-%m-%d %H:%M:%S")

    minutes = int(training_duration // 60)
    seconds = int(training_duration % 60)
    training_time_str = f"{minutes}m {seconds}s"

    results_path = os.environ.get("ITSPM_CLASSIFICATION_RESULTS_FILE", "Classification/results/Classification_results.txt")
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_path, "a") as res_f:
        res_f.write(f"Model: {opt.model}, Task: Classification, Dataset: {opt.task}, Best Epoch: {best_epoch}, Seed: {opt.seed}, lr: {opt.lr}, batch_size: {opt.batch_size}, {hparams_str}\n")
        res_f.write(f"{metrics_str}\n")
        res_f.write(f"Time now: {time_now_str}, Time for training: {training_time_str}\n\n")

if __name__ == '__main__':
    main()
