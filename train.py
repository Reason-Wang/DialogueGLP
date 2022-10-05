from utils.util import AverageMeter, get_logger, seed_everything, get_optimizer_params
import time
import torch
from tqdm import tqdm
import numpy as np
import sys
from utils.options import Options
from sklearn import metrics
from data.load_data import load_data
from data.dataset import dataset_map, collator_map
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model.base_models import model_class_map
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.nn as nn
import gc
import os

def validate_fn(model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    # tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    start = end = time.time()

    with torch.no_grad():
        # for idx, (inputs, labels) in enumerate(tbar):
        for step, (inputs, labels) in enumerate(val_loader):
            # inputs, labels = read_data(data)
            # for k, v in inputs.items():
            #     inputs[k] = v.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            pred = model(inputs)
            # print(pred.shape)
            preds.append(pred.detach().cpu().numpy())
            loss = criterion(pred, labels)
            losses.update(loss.item(), batch_size)
            end = time.time()
    predictions = np.concatenate(preds, axis=0)
    return losses.avg, predictions


def train_fn(model, train_loader, val_loader, val_ds, criterion, optimizer, epoch, scheduler, device):
    model.train()
    tbar = tqdm(train_loader, file=sys.stdout)
    losses = AverageMeter()
    global_step = 0
    for step, (inputs, labels) in enumerate(tbar):
        batch_size = labels.size(0)
        pred = model(inputs)
        loss = criterion(pred, labels)
        if opt.gradient_accumulation_steps > 1:
            loss = loss / opt.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        loss.backward()

        del inputs, labels
        torch.cuda.empty_cache()
        if opt.gradient_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        else:
            grad_norm = 0
        if (step + 1) % opt.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if opt.scheduler != 'fixed':
                scheduler.step()

        tbar.set_description(
            f"Epoch {epoch + 1} Loss: {losses.avg:.4f} lr: {scheduler.get_last_lr()[0]:.8f} grad_norm: {grad_norm:.2f}")
        # tbar.set_description(f"Epoch {epoch+1} Loss: {losses.avg:.4f} lr: {CFG.lr:.8f} grad_norm: {grad_norm:.2f}")

    return losses.avg


def train_loop(train_ds, val_ds, opt):
    LOGGER.info(f"========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model_class_map[opt.model](opt)
    model.to(device)

    collator = collator_map[opt.model](opt, device)

    val_loader = DataLoader(val_ds,
                            batch_size=opt.batch_size * 2,
                            shuffle=False,
                            collate_fn=collator,
                            num_workers=opt.num_workers,
                            pin_memory=False,
                            drop_last=False)

    train_loader = DataLoader(train_ds,
                              batch_size=opt.batch_size,
                              shuffle=not opt.no_shuffle_train,
                              collate_fn=collator,
                              num_workers=opt.num_workers,
                              pin_memory=False,
                              drop_last=True)

    # ====================================================
    # model & optimizer
    # ====================================================



    optimizer_parameters = get_optimizer_params(model, opt)
    optimizer = AdamW(optimizer_parameters)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear' or 'fixed':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles
            )
        return scheduler


    num_train_steps = int(opt.epochs * len(train_ds) / (opt.batch_size * opt.gradient_accumulation_steps))
    scheduler = get_scheduler(opt, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.CrossEntropyLoss(weight=loss_weights)
    criterion = nn.CrossEntropyLoss()
    best_score = 0.

    for epoch in range(opt.epochs):
        if opt.model == 'BaseModel':
            if epoch == 0 and opt.freeze_epochs > 0:
                model.freeze_plm()
            elif opt.freeze_epochs > 0 and epoch == opt.freeze_epochs:
                model.unfreeze_plm()

        start_time = time.time()
        # train
        avg_loss = train_fn(model, train_loader, val_loader, val_ds, criterion, optimizer, epoch, scheduler, device)
        # avg_loss = train_fn(model, train_loader, val_loader, val_ds, criterion, optimizer, epoch, device)
        # eval
        avg_val_loss, predictions = validate_fn(model, val_loader, criterion, device)

        # scoring
        score = get_score(val_ds, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), OUTPUT_DIR + f"{opt.model}.pth")

    torch.cuda.empty_cache()
    gc.collect()

    return val_ds


def get_score(val_ds, preds):
    gold = []
    for i in range(len(val_ds)):
        gold_label = val_ds[i]['label']
        gold.append(gold_label)
    preds = np.argmax(preds, axis=1)
    return metrics.f1_score(gold, preds, average=opt.metric)


def get_final_score(ds, opt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    collator = collator_map[opt.model](opt, device)
    data_loader = DataLoader(
        ds,
        batch_size=opt.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=opt.num_workers,
        pin_memory=False,
        drop_last=False)

    criterion = nn.CrossEntropyLoss()
    model_state = torch.load(OUTPUT_DIR+f"{opt.model}.pth")
    model = model_class_map[opt.model](opt)
    model.to(device)
    model.load_state_dict(model_state)
    avg_val_loss, predictions = validate_fn(model, data_loader, criterion, device)
    score = (get_score(ds, predictions))

    LOGGER.info(f'metric: {opt.metric}, score: {score}')
    return score

if __name__ == '__main__':
    options = Options()
    opt = options.parse()[0]
    if opt.model == 'BaseModel':
        options.add_basemodel_options()
    elif opt.model == 'DialogueInfer':
        options.add_dialogue_infer_options()
    elif opt.model == 'DialogueRNN':
        options.add_dialogue_rnn_options()
    elif opt.model == 'DialogueGCN':
        options.add_dialogue_gcn_options()
    opt = options.parse()[0]
    if opt.cls_3:
        opt.target_size = 3

    OUTPUT_DIR = 'data/ckpts/' + opt.name + '/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    LOGGER = get_logger(OUTPUT_DIR)
    seed_everything(opt.seed)

    train_data = load_data(opt.dataset, 'train', opt.feature_metric, opt.knowledge, opt.cls_3)
    dev_data = load_data(opt.dataset, 'val', opt.feature_metric, opt.knowledge, opt.cls_3)
    test_data = load_data(opt.dataset, 'test', opt.feature_metric, opt.knowledge, opt.cls_3)
    LOGGER.info(f"Loaded {opt.dataset} with {opt.feature_metric} feature.")

    dataset_class = dataset_map[opt.model]
    train_ds = dataset_class(train_data, opt)
    dev_ds = dataset_class(dev_data, opt)
    test_ds = dataset_class(test_data, opt)
    LOGGER.info(train_ds[0])

    train_loop(train_ds, dev_ds, opt)
    dev_score = get_final_score(dev_ds, opt)
    test_score = get_final_score(test_ds, opt)

    paras_str = options.get_options(opt)
    paras_str = paras_str + '\n' + f'dev: {dev_score} \ntest: {test_score}'
    LOGGER.info(paras_str)
    with open(OUTPUT_DIR + '/result.txt', "w") as file:
        file.write(paras_str)