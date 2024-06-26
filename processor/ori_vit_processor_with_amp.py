import logging
import os
import time
import torch
import torch.nn as nn
from loss.domain_SCT_loss import domain_SCT_loss
from loss.triplet_loss import TripletLoss, euclidean_dist, hard_example_mining
from loss.triplet_loss_for_mixup import hard_example_mining_for_mixup
from model.make_model import make_model
from processor.inf_processor import do_inference, do_inference_multi_targets
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def ori_vit_do_train_with_amp(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             patch_centers = None,
             pc_criterion= None,
             train_dir=None,
             num_pids=None,
             memories=None,
             sour_centers=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid.train")
    logger.info('start training')
    log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_id_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
    loss_center_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512) # altered by lyk
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    num_ins = cfg.DATALOADER.NUM_INSTANCE
    classes = len(train_loader.dataset.pids)
    center_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT

    best = 0.0
    best_index = 1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_id_meter.reset()
        loss_tri_meter.reset()
        loss_center_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        ##### for fixed BN test
        if cfg.MODEL.FIXED_RES_BN:
            if 'res' in cfg.MODEL.NAME or 'ibn' in cfg.MODEL.NAME:
                for name, mod in model.base.named_modules():
                    if 'bn' in name:
                        mod.eval()
                        # totally freezed BN
                        # mod.weight.requires_grad_(False)
                        # mod.bias.requires_grad_(False)
                print("====== freeze BNs ======")
            else:
                for name, mod in model.base.named_modules():
                    if 'norm' in name:
                        mod.eval()
                        # totally freezed LN
                        mod.weight.requires_grad_(False)
                        mod.bias.requires_grad_(False)
                print("====== freeze LNs ======")
            
        
        for n_iter, informations in enumerate(train_loader):
            img = informations['images'].to(device)
            # img = MixStyle_2d()(img) #### test
            vid = informations['targets'].to(device)
            target_cam = informations['camid'].to(device)
            # ipath = informations['img_path']
            ori_label = informations['ori_label'].to(device)
            # t_domains = informations['others']['domains'].to(device)

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            ori_label = ori_label.to(device)
            # t_domains = t_domains.to(device)

            targets = torch.zeros((bs, classes)).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(device)
            model.to(device)
            with amp.autocast(enabled=True):
                # score, feat, target, score_, loss_tri_hard = model(img, target, t_domains)
                loss_tri_hard = torch.tensor(0.,device=device)
                score, feat = model(img)
                ### id loss
                targets = 0.9 * targets + 0.1 / classes # label smooth
                if isinstance(score, list):
                    log_probs = [nn.LogSoftmax(dim=1)(s) for s in score[1:]]
                    loss_id = [(- targets * p).mean(0).sum() for p in log_probs]
                    loss_id = sum(loss_id) / len(loss_id)
                    loss_id = 0.5*loss_id + 0.5 * (- targets * nn.LogSoftmax(dim=1)(score[0])).mean(0).sum()
                else:
                    log_probs = nn.LogSoftmax(dim=1)(score)
                    loss_id = (- targets * log_probs).mean(0).sum()

                #### triplet loss
                triplet = TripletLoss()
                if isinstance(feat, list):
                    loss_tri = [triplet(f, target)[0] for f in feat[1:]]
                    loss_tri = sum(loss_tri) / len(loss_tri)
                    loss_tri = 0.5*loss_tri + 0.5*triplet(feat[0], target)[0]
                else:
                    loss_tri = triplet(feat, target)[0]
                # loss_tri = torch.tensor(0.0, device=device)

                #### scatter loss
                # styles = torch.arange(16).repeat(4).to(device)
                # loss_sct = domain_SCT_loss(feat, styles)
                loss_sct = torch.tensor(0.0, device=device)

                #### center loss
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    if isinstance(feat, list):
                        loss_center = [center_criterion(f, target) for f in feat[1:]]
                        loss_center = sum(loss_center) / len(loss_center)
                        loss_center = 0.5*loss_center + 0.5*center_criterion(feat[0], target)
                        for f in feat:
                            loss_center += center_criterion(f, target)
                    else:
                        loss_center = center_criterion(feat, target)
                else:
                    loss_center = torch.tensor(0.0, device=device)
                #### XDED loss
                if cfg.MODEL.DISTILL.DO_XDED and epoch > 5:
                    probs = nn.Softmax(dim=1)(score / 0.2) # tao
                    probs_mean = probs.reshape(bs//num_ins,num_ins,classes).mean(1,True)
                    probs_xded = probs_mean.repeat(1,num_ins,1).view(-1,classes).detach()
                    loss_xded = (- probs_xded * log_probs).mean(0).sum()
                else:
                    loss_xded = torch.tensor(0.0, device=device)

                loss = loss_id + loss_tri\
                    + center_weight * loss_center\

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), bs)
            loss_id_meter.update(loss_id.item(), bs)
            loss_tri_meter.update(loss_tri.item(), bs)
            loss_center_meter.update(center_weight*loss_center.item(), bs)
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, id:{:.3f}, tri:{:.3f}, cen:{:.3f} Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg,
                loss_id_meter.avg, loss_tri_meter.avg,
                loss_center_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/loss', loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            if 'DG' in cfg.DATASETS.TEST[0]:
                cmc, mAP = do_inference_multi_targets(cfg, model, num_query, logger)
            else:
                cmc, mAP = do_inference(cfg, model, val_loader, num_query)
            tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
            tbWriter.add_scalar('val/mAP', mAP, epoch)
            torch.cuda.empty_cache()
            if best < mAP + cmc[0]:
                best = mAP + cmc[0]
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
                else:
                    torch.save(model.state_dict(),
                            os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
        torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_best.pth')
    # eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0)
    model.load_param(load_path)
    logger.info('load weights from best.pth')
    if 'DG' in cfg.DATASETS.TEST[0]:
        do_inference_multi_targets(cfg, model, logger)
    else:
        for testname in cfg.DATASETS.TEST:
            _, _, val_loader, num_query = build_reid_test_loader(cfg, testname)
            do_inference(cfg, model, val_loader, num_query, multi_query=cfg.TEST.MULTI_QUERY, reranking=cfg.TEST.RE_RANKING)