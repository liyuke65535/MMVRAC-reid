import logging
import os
import time
import torch
import torch.nn as nn
from loss.domain_SCT_loss import domain_SCT_loss
from loss.triplet_loss import euclidean_dist, hard_example_mining
from loss.triplet_loss_for_mixup import hard_example_mining_for_mixup
from model.make_model import make_model
from processor.inf_processor import do_inference, do_inference_multi_targets,do_inference_only_attr
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def attr_vit_do_train_with_amp(cfg,
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
             sour_centers=None,
             attr_recognition=False):
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
    loss_attr_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512) # altered by lyk
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    num_ins = cfg.DATALOADER.NUM_INSTANCE
    classes = len(train_loader.dataset.pids)
    center_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT
    id_weight = cfg.MODEL.ID_LOSS_WEIGHT
    tri_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
    attr_weight = cfg.MODEL.ATTRIBUTE_LOSS_WEIGHT

    best = 0.0
    best_attr = 0.0
    best_index = 1
    best_attr_index = 1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_id_meter.reset()
        loss_tri_meter.reset()
        loss_center_meter.reset()
        loss_attr_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, informations in enumerate(train_loader):
            img = informations['images'].to(device)
            vid = informations['targets'].to(device)
            target_cam = informations['camid'].to(device)
            ori_label = informations['ori_label'].to(device)

            # attributes
            attrs = informations['others']
            attributes = []
            for k in attrs.keys():
                attrs[k] = attrs[k].to(device)
                attributes.append(attrs[k])

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            ori_label = ori_label.to(device)

            targets = torch.zeros((bs, classes)).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(device)
            model.to(device)
            with amp.autocast(enabled=True):
                loss_tri_hard = torch.tensor(0.,device=device)
                if cfg.MODEL.HAS_ATTRIBUTE_EMBEDDING:
                    score, feat, attr_scores = model(img, attrs=attributes)
                else:
                    score, feat, attr_scores = model(img)
                # import ipdb; ipdb.set_trace()
                #### id loss
                log_probs = nn.LogSoftmax(dim=1)(score)
                targets = 0.9 * targets + 0.1 / classes # label smooth
                loss_id = (- targets * log_probs).mean(0).sum()

                #### attr loss
                attr_targets = [
                    torch.zeros((bs, 2)).to(device).scatter_(1, attributes[0].unsqueeze(1), 1),
                    torch.zeros((bs, 5)).to(device).scatter_(1, attributes[1].unsqueeze(1), 1),
                    torch.zeros((bs, 5)).to(device).scatter_(1, attributes[2].unsqueeze(1), 1),
                    torch.zeros((bs, 12)).to(device).scatter_(1, attributes[3].unsqueeze(1), 1),
                    torch.zeros((bs, 4)).to(device).scatter_(1, attributes[4].unsqueeze(1), 1),
                    torch.zeros((bs, 12)).to(device).scatter_(1, attributes[5].unsqueeze(1), 1),
                    torch.zeros((bs, 4)).to(device).scatter_(1, attributes[6].unsqueeze(1), 1),
                    ]
                # attr_targets = torch.tensor(attr_targets).to(device)
                attr_log_probs = [nn.LogSoftmax(dim=1)(s) for s in attr_scores] # attr
                loss_attr = [-attr_targets[i] * attr_log_probs[i] for i in range(7)]
                loss_attr = sum([l.mean(0).sum() for l in loss_attr])
                # import ipdb; ipdb.set_trace()

                #### triplet loss
                # target = targets.max(1)[1] ###### for mixup
                N = feat.shape[0]
                dist_mat = euclidean_dist(feat, feat)
                target_new = target
                is_pos = target_new.expand(N, N).eq(target_new.expand(N, N).t())
                is_neg = target_new.expand(N, N).ne(target_new.expand(N, N).t())
                dist_ap, relative_p_inds = torch.max(
                    dist_mat[is_pos].contiguous().view(bs, -1), 1, keepdim=True)
                dist_an, relative_n_inds = torch.min(
                    dist_mat[is_neg].contiguous().view(bs, -1), 1, keepdim=True)
                y = dist_an.new().resize_as_(dist_an).fill_(1)
                loss_tri = nn.SoftMarginLoss()(dist_an - dist_ap, y)
                # loss_tri = torch.tensor(0.0, device=device)

                #### center loss
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    loss_center = center_criterion(feat, target)
                else:
                    loss_center = torch.tensor(0.0, device=device)

                loss = id_weight * loss_id + tri_weight * loss_tri + center_weight * loss_center + attr_weight * loss_attr

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
            loss_id_meter.update(id_weight*loss_id.item(), bs)
            loss_tri_meter.update(tri_weight*loss_tri.item(), bs)
            loss_center_meter.update(center_weight*loss_center.item(), bs)
            loss_attr_meter.update(attr_weight*loss_attr.item(), bs)
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, id:{:.3f}, tri:{:.3f}, cen:{:.3f}, attr:{:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg, loss_id_meter.avg, loss_tri_meter.avg,
                loss_center_meter.avg, loss_attr_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
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
                cmc, mAP = do_inference(cfg, model, val_loader, num_query, attr_recognition=True)
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
        # # The best model for attribute recognition is retained while training reid
        # if attr_recognition:
        #     if epoch % 1 == 0 and epoch <= 25:
        #         accuracy_per_attribute = do_inference_only_attr(cfg, model, val_loader, num_query, attr_recognition=True)
        #         if best_attr < sum(accuracy_per_attribute):
        #             best_attr = sum(accuracy_per_attribute)
        #             best_attr_index = epoch
        #             logger.info("=====best epoch: {}=====".format(best_attr_index))
        #             logger.info("=====best accuracy: {:.2%}=====".format(best_attr))
        #             if cfg.MODEL.DIST_TRAIN:
        #                 if dist.get_rank() == 0:
        #                     torch.save(model.state_dict(),
        #                             os.path.join(log_path,'attr_best.pth'))
        #             else:
        #                 torch.save(model.state_dict(),
        #                         os.path.join(log_path, 'attr_best.pth'))
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