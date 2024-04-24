import logging
import time

import torch
import torch.nn as nn
from data.build_DG_dataloader import build_reid_test_loader

from utils.metrics import R1_mAP_eval, R1_mAP_eval_ensemble
from utils.attribute_recognition import Attribute_Recognition

from prettytable import PrettyTable
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import os
import json

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 reranking=False,
                 iflog=True,
                 query=None,
                 gallery=None,
                 gen_result=False,
                 multi_query=False,
                 attr_recognition=False,
                ):
    device = "cuda"
    if iflog:
        logger = logging.getLogger("reid.test")
        logger.info("Enter inferencing")

    log_path = cfg.LOG_ROOT + cfg.LOG_NAME
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking = reranking, query = query, gallery = gallery, log_path = log_path, gen_result=gen_result, multi_query=multi_query)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    torch.cuda.synchronize()
    t0 = time.time()
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    
    attributes = []
    attr_classes = []
    data_list = []
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']

        # attributes
        attrs = informations['others']
        
        data_list.append(informations)
        for k in attrs.keys():
            attrs[k] = attrs[k].to(device)
            attributes.append(attrs[k])
        with torch.no_grad():
            img = img.to(device)
            outputs  = model(img, attr_recognition)
            if attr_recognition:
                feat, attr_scores = outputs
                feat = feat[:, 0]
                for scores in attr_scores:
                    class_indices = torch.argmax(scores, dim=1)
                    attr_classes.append(class_indices.tolist())
            else:
                feat = outputs
                
            evaluator.update((feat, pid, camids))
            img_path_list.extend(imgpath)

    if attr_recognition:
        # if want to get attribute recognition wrong result, set "gen_attr_result = True"
        accuracy_per_attribute = Attribute_Recognition(cfg,attributes,attr_classes,data_list,gen_attr_reslut = False)
        table = PrettyTable(["task", "gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style'])
        formatted_accuracy = ["{:.2%}".format(accuracy) for accuracy in accuracy_per_attribute]
        table.add_row(["Attribute Recognition"] + formatted_accuracy)
        logger.info('\n' + str(table))

    total_f_time = time.time() - t0
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    if iflog:
        logger.info("Total feature time: {:.2f}s".format(total_f_time))
        logger.info("single feature time: {:.5f}s".format(single_f_time))
        logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if iflog:
        logger.info("Validation Results ")
        table = PrettyTable(["task", "mAP", "R1", "R5", "R10"])
        table.add_row(['Reid', mAP, cmc[0],cmc[4], cmc[9]])
        table.custom_format["R1"] = lambda f, v: f"{v:.2%}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2%}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2%}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2%}"
        logger.info('\n' + str(table))
        logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP

def do_inference_only_attr(cfg,
                 model,
                 val_loader,
                 iflog=True,
                 attr_recognition=True,
                ):
    device = "cuda"
    if iflog:
        logger = logging.getLogger("reid.test")
        logger.info("Enter inferencing")

    log_path = cfg.LOG_ROOT + cfg.LOG_NAME

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    torch.cuda.synchronize()
    t0 = time.time()
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    
    attributes = []
    attr_classes = []
    data_list = []
    save_data = []
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        attrs = informations['others']
        paths = informations['img_path']
        # import ipdb; ipdb.set_trace()
        data_list.append(informations)
        for k in attrs.keys():
            attrs[k] = attrs[k].to(device)
            attributes.append(attrs[k])
        with torch.no_grad():
            img = img.to(device)
            outputs  = model(img, attr_recognition)
            if attr_recognition:
                feat, attr_scores = outputs
                feat = feat[:, 0]
                for scores in attr_scores:
                    class_indices = torch.argmax(scores, dim=1)
                    attr_classes.append(class_indices.tolist())
            else:
                feat = outputs

        # import ipdb; ipdb.set_trace()
        for i in range(img.shape[0]):
            data = {}
            data['file_path'] = paths[i]
            data['attrs_cls'] = {
                "gender": attr_classes[0][i],
                "backpack": attr_classes[1][i],
                "hat": attr_classes[2][i],
                "ucc": attr_classes[3][i],
                "ucs": attr_classes[4][i],
                "lcc": attr_classes[5][i],
                "lcs": attr_classes[6][i],
            }
            data['attrs'] = {
                "gender": attr_dicts[0][attr_classes[0][i]],
                "backpack": attr_dicts[1][attr_classes[1][i]],
                "hat": attr_dicts[2][attr_classes[2][i]],
                "ucc": attr_dicts[3][attr_classes[3][i]],
                "ucs": attr_dicts[4][attr_classes[4][i]],
                "lcc": attr_dicts[5][attr_classes[5][i]],
                "lcs": attr_dicts[6][attr_classes[6][i]],
            }
            save_data.append(data)
    save_folder = cfg.LOG_ROOT + cfg.LOG_NAME + f"/attr_res.json"
    with open(save_folder,'w') as f:
        # json.dump(save_data, f)
        for obj in save_data:
            f.write(json.dumps(obj)+'\n')
            f.flush()
    logger.info("save attribute results in {}".format(save_folder))
    
    total_f_time = time.time() - t0
    # if want to get attribute recognition wrong result, set "gen_attr_result = True"
    accuracy_per_attribute = Attribute_Recognition(cfg,attributes,attr_classes,data_list,gen_attr_reslut = False)
    logger.info("Validation Results ")
    table = PrettyTable(["task", "gender", "backpack", "hat", "upper_color", "upper_style","lower_color",'lower_style'])
    formatted_accuracy_per_attribute = ["{:.2%}".format(accuracy) for accuracy in accuracy_per_attribute]
    table.add_row(["Attribute Recognition"] + formatted_accuracy_per_attribute)
    logger.info('\n' + str(table))
    logger.info("=====attribute recognition accuracy: {:.2%}=====".format(sum(accuracy_per_attribute) / len(accuracy_per_attribute)))

    
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    if iflog:
        logger.info("Total feature time: {:.2f}s".format(total_f_time))
        logger.info("single feature time: {:.5f}s".format(single_f_time))
        logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))
        logger.info("total inference time: {:.2f}".format(time.time() - t0))

    return accuracy_per_attribute


def do_inference_ensemble(cfg,
                 models,
                 val_loader,
                 num_query,
                 reranking=False,
                 iflog=True,
                 query=None,
                 gallery=None,
                 gen_result=False,
                 multi_query=False,
                 attr_recognition=False,
                 threshold=0,
                ):
    device = "cuda"
    if iflog:
        logger = logging.getLogger("reid.test")
        logger.info("Enter inferencing")

    log_path = cfg.LOG_ROOT + cfg.LOG_NAME
    evaluator = R1_mAP_eval_ensemble(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking = reranking, query = query, gallery = gallery, log_path = log_path, gen_result=gen_result, multi_query=multi_query, num_models=len(models), threshold=threshold)

    evaluator.reset()

    if device:
        for model in models.values():
            model.to(device)

    for k in models.keys():
        models[k] = models[k].eval()
    img_path_list = []
    torch.cuda.synchronize()
    t0 = time.time()
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    
    attributes = []
    attr_classes = []
    data_list = []
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        img_384 = informations['images_384']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        resolutions = informations['resolutions']

        # attributes
        attrs = informations['others']
        
        data_list.append(informations)
        if attrs is not None:
            for k in attrs.keys():
                attrs[k] = attrs[k].to(device)
                attributes.append(attrs[k])
        with torch.no_grad():
            img = img.to(device)
            img_384 = img_384.to(device)
            feats = []
            for name in models.keys():
                if '384' in name:
                    outputs = models[name](img_384)
                else:
                    outputs  = models[name](img)
                if attr_recognition:
                    feat, attr_scores = outputs
                    feat = feat[:, 0]
                    for scores in attr_scores:
                        class_indices = torch.argmax(scores, dim=1)
                        attr_classes.append(class_indices.tolist())
                else:
                    feat = outputs
                feats.append(feat)
                
            evaluator.update((feats, pid, camids, resolutions, attrs))
            img_path_list.extend(imgpath)

    total_f_time = time.time() - t0
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    if iflog:
        logger.info("Total feature time: {:.2f}s".format(total_f_time))
        logger.info("single feature time: {:.5f}s".format(single_f_time))
        logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if iflog:
        logger.info("Validation Results ")
        table = PrettyTable(["task", "mAP", "R1", "R5", "R10"])
        table.add_row(['Reid', mAP, cmc[0],cmc[4], cmc[9]])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        logger.info('\n' + str(table))
        logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP


def do_inference_ensemble_concat(cfg,
                 models,
                 val_loader,
                 num_query,
                 reranking=False,
                 iflog=True,
                 query=None,
                 gallery=None,
                 gen_result=False,
                 multi_query=False,
                 attr_recognition=False,
                 threshold=0,
                 swin_loader=None,
                ):
    device = "cuda"
    if iflog:
        logger = logging.getLogger("reid.test")
        logger.info("Enter inferencing")

    log_path = cfg.LOG_ROOT + cfg.LOG_NAME
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking = reranking, query = query, gallery = gallery, log_path = log_path, gen_result=gen_result, multi_query=multi_query)

    evaluator.reset()

    if device:
        for model in models.values():
            model.to(device)

    for k in models.keys():
        models[k] = models[k].eval()
    img_path_list = []
    torch.cuda.synchronize()
    t0 = time.time()
    bs = cfg.SOLVER.IMS_PER_BATCH # altered by lyk
    
    attributes = []
    attr_classes = []
    data_list = []
    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        resolutions = informations['resolutions']

        # attributes
        attrs = informations['others']
        
        data_list.append(informations)
        for k in attrs.keys():
            attrs[k] = attrs[k].to(device)
            attributes.append(attrs[k])
        with torch.no_grad():
            img = img.to(device)
            img_swin = nn.functional.interpolate(img, [224, 224])
            feats = []
            for name in models.keys():
                if 'swin' in name:
                    outputs = models[name](img_swin)
                else:
                    outputs  = models[name](img)
                if attr_recognition:
                    feat, attr_scores = outputs
                    feat = feat[:, 0]
                    for scores in attr_scores:
                        class_indices = torch.argmax(scores, dim=1)
                        attr_classes.append(class_indices.tolist())
                else:
                    feat = outputs
                feats.append(feat)
                
            evaluator.update((torch.cat(feats, dim=1), pid, camids))
            img_path_list.extend(imgpath)

    total_f_time = time.time() - t0
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    if iflog:
        logger.info("Total feature time: {:.2f}s".format(total_f_time))
        logger.info("single feature time: {:.5f}s".format(single_f_time))
        logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if iflog:
        logger.info("Validation Results ")
        table = PrettyTable(["task", "mAP", "R1", "R5", "R10"])
        table.add_row(['Reid', mAP, cmc[0],cmc[4], cmc[9]])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        logger.info('\n' + str(table))
        logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP

def do_inference_feat_fusion(cfg,
                 model,
                 val_loader,
                 num_query,
                 reranking=False,
                 iflog=True,
                 query=None,
                 gallery=None,
                 gen_result=False,
                ):
    device = "cuda"
    if iflog:
        logger = logging.getLogger("reid.test")
        logger.info("Enter inferencing")

    log_path = cfg.LOG_ROOT + cfg.LOG_NAME
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking = reranking, query = query, gallery = gallery, log_path = log_path, gen_result=False, multi_query=cfg.TEST.MULTI_QUERY)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    torch.cuda.synchronize()
    t0 = time.time()

    for n_iter, informations in enumerate(val_loader):
        img = informations['images']
        pid = informations['targets']
        camids = informations['camid']
        imgpath = informations['img_path']
        # domains = informations['others']['domains']
        with torch.no_grad():
            img = img.to(device)
            # camids = camids.to(device)
            feat, attr_score = model(img, attr_recognition=True)
            feat_sync = feat[:, 0]*0.5 + (feat[:, 1] + feat[:, 3])/2*0.5
            # feat_sync = torch.cat([feat[:, 0], feat[:, 1], feat[:, 3]], dim=1)
            # feat_sync = (feat[:, 1] + feat[:, 2] + feat[:, 3])/3
            evaluator.update((feat_sync, pid, camids))
            img_path_list.extend(imgpath)

    total_f_time = time.time() - t0
    single_f_time = total_f_time / (len(val_loader) * img.shape[0])
    num_imgs_per_sec = (len(val_loader) * img.shape[0]) / total_f_time
    if iflog:
        logger.info("Total feature time: {:.2f}s".format(total_f_time))
        logger.info("single feature time: {:.5f}s".format(single_f_time))
        logger.info("number of images per sec: {:.2f}img/s".format(num_imgs_per_sec))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    if iflog:
        logger.info("Validation Results ")
        table = PrettyTable(["task", "mAP", "R1", "R5", "R10"])
        table.add_row(['Reid', mAP, cmc[0],cmc[4], cmc[9]])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        logger.info('\n' + str(table))
        logger.info("total inference time: {:.2f}".format(time.time() - t0))
    return cmc, mAP

def do_inference_multi_targets(cfg,
                 model,
                 logger):

    cmc_all, mAP_all = [0 for i in range(50)], 0
    for testname in cfg.DATASETS.TEST:
        cmc_avg, mAP_avg = [0 for i in range(50)], 0
        for split_id in range(10):
            if testname == 'DG_VIPeR':
                split_id = 'split_{}a'.format(split_id+1)
            val_loader, num_query = build_reid_test_loader(cfg, testname, opt=split_id)
            cmc, mAP = do_inference(cfg, model, val_loader, num_query, False)
            cmc_avg += cmc
            mAP_avg += mAP
        cmc_avg /= 10
        mAP_avg /= 10
        cmc_all += cmc_avg
        mAP_all += mAP_avg
        logger.info("===== Avg Results for 10 splits of {} =====".format(testname))
        logger.info("mAP: {:.1%}".format(mAP_avg))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_avg[r - 1]))

    logger.info("===== Mean Results on 4 target datasets =====")
    logger.info("mAP: {:.1%}".format(mAP_all / len(cfg.DATASETS.TEST)))
    for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_all[r - 1] / len(cfg.DATASETS.TEST)))

    return cmc_all, mAP_all




attr_dicts = [
    {0: 'male', 1: 'female'}, # gender
    {0:'n/a', 1:'red', 2:'black', 3:'green', 4:'yellow'}, # backpack
    {0:'n/a', 1:'red', 2:'black', 3:'yellow', 4:'white'}, # hat
    {0:'n/a', 1:'red', 2:'black', 3:'blue', 4:'green', 5:'multicolor', 6:'grey', 7:'white', 8:'yellow', 9:'dark brown', 10:'purple', 11:'pink'}, # ucc
    {0:'n/a', 1:'long', 2:'short', 3:'skirt'}, # ucs
    {0:'n/a', 1:'red', 2:'black', 3:'blue', 4:'green', 5:'multicolor', 6:'grey', 7:'white', 8:'yellow', 9:'dark brown', 10:'purple', 11:'pink'}, # lcc
    {0:'n/a', 1:'long', 2:'short', 3:'skirt'}, # lcs
]