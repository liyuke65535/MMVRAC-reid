import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
from model.make_model import build_attr_vit, Backbone
from model.backbones.swin_transformer import swin_base_patch4_window7_224
from model.backbones.vit_pytorch import attr_vit_base_patch16_224_TransReID, attr_vit_large_patch16_224_TransReID, attr_vit_small_patch16_224_TransReID, vit_large_patch16_224_TransReID
from processor.inf_processor import do_inference, do_inference_ensemble, do_inference_multi_targets
from utils.logger import setup_logger
import torchvision.transforms as T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/uavhuman.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    output_dir = "debug"
    logger = setup_logger("reid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model_vitb = build_attr_vit(619, cfg, 'attr_vit_base_patch16_224_TransReID', stride_size=12, model_path="/home/liyuke/data5/exp/attr_vit_b12_rea_256x128_centerLoss_lr1e-2/attr_vit_best.pth")
    model_vitb_pretrained = build_attr_vit(619, cfg, 'attr_vit_base_patch16_224_TransReID', stride_size=12, model_path="/home/liyuke/data5/exp/attr_vit_multi_source_b12_rea_256x128_centerLoss_lr1e-2/attr_vit_best.pth", pretrain_choice='self')
    model_vitb_pretrained_384 = build_attr_vit(619, cfg, 'attr_vit_base_patch16_224_TransReID', stride_size=12, model_path="/home/liyuke/data5/exp/from_995/attr_vit_pretrained_b12_rea_384x192_centerLoss_lr1e-2/attr_vit_best.pth", pretrain_choice='self', img_size=[384,192])
    model_vitb_pretrained_soft_rea_384 = build_attr_vit(619, cfg, 'attr_vit_base_patch16_224_TransReID', stride_size=12, model_path="/home/liyuke/data5/exp/from_995/attr_vit_pretrained_b12_soft_rea_384x192_centerLoss_lr1e-2/attr_vit_best.pth", pretrain_choice='self', img_size=[384,192])
    model_vitb_pretrained_dirty_soft_rea_384 = build_attr_vit(619, cfg, 'attr_vit_base_patch16_224_TransReID', stride_size=12, model_path="/home/liyuke/data5/exp/from_995/attr_vit_pretrained_dirty_images_b12_soft_rea_384x192_centerLoss_lr1e-2/attr_vit_best.pth", pretrain_choice='self', img_size=[384,192])

    
    models = {
        "vit_b": model_vitb, # 1
        "vit_b_pre": model_vitb_pretrained, # 2
        "vit_b_pre_384": model_vitb_pretrained_384, # 3
        "vit_b_pre_soft_rea_384": model_vitb_pretrained_soft_rea_384, # 4
        "vit_b_pre_dirt_soft_rea_384": model_vitb_pretrained_dirty_soft_rea_384, # 5
        }


    for testname in cfg.DATASETS.TEST:
        _, _, val_loader, num_query = build_reid_test_loader(cfg, testname)

        do_inference_ensemble(
            cfg,
            models,
            val_loader,
            num_query,
            reranking=True,
            query_aggregate=True,
            threshold=0,
        )