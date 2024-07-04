import torch
import os
from collections import OrderedDict

def reverse_correct_unfold_reduction_order(x):
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, in_channel // 4, 4)
    x = x[:, :, [0, 2, 1, 3]].transpose(1, 2).reshape(out_channel, in_channel)
    return x


def reverse_correct_unfold_norm_order(x):
    in_channel = x.shape[0]
    x = x.reshape(in_channel // 4, 4)
    x = x[:, [0, 2, 1, 3]].transpose(0, 1).reshape(in_channel)
    return x


def print_layers(state_dict, max_depth):
    layers_printed = set()  # To avoid printing layers of the same depth multiple times
    for key, value in state_dict.items():
        depth = key.count('.')
        if depth < max_depth and key not in layers_printed:
            print(f"{key}: {value.shape}")
            layers_printed.add(key)


def get_state_dict(state_dict, depth):
        # 用于存储前两层的参数名称
    layer_names = set()

    # 遍历 state_dict 中的所有参数名称
    for key in state_dict.keys():
        # 分割参数名称以获取层级信息
        parts = key.split('.')
        # 只考虑前两个层级
        if len(parts) >= depth:
            # 将层级名称组合回去，但只包含前两个部分
            layer_name = '.'.join(parts[:depth])
            layer_names.add(layer_name)
    return layer_names


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def mmdet_to_groundingdino(ckpt, swin_b=False):
    new_ckpt = OrderedDict()
    for k, v in list(ckpt.items()):
        new_v = v
        
        if  "language_model.language_backbone.body.model" in k:
            new_k = k.replace("language_model.language_backbone.body.model","module.bert")
        elif "backbone" in k:
            new_k = k.replace("backbone","module.backbone.0")
            if "patch_embed.projection" in new_k:
                new_k = new_k.replace("patch_embed.projection","patch_embed.proj")
            elif "drop_after_pos" in new_k:
                new_k = new_k.replace( "drop_after_pos","pos_drop")

            if  "stages" in new_k:
                new_k = new_k.replace("stages","layers")
                if "ffn.layers.0.0" in new_k:
                    new_k = new_k.replace("ffn.layers.0.0","mlp.fc1")
                elif "ffn.layers.1" in new_k:
                    new_k = new_k.replace( "ffn.layers.1","mlp.fc2")
                elif "attn.w_msa" in new_k:
                    new_k = new_k.replace("attn.w_msa","attn")

                if "downsample" in k:
                    if "reduction." in k:
                        new_v = reverse_correct_unfold_reduction_order(v)
                    elif "norm." in k:
                        new_v = reverse_correct_unfold_norm_order(v)
        elif  "text_feat_map" in k:
            new_k = k.replace( "text_feat_map","module.feat_map")
        elif "neck.extra_convs.0" in k or "neck.convs" in k:
            # extra convs for 4th scale
            new_k = k.replace("neck.extra_convs.0","neck.convs.3")
            if "neck.convs" in new_k:
                new_k = new_k.replace("neck.convs","module.input_proj")
                if "conv.weight" in new_k:
                    # 0.weight -> conv.weight
                    new_k = new_k.replace("conv.weight","0.weight")
                if "conv.bias" in new_k:
                    # 0.bias -> conv.bias
                    new_k = new_k.replace("conv.bias","0.bias")
                if "gn.weight" in new_k:
                    # 1.weight -> gn.weight
                    new_k = new_k.replace("gn.weight","1.weight")
                if  "gn.bias" in new_k:
                    # 1.bias -> gn.bias
                    new_k = new_k.replace("gn.bias","1.bias")

        elif "level_embed" in k:
            # module.transformer.level_embed -> level_embed
            new_k = k.replace("level_embed","module.transformer.level_embed")

        elif  "encoder"  in k:
            # if ".layers" in k:
            new_k = k.replace("encoder","module.transformer.encoder",)
            if "norms.0" in new_k:
                new_k = new_k.replace("norms.0", "norm1")
            if "norms.1" in new_k:
                new_k = new_k.replace("norms.1", "norm2")
            if "norms.2" in new_k:
                new_k = new_k.replace("norms.2", "norm3")
            if "norms.w" in new_k:
                new_k = new_k.replace("norms.w", "normse")
            if "ffn.layers.0.0" in new_k:
                new_k = new_k.replace("ffn.layers.0.0","linear1")
            if "ffn.layers.1" in new_k:
                new_k = new_k.replace("ffn.layers.1","linear2")

            if "text_layers" in new_k and "self_attn.attn" in new_k:
                new_k = new_k.replace("self_attn.attn", "self_attn")
        elif "memory_trans_" in k:
            if "memory_trans_fc" in k and "norm" not in k:
                new_k = k.replace("memory_trans_fc","module.transformer.enc_output")
            new_k = k.replace("memory_trans_fc", "module.transformer.enc_output")
            if "memory_trans_norm" in k:
                new_k = k.replace("memory_trans_norm","module.transformer.enc_output_norm")
        elif "bbox_head.reg_branches.6" in k:
            if "bbox_head.reg_branches.6.0" in k:
                new_k = k.replace(
                    "bbox_head.reg_branches.6.0",
                    "module.transformer.enc_out_bbox_embed.layers.0")
            if "bbox_head.reg_branches.6.2" in k:
                new_k = k.replace(
                    "bbox_head.reg_branches.6.2",
                    "module.transformer.enc_out_bbox_embed.layers.1")
            if "bbox_head.reg_branches.6.4" in k:
                new_k = k.replace(
                    "bbox_head.reg_branches.6.4",
                    "module.transformer.enc_out_bbox_embed.layers.2")
        elif "bbox_head.reg_branches" in k:
            parts = k.split(".")
            weight_or_bias = parts[-1]
            linear_id = int(parts[-2]) // 2  # Undo the multiplication by 2
            reg_layer_id = int(parts[-3])
            new_k = f"module.transformer.decoder.bbox_embed.{reg_layer_id}.layers.{linear_id}.{weight_or_bias}"

        elif "query_embedding" in k:
                new_k = k.replace("query_embedding",
                                "module.transformer.tgt_embed")


        elif k.startswith("decoder"):
            new_k = k.replace("decoder","module.transformer.decoder", 1)
            if  "norms.2" in new_k:
                # norm1 in official GroundingDINO is the third norm in decoder
                new_k = new_k.replace( "norms.2","norm1")
            if "norms.1" in new_k:
                # catext_norm in official GroundingDINO is the
                # second norm in decoder
                new_k = new_k.replace("norms.1","catext_norm")
            if "norms.0" in new_k:
                # norm2 in official GroundingDINO is the first norm in decoder
                new_k = new_k.replace("norms.0", "norm2")
            if "norms.3" in new_k:
                new_k = new_k.replace("norms.3", "norm3")
            if "cross_attn_text" in new_k:
                new_k = new_k.replace( "cross_attn_text","ca_text")
                if "attn.in_proj_weight" in new_k:
                    new_k = new_k.replace("attn.in_proj_weight","in_proj_weight")
                if "attn.in_proj_bias" in new_k:
                    new_k = new_k.replace("attn.in_proj_bias","in_proj_bias")
                if "attn.out_proj.weight" in new_k:
                    new_k = new_k.replace("attn.out_proj.weight","out_proj.weight")
                if "attn.out_proj.bias" in new_k:
                    new_k = new_k.replace("attn.out_proj.bias","out_proj.bias")
            if  "ffn.layers.0.0" in new_k:
                new_k = new_k.replace( "ffn.layers.0.0","linear1")
            if "ffn.layers.1" in new_k:
                new_k = new_k.replace("ffn.layers.1","linear2")
            if "self_attn.attn" in new_k:
                new_k = new_k.replace("self_attn.attn","self_attn")
        else:
            print("skip:", k)
            continue

        if swin_b and new_k.startswith("module"):
            new_k = new_k.replace("module.", "")
        new_ckpt[new_k] = new_v
        
        if "transformer.decoder.bbox_embed" in new_k:
            new_ckpt[new_k.replace("transformer.decoder.bbox_embed", "bbox_embed")] = new_v
    return new_ckpt

model_root = '/media/gpuadmin/rcao/result/uois/detection/uoais_v0.2'
model_path = os.path.join(model_root, 'best_coco_bbox_mAP_epoch_2.pth')
ckpt = torch.load(model_path, map_location="cpu")

ckpt_converted = mmdet_to_groundingdino(ckpt["state_dict"])
save_dict = {"model": ckpt_converted}
torch.save(save_dict, 'groundingdino_swint_uoais_sim_tune_v2.pth')

# print(checkpoint1.keys())

# model_path = 'groundingdino_swint_ogc.pth'
# checkpoint2 = torch.load(model_path, map_location="cpu")

# # print(checkpoint2["model"].keys())
# state_dict1 = clean_state_dict(checkpoint1)
# state_dict2 = clean_state_dict(checkpoint2["model"])

# print("State dict from checkpoint 1:")
# print(get_state_dict(state_dict1, 3))
# print("------------------------------")
# print("------------------------------")
# print("------------------------------")
# print("State dict from checkpoint 2:")
# print(get_state_dict(state_dict2, 3))

# print("State dict from checkpoint 1:")
# for key, value in state_dict1.items():
#     print(f"{key}: {value.shape}")

# # 打印第二个 checkpoint 的 state_dict
# print("\nState dict from checkpoint 2:")
# for key, value in state_dict2.items():
#     print(f"{key}: {value.shape}")
# # print(len(clean_state_dict(checkpoint2["model"]).keys()))


