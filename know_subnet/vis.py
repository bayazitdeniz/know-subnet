import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["WORLD_SIZE"] = "1"

import glob

from know_subnet.utils import *
from know_subnet.lm.lm_utils import *
from matplotlib import pyplot as plt
import os
from know_subnet.lm.mask import MaskedHuggingfaceConv1D

def get_model_and_config(
        exper_dir,
        checkpoint_step
    ):
    config_paths = list(glob.iglob('{}/config*'.format(exper_dir), recursive=True))
    checkpoint_paths = list(glob.iglob('{}/checkpoints/ckpt-step={}-*'.format(exper_dir, checkpoint_step), recursive=True))

    assert (len(checkpoint_paths) == 1)

    for config_path in config_paths:
        config = parse_json_to_dict(config_path)

    for checkpoint_path in checkpoint_paths:
        print("Loading model from:", checkpoint_path)
        checkpoint_title = checkpoint_path.split("/")[-1].split("-")
        print(checkpoint_title)
        checkpoint_step = checkpoint_title[1].split("=")[1]
        print("Checkpoint step:", checkpoint_step)
        checkpoint_sparsity = checkpoint_title[2].split("=")[1]
        print("Checkpoint sparsity:", checkpoint_sparsity)  
        model = load_from_checkpoint(config, checkpoint_path, torch.cuda.is_available())
        model.eval()
        model.set_is_inverse_mask(is_inverse_mask=False)
    return model, checkpoint_sparsity, config

def visualize_head_density(model, path, block=False, layers=12, heads=12, model_name=""): 
    """
    bert_lm.bert.encoder.layer.0.attention.self.query.mask_scores
    percentage of each head in each model that is on, displayed as a grid
    """
    plt.clf()
    sparsity_vals = np.zeros((layers - 6, heads))
    
    cnt = 0
    for layer in range(6, 12):
        for head in range(heads):
            q, k, v = model.get_mask_sparsity_head_qkv(layer, head, density=True)
            q = round(q * 100, 4) 
            k = round(k * 100, 4) 
            v = round(v * 100, 4) 
            sparsity_vals[6 - 1 - cnt, head] = (q + k + v) / 3
        cnt += 1
            
    plt.rcParams['font.size'] = '20'
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    print(sparsity_vals)
    im = ax.imshow(sparsity_vals, aspect='equal', cmap='viridis')
    threshold = ((im.get_array().max() - im.get_array().min()) / 2) + im.get_array().min()
    for (j,i),label in np.ndenumerate(sparsity_vals):
        if label > threshold:
            color = 'black'
        else:
            color = 'white'
        ax.text(i, j, round(label, 2), ha='center', va='center', size='small', color=color)

    ax.set_xticks(np.arange(heads))
    ax.set_yticks(np.arange(layers - 6))
    ax.set_xticklabels(np.arange(start=1, stop=heads + 1, step=1))
    ax.set_yticklabels(np.arange(start=layers, stop=6, step=-1))
    ax.set_title(model_name)
    plt.xlabel('Heads')
    plt.ylabel('Layers')
    
    for pcm in ax.get_children():
        if isinstance(pcm, plt.cm.ScalarMappable):
            break
    im_ratio = sparsity_vals.shape[0]/sparsity_vals.shape[1]
    plt.colorbar(im, ax=ax, fraction=0.046*im_ratio, pad=0.02, shrink=0.85)
    fig.tight_layout()
    if block:
        plt.show()
    else:
        fname = path + '_{}.pdf'.format('head_density')
        plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.show(block=False)
        plt.close()
            
def visualize_qkv_density(model, path, block=False, layers=12, model_name=""): 
    """
    bert_lm.bert.encoder.layer.0.attention.self.query.mask_scores
    percentage of each head in each model that is on, displayed as a grid
    """
    plt.clf()
    sparsity_vals = np.zeros((layers - 6, 3))
    cnt = 0
    for layer in range(6, 12):
        q, k, v = model.get_mask_sparsity_head_qkv(layer, head=None, density=True)
        sparsity_vals[(6 - 1) - cnt][0] = round(q * 100, 4) 
        sparsity_vals[(6 - 1) - cnt][1] = round(k * 100, 4) 
        sparsity_vals[(6 - 1) - cnt][2] = round(v * 100, 4) 
        cnt += 1
        
    plt.rcParams['font.size'] = '20'
    fig, ax = plt.subplots(1,1, figsize=(10, 7))
    im = ax.imshow(sparsity_vals, aspect='equal', cmap='viridis')
    threshold = ((im.get_array().max() - im.get_array().min()) / 2) + im.get_array().min()
    for (j,i),label in np.ndenumerate(sparsity_vals):
        if label > threshold:
            color = 'black'
        else:
            color = 'white'
        ax.text(i, j, round(label, 2), ha='center', va='center', size='small', color=color)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(layers - 6))
    ax.set_xticklabels(np.array(["$W_q$", "$W_k$", "$W_v$"]))
    ax.set_yticklabels(np.arange(start=layers, stop=6, step=-1))
    ax.set_title(model_name)
    plt.ylabel('Layers')
    
    im_ratio = sparsity_vals.shape[0]/sparsity_vals.shape[1]
    plt.colorbar(im, ax=ax, fraction=0.046*im_ratio, pad=0.02, shrink=0.85)
    fig.tight_layout()
    if block:
        plt.show()
    else:
        fname = path + '_{}.pdf'.format('qkv_density')
        plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.show(block=False)
        plt.close()

def visualize_dense_density(model, path, block=False, layers=12, model_name=""): 
    """
    bert_lm.bert.encoder.layer.0.attention.self.query.mask_scores
    percentage of each head in each model that is on, displayed as a grid
    """
    plt.clf()
    sparsity_vals = np.zeros((layers - 6, 3))
    cnt = 0
    for layer in range(6, 12):
        a, b, c = model.get_mask_sparsity_mlp(layer, density=True)
        sparsity_vals[(6  - 1) - cnt, 0] = round(a * 100, 4) 
        sparsity_vals[(6  - 1) - cnt, 1] = round(b * 100, 4) 
        sparsity_vals[(6  - 1) - cnt, 2] = round(c * 100, 4) 
        cnt += 1
    
    fig, ax = plt.subplots(1,1, figsize=(10, 7))
    im = ax.imshow(sparsity_vals, aspect='equal', cmap='viridis')
    threshold = ((im.get_array().max() - im.get_array().min()) / 2) + im.get_array().min()
    for (j,i),label in np.ndenumerate(sparsity_vals):
        if label > threshold:
            color = 'black'
        else:
            color = 'white'
        ax.text(i, j, round(label, 2), ha='center', va='center', size='small', color=color)
    
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Att-Out", "FF-1", "FF-2"])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    ax.set_yticks(np.arange(layers - 6))
    ax.set_yticklabels(np.arange(start=layers, stop=6, step=-1))
    plt.ylabel('Layers')
    ax.set_title(model_name)
    
    for pcm in ax.get_children():
        if isinstance(pcm, plt.cm.ScalarMappable):
            break
    im_ratio = sparsity_vals.shape[0]/sparsity_vals.shape[1]
    plt.colorbar(im, ax=ax, fraction=0.046*im_ratio, pad=0.02, shrink=0.85)
    
    fig.tight_layout()
    if block:
        plt.show()
    else:
        fname = path + '_{}.pdf'.format('ff_density')
        plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.show(block=False)
        plt.close()


def visualize_all_layers_density_multi_model(model_dict, path, block=False, neuron_specific=False, neuron_type=""):     
    plt.clf()
    fig, ax = plt.subplots(1,1, figsize=(25, 20))
    plt.rcParams['font.size'] = '20'
    
    df_dict = {}
    new_index = None
    
    for model_name, model in model_dict.items():
        model_modules = model.named_modules() 
        model_produced_mask = {k: v.produce_mask_reshaped() for k, v in model_modules if type(v) == MaskedHuggingfaceConv1D}
        model_mask_density = {}
        if neuron_specific:
            print(neuron_type)
            if neuron_type == "in_features":
                for k, v in model_produced_mask.items():
                    out_feats_num = v.shape[1]
                    pruned_in_feats = 0.0
                    total_in_feats = 0.0
                    for in_feat in v:
                        assert (len(in_feat) == out_feats_num)
                        if torch.equal(in_feat, torch.ones(out_feats_num)):
                            pruned_in_feats += 1.0
                        total_in_feats += 1.0
                    model_mask_density[k] = pruned_in_feats / total_in_feats
                print(in_feat)
                print(torch.ones(out_feats_num))
            elif neuron_type == "out_features":
                for k, v in model_produced_mask.items():
                    in_feats_num = v.shape[0]
                    pruned_out_feats = 0.0
                    total_out_feats = 0.0
                    for out_feat in v.t():
                        assert (len(out_feat) == in_feats_num)
                        if torch.equal(out_feat, torch.ones(in_feats_num)):
                            pruned_out_feats += 1.0
                        total_out_feats += 1.0
                    model_mask_density[k] = pruned_out_feats / total_out_feats
                print(out_feat)
                print(torch.ones(in_feats_num))
            else:
                print("Doesn't match the right neuron pruning types!")
        else:
            model_mask_density = {k: ((v == 1.0).sum().item()) / v.numel() for k, v in model_produced_mask.items()}
        
        model_mask_density = {k: round(v * 100, 4) for k, v in model_mask_density.items()}
        edited_model_name = model_name.replace(".n.02", "").replace(".n.01", "")
        df_dict[edited_model_name] = model_mask_density
        new_index = list(model_produced_mask.keys())
        
    df = pd.DataFrame.from_dict(df_dict)
    new_index.reverse()
    df = df.reindex(new_index)
    data = df.to_numpy()
    
    def layer_name_mapper(x):
        my_str = ""
        split_name = x.split(".")[3:]
        layer_num = str(int(split_name[0]) + 1)
        my_str += f"L{layer_num}"
        layer_type = ".".join(split_name[1:])
        if layer_type == "attn.c_attn":
            my_str += " Attn-$W_q,W_k,W_v$"
        elif layer_type == "attn.c_proj":
            my_str += " Attn-Out"
        elif layer_type == "mlp.c_fc":
            my_str += " FF-1"
        elif layer_type == "mlp.c_proj":
            my_str += " FF-2"
        return my_str
    
    columns = df.columns.values.tolist()
    rows = [layer_name_mapper(i) for i in df.index.values.tolist()]
    im = ax.imshow(data, aspect='equal', cmap='viridis', vmax=4.5) #, extent=(-0.5, len(columns) - 0.5, -0.5, len(rows) - 0.5))

    # # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    threshold = ((im.get_array().max() - im.get_array().min()) / 2) + im.get_array().min()
    # # Loop over data dimensions and create text annotations.
    for i in range(len(rows)):
        for j in range(len(columns)):
            label = round(data[i, j],2)
            if label > threshold:
                color = 'black'
            else:
                color = 'white'
            ax.text(j, i, label, ha='center', va='center', size='small', color=color)
            
    for pcm in ax.get_children():
        if isinstance(pcm, plt.cm.ScalarMappable):
            break

    ax.set_title("Density % for all layer masks") #, pad=30)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.8)
        
    if block:
        plt.show()
    else:
        prefix = "all_layer_density"
        if neuron_specific:
            fname = os.path.join(path, f"{prefix}_{neuron_type}.pdf")
        else:
            fname = os.path.join(path, f"{prefix}.pdf")
        plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.show(block=False)
        plt.close()

def calculate_tensor_subsets(mask1, mask2, mask3):
    return {
        "100": torch.logical_and(mask1, torch.logical_not(torch.logical_or(mask2, mask3))).sum().item(),
        "010": torch.logical_and(mask2, torch.logical_not(torch.logical_or(mask1, mask3))).sum().item(),
        "110": torch.logical_and(torch.logical_and(mask1, mask2), torch.logical_not(mask3)).sum().item(),
        "001": torch.logical_and(mask3, torch.logical_not(torch.logical_or(mask1, mask2))).sum().item(),
        "101": torch.logical_and(torch.logical_and(mask1, mask3), torch.logical_not(mask2)).sum().item(),
        "011": torch.logical_and(torch.logical_and(mask2, mask3), torch.logical_not(mask1)).sum().item(),
        "111": torch.logical_and(torch.logical_and(mask1, mask2), mask3).sum().item(),
    }

def jaccard_index_for_three(mask1, mask2, mask3):
    sum_per_elem = mask1.float() + mask2.float() + mask3.float()
    intersection = (sum_per_elem == 3).sum().item()
    union = (sum_per_elem >= 1).sum().item()
    if union == 0 or intersection == 0:
        return 0.0
    return intersection / float(union)

def plot_jaccard_heatmap(df_dict, distance_name, save_path, filename, title, column_formatter=None, x_rotation=30):
    def layer_name_mapper(x):
        my_str = ""
        split_name = x.split(".")[3:]
        layer_num = str(int(split_name[0]) + 1)
        my_str += f"L{layer_num}"
        layer_type = ".".join(split_name[1:])
        if layer_type == "attn.c_attn":
            my_str += " Attn-$W_q,W_k,W_v$"
        elif layer_type == "attn.c_proj":
            my_str += " Attn-Out"
        elif layer_type == "mlp.c_fc":
            my_str += " FF-1"
        elif layer_type == "mlp.c_proj":
            my_str += " FF-2"
        return my_str

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(25, 20))
    plt.rcParams["font.size"] = "20"

    df = pd.DataFrame.from_dict(df_dict)
    new_index = df.index.values.tolist()
    new_index.reverse()
    df = df.reindex(new_index)
    data = df.to_numpy()

    columns = df.columns.values.tolist()
    if column_formatter is not None:
        columns = [column_formatter(col) for col in columns]
    rows = [layer_name_mapper(i) for i in df.index.values.tolist()]
    im = ax.imshow(data, aspect="equal", cmap="viridis")

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)

    plt.setp(ax.get_xticklabels(), rotation=x_rotation, ha="right", rotation_mode="anchor")

    threshold = ((im.get_array().max() - im.get_array().min()) / 2) + im.get_array().min()
    for i in range(len(rows)):
        for j in range(len(columns)):
            label = round(data[i, j], 2)
            color = "black" if label > threshold else "white"
            ax.text(j, i, label, ha="center", va="center", size="x-small", color=color)

    ax.set_title("{} {}".format(distance_name, title), pad=30)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.8)
    fig.savefig(os.path.join(save_path, filename), format="pdf", bbox_inches="tight")
    plt.close()

def plot_venn3_overlap(subset_count, labels, title, save_path, filename, set_colors=None, font_size=7):
    try:
        from matplotlib_venn import venn3, venn3_circles
    except ImportError as exc:
        raise ImportError("matplotlib_venn is required for Venn diagrams.") from exc

    plt.rcParams.update({"font.size": font_size})
    venn3(
        subset_count,
        set_colors=set_colors or ["#51954D", "#1F71DA", "#C77CC4"],
        set_labels=labels,
        subset_label_formatter=lambda x: "{:,.0f}".format(x),
        alpha=0.75,
    )
    venn3_circles(subset_count, lw=0.7)
    plt.title(title)
    plt.savefig(os.path.join(save_path, filename), format="pdf", bbox_inches="tight")
    plt.close()




