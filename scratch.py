# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model: HookedTransformer = HookedTransformer.from_pretrained("stanford-gpt2-medium-a")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab

# %%
vocab_df = pd.read_csv("vocab_df.csv")
display(vocab_df.head())
neuron_df = pd.read_csv("neuron_df.csv")
neuron_df["out_norm"] = to_numpy(model.W_out.norm(dim=-1).flatten())
neuron_df["in_norm"] = to_numpy(model.W_in.norm(dim=-2).flatten())
neuron_df["label"] = [f"L{l}N{n}" for l in range(n_layers) for n in range(d_mlp)]
display(neuron_df.head())
neuron_stats = pd.read_csv("neuron_stats.csv")
display(neuron_stats.head())
# %%
nutils.show_df(neuron_df.sort_values("vocab_var", ).head(50))

# %%
px.histogram(neuron_df, x="out_norm", marginal="box", hover_name="label")
# %%
data = load_dataset("stas/openwebtext-10k")
# %%
tokenized_data = utils.tokenize_and_concatenate(data["train"], model.tokenizer, max_length=256)
tokenized_data = tokenized_data.shuffle(42)
# %%
neuron_df.query("vocab_kurt>15").sort_values("vocab_skew")
# %%
# Not number
layer = 23
ni = 3111
temp_df = nutils.create_vocab_df(model.W_out[layer, ni] @ model.W_U)
# nutils.show_df(temp_df)
px.histogram(temp_df, x="logit", hover_name="token", marginal="box", title="# Not number").show()

# Not year
layer = 23
ni = 2260
temp_df = nutils.create_vocab_df(model.W_out[layer, ni] @ model.W_U)
# nutils.show_df(temp_df)
px.histogram(temp_df, x="logit", hover_name="token", marginal="box", title="# Not year").show()
# Not single digit number
layer = 23
ni = 2110
temp_df = nutils.create_vocab_df(model.W_out[layer, ni] @ model.W_U)
# nutils.show_df(temp_df)
px.histogram(temp_df, x="logit", hover_name="token", marginal="box", title="# Not single digit number").show()
# Not male pronoun
layer = 23
ni = 2774
temp_df = nutils.create_vocab_df(model.W_out[layer, ni] @ model.W_U)
# nutils.show_df(temp_df)
px.histogram(temp_df, x="logit", hover_name="token", marginal="box", title="# Not male pronoun").show()
# Not female pronoun
layer = 23
ni = 2330
temp_df = nutils.create_vocab_df(model.W_out[layer, ni] @ model.W_U)
# nutils.show_df(temp_df)
px.histogram(temp_df, x="logit", hover_name="token", marginal="box", title="# Not female pronoun").show()
# Not open paren
layer = 23
ni = 2042
temp_df = nutils.create_vocab_df(model.W_out[layer, ni] @ model.W_U)
# nutils.show_df(temp_df)
px.histogram(temp_df, x="logit", hover_name="token", marginal="box", title="# Not open paren").show()
# %%
# Not number
layer = 23
ni = 3111
temp_df = nutils.create_vocab_df(model.W_out[layer, ni] @ model.W_U)
# nutils.show_df(temp_df)
temp_df["is_number"] = [s.replace("·", " ").strip().isdigit() for s in temp_df.token]
px.histogram(temp_df, x="logit", hover_name="token", marginal="box", title="# Not number", color="is_number").show()

# %%
nutils.show_df(temp_df.query("is_number").head(100))
# %%
layer = 23
ni = 3111
tokens = tokenized_data[:64]["tokens"]
logits, cache = model.run_with_cache(tokens, names_filter=[utils.get_act_name("post", layer), utils.get_act_name("pre", layer),  utils.get_act_name("resid_mid", layer)])
neuron_acts = cache["post", layer][:, :, ni]
token_df = nutils.make_token_df(tokens, drop_last=True)
token_df["act"] = to_numpy(neuron_acts[:, :-1].flatten())
token_df["pre_act"] = to_numpy(cache["pre", layer][:, :, ni][:, :-1].flatten())
token_df["next_is_number"] = to_numpy([[i.strip().isdigit() for i in model.to_str_tokens(tokens[j, 1:])] for j in range(len(tokens))]).flatten()
token_df["curr_is_number"] = to_numpy([[i.strip().isdigit() for i in model.to_str_tokens(tokens[j, :-1])] for j in range(len(tokens))]).flatten()

resid_mid = cache["resid_mid", layer][:, :-1, :]
vocab_is_number = np.array([i.strip().isdigit() for i in model.to_str_tokens(torch.arange(d_vocab))])
logit_lens_probs = (resid_mid @ model.W_U).softmax(dim=-1)[:, :, vocab_is_number].sum(-1)
token_df["lens_num"] = to_numpy(logit_lens_probs.flatten().log())

nutils.show_df(token_df.sort_values("act", ascending=False).head(20))
# %%
px.histogram(token_df, x="act", color="next_is_number", marginal="box", barmode="overlay", histnorm="percent").show()
px.histogram(token_df, x="act", color="curr_is_number", marginal="box", barmode="overlay", histnorm="percent").show()
# %%
px.scatter(token_df.query("lens_num>-20"), x="pre_act", y="lens_num", marginal_x="histogram", marginal_y="histogram", color="next_is_number", trendline="ols").show()
# %%

token_df["mode"] = [f"next {'yes' if n else 'no'} curr {'yes' if c else 'no'}" for n, c in zip(token_df.next_is_number, token_df.curr_is_number)]
px.histogram(token_df, x="pre_act", color="mode", marginal="box", barmode="overlay", log_y=True, hover_name="context").show()
# %%
def delete_num_neuron(mlp_post, hook):
    mlp_post[:, :, 3111] = -F.relu(-mlp_post[:, :, 3111])
    return mlp_post
batch_size = 32
for i in range(5):
    tokens = tokenized_data[i*batch_size:(i+1)*batch_size]["tokens"]
    baseline_loss = model(tokens, return_type="loss")
    ablated_loss = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 23), delete_num_neuron)], return_type="loss")
    print(f"{(ablated_loss/baseline_loss - 1).item():.4%}")
# %%
def logits_to_prob_num(logits, return_per_token=False):
    per_token_num_prob = logits.softmax(dim=-1)[..., vocab_is_number].sum(-1)
    if return_per_token:
        return per_token_num_prob
    else:
        return per_token_num_prob.mean(), per_token_num_prob.log().mean()

gpt2_xl = HookedTransformer.from_pretrained("gpt2-xl")

batch_size = 16
baseline_prob_nums = []
true_prob_nums = []
abl_prob_nums = []
for i in tqdm.trange(20):
    tokens = tokenized_data[i*batch_size:(i+1)*batch_size]["tokens"]
    baseline_logits = model(tokens)
    baseline_prob_nums.append(logits_to_prob_num(baseline_logits, True))
    abl_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 23), delete_num_neuron)])
    abl_prob_nums.append(logits_to_prob_num(abl_logits, True))
    true_logits = gpt2_xl(tokens)
    true_prob_nums.append(logits_to_prob_num(true_logits, True))
    print(true_prob_nums[-1].mean(), baseline_prob_nums[-1].mean(), abl_prob_nums[-1].mean())
baseline_prob_nums = torch.cat(baseline_prob_nums, dim=0)
true_prob_nums = torch.cat(true_prob_nums, dim=0)
abl_prob_nums = torch.cat(abl_prob_nums, dim=0)
# %%
baseline_delta = baseline_prob_nums - true_prob_nums
abl_delta = abl_prob_nums - true_prob_nums
scatter(baseline_delta.flatten()[::107], abl_delta.flatten()[::107])
# %%
histogram((baseline_prob_nums - abl_prob_nums).flatten(), marginal="box", log_y=True)
# %%
abl_delta = abl_prob_nums.log() - baseline_prob_nums.log()
true_delta = true_prob_nums.log() - baseline_prob_nums.log()
temp_df = pd.DataFrame({"abl_delta": to_numpy(abl_delta.flatten()), "true_delta": to_numpy(true_delta.flatten())})
temp_df["abl_delta_abs"] = temp_df.abl_delta.abs()
px.scatter(temp_df.query("abl_delta_abs>1e-2 & true_delta>0"), x="abl_delta", y="true_delta", marginal_x="histogram", marginal_y="histogram", trendline="ols").show()
px.scatter(temp_df.query("abl_delta_abs>1e-2 & true_delta<0"), x="abl_delta", y="true_delta", marginal_x="histogram", marginal_y="histogram", trendline="ols").show()
# %%
def delete_num_neuron(mlp_post, hook):
    mlp_post[:, :, 3111] = -F.relu(-mlp_post[:, :, 3111])
    return mlp_post
layer = 23
ni = 3111
tokens = tokenized_data[:64]["tokens"]
logits, cache = model.run_with_cache(tokens, names_filter=[utils.get_act_name("post", layer), utils.get_act_name("pre", layer),  utils.get_act_name("resid_mid", layer)])
neuron_acts = cache["post", layer][:, :, ni]
token_df = nutils.make_token_df(tokens, drop_last=True)
token_df["act"] = to_numpy(neuron_acts[:, :-1].flatten())
token_df["pre_act"] = to_numpy(cache["pre", layer][:, :, ni][:, :-1].flatten())
token_df["next_is_number"] = to_numpy([[i.strip().isdigit() for i in model.to_str_tokens(tokens[j, 1:])] for j in range(len(tokens))]).flatten()
token_df["curr_is_number"] = to_numpy([[i.strip().isdigit() for i in model.to_str_tokens(tokens[j, :-1])] for j in range(len(tokens))]).flatten()

resid_mid = cache["resid_mid", layer][:, :-1, :]
vocab_is_number = np.array([i.strip().isdigit() for i in model.to_str_tokens(torch.arange(d_vocab))])
logit_lens_probs = (resid_mid @ model.W_U).softmax(dim=-1)[:, :, vocab_is_number].sum(-1)
token_df["lens_num"] = to_numpy(logit_lens_probs.flatten().log())


abl_logits = model.run_with_hooks(tokens, fwd_hooks=[(utils.get_act_name("post", 23), delete_num_neuron)])
abl_clps = model.loss_fn(abl_logits, tokens, True)
base_clps = model.loss_fn(logits, tokens, True)
abl_delta = abl_clps - base_clps
token_df["abl_delta"] = to_numpy(abl_delta.flatten())

nutils.show_df(token_df.sort_values("act", ascending=False).head(20))

px.histogram(token_df, x="abl_delta", color="next_is_number", marginal="box", barmode="overlay", log_y=True, hover_name="context").show()
px.histogram(token_df, x="abl_delta", color="curr_is_number", marginal="box", barmode="overlay", log_y=True, hover_name="context").show()
# %%
px.scatter(token_df.query("lens_num>-20"), x="abl_delta", y="lens_num", marginal_x="histogram", marginal_y="histogram", color="next_is_number", trendline="ols").show()
