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
# tokens = tokenized_data[:64]["tokens"]
# logits, cache = model.run_with_cache(tokens)

# temp_df = neuron_df.sort_values("vocab_var", )
# layers = temp_df.head(6).layer.values
# neurons = temp_df.head(6).neuron.values
# labels = temp_df.head(6).label.values
# top_neuron_acts = cache.stack_activation("post")[layers, :, :, neurons]
# histogram(top_neuron_acts.reshape(6, -1).T, marginal="box", barmode="overlay", histnorm="percent")
# print(labels)
# # %%
# token_df = nutils.make_token_df(tokens)

# for i in range(6):
#     token_df[labels[i]] = to_numpy(top_neuron_acts[i].flatten())
# px.histogram(token_df, x=labels, marginal="box", hover_name="context", barmode="overlay", histnorm="percent")
# # %%
# layer = 22
# neuron = 1544
# nutils.create_vocab_df(model.W_out[layer, neuron, :] @ model.W_U)
# %%
tokens = tokenized_data[128:192]["tokens"]
layer = 23
ni = 945
def neuron_set_hook(mlp_pre, hook, new_value, ni):
    mlp_pre[:, :, ni] = new_value
    return mlp_pre
baseline_loss = model(tokens, return_type="loss")
print(baseline_loss)
for new_value in [ -2., -1., 0., 1., 2., 3., 4.]:
    ablated_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("pre", layer, "mlp"), partial(neuron_set_hook, new_value=new_value, ni=ni))])
    print(f"{new_value}: {(ablated_loss/baseline_loss).item() - 1:.2%}")


# %%
ni = 945

baseline_logits = model(tokens, return_type="logits").cpu()
baseline_clps = model.loss_fn(baseline_logits, tokens, True)

new_logits = []
new_clps = []
for new_value in [-3.5, 1., 20.]:
    ablated_logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=[(utils.get_act_name("pre", layer, "mlp"), partial(neuron_set_hook, new_value=new_value, ni=ni))]).cpu()
    new_logits.append(ablated_logits)
    ablated_loss = model.loss_fn(ablated_logits, tokens)
    ablated_clps = model.loss_fn(ablated_logits, tokens, True)
    new_clps.append(ablated_clps)
    print(f"{new_value}: {(ablated_loss/baseline_loss).item() - 1:.2%}")
new_clps = torch.stack(new_clps)
new_logits = torch.stack(new_logits)
loss_delta = new_clps.reshape(new_clps.shape[0], -1) - baseline_clps.flatten()
token_df = nutils.make_token_df(tokens)
token_df = token_df.query("pos<255")
token_df["set_-3"] = to_numpy(loss_delta[0])
token_df["set_1"] = to_numpy(loss_delta[1])
temp_logits, temp_cache = model.run_with_cache(tokens, names_filter=utils.get_act_name("resid_post", 23))
token_df["uncertainty_proj"] = to_numpy(temp_cache["resid_post", 23][:, :-1, :] @ model.W_out[23, 945, :]).flatten()


nutils.show_df(token_df.sort_values("set_1", ascending=False).head(20))
nutils.show_df(token_df.sort_values("set_1", ascending=False).tail(20))
# %%
nutils.show_df(token_df.sort_values("uncertainty_proj", ascending=False).head(20))
nutils.show_df(token_df.sort_values("uncertainty_proj", ascending=False).tail(20))

# %%
# Show that KL to the LN bias distn improves as we increase the neuron's activation
def get_kl(P, Q, return_per_token=False):
    log_prob_P = P.log_softmax(dim=-1)
    log_prob_Q = Q.log_softmax(dim=-1)
    if return_per_token:
        return (log_prob_P.exp() * (log_prob_P - log_prob_Q)).sum(-1)
    else:
        return (log_prob_P.exp() * (log_prob_P - log_prob_Q)).sum(-1).mean()
def get_entropy(P, return_per_token=False):
    log_prob_P = P.log_softmax(dim=-1)
    if return_per_token:
        return -(log_prob_P.exp() * log_prob_P).sum(-1)
    else:
        return -(log_prob_P.exp() * log_prob_P).sum(-1).mean()
get_kl(baseline_logits, new_logits[2])

ln_bias_logits = model.b_U.cpu()
print("baseline", get_kl(ln_bias_logits, baseline_logits))
print("set -3.5", get_kl(ln_bias_logits, new_logits[0]))
print("set 1.", get_kl(ln_bias_logits, new_logits[1]))
print("set +20.", get_kl(ln_bias_logits, new_logits[2]))

print(get_entropy(baseline_logits))
print(get_entropy(new_logits[0]))
print(get_entropy(new_logits[1]))
print(get_entropy(new_logits[2]))

# %%
temp_logits, temp_cache = model.run_with_cache(tokens, names_filter=utils.get_act_name("post", 23))
token_df["act"] = to_numpy(temp_cache["post", 23][:, :-1, 945]).flatten()
token_df["relu_act"] = to_numpy(F.relu(temp_cache["post", 23][:, :-1, 945])).flatten()

token_df["entropy"] = get_entropy(baseline_logits[:, :-1], True).flatten()

px.scatter(token_df.query("act>0."), x="entropy", y="act", trendline="ols")

# %%
# %%
short_tokens = tokenized_data[:16]["tokens"]
short_logits, short_cache = model.run_with_cache(short_tokens)
histogram((short_cache["resid_post", 23] @ model.W_out[23, 945, :]).flatten())
# %%
