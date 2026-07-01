**The core idea: learned feature recalibration**

A standard conv layer applies the same spatial filter across all positions and treats all channels with equal implicit weight. The output at any point is a sum over all input channels — there's no mechanism for the network to say "channel 7 is more informative for this input than channel 3." SENet adds exactly that mechanism, but it does it _conditionally on the input_ — the excitation weights are a function of the current feature maps, not fixed parameters.

Formally, if your conv output is $U \in \mathbb{R}^{B \times C \times H \times W}$, SE computes:

$$\tilde{U}_c = \sigma\left(W_2 \cdot \delta\left(W_1 \cdot \frac{1}{HW}\sum_{h,w} U_c^{h,w}\right)\right) \cdot U_c$$

where $W_1 \in \mathbb{R}^{C/r \times C}$, $W_2 \in \mathbb{R}^{C \times C/r}$, $\delta$ is ReLU, $\sigma$ is sigmoid. The squeeze aggregates spatial information into a channel descriptor; the excitation produces a gating vector.

**Why squeeze with global average pooling?**

GAP computes $z_c = \frac{1}{HW} \sum_{h,w} U_c^{h,w}$, giving you a single scalar per channel. This is the channel's "average activation energy" — a rough summary of how much that filter fired across the spatial extent of the input. It works because conv filters tend to be semantically coherent: if you have a "diagonal edge detector" filter, its GAP value tells you how edge-heavy this image region is. The spatial _where_ is discarded, but the channel _what_ is preserved.

You could use GMP (global max pool) instead — SENet ablations show GAP wins slightly, probably because max is more sensitive to noise.

**Why the bottleneck MLP?**

Two reasons:

1. **Complexity control.** A direct $C \to C$ linear map would have $C^2$ parameters per SE block. With $r=16$ you get $2C^2/r$ — much cheaper, and in practice the bottleneck forces the network to learn a compressed representation of inter-channel relationships rather than memorizing per-channel lookup tables.

2. **It models channel dependencies.** The two-layer MLP with nonlinearity can represent non-linear interactions between channels — e.g., "suppress channel 3 when channel 7 and channel 12 are both active." A single linear layer can't capture that. This is the actual mechanism by which SE learns things like "if texture features are firing, down-weight color features."

**Why sigmoid instead of softmax?**

Softmax would force the channel weights to sum to 1 — zero-sum competition. Sigmoid allows each channel to be gated independently in $[0,1]$, so the network can say "amplify these three, suppress everything else" without the amplification being cancelled by the suppression arithmetically. It's a non-exclusive selection.

**Why does it help at residual connections specifically?**

In ResNets, the residual stream is an additive accumulation. If irrelevant features get added into the stream at every block, they accumulate and can't be easily undone downstream. SE at each block acts as a filter on what gets contributed to the residual — it's cheap online cleanup of the feature representation before it compounds.

**The linear algebra view**

If you think of $U_c$ as a set of $C$ vectors in $\mathbb{R}^{HW}$, the SE operation is a learned diagonal rescaling in channel space: $\tilde{U} = \text{diag}(s) \cdot U$ where $s \in [0,1]^C$ depends on $U$ itself. It's a content-adaptive projection that keeps the same basis (channels) but reweights them. A full attention mechanism would learn an arbitrary rotation in channel space — SE is the cheap approximation where you only allow axis-aligned rescaling.

That's also why it's so parameter-efficient: $2C^2/r$ parameters to buy you input-dependent channel selection, which is a qualitatively different capability from what conv weights alone can express.
