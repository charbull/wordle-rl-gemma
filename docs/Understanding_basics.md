# Understanding Policy Optimization

This section contains my notes to understand better how Policy Optimizations work:
* Direct Policy Optimization [DPO paper](https://arxiv.org/pdf/2305.18290)
* Group Sequence Policy Optimization [GSPO paper](https://arxiv.org/pdf/2507.18071).
* Group Relative Policy Optimization [DeepSeek GRPO related paper](https://arxiv.org/abs/2402.03300)
* Group Sequence Policy Optimization [GSPO paper](https://arxiv.org/pdf/2507.18071).

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/dpo/3.webp?123" width=600px>

## References
I read, borrowed, and copied code from these references.

* [understanding-the-math-behind-grpo-deepseek-r1-zero](https://medium.com/yugen-ai-technology-blog/understanding-the-math-behind-grpo-deepseek-r1-zero-9fb15e103a0a)
* [The Illustrated Deepseek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)
* [Huggingface LLM RL course](https://huggingface.co/learn/llm-course/chapter12/1?fw=pt)
* [unsloth: RL Slides](https://docs.google.com/presentation/d/1Jh5p_JDXt4eLD0ireaHJjJNpzqSF8E1WTwIHeojyjNU/edit?usp=sharing)
* [unsloth: RL Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
* [Magistral paper](https://mistral.ai/static/research/magistral.pdf)
* [Gemma3 blog](https://huggingface.co/blog/gemma3)
* [TRL GRPO with HF](https://colab.sandbox.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_llm_grpo_trl.ipynb)

## KL Divergence and Cross Entropy

The first question I asked myself is why do we use KL divergence  instead of cross entropy in the policy optimization?
Lets go through the basics:

### Cross Entropy
The Core Idea is Measuring the "Surprise" or "Difference" between two proability distributions.

If we have:

* The True Distribution ($P$): This represents the actual, real-world probabilities of different events or classes. In supervised learning, this is our ground truth label.
* The Predicted Distribution ($Q$): This is what the model predicts for the probabilities of those same events or classes.

Cross-entropy measures how different the predicted distribution $Q$ is from the true distribution $P$.
* If $Q$ is very similar to $P$, the cross-entropy will be low.
* If $Q$ is very different from $P$, the cross-entropy will be high.


*   In typical SFT for LLMs, cross-entropy loss (often simplified to $$-\log P(\text{correct token})$$ or $$-\log P(\text{correct class})$$) is used when we have a ground truth target.
*   For SFT, the "ground truth" is the next token in a human-written demonstration. We are trying to make the model $\pi_\theta$ predict that specific token with high probability.
    *   Example: $$\text{Loss} = -\log \pi_\theta(y_{\text{target}} | x)$$ (simplified for a whole sequence).
*   Here, $\pi_\theta$ is being directly compared to a fixed, known "correct" output. The goal is for the model's predicted distribution $Q$ to match the true distribution $P$

### 2. The Goal of DPO/GRPO: Relative Preference & Regularization

*  DPO/GRPO don't operate with a single "correct" answer. They work with **preference pairs**: $y_c$ (chosen) is preferred over $y_r$ (rejected) for a given prompt $x$.
*   The objective isn't just to maximize the probability of $y_c$. It's to:
    1.   <font color='red'> Maximize the probability of </font> $y_c$ *relative to* $y_r$.
    2.  Do so in a way that doesn't drastically change the model from its initial, well-behaved state (the reference model $\pi_{\text{ref}}$, usually the SFT model).
*   <font color='red'> This "not drastically changing" part is crucial</font>. It's where the idea of **KL Divergence** becomes central. In the original Reinforcement Learning from Human Feedback (RLHF) formulation, the objective is to maximize reward *while constraining the KL divergence between the policy $\pi_\theta$ and the reference policy $\pi_{\text{ref}}$*:
    *   RLHF Objective (conceptual): $$\text{Maximize } E[\text{Reward}(y)] \text{ subject to } D_{KL}(\pi_\theta || \pi_{\text{ref}}) \le \beta_{KL}$$
    *   <font color='red'>This KL constraint prevents "reward hacking" (where the model finds trivial solutions to get high reward while destroying its general language abilities) by keeping it close to </font> $\pi_{\text{ref}}$.




### 3. DPO's Key Insight: Implicit Reward and the KL Connection

*   DPO provides a way to achieve the goals of KL-regularized RLHF *without* explicitly training a reward model or performing reinforcement learning.
*   It shows that the optimal policy $\pi^*$ for the KL-constrained reward maximization problem above can be written as: $$ \pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta_{DPO}} r(x,y)\right)$$
where $r(x,y)$ is the (unknown) true reward function, $\beta_{DPO}$ is a scaling factor (like temperature), and $Z(x)$ is a partition function (normalizer).
*   Rearranging this, we can express the reward function in terms of the policies:
$$r(x,y) \propto \log\left(\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}\right) $$
Ignoring $\beta_{DPO}$ and $Z(x)$ for proportionality as $Z(x)$ is constant for a given $x$ when comparing two responses).
*   Effectively, the "advantage" or implicit reward of a response $y$ by policy $\pi_\theta$ over $\pi_{\text{ref}}$ is proportional to:
$\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$ (which is $\log\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)$)
*   This **log-probability ratio** is a core component that also appears in the definition of KL divergence:
$$ D_{KL}(\pi_\theta || \pi_{\text{ref}}) = \sum_y \pi_\theta(y|x) \log\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right) $$

### 4. How the DPO/GRPO Loss Uses This KL-Related Term

The DPO/GRPO loss is (for a single chosen $y_c$ and rejected $y_r$ for simplicity, GRPO averages this over multiple $y_r$):
$$
\begin{aligned}
\text{Loss} = -\log\Bigg(\sigma\Bigg( \beta_{DPO} \bigg[ & \left(\log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x)\right) \\
& - \left(\log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x)\right) \bigg] \Bigg)\Bigg)
\end{aligned}
$$
(where $\sigma$ denotes the sigmoid function)

*   The terms $$(\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x))$$ are precisely these implicit reward estimates, derived from the logic connecting to KL-regularized RL.
*   <font color='red'>  The loss function aims to maximize the difference between the implicit reward of  </font> $y_c$ and $y_r$.
*   By optimizing these log-probability ratios, DPO/GRPO implicitly finds a policy $\pi_\theta$ that is optimal under the Bradley-Terry model (which models preferences based on reward differences) *and* inherently respects the KL divergence constraint from the original RLHF problem.

### Why KL Divergence is "Under the Hood" and Not Just Cross-Entropy:

1.  **No Absolute Target:**  We aren't saying $y_c$ is the 100% correct answer to be matched via cross-entropy. We're saying it's *better than* $y_r$. This is a relative judgment.
2.  **Reference Model $\pi_{\text{ref}}$ is Crucial:**
    *   The $\pi_{\text{ref}}$ is not just an arbitrary starting point; it's the **anchor**. The DPO/GRPO loss structure encourages $\pi_\theta$ to learn the preferences while staying "close" (in a KL-divergence sense) to $\pi_{\text{ref}}$.
    *   If $\pi_{\text{ref}}$ already gives $y_c$ a high probability and $y_r$ a low one, $\pi_\theta$ doesn't need to change much for that pair.
    *   If $\pi_{\text{ref}}$ gives $y_r$ a high probability (i.e., the SFT model prefers the bad response), $\pi_\theta$ needs to work harder (increase its log-probability ratio for $y_c$ and decrease it for $y_r$) to satisfy the preference.
3.  **Implicit KL Regularization:**
    *   The DPO/GRPO loss formulation implicitly optimizes for a reward model that explains the human preferences while ensuring the resulting policy $\pi_\theta$ doesn't stray too far (in terms of KL divergence) from the reference SFT policy $\pi_{\text{ref}}$.
    *  <font color='red'>This regularization is key to stable and effective alignment</font>. If you just used a cross-entropy-like loss to maximize $P(y_{c})$ and minimize $P(y_r)$ without the $\pi_{\text{ref}}$ normalization, the model could easily overfit to the preference data and suffer from "catastrophic forgetting" of its general abilities

### In Essence:

*   **Cross-entropy** is suitable when you want your model's distribution $Q$ to directly match a fixed target distribution $P$ (e.g., one-hot labels in SFT or classification).
*   **KL divergence** (or objectives derived from it, as in DPO/GRPO) is suitable when you want to optimize a model $\pi_\theta$ relative to a reference $\pi_{\text{ref}}$ based on preferences or rewards, implicitly controlling how far $\pi_\theta$ moves from $\pi_{\text{ref}}$.

The GRPO/DPO loss structure cleverly bakes in this KL-regularized reward optimization directly from preference data, without needing to explicitly train a reward model or compute KL divergence during the loss calculation. The $\log(\pi_\theta(y|x)/\pi_{\text{ref}}(y|x))$ terms are the signature of this underlying KL-divergence-aware objective.

### Magistral RL changes

The [Magistral paper](https://arxiv.org/pdf/2506.10910) outlines few modifications for GRPO:

* KL divergence is eliminated: The KL divergence penalty constrains the online policy from deviating too much from a reference policy, helping to maintain alignment with the initial model. However, in GRPO, the policy diverges substantially regardless, and maintaining a copy of the reference model
for KL computation incurs a compute cost we find unjustified. We remove the KL penalty entirely.

* Loss normalization. To avoid introducing length biases between generations in one group, we normalize the loss by first adding token-wise loss for all tokens and all generations and then dividing by the total length of generations in the group $\sum_{i=1}^G |o_i|$. Why they do it? In our batch, some generated responses are long and some are short. If we just average the loss per sequence, the longer sequences will have a much larger impact on the final gradient because their loss is summed over more tokens. How they do it? They change the final loss calculation from a mean-per-sequence to a mean-per-token. This is done by taking a length-weighted average of the per-sequence losses. Each token across the entire batch contributes equally to the loss, regardless of which sequence it came from.



    1. The Problem: "Length Biases"

    When training an AI with Reinforcement Learning (RL), the model generates an answer (a "generation") and receives a "reward" or "loss" (a penalty).

    Imagine the model is asked a complex math problem. It makes two attempts (two "generations" in a "group"):

    *   **Generation 1 (Short & Wrong):** "The answer is 5."
        *   *Length:* 4 tokens.
        *   *Loss (Penalty):* Let's say it gets a high penalty of **-10**.

    *   **Generation 2 (Long & Wrong):** "Let me think... if I take the first number and multiply it by the second, and then add the third, I believe the final result, after careful consideration, is probably 5."
        *   *Length:* 30 tokens.
        *   *Loss (Penalty):* Because this answer is also wrong, it should also get a high penalty. But if the penalty is calculated for each token, the total penalty might be huge, say **-75**.

    **The Bias:** Without normalization, the training algorithm sees a penalty of -75 for the long answer and only -10 for the short one. It might incorrectly conclude: **"Generating long answers is extremely bad! I should always be brief, even if I'm wrong."**

    This is a "length bias." The model is being unfairly punished for its verbosity, not its incorrectness. This is especially bad for "reasoning models" which are *supposed* to generate long, step-by-step chains of thought.

    2. The Solution: "Loss Normalization"

    The paper's method fixes this by calculating the **average loss per token**.

    Here's how it works, using the same example:

      **"first adding token-wise loss for all tokens and all generations"**
        *   They take the total penalty from all attempts.
        *   **Total Loss** = (Loss from Gen 1) + (Loss from Gen 2) = (-10) + (-75) = **-85**.

      **"and then dividing by the total length of generations in the group"**
        *   They count the total number of tokens (words) across all attempts.
        *   **Total Length** = (Length of Gen 1) + (Length of Gen 2) = 4 + 30 = **34 tokens**.

      **Calculate the Normalized Loss:**
        *   **Normalized Loss** = Total Loss / Total Length = -85 / 34 = **-2.5 per token**.


* Advantage normalization. We estimate the advantage of each token simply as
$\hat{A}_{i, t} = \hat{A}_i = r_i - \mu$, where $\mu$ is the mean of rewards within a group.
Following [andrychowicz2020](https://arxiv.org/abs/2006.05990), we additionally normalize the advantages in each minibatch as $\hat{A}_{i, t}^{\text{norm}} = (\hat{A}_i - \hat{A}^{\text{mean}}) / \hat{A}^{\text{std}}$ where $\hat{A}^{\text{mean}}$ and $\hat{A}^{\text{std}}$ are the sequence-wise mean and standard deviation of the advantages $\hat{A}_i$ in a minibatch.

### Appendix Loss function
How did we get to this loss function?

$$
\begin{aligned}
\text{Loss} = -\log\Bigg(\sigma\Bigg( \beta_{DPO} \bigg[ & \left(\log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x)\right) \\
& - \left(\log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x)\right) \bigg] \Bigg)\Bigg)
\end{aligned}
$$


The derivation combines a few key ideas:
1.  **Modeling Preferences with a Latent Reward Model:** We assume there's some underlying (latent) reward function $r(x,y)$ that humans use to judge responses.
2.  **Bradley-Terry Model:** This model is commonly used for pairwise comparisons. It states that the probability of item $A$ being preferred over item $B$ is a logistic function of the difference in their underlying "scores" or "strengths."
3.  **DPO's Connection to RLHF:** DPO showed that the optimal policy $\pi^*$ for a KL-regularized reward maximization objective (like in RLHF) can be expressed in terms of a reference policy $\pi_{\text{ref}}$ and this latent reward $r(x,y)$.
4.  **Maximum Likelihood Estimation:** We want to find model parameters that maximize the likelihood of observing the human preference data.

Here's the step-by-step derivation:

**Step 1: The Bradley-Terry Model for Preferences**

Let $r(x,y)$ be the (unknown) true reward score for a response $y$ given prompt $x$. The Bradley-Terry model posits that the probability of a human preferring $y_c$ (chosen) over $y_r$ (rejected) is given by a sigmoid function of the difference in their rewards: $\sigma(\text{score\_difference})$

$$ P(y_c \succ y_r | x) = \sigma(\beta_{DPO} [r(x, y_c) - r(x, y_r)]) $$

Where:
*   $y_c \succ y_r$ means $y_c$ is preferred over $y_r$.
*   $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function.
*   $\beta_{DPO}$ is a hyperparameter (like an inverse temperature) that scales the reward difference. A higher $\beta_{DPO}$ means the preferences are more deterministic based on the reward difference.

 **Converting Scores/Values to Probabilities:**
    *   In DPO/GRPO, the term inside the sigmoid function:
        $\beta_{DPO} \left[ (\log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x)) - (\log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x)) \right]$
        represents a "score difference" or "advantage" of the chosen response $y_c$ over the rejected response $y_r$, according to the current policy $\pi_\theta$ relative to the reference $\pi_{\text{ref}}$, scaled by $\beta_{DPO}$.
    *   This score difference can be any real number (large positive, large negative, or near zero).
    *   The logistic function (sigmoid) takes this score difference and converts it into a probability: the probability that $y_c$ is preferred over $y_r$.
        *   If the score difference is very large and positive (meaning $y_c$ is much "better" than $y_r$), the sigmoid output will be close to 1.
        *   If the score difference is very large and negative (meaning $y_c$ is much "worse" than $y_r$), the sigmoid output will be close to 0.
        *   If the score difference is close to 0 (meaning $y_c$ and $y_r$ are similarly "good"), the sigmoid output will be close to 0.5.


**Step 2: DPO's Insight – Expressing Reward in Terms of Policies**

The DPO paper shows (building on the optimal solution to KL-regularized RL) that the implicit reward can be related to the policy $\pi_\theta$ and the reference policy $\pi_{\text{ref}}$ as follows. The optimal policy $\pi^*$ is:

$$ \pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta_{DPO}} r(x,y)\right) $$

Solving for $r(x,y)$:

$$ \frac{1}{\beta_{DPO}} r(x,y) = \log\left(\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}\right) - \log Z(x) $$

So, $r(x,y) = \beta_{DPO} \left( \log\left(\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}\right) - \log Z(x) \right)$.

When we consider the *difference* in rewards $r(x, y_c) - r(x, y_r)$, the $\log Z(x)$ term (which is a normalizer constant for a given $x$) cancels out:

$$ r(x, y_c) - r(x, y_r) = \beta_{DPO} \left[ \log\left(\frac{\pi^*(y_c|x)}{\pi_{\text{ref}}(y_c|x)}\right) - \log\left(\frac{\pi^*(y_r|x)}{\pi_{\text{ref}}(y_r|x)}\right) \right] $$

Let's define a shorthand for the log-policy ratio term, which acts as our model's estimate of the scaled reward (or "advantage" over reference):
Let $\hat{r}_\theta(x,y) = \log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)$.
This term represents how much more (or less) likely the current policy $\pi_\theta$ makes the response $y$ compared to the baseline $\pi_{\text{ref}}$, in log-space.

So, the difference in *our model's estimate* of the (unscaled) rewards is:
$$ \hat{r}_\theta(x, y_c) - \hat{r}_\theta(x, y_r) = \left( \log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x) \right) - \left( \log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x) \right) $$

**Step 3: Plugging the Model's Reward Estimate into the Bradley-Terry Model**

We now substitute our model's estimate of the reward difference into the Bradley-Terry preference probability:

$$ P(y_c \succ y_r | x; \theta) = \sigma\left( \beta_{DPO} \left[ (\log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x)) - (\log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x)) \right] \right) $$

This equation now gives the probability of observing the preference $(y_c, y_r)$ given our current policy $\pi_\theta$, the reference policy $\pi_{\text{ref}}$, and the hyperparameter $\beta_{DPO}$.

**Step 4: Maximum Likelihood Estimation and the Loss Function**

To train our policy $\pi_\theta$, we want to maximize the likelihood of the observed human preferences. For a dataset $D = \{(x^{(i)}, y_c^{(i)}, y_r^{(i)})\}$ of preference pairs, we want to maximize:

$$ L(\theta) = \prod_{i} P(y_c^{(i)} \succ y_r^{(i)} | x^{(i)}; \theta) $$

Maximizing the likelihood is equivalent to maximizing the log-likelihood (because $\log$ is a monotonic function):

$$ \log L(\theta) = \sum_{i} \log P(y_c^{(i)} \succ y_r^{(i)} | x^{(i)}; \theta) $$

In machine learning, we typically *minimize* a loss function. So, we minimize the negative log-likelihood (NLL):

$$ \text{Loss}(\theta) = -\log L(\theta) = -\sum_{i} \log P(y_c^{(i)} \succ y_r^{(i)} | x^{(i)}; \theta) $$

For a single preference pair $(x, y_{c}, y_{r})$, the loss contribution is:

$$ \text{Loss}_{\text{single\_pair}} = -\log P(y_{c} \succ y_{r} | x; \theta) $$

Substituting the expression for $P(y\_c \succ y\_r | x; \theta)$ from Step 3:

$$
\begin{aligned}
\text{Loss}_{\text{single\_pair}} = -\log\Bigg(\sigma\Bigg( \beta_{DPO} \bigg[ & \left(\log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x)\right) \\
& - \left(\log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x)\right) \bigg] \Bigg)\Bigg)
\end{aligned}
$$

This is precisely the DPO loss function.

**Summary of the Components:**

*   **$-\log(\cdot)$:** The negative log-likelihood part. Minimizing this maximizes the probability of the observed preferences.
*   **$\sigma(\cdot)$:** The sigmoid function, which maps the difference in "scores" to a probability between 0 and 1, as per the Bradley-Terry model.
*   **$\beta_{DPO}$:** A hyperparameter (temperature) that scales the difference. It controls how strongly the model should adhere to the preferences.
*   **$(\log \pi_\theta(y_c|x) - \log \pi_{\text{ref}}(y_c|x))$**: The "score" or "advantage" of the chosen response under the current policy $\pi_\theta$ relative to the reference policy $\pi_{\text{ref}}$. We want to make this term larger.
*   **$(\log \pi_\theta(y_r|x) - \log \pi_{\text{ref}}(y_r|x))$**: The "score" or "advantage" of the rejected response. We want to make this term smaller.
*   **The subtraction between these two "score" terms**: This is the core difference that the sigmoid acts upon. We want this difference to be positive and large, indicating that the chosen response is significantly "better" than the rejected one according to our policy (relative to the reference).

For GRPO, you would compute this loss for each $(y_c, y_{r_j})$ pair within a group (one chosen, multiple rejected) and then average these losses.


## Putting everything together: GRPO in the DeepSeek paper

$$
\begin{aligned}
J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \Bigg[ & \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg( \min \Bigg( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \\
& \text{clip} \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \Bigg) \Bigg) \\
& - \beta D_{KL}(\pi_\theta || \pi_{ref}) \Bigg]
\end{aligned}
$$

Here's the dissection:

1.  **$J_{GRPO}(\theta)$**:
    *   The objective function to be maximized.
    *   $\theta$ are the parameters of the policy $\pi_\theta$ being optimized.

2.  **$\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} [\dots]$**:
    *   $\mathbb{E}$: Expectation over samples.
    *   $q \sim P(Q)$: A query/prompt $q$ is sampled.
    *   The remaining part suggests that for a single query $q$, a **group or batch of $G$ different output sequences $o_i$ are sampled using the old policy $\pi_{\theta_{old}}$**. This is a key difference from the standard PPO notation which typically shows one $o$ per $q$ in the outer expectation. This "G" might hint at the "Grouped" aspect of "GRPO," but the original GRPO from DeepSeek is a DPO variant and doesn't directly use this PPO-style objective.

3.  **$\frac{1}{G} \sum_{i=1}^{G} \dots$**:
    *   This is an **average over the $G$ sampled output sequences** for the given query $q$.

4.  **$\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \dots$**:
    *   For each of the $G$ sequences $o_i$, this is an average over all timesteps (tokens) $t$ within that sequence.

5.  **$\min \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right)$**:
    *   This is the **standard PPO-Clip term**, applied at the token level for each sequence $o_i$ in the group.
    *   $\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$: The probability ratio $r_{i,t}(\theta)$ for token $o_{i,t}$ of sequence $o_i$.
    *   A is the advantage estimate for token $o_{i,t}$ of sequence $o_i$. This advantage would likely be derived from a reward model that scores the entire sequence $o_i$.
    *   $\epsilon$: The clipping hyperparameter.

6.  **$- \beta D_{KL}(\pi_\theta || \pi_{ref})$**:
    *   This is an **explicit KL divergence penalty term**.
    *   $D_{KL}(\pi_\theta || \pi_{ref})$: The Kullback-Leibler divergence between the current policy $\pi_\theta$ and some **reference policy $\pi_{ref}$**.
        *   **Crucially, this $\pi_{ref}$ is distinct from $\pi_{\theta_{old}}$**.
        *   $\pi_{\theta_{old}}$ is the policy used for data collection in the current iteration.
        *   $\pi_{ref}$ is typically a fixed, trusted policy, often the original SFT model. This term aims to prevent $\pi_\theta$ from straying too far from this trusted reference, maintaining its general capabilities and style.
    *   $\beta$: A hyperparameter that controls the strength of this KL penalty. A larger $\beta$ means a stronger penalty for deviating from $\pi_{ref}$.
    *   The negative sign means we are *subtracting* this penalty, so we are trying to *minimize* the KL divergence (since the overall objective $J$ is maximized).

**Interpretation and Potential Context:**

This objective function describes a PPO-like algorithm that:
1.  **Uses PPO's clipped surrogate objective** to encourage actions leading to higher advantage, while limiting the magnitude of policy updates.
2.  **Processes a group of $G$ responses per prompt** in its expectation calculation (this is unusual for the standard PPO formulation, which usually averages over state-action pairs without an explicit "group" like this for a single prompt, unless $G=1$). If $G > 1$, this could be a way to get a more stable estimate of the expected clipped advantage for a given prompt by sampling multiple continuations.
3.  **Includes an explicit KL divergence penalty** against a fixed reference policy $\pi_{ref}$. This is a common addition to PPO in RLAIF/RLHF to provide stronger regularization and prevent the policy from drifting too far from the base SFT model. The PPO clipping itself provides some regularization against $\pi_{\theta_{old}}$, but the explicit KL penalty against a more global $\pi_{ref}$ is often beneficial.

**How it differs from the actual GRPO (DeepSeek's DPO variant):**

*   **Core Mechanism:** The GRPO you asked about earlier is a **Direct Preference Optimization (DPO)** variant. DPO directly optimizes a policy based on preference pairs (chosen vs. rejected) and a loss function derived from the Bradley-Terry model and the relationship between optimal policies and reward functions. It *doesn't* use advantage estimates, explicit PPO-style clipping, or explicit KL divergence terms in its loss function (though the KL divergence is implicitly controlled).
*   **Data Source:** DPO/GRPO learn from fixed datasets of (prompt, chosen\_response, rejected\_response(s)). The formula you've given now implies data collection using $\pi_{\theta_{old}}$ and calculation of advantage estimates $\hat{A}_{i,t}$, which is characteristic of RL algorithms like PPO that typically use a reward model.
*   **"Grouped" aspect:**
    *   In DeepSeek's GRPO, "Grouped" refers to using one chosen response and *multiple* rejected responses for the *same prompt* in the preference data, and the loss is averaged over these chosen-rejected pairs.
    *   In the formula you just provided, the "Grouped" aspect ($G$ sequences) seems to be about sampling multiple outputs from $\pi_{\theta_{old}}$ for a given prompt during the PPO update phase.

**Conclusion for the new formula:**

The formula you've provided represents a sophisticated PPO-based objective that incorporates:
*   Standard PPO clipping for stable updates relative to the data-collection policy ($\pi_{\theta_{old}}$).
*   An additional, explicit KL divergence penalty to regularize the policy against a fixed reference model ($\pi_{ref}$).
*   A potential modification to sample and average over a group of $G$ responses generated by $\pi_{\theta_{old}}$ for each prompt, perhaps for variance reduction or a more robust gradient estimate.

This is a plausible objective for RLAIF/RLHF, but it's important to distinguish it from the DPO-based GRPO algorithm developed by DeepSeek. If this formula *is* being called "GRPO" by someone, it would be a different algorithm using the same acronym.

### Appendix on Clipping


1.  **`value_to_clip`**:
    $\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}$

    *   This is the **probability ratio**, often denoted as $r_t(\theta)$.
    *   $\pi_\theta(o_t|q, o_{<t})$: The probability of taking action $o_t$ (generating token $o_t$) in the current state (given query $q$ and previous tokens $o_{<t}$) according to the **current policy $\pi_\theta$** (the one being updated).
    *   $\pi_{\theta_{old}}(o_t|q, o_{<t})$: The probability of taking the *same* action $o_t$ in the *same* state according to the **old policy $\pi_{\theta_{old}}$** (the policy that was used to generate the experience/trajectory).
    *   **Interpretation of the ratio:**
        *   If $r_t(\theta) > 1$: The current policy makes this action more likely than the old policy did.
        *   If $r_t(\theta) < 1$: The current policy makes this action less likely than the old policy did.
        *   If $r_t(\theta) = 1$: The current policy assigns the same probability to this action as the old policy did.

2.  **`lower_bound`**:
    $1-\epsilon$

    *   $\epsilon$ (epsilon) is a small, positive hyperparameter (e.g., 0.1, 0.2).
    *   So, $1-\epsilon$ is a value slightly less than 1 (e.g., $1 - 0.2 = 0.8$).
    *   This sets the minimum allowed value for the probability ratio $r_t(\theta)$ in this clipped term.

3.  **`upper_bound`**:
    $1+\epsilon$

    *   This is a value slightly greater than 1 (e.g., $1 + 0.2 = 1.2$).
    *   This sets the maximum allowed value for the probability ratio $r_t(\theta)$ in this clipped term.

**How the `clip` function works here:**

The `clip` function takes the calculated probability ratio $r_t(\theta)$ and constrains it:

*   If $r_t(\theta) < (1-\epsilon)$: The output of `clip(...)` becomes $1-\epsilon$.
*   If $r_t(\theta) > (1+\epsilon)$: The output of `clip(...)` becomes $1+\epsilon$.
*   If $(1-\epsilon) \le r_t(\theta) \le (1+\epsilon)$: The output of `clip(...)` is just $r_t(\theta)$ itself (it's already within the bounds).

**What happens after the clipping:**

The result of this `clip` operation (the clipped probability ratio) is then multiplied by the advantage estimate $A_t$:

Clipped Term = $\text{clipped\_ratio} \times A_t$