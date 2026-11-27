# Mid-term report draft (plain-language version)

Fill in the title page and declaration yourself. Reword anything you copy, and check every
number against the logs before you submit. Items marked [verify] need checking.

---

## Title (suggestion)

Choosing Which Links to Check: Active, Task-Aware Link Acquisition for Node Classification

## Abstract

In many networks we only see part of the links, and checking whether a link exists costs time
or money. This project asks a simple question: if we can check only a limited number of
links, which ones should we check so that a chosen group of nodes gets classified correctly?
We compared several simple strategies with a new strategy that scores each possible link by
how much it would improve the classifier's confidence on those nodes. In simulated networks,
strategies that simply find many real links did not improve classification. The new strategy
closed about half of the possible improvement within 200 checks, while the simple strategies
closed at most about 13%. The result held on new random seeds and when links were hidden in
different ways. We also list the limits of this study and the work planned next.

---

## 1 Introduction: Background and Motivation

Many real networks are only partly known. Think of friendships on a social network, citations
between papers, or interactions between proteins. To use such a network for a task, for
example guessing the group or topic of a node, we need its links. But finding out whether two
nodes are really linked can be expensive. It may need an expert, a lab experiment or a paid
data request.

So we have a limited budget of "checks" (we call them queries). Each query asks one question:
"are these two nodes linked?" The answer is yes or no. The question for this project is which
pairs to ask about.

A natural idea is to ask about the pairs that are most likely to be linked. This is what link
prediction does. Early in this project we found that it is not enough. A strategy that found
more real links did not classify the nodes any better. A link can be real and still useless,
for example when it joins two nodes that are already classified, or two nodes that no known
label can reach.

This project therefore studies two ideas. The first is **task awareness**: choose links that
help classify the nodes we care about (the "target nodes"). The second is **active
learning**: choose one query at a time and use each answer to decide the next one.

---

## 2 Literature Review

**2.1 Active link prediction.** The closest work is ALPINE by Chen, Kang, Lijffijt and De
Bie [1]. It chooses which links to check using a network-embedding model of link probability
and a mathematical criterion for the expected benefit [verify]. Its simple selection rules
(highest probability, highest degree, PageRank) are the source of some of our baselines. The
difference is the goal. ALPINE aims to predict links well, while we aim to classify a chosen
set of nodes well. We did not implement its embedding method. ActiveLink [2] applies deep
active learning to link prediction in knowledge graphs [verify].

**2.2 Active learning on graphs.** Hu et al. [3] train a policy with reinforcement learning to
choose which nodes to label, and the policy can transfer to other graphs [verify]. They query
labels, while we query links. Bevilacqua et al. [4] learn which parts of a graph to select
for a graph neural network; this is not active learning, but it supports the idea of learning
what to select [verify]. ActiveReach [5] uses active learning to answer approximate
reachability questions on large graphs [verify]. Reachability, whether a node can be reached
from a labelled node, is also something we measure.

**2.3 Scoring the value of a query.** The usual idea is to pick the query that is expected to
help the model most. Roy and McCallum [6] call this expected error reduction. It comes from
the older field of Bayesian experimental design [7]. Cheaper approximations include BALD [8]
and influence functions [9]. Sun et al. [10] use a perturbation-based signal for graph active
learning [verify]. Our scorer is also a perturbation: it pretends the link exists and checks
how much the classifier's confidence changes.

**2.4 Surveys and framing.** Settles [11], Ren et al. [12] and Li et al. [13] survey active
learning. Li et al. group query strategies into uncertainty-based, representative, influence,
Bayesian and hybrid families [verify]. Wilder et al. [14] argue that data collection should
be judged by the final decision it supports. This is the idea behind our task-aware scoring.
Foster et al. [15] study learning to design experiments one after another, which is relevant
to future work on speeding up our method.

**2.5 What is missing.** We did not find work that picks links, one at a time, to improve node
classification on a chosen target set, and compares this with both link-prediction heuristics
and generic uncertainty sampling. This is the gap we address. (State this carefully; it is
your claim after your own search.)

---

## 3 Methodology

### 3.1 The problem in simple terms

There is a hidden network. We can see a random 15% of its links. A small share of the nodes
(15% or 40%) have known labels. We are given 15 **target nodes** whose labels are unknown. We
can ask up to B queries; the oracle replies whether a pair is linked, and a real link is
added to what we can see. After the queries, a classifier predicts the targets' labels. We
want as many of those predictions as possible to be right.

*Figure 1 (draw this): observed network -> choose a pair -> ask the oracle -> update the
network -> classify the targets -> repeat until the budget is used.*

### 3.2 Phase 1: Test networks

We generate artificial networks with a standard community model. Each has 90 nodes in 3
groups. Nodes in the same group are linked with probability 0.13 and nodes in different
groups with probability 0.006, giving about 186 links. Each node also has 12 numbers
(attributes), but the group only slightly affects them, so the attributes are nearly
useless. This is deliberate: the links must carry the information. A new network is made for
each random seed.

### 3.3 Phase 2: The classifier

We use label propagation. Each unlabelled node takes on the average belief of its neighbours,
and this repeats until the beliefs settle, with the labelled nodes held fixed. It is simple
and fast to recompute, which lets us test every candidate link exactly. A node that no label
can reach gets a fixed guess, and that guess counts in the accuracy.

### 3.4 Phase 3: Query strategies

| Strategy | Idea in plain words |
|---|---|
| Random | pick a pair at random |
| Common neighbours, Adamic-Adar | pick pairs that already share friends (likely to be linked) |
| Max-degree, PageRank | pick pairs of well-connected nodes (ALPINE-style rules) |
| Uncertainty | pick pairs of nodes the classifier is least sure about |
| **Value (our method)** | pretend each link exists, measure how much the classifier's confidence on the targets improves, and weight by how likely the link is real |
| Value on all nodes | same, but measuring all unlabelled nodes instead of the targets |

The value strategy needs the chance that a link is real. We estimate it from the visible
network, and we never use the true probabilities. Strategies can run **static** (choose all
queries at the start) or **adaptive** (choose the next query after seeing each answer).

### 3.5 Phase 4: Measuring and checking

For each run we record how much the accuracy on the targets improves ("accuracy gain"), and
"**gap closed**": the share of the possible improvement achieved. (Possible improvement =
accuracy with the full network minus accuracy at the start.) We also record how many targets
are connected to a labelled node ("reach") and how many real links were found.

We used 30 random seeds, and every strategy faced the same networks, so differences are
paired. We used t-tests, Wilcoxon tests, bootstrap confidence intervals and a correction for
running many tests. The design was tuned on a few early seeds, so the main result was
**repeated on 30 new seeds**. We also tested that no strategy uses hidden labels (we replaced
them with random labels and the chosen queries did not change), and that repeating a run
gives identical results.

---

## 4 Results so far

**4.1 Finding more links is not enough.** In our first phase (50 seeds), common neighbours
found significantly more real links than random (1.94 vs 0.78 after 25 queries, p < 0.0001),
but classification did not improve. In the main runs, common neighbours found about 10 real
links in 200 queries and our method about 8, yet our method improved accuracy far more.

**4.2 Do node attributes help?** When the attributes are nearly noise (our setting), a
classifier using only attributes is no better than guessing (0.317 vs 0.333 at 15% labels),
and mixing them in makes label propagation worse (0.434 down to 0.379). Attributes start to
help when they are more informative (accuracy 0.41, 0.56 and 0.87 for stronger settings 0.3,
0.5 and 1.0). On our network at 15% labels, 85% of the unlabelled nodes have no path to any
label at the start.

**4.3 Main comparison.** Table 1 shows 200 queries on the 30 new seeds. The original seeds
gave numbers within about 0.03.

| Strategy | 15% labels: gain / gap closed | 40% labels: gain / gap closed |
|---|---|---|
| Random | +0.007 / 0.01 | +0.044 / 0.10 |
| Common neighbours | +0.033 / 0.07 | +0.060 / 0.13 |
| Max-degree | +0.047 / 0.10 | +0.031 / 0.07 |
| Value on all nodes (adaptive) | +0.087 / 0.18 | +0.111 / 0.25 |
| Value on targets (static) | +0.156 / 0.33 | +0.187 / 0.41 |
| **Value on targets (adaptive)** | **+0.209 / 0.45** | **+0.238 / 0.53** |

*Table 1: accuracy gain on the target nodes and gap closed after 200 queries.*

The value strategy beats each simple strategy by +0.16 to +0.21 accuracy (p < 0.001). Being
aimed at the targets adds about +0.12 over measuring all nodes. Choosing queries adaptively
instead of in advance adds about +0.05 (p = 0.02 and 0.006); this extra gain is small and is
not clear before about 200 queries.

**4.4 How many queries are enough?** With more queries (up to 400, original seeds), the value
strategy closes half of the possible improvement after about 200 queries at 15% labels and
about 160 at 40% labels. The static version needs about 290 and 310. Extra queries help less
and less: the improvement added per 100 queries goes +0.13, +0.09, +0.05, +0.02 (15% labels).
No strategy reached 75% of the possible improvement within 400 queries. Even when 96% of the
targets are connected to a label, accuracy is still below the full-network level.

**4.5 Asking only near the targets.** We also tried allowing queries only about nodes already
connected to the targets, like exploring outward from a question. This did not hurt our method
(+0.229 vs +0.227 at 15% labels) and slightly helped random queries, but our method still
beat random queries by +0.13 to +0.17.

**4.6 When the targets start with no links.** We hid every visible link touching the targets.
Our method closed 47% and 56% of the possible improvement against at most 12% for the simple
strategies (200 queries, original seeds). Adaptive queries helped more here (+0.138 over
static at 40% labels, p < 0.001).

**4.7 A picture of clusters forming.** Figure 2 follows one run (an example, not an average).
With our method the new links connect target nodes to labelled nodes and accuracy rises from
0.60 to 0.80. With random queries it stays at 0.60.

**4.8 A side study that did not hold up.** We also tried giving links strengths (strong within
a group, weak between groups). Simple strategies found about 15% more real links, but accuracy
did not improve. A classifier that treated weak links as evidence of different groups did much
better, but only because the strengths were computed from the true groups and so leaked the
answer. We do not count this as a result; it motivates using noisy weights later.

**4.9 Summary of work done.**

| Step | What was done |
|---|---|
| Baseline | common neighbours vs random, 50 seeds |
| Node attributes | 5 strengths x 2 network types x 2 label levels, 30 seeds each |
| Link-weight study | 2 experiments, 10 seeds each |
| Main comparison | 17 strategy variants, 2 label levels, 30 seeds, 200 queries |
| Longer budget | 4 strategies, 400 queries, 30 seeds |
| Exploring near targets | 5 strategies, 30 seeds |
| Three ways of hiding links | 6 strategies x 3 settings, 30 seeds |
| Repeat on new seeds | 6 strategies, 30 new seeds |
| Checks | label-leakage test, exact-reproduction test, extra statistics |
| Picture of cluster formation | one illustrative run |

*Table 2: roughly 3,000 or more simulated query runs in total (approximate count).*

---

## 5 Limits of this study

- All networks are small artificial ones of one type, and the classifier is label
  propagation, the same model our scorer uses internally.
- Our method scores links with a measure that matches how we judge it, so the comparison is
  not equally friendly to every strategy. Stronger competitors are needed.
- We did not implement ALPINE's embedding method, so we cannot say we beat ALPINE.
- Part of the gain from aiming at the targets may come from the way the method narrows its
  candidate list; we have not separated the two.
- The 15 target nodes make accuracy move in coarse steps, and the adaptive gain is small.
- Run-time measurements were unreliable, so we make no claim about computing cost beyond
  "choosing in advance is much cheaper than choosing adaptively".

## 6 Planned Work

1. **Real data and a GCN.** Test on a real citation network (for example Cora) with a graph
   neural network as the classifier.
2. **An upper bound.** Add an "oracle" strategy that uses the true labels, to show how much
   room for improvement remains, and test the candidate-list effect separately.
3. **Different tasks.** Vary the number and kind of target nodes.
4. **Stronger competitors.** Implement ALPINE's full method and query by committee.
5. **Realistic extras.** Noisy link weights, informative attributes, and different costs for
   different kinds of queries (for example asking for a label vs asking for a link).
6. **Speed.** Learn a policy that imitates the scorer so it scales to larger networks.

## 7 Conclusion

On artificial partly-hidden networks, finding more real links did not help classification,
but choosing links by how much they help the target nodes did: it closed about half of the
possible improvement in 200 queries, and this held on new random seeds and when links were
hidden in different ways. Being aimed at the task gave most of the gain, and choosing queries
one at a time gave a smaller, later gain. These results are from simulations only, so the next
steps are real data, a graph neural network and stronger comparison methods.

---

## References (check every detail against the source)

[1] Chen, Kang, Lijffijt, De Bie. "Active Link Prediction Using Network Embedding."
    arXiv:2002.01227, 2020; Applied Sciences 11(11):5043, 2021.
[2] Ostapuk, Yang, Cudre-Mauroux. "Deep Active Learning for Link Prediction in Knowledge
    Graphs." The Web Conference (WWW), 2019.
[3] Hu et al. "Graph Policy Network for Transferable Active Learning on Graphs." NeurIPS 2020.
    arXiv:2006.13463.
[4] Bevilacqua, Eliasof, Meirom, Ribeiro, Maron. "Efficient Subgraph GNNs by Learning
    Effective Selection Policies." ICLR 2024.
[5] Raghebi, Banaei-Kashani. "An active learning framework for approximate reachability
    query answering in large-scale graphs." Frontiers in Big Data, 2024.
[6] Roy, McCallum. "Toward Optimal Active Learning through Sampling Estimation of Error
    Reduction." ICML 2001. [verify]
[7] Chaloner, Verdinelli. "Bayesian Experimental Design: A Review." Statistical Science,
    1995; Lindley, 1956.
[8] Houlsby et al., 2011; Gal, Islam, Ghahramani. "Deep Bayesian Active Learning with Image
    Data." ICML 2017.
[9] Koh, Liang. "Understanding Black-box Predictions via Influence Functions." ICML 2017.
[10] Sun et al. "Perturbation-based Graph Active Learning for Weakly-Supervised Belief
     Representation Learning." arXiv:2410.19176, 2024.
[11] Settles. "Active Learning Literature Survey." 2009.
[12] Ren et al. "A Survey of Deep Active Learning." ACM Computing Surveys, 2021.
     arXiv:2009.00236.
[13] Li et al. "A Survey on Deep Active Learning: Recent Advances and New Frontiers." IEEE
     TNNLS, 2024. arXiv:2405.00334.
[14] Wilder, Dilkina, Tambe. "Melding the Data-Decisions Pipeline: Decision-Focused Learning
     for Combinatorial Optimization." AAAI 2019. arXiv:1809.05504.
[15] Foster, Ivanova, Malik, Rainforth. "Deep Adaptive Design: Amortizing Sequential Bayesian
     Experimental Design." ICML 2021. arXiv:2103.02438.

Used in our code but not yet cited in the text above (add where you describe the
classifier, the baselines and the test networks; verify details):
- Zhu, Ghahramani, Lafferty. Label propagation (Gaussian fields and harmonic functions),
  ICML 2003.
- Adamic, Adar. "Friends and neighbors on the web." Social Networks, 2003.
- Deshpande et al. Contextual stochastic block models. NeurIPS 2018.
- Kipf, Welling. GCN. ICLR 2017 (for future work).
