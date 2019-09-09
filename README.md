# COMER: Sclable and Accurate Dialogue State Tracking via Hierachical Sequence Generation

This is the PyTorch implementation of the paper:
**Sclable and Accurate Dialogue State Tracking via Hierachical Sequence Generation**. **Liliang Ren**,Jianmo Ni, Julian McAuley. ***EMNLP 2019***
[[PDF]](https://arxiv.org/abs/1909.00754)

The code is written and tested with PyTorch == 1.1.0. The minimum memory requirement for the graphic card is 8GB.

## Abstract
Existing approaches to dialogue state tracking rely on pre-defined ontologies consisting of a set of all possible slot types and values. Though such approaches exhibit promising performance on single-domain benchmarks, they suffer from computational complexity that increases proportionally to the number of pre-defined slots that need tracking. This issue becomes more severe when it comes to multi-domain dialogues which include larger numbers of slots. In this paper, we investigate how to approach DST using a generation framework without the pre-defined ontology list. Given each turn of user utterance and system response, we directly generate a sequence of belief states by applying a hierarchical encoder-decoder structure. In this way, the computational complexity of our model will be a constant regardless of the number of pre-defined slots. Experiments on both the multi-domain and the single domain dialogue state tracking dataset show that our model not only scales easily with the increasing number of pre-defined domains and slots but also reaches the state-of-the-art performance.


## Create Data
```
python create_data.py 
```
This Python script has to be run under the environment, Python == 2.7, for reproducibility.
***************************************************************


## Preprocessing
```
python3 convert_mw.py
python3 preprocess_mw.py 
python3 make_emb.py
```

***************************************************************

## Training
```
bash run.sh
```

****************************************************************

## Evaluation
```
python3 predict.py 
```

*******************************************************************

