# ExplagraphGen

PyTorch code for the reproduction of the following ACL 2022 paper done by Zhivar Sourati and Omey Manyar which is basically a fork from the [main repository](https://github.com/swarnaHub/ExplagraphGen):

[Explanation Graph Generation via Pre-trained Language Models: An Empirical Study with Contrastive Learning](https://arxiv.org/abs/2204.04813)

[Swarnadeep Saha](https://swarnahub.github.io/), [Prateek Yadav](https://prateek-yadav.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)


the list of content can be found below:

- [Installation](#installation)
- [ExplaGraphs Dataset](#explagraphs-dataset)
- [Contrastive Graph Data](#contrastive-graph-data)
- [Models](#models)
- [Running the scripts on an GPU Cluster](#running-the-scripts-on-an-gpu-cluster)
- [Predictions](#predictions)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [Related Citation](#related-citation)


## Installation
This repository is tested on Python 3.8.3.  
You should install the repository on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## ExplaGraphs Dataset
ExplaGraphs dataset can be found inside the ```data``` folder. For more details, check out the ExplaGraphs website hosted [here](https://explagraphs.github.io/).

It contains the training data in ```train.tsv``` and the validation samples in ```dev.tsv```.

Each training sample contains four tab-separated entries -- belief, argument, stance label and the explanation graph.

The graph is organized as a bracketed string ```(edge_1)(edge_2)...(edge_n)```, where each edge is of the form ```concept_1; relation; concept_2```. 

## Contrastive Graph Data
All the negatively perturbed graphs for training our contrastive models can be found in ```contrastive_data/train.neg_target```.

The corresponding gold samples (belief, argument, stance) and the gold graphs are contained in ```contrastive_data/train.source``` and ```contrastive_data/train.target```. The files are created in a way to be directly used for training the models.

The positively perturbed graphs can be found in ```contrastive_data/train.pos_target```.

## Models

We experiment with rationalizing models that first predict the stance and then conditions on it to generate the graph.

For training the stance prediction model, run the following script
```
bash scripts/train_stance.sh
```
The validation samples in ```contrastive_data/val.source``` are already appended with the predicted stances from our best model. If you train your own stance prediction model, replace the stances in ```contrastive_data/val.source``` with your predictions so that you condition on the predicted stances before generating the graph.

Next, the Max-margin Graph Generation model and the Contrastive model can be trained using the following scripts.
```
bash scripts/train_graph_max_margin.sh
bash scripts/train_graph_contrastive.sh
```

Also for training of the other models, including the model that uses the positive graphs aside the original data to train the base model and also training of the base model, you can use the following scripts: 

```
bash scripts/train_graph_gen.sh

bash scripts/train_graph_gen_pos_perturbed.sh
```

Note that as the last experiment we did replacing the base T5 with a T5 that was fine-tuned on commonsense knowledge graphs uses the model from [here](https://arxiv.org/abs/2205.10661), in order to do the last experiment we advice you to reach out to the author and get the model and then just change the value for the field `model_name_or_path` in the training scripts mentioned above with the path to the model.


All trained models will be saved in the ```models``` folder. The scripts to evaluate your models can also be found in the ```scripts``` folder.

We'll share our trained models soon. Stay tuned!

## Running the scripts on an GPU Cluster

If you want to run the scripts on a GPU cluster, you can use `general.sh` that is basically a wrapper for executing the scripts on a GPU cluster that has SLURM as a job scheduler. After adjusting the parameters in `general.sh` such as number and type of GPU and also the amount of RAM that you need, you can submit your job with `sbatch general.sh`

## Predictions
You can find the predicted stances and the generated graphs from our max-margin model in ```output/preds.tsv```.

## Evaluation Metrics
For running the evaluation metrics, refer to the detailed steps outlined in the original [ExplaGraphs](https://github.com/swarnaHub/ExplaGraphs) repository. If you wish to reproduce our results on the validation set of ExplaGraphs, run the evaluation scripts on ```output/preds.tsv```. 

Note that the test set is hidden and if you wish to evaluate your own model on it, refer to the instructions [here](https://github.com/swarnaHub/ExplaGraphs).

### Citation
```
@inproceedings{saha2022explagraphsgen,
  title={Explanation Graph Generation via Pre-trained Language Models: An Empirical Study with Contrastive Learning},
  author={Saha, Swarnadeep and Yadav, Prateek and Bansal, Mohit},
  booktitle={ACL},
  year={2022}
}
```

### Related Citation
```
@inproceedings{saha2021explagraphs,
  title={ExplaGraphs: An Explanation Graph Generation Task for Structured Commonsense Reasoning},
  author={Saha, Swarnadeep and Yadav, Prateek and Bauer, Lisa and Bansal, Mohit},
  booktitle={EMNLP},
  year={2021}
}
```
