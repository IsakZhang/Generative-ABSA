# Generative ABSA

This repo contains the data and code for our paper [Towards Generative Aspect-Based Sentiment Analysis](https://aclanthology.org/2021.acl-short.64.pdf) in ACL 2021.


## Requirements

Pls note that some packages (such as transformers) are under highly active development, so we highly recommend you to install the specified version of the following packages:
- transformers==4.0.0
- sentencepiece==0.1.91
- pytorch_lightning==0.8.1



## Quick Start

- Set up the environment as described in the above section
- Download the pre-trained T5-base model (you can also use larger versions for better performance depending on the availability of the computation resource), put it under the folder `T5-base`.
  - You can also skip this step and the pre-trained model would be automatically downloaded to the cache in the next step
- Run command `sh run.sh`, which runs the `UABSA` task on the `laptop14` dataset.



## Detailed Usage
We conduct experiments on four ABSA tasks with four datasets in the paper, you can change the parameters in `run.sh` to try them:
```
python main.py --task $task \
            --dataset $dataset \
            --model_name_or_path t5-base \
            --paradigm $paradigm \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20 
```
- `$task` refers to one of the ABSA task in [`aope`, `uabsa`, `aste`, `tasd`] 
- `$dataset` refers to one of the four datasets in [`laptop14`, `rest14`, `rest15`, `rest6`]
- `$paradigm` refers to one of the two paradigms proposed in the model. 

More details can be found in the paper and the help info in the `main.py`.



## Citation

If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{zhang-etal-2021-towards,
    title = "Towards Generative Aspect-Based Sentiment Analysis",
    author = "Zhang, Wenxuan  and
      Li, Xin  and
      Deng, Yang  and
      Bing, Lidong  and
      Lam, Wai",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    year = "2021",
    url = "https://aclanthology.org/2021.acl-short.64",
    pages = "504--510",
}
```
