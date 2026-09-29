# CyIN

> **Official Pytorch Implementation of "CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning"**

## 🔧 Setup

Clone the repository and create the environment:

```bash
conda create -n cyin python=3.10
conda activate cyin

pip install -r requirements.txt
```

Prepare the required **datasets and configuration files**.

## 🔥 Train & Eval

```bash
bash train_cyin.sh
bash eval_cyin.sh
```


## 📝 Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@inproceedings{NEURIPS2025_19947989,
 author = {Lin, Ronghao and He, Qiaolin and Mai, Sijie and Zeng, Ying and Xiong, Aolin and Huang, Li and Tan, Yap-Peng and Hu, Haifeng},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {17605--17642},
 title = {CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning},
 volume = {38, Main Conference},
 year = {2025}
}
```

## 📄 Acknowledgments

We would like to express our gratitude to [huggingface](https://huggingface.co/) and [mmsa](https://github.com/thuiar/MMSA), which are of great help to our work.

