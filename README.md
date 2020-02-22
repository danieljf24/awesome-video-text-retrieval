# Awesome Video-Text Retrieval by Deep Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of deep learning resources for video-text retrieval.

## Contributing
Please feel free to [pull requests](https://github.com/danieljf24/awesome-video-text-retrieval/pulls) to add papers.

Markdown format:

```markdown
- `[Conference/Trans Year]` Author. Title. Trans Year. [[paper]](link) [[code]](link) [[homepage]](link)
```


## Table of Contents
- [Popular Implementations](#popular-implementations)
  - [PyTorch](#pytorch)
  - [TensorFlow](#tensorflow)
  - [Others](#others)
- [Papers](#papers)
  - [2019](#2019) - [2018](#2018) - [Before](#before)
  - [Ad-hoc Video Search](#ad-hoc-video-search)
  - [Other Related](#other-related)
- [Datasets](#datasets)



## Popular Implementations

#### PyTorch
- [dual_encoding](https://github.com/danieljf24/dual_encoding)
- [w2vvpp](https://github.com/li-xirong/w2vvpp)
- [Mixture-of-Embedding-Experts](https://github.com/antoine77340/Mixture-of-Embedding-Experts)
- [howto100m](https://github.com/antoine77340/howto100m)

#### TensorFlow
- [jsfusion](https://github.com/yj-yu/lsmdc)

#### Others
- [w2vv](https://github.com/danieljf24/w2vv)(Keras)



## Papers

### 2019
* `[CVPR2019]` Jianfeng Dong, Xirong Li, Chaoxi Xu, Shouling Ji, Yuan He, Gang Yang, Xun Wang. Dual Encoding for Zero-Example Video Retrieval. CVPR, 2019. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Dual_Encoding_for_Zero-Example_Video_Retrieval_CVPR_2019_paper.pdf) [[code]](https://github.com/danieljf24/dual_encoding)
* `[CVPR2019]` Yale Song, and Mohammad Soleymani. Polysemous visual-semantic embedding for cross-modal retrieval. CVPR, 2019. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Song_Polysemous_Visual-Semantic_Embedding_for_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)
* `[ICCV2019]` Michael Wray, Diane Larlus, Gabriela Csurka, and Dima Damen. Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings. ICCV, 2019. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf)
* `[ICCV2019]` Yu Xiong, Qingqiu Huang, Lingfeng Guo, Hang Zhou, Bolei Zhou, and Dahua Lin. A Graph-Based Framework to Bridge Movies and Synopses. ICCV, 2019. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_A_Graph-Based_Framework_to_Bridge_Movies_and_Synopses_ICCV_2019_paper.pdf)
* `[ACMMM2019]` Xirong Li, Chaoxi Xu, Gang Yang, Zhineng Chen, and Jianfeng Dong. W2VV++ Fully Deep Learning for Ad-hoc Video Search. ACM Multimedia, 2019. [[paper]](http://lixirong.net/pub/mm2019-w2vvpp.pdf) [[code]](https://github.com/li-xirong/w2vvpp)
* `[BMVC2019]` Yang Liu, Samuel Albanie, Arsha Nagrani, Andrew Zisserman. Use What You Have: Video Retrieval Using Representations From Collaborative Experts. MBVC, 2019. [[paper]](https://arxiv.org/abs/1907.13487) [[code]](https://github.com/albanie/collaborative-experts)
* `[BigMM2019]` Jaeyoung Choi, Martha Larson, Gerald Friedland, and Alan Hanjalic. From Intra-Modal to Inter-Modal Space: Multi-Task Learning of Shared Representations for Cross-Modal Retrieval. International Conference on Multimedia Big Data, 2019. [[paper]](https://repository.ubn.ru.nl/bitstream/handle/2066/209215/209215.pdf?sequence=1)


### 2018
* `[TMM2018]` Jianfeng Dong, Xirong Li, Cees GM Snoek. Predicting visual features from text for image and video caption retrieval. IEEE Transactions on Multimedia, 2018. [[paper]](https://arxiv.org/pdf/1709.01362) [[code]](https://github.com/danieljf24/w2vv)
* `[ECCV2018]` Bowen Zhang, Hexiang Hu, Fei Sha. Cross-Modal and Hierarchical Modeling of Video and Text. ECCV, 2018. [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bowen_Zhang_Cross-Modal_and_Hierarchical_ECCV_2018_paper.pdf) [[code]](https://github.com/zbwglory/CMHSE)
* `[ECCV2018]` Youngjae Yu, Jongseok Kim, Gunhee Kim. A Joint Sequence Fusion Model for Video Question Answering and Retrieval. ECCV, 2018. [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Youngjae_Yu_A_Joint_Sequence_ECCV_2018_paper.pdf)
* `[ECCV2018]` Dian Shao, Yu Xiong, Yue Zhao, Qingqiu Huang, Yu Qiao, and Dahua Lin. Find and focus: Retrieve and localize video events with natural language queries. ECCV, 2018. [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dian_SHAO_Find_and_Focus_ECCV_2018_paper.pdf)
* `[ICMR2018]` Niluthpol Chowdhury Mithun, Juncheng Li, Florian Metze, Amit K. Roy-Chowdhury. Learning Joint Embedding with Multimodal Cues for Cross-Modal Video-Text Retrieval. ICMR, 2018. [[paper]](https://dl.acm.org/citation.cfm?id=3206064) [[code]](https://github.com/niluthpol/multimodal_vtt)
* `[arXiv2018]` Antoine Miech, Ivan Laptev, Josef Sivic. Learning a Text-Video Embedding from Incomplete and Heterogeneous Data. arXiv preprint arXiv:1804.02516, 2018. [[paper]](https://arxiv.org/abs/1809.06181) [[code]](https://github.com/antoine77340/Mixture-of-Embedding-Experts)


### Before
* `[CVPR2017]` Youngjae Yu, Hyungjin Ko, Jongwook Choi, Gunhee Kim. End-to-end concept word detection for video captioning, retrieval, and question answering. CVPR, 2017. [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_End-To-End_Concept_Word_CVPR_2017_paper.pdf) [[code]](https://gitlab.com/fodrh1201/CT-SAN/tree/master)
* `[ECCVW2016]` Mayu OtaniEmail, Yuta NakashimaEsa, RahtuJanne Heikkilä, Naokazu Yokoya. Learning joint representations of videos and sentences with web image search. ECCV Workshop, 2016. [[paper]](https://arxiv.org/pdf/1608.02367)
* `[AAAI2015]` Ran Xu, Caiming Xiong, Wei Chen, Jason J Corso. Jointly modeling deep video and compositional text to bridge vision and language in a unified framework. AAAI, 2015. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9734/9563)


### Ad-hoc Video Search
* For the papers targeting at ad-hoc video search in the context of [[TRECVID]](https://trecvid.nist.gov/), please refer to [[here]](https://github.com/li-xirong/video-retrieval)


### Other Related
* `[arXiv2020]` Tianhao Li, and Limin Wang. Learning Spatiotemporal Features via Video and Text Pair Discrimination. arXiv preprint arXiv:2001.05691, 2020. [[paper]](https://arxiv.org/pdf/2001.05691) 
* `[arXiv2019]` Hazel Doughty, Ivan Laptev, Walterio Mayol-Cuevas, and Dima Damen. Action Modifiers: Learning from Adverbs in Instructional Videos. arXiv preprint arXiv:1912.06617, 2019. [[paper]](https://arxiv.org/abs/1912.06617)
* `[arXiv2019]` Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-End Learning of Visual Representations from Uncurated Instructional Videos. arXiv preprint arXiv:1912.06430, 2019. [[paper]](https://arxiv.org/abs/1912.06430)




## Datasets
* `[MSVD]`  David L. Chen and William B. Dolan. Collecting Highly Parallel Data for Paraphrase Evaluation. ACL, 2011. [[paper]](https://www.aclweb.org/anthology/P11-1020) [[dataset]](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
* `[MSRVTT]` Jun Xu Tao Mei Ting Yao Yong Rui. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. CVPR, 2016. [[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf) [[dataset]](http://ms-multimedia-challenge.com/2017/dataset)
* `[TGIF]` Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. TGIF: A new dataset and benchmark on animated GIF description. CVPR, 2016. [[paper]](https://hal.archives-ouvertes.fr/hal-01854776/document) [[homepage]](http://raingo.github.io/TGIF-Release/)
* `[AVS]` George Awad, et al. Trecvid 2016: Evaluating video search, video event detection, localization, and hyperlinking. TRECVID Workshop, 2016. [[paper]](https://hal.archives-ouvertes.fr/hal-01854776/document) [[dataset]](https://github.com/li-xirong/avs)
* `[LSMDC]` Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. IJCV, 2017. [[paper]](https://link.springer.com/article/10.1007/s11263-016-0987-1) [[dataset]](https://sites.google.com/site/describingmovies/download)
* `[ActivityNet Captions]` Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. ICCV, 2017. [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Krishna_Dense-Captioning_Events_in_ICCV_2017_paper.pdf) [[dataset]](https://cs.stanford.edu/people/ranjaykrishna/densevid/)
* `[DiDeMo]` Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, Bryan Russell. Localizing Moments in Video with Natural Language. ICCV, 2017. [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hendricks_Localizing_Moments_in_ICCV_2017_paper.pdf) [[code]](https://github.com/LisaAnne/LocalizingMoments) 
* `[HowTo100M]` Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, Josef Sivic. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. ICCV, 2019. [[homepage]](https://www.di.ens.fr/willow/research/howto100m/) [paper](https://arxiv.org/pdf/1906.03327.pdf)
* `[VATEX]` Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, William Yang Wang. VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research. ICCV, 2019. [[paper]](https://arxiv.org/abs/1904.03493) [[homepage]](http://vatex.org/main/index.html)



## Licenses

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [danieljf24](https://github.com/danieljf24) all copyright and related or neighboring rights to this repository.