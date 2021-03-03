# Awesome Video-Text Retrieval by Deep Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of deep learning resources for video-text retrieval.

## Contributing
Please feel free to [pull requests](https://github.com/danieljf24/awesome-video-text-retrieval/pulls) to add papers.

Markdown format:

```markdown
- `[Author Journal/Booktitle Year]` Title. Journal/Booktitle, Year. [[paper]](link) [[code]](link) [[homepage]](link)
```


## Table of Contents
- [Implementations](#implementations)
  - [PyTorch](#pytorch)
  - [TensorFlow](#tensorflow)
  - [Others](#others)
- [Papers](#papers)
  - [2021](#2021) - [2020](#2020) - [2019](#2019) - [2018](#2018) - [Before](#before)
  - [Ad-hoc Video Search](#ad-hoc-video-search)
  - [Other Related](#other-related)
- [Datasets](#datasets)



## Implementations

#### PyTorch
- [hybrid_space](https://github.com/danieljf24/hybrid_space)
- [dual_encoding](https://github.com/danieljf24/dual_encoding)
- [w2vvpp](https://github.com/li-xirong/w2vvpp)
- [Mixture-of-Embedding-Experts](https://github.com/antoine77340/Mixture-of-Embedding-Experts)
- [howto100m](https://github.com/antoine77340/howto100m)
- [collaborative]https://github.com/albanie/collaborative-experts
- [hgr](https://github.com/cshizhe/hgr_v2t)
- [coot](https://github.com/gingsi/coot-videotext)
- [mmt](https://github.com/gabeur/mmt)

#### TensorFlow
- [jsfusion](https://github.com/yj-yu/lsmdc)

#### Others
- [w2vv](https://github.com/danieljf24/w2vv)(Keras)

#### Useful Toolkit
- [Extracting CNN features from video frames by MXNet](https://github.com/xuchaoxi/video-cnn-feat)


## Papers

### 2021
* `[Dong et al. TPAMI21]` Dual Encoding for Video Retrieval by Text. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020. [[paper]](https://arxiv.org/abs/2009.05381) [[code]](https://github.com/danieljf24/hybrid_space) 
* `[Patrick et al. ICLR21]` Support-set Bottlenecks for Video-text Representation Learning. ICLR, 2021. [[paper]](https://arxiv.org/abs/2010.02824)

### 2020
* `[Yang et al. SIGIR20]` Tree-Augmented Cross-Modal Encoding for Complex-Query Video Retrieval. SIGIR, 2020. [[paper]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401151) 
* `[Ging et al. NeurIPS20]` COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning. NeurIPS, 2020. [[paper]](https://proceedings.neurips.cc/paper/2020/file/ff0abbcc0227c9124a804b084d161a2d-Paper.pdf)  [[code]](https://github.com/gingsi/coot-videotext)
* `[Gabeur et al. ECCV20]` Multi-modal Transformer for Video Retrieval. ECCV, 2020. [[paper]](https://arxiv.org/abs/2007.10639) [[code]](https://github.com/gabeur/mmt)[[homepage]](link)
* `[Li et al. TMM20]` SEA: Sentence Encoder Assembly for Video Retrieval by Textual Queries. IEEE Transactions on Multimedia, 2020. [[paper]](https://arxiv.org/abs/2011.12091)
* `[Wang et al. TMM20]` Learning Coarse-to-Fine Graph Neural Networks for Video-Text Retrieval. IEEE Transactions on Multimedia, 2020. [[paper]](https://ieeexplore.ieee.org/abstract/document/9147074)
* `[Chen et al. TMM20]` Interclass-Relativity-Adaptive Metric Learning for Cross-Modal Matching and Beyond. IEEE Transactions on Multimedia, 2020. [[paper]](https://ieeexplore.ieee.org/abstract/document/9178501)
* `[Wu et al. ACMMM20]` Interpretable Embedding for Ad-Hoc Video Search. ACM Multimedia, 2020. [[paper]](http://vireo.cs.cityu.edu.hk/papers/MM2020_dual_task_video_retrieval.pdf) 
* `[Feng et al. IJCAI20]` Exploiting Visual Semantic Reasoning for Video-Text Retrieval. IJCAI, 2020. [[paper]](https://arxiv.org/abs/2006.08889) 
* `[Wei et al. CVPR20]` Universal Weighting Metric Learning for Cross-Modal Retrieval. CVPR, 2020. [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Universal_Weighting_Metric_Learning_for_Cross-Modal_Matching_CVPR_2020_paper.pdf)
* `[Doughty et al. CVPR20]` Action Modifiers: Learning from Adverbs in Instructional Videos. CVPR, 2020. [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Doughty_Action_Modifiers_Learning_From_Adverbs_in_Instructional_Videos_CVPR_2020_paper.pdf)
* `[Chen et al. CVPR20]` Fine-grained Video-Text Retrieval with Hierarchical Graph Reasoning. CVPR, 2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Fine-Grained_Video-Text_Retrieval_With_Hierarchical_Graph_Reasoning_CVPR_2020_paper.pdf)
* `[Zhu et al. CVPR20]` ActBERT: Learning Global-Local Video-Text Representations. CVPR, 2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.pdf)
* `[Zhao et al. ICME20]` Stacked Convolutional Deep Encoding Network For Video-Text Retrieval. ICME, 2020. [[paper]](https://arxiv.org/pdf/2004.04959.pdf)
* `[Luo et al. ARXIV20]` UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation. arXiv:2002.06353, 2020. [[paper]](https://arxiv.org/abs/2002.06353v3)


### 2019
* `[Dong et al. CVPR19]` Dual Encoding for Zero-Example Video Retrieval. CVPR, 2019. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Dual_Encoding_for_Zero-Example_Video_Retrieval_CVPR_2019_paper.pdf) [[code]](https://github.com/danieljf24/dual_encoding)
* `[Song et al. CVPR19]` Polysemous visual-semantic embedding for cross-modal retrieval. CVPR, 2019. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Song_Polysemous_Visual-Semantic_Embedding_for_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)
* `[Wray et al. ICCV19]` Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings. ICCV, 2019. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf)
* `[Xiong et al. ICCV19]` A Graph-Based Framework to Bridge Movies and Synopses. ICCV, 2019. [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_A_Graph-Based_Framework_to_Bridge_Movies_and_Synopses_ICCV_2019_paper.pdf)
* `[Li et al. ACMMM19]` W2VV++ Fully Deep Learning for Ad-hoc Video Search. ACM Multimedia, 2019. [[paper]](http://lixirong.net/pub/mm2019-w2vvpp.pdf) [[code]](https://github.com/li-xirong/w2vvpp)
* `[Liu et al. BMVC19]` Use What You Have: Video Retrieval Using Representations From Collaborative Experts. MBVC, 2019. [[paper]](https://arxiv.org/abs/1907.13487) [[code]](https://github.com/albanie/collaborative-experts)
* `[Choi et al. BigMM19]` From Intra-Modal to Inter-Modal Space: Multi-Task Learning of Shared Representations for Cross-Modal Retrieval. International Conference on Multimedia Big Data, 2019. [[paper]](https://repository.ubn.ru.nl/bitstream/handle/2066/209215/209215.pdf?sequence=1)


### 2018
* `[Dong et al. TMM18]` Predicting visual features from text for image and video caption retrieval. IEEE Transactions on Multimedia, 2018. [[paper]](https://arxiv.org/pdf/1709.01362) [[code]](https://github.com/danieljf24/w2vv)
* `[Zhang et al. ECCV18]` Cross-Modal and Hierarchical Modeling of Video and Text. ECCV, 2018. [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bowen_Zhang_Cross-Modal_and_Hierarchical_ECCV_2018_paper.pdf) [[code]](https://github.com/zbwglory/CMHSE)
* `[Yu et al. ECCV18]` A Joint Sequence Fusion Model for Video Question Answering and Retrieval. ECCV, 2018. [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Youngjae_Yu_A_Joint_Sequence_ECCV_2018_paper.pdf)
* `[Shao et al. ECCV18]` Find and focus: Retrieve and localize video events with natural language queries. ECCV, 2018. [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dian_SHAO_Find_and_Focus_ECCV_2018_paper.pdf)
* `[Mithun et al. ICMR18]` Learning Joint Embedding with Multimodal Cues for Cross-Modal Video-Text Retrieval. ICMR, 2018. [[paper]](https://dl.acm.org/citation.cfm?id=3206064) [[code]](https://github.com/niluthpol/multimodal_vtt)
* `[Miech et al. arXiv18]` Learning a Text-Video Embedding from Incomplete and Heterogeneous Data. arXiv preprint arXiv:1804.02516, 2018. [[paper]](https://arxiv.org/abs/1809.06181) [[code]](https://github.com/antoine77340/Mixture-of-Embedding-Experts)


### Before
* `[Yu et al. CVPR17]` End-to-end concept word detection for video captioning, retrieval, and question answering. CVPR, 2017. [[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_End-To-End_Concept_Word_CVPR_2017_paper.pdf) [[code]](https://gitlab.com/fodrh1201/CT-SAN/tree/master)
* `[OtaniEmail et al. ECCVW2016]` Learning joint representations of videos and sentences with web image search. ECCV Workshop, 2016. [[paper]](https://arxiv.org/pdf/1608.02367)
* `[Xu et al. AAAI15]` Jointly modeling deep video and compositional text to bridge vision and language in a unified framework. AAAI, 2015. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9734/9563)


### Ad-hoc Video Search
* For the papers targeting at ad-hoc video search in the context of [TRECVID](https://trecvid.nist.gov/), please refer to [here](https://github.com/li-xirong/video-retrieval).


### Other Related
* `[Li et al. arXiv20]` Learning Spatiotemporal Features via Video and Text Pair Discrimination. arXiv preprint arXiv:2001.05691, 2020. [[paper]](https://arxiv.org/abs/2001.05691) 
* `[Miech et al. CVPR20]` End-to-End Learning of Visual Representations from Uncurated Instructional Videos. CVPR, 2020. [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Miech_End-to-End_Learning_of_Visual_Representations_From_Uncurated_Instructional_Videos_CVPR_2020_paper.pdf)




## Datasets
* `[MSVD]`  David et al. Collecting Highly Parallel Data for Paraphrase Evaluation. ACL, 2011. [[paper]](https://www.aclweb.org/anthology/P11-1020) [[dataset]](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
* `[MSRVTT]` Xu et al. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. CVPR, 2016. [[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_-1.pdf) [[dataset]](http://ms-multimedia-challenge.com/2017/dataset)
* `[TGIF]` Li et al. TGIF: A new dataset and benchmark on animated GIF description. CVPR, 2016. [[paper]](https://hal.archives-ouvertes.fr/hal-01854776/document) [[homepage]](http://raingo.github.io/TGIF-Release/)
* `[AVS]` Awad et al. Trecvid 2016: Evaluating video search, video event detection, localization, and hyperlinking. TRECVID Workshop, 2016. [[paper]](https://hal.archives-ouvertes.fr/hal-01854776/document) [[dataset]](https://github.com/li-xirong/avs)
* `[LSMDC]` Rohrbach et al. Movie description. IJCV, 2017. [[paper]](https://link.springer.com/article/10.1007/s11263-016-0987-1) [[dataset]](https://sites.google.com/site/describingmovies/download)
* `[ActivityNet Captions]` Krishna et al. Dense-captioning events in videos. ICCV, 2017. [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Krishna_Dense-Captioning_Events_in_ICCV_2017_paper.pdf) [[dataset]](https://cs.stanford.edu/people/ranjaykrishna/densevid/)
* `[DiDeMo]` Hendricks et al. Localizing Moments in Video with Natural Language. ICCV, 2017. [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hendricks_Localizing_Moments_in_ICCV_2017_paper.pdf) [[code]](https://github.com/LisaAnne/LocalizingMoments) 
* `[HowTo100M]` Miech et al. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. ICCV, 2019. [[paper]](https://arxiv.org/pdf/1906.03327.pdf) [[homepage]](https://www.di.ens.fr/willow/research/howto100m/) 
* `[VATEX]` Wang et al. VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research. ICCV, 2019. [[paper]](https://arxiv.org/abs/1904.03493) [[homepage]](http://vatex.org/main/index.html)



## Licenses

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [danieljf24](https://github.com/danieljf24) all copyright and related or neighboring rights to this repository.
