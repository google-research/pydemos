# Copyright 2023 The pydemos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""File containing text extracted from the OWL-VIT paper."""

import streamlit as st

# Allows backslash continuation to be able to read and modify text in the editor
# pylint: disable=g-backslash-continuation


def abstract():
  """Returns abstract."""
  st.markdown(
      '<p align="center"><a href="https://scholar.google.co.uk/citations?user=57BFBY0AAAAJ&hl=en">Matthias Minderer</a>,\
    <a href="https://scholar.google.nl/citations?user=zTy9cUwAAAAJ&hl=en">Alexey Gritsenko</a>,\
    Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy,<br>\
    Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen,\
    Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby</p>',
      unsafe_allow_html=True)
  st.markdown(
      '<p align="justify">Combining simple architectures with large-scale pre-training\
        has led to massive improvements in image classification. For object detection,\
        pre-training and scaling approaches are less well established,\
        especially in the long-tailed and open-vocabulary setting, where training data\
        is relatively scarce. In this paper, we propose a strong recipe\
        for transferring image-text models to open-vocabulary object detection.\
        We use a standard Vision Transformer architecture with minimal modifications,\
        contrastive image-text pre-training, and end-to-end detection\
        fine-tuning. Our analysis of the scaling properties of this setup shows\
        that increasing image-level pre-training and model size yield consistent\
        improvements on the downstream detection task. We provide the adaptation\
        strategies and regularizations needed to attain very strong performance\
        on zero-shot text-conditioned and one-shot image-conditioned object detection.\
        Code and models are available on GitHub.</p>',
      unsafe_allow_html=True)

  st.markdown('###')


def introduction():
  """Returns introduction."""
  st.subheader('Introduction')
  st.markdown(
      '<p align="justify">Object detection is a fundamental task in computer vision. Until recently, detection models were typically limited to a small, fixed set of semantic categories, \
because obtaining localized training data with large or open label spaces is costly and time-consuming. This has changed with the development of powerful language encoders and contrastive image-text training. These models learn a shared \
representation of image and text from loosely aligned image-text pairs, which are abundantly available on the web. By leveraging large amounts of image-text \
data, contrastive training has yielded major improvements in zero-shot classification performance and other language-based tasks [39, 19, 44]. <br><br>\
Many recent works aim to transfer the language capabilities of these models to object detection [12, 26, 45, 46, 20]. These methods, for example, \
use distillation against embeddings of image crops [12], weak supervision with image-level labels [46], or self-training [26, 45]. Here, we provide a simple \
architecture and end-to-end training recipe that achieves strong open-vocabulary detection without these methods, even on categories not seen during training. <br><br> \
We start with the Vision Transformer architecture [22], which has been shown to be highly scalable, and pre-train it contrastively on a \
large image-text dataset [44, 19]. To transfer the model to detection, we make a minimal set of changes: We remove the final token pooling layer and instead attach a lightweight \
classification and box head to each transformer output token. Open-vocabulary classification is enabled by replacing the fixed classification layer weights with the class-name embeddings obtained from the \
text model [2] (Figure 1). We fine-tune the pre-trained model on standard detection datasets using a bipartite matching loss [6]. Both image and text model are fine-tuned end-to-end. \
<br><br>We analyze the scaling properties of this approach and find that increasing model size and pre-training duration continue to yield improvements in detection performance beyond 20 billion image-text pairs. \
This is important since imagetext pairs, in contrast to detection data, are abundant and allow further scaling. <br><br>A key feature of our model is its simplicity and modularity. Since the image and text components \
of our model are not fused, our model is agnostic to the source of query representations. We can therefore use our model without modification as a one-shot detection learner simply by querying it with imagederived \
embeddings. One-shot object detection is the challenging problem of detecting novel objects solely based on a query image patch showing the object [16,4,31]. The image-conditioned one-shot ability is a powerful extension to \
text-conditioned detection because it allows detecting objects that are difficult to describe through text (yet easy to capture in an image), such as specialized technical parts. Despite using a generic architecture \
not specialized for this problem, we improve the state of the art for one-shot detection on unseen COCO categories (held out during training) from 26.0 to 41.8 AP50, an improvement of 72%. \
<br><br>For open-vocabulary text-conditioned detection, our model achieves 34.6% AP overall and 31.2% AP<sub>rare</sub> on unseen classes on the LVIS dataset. \
<br>In summary, we make the following contributions:\
<ol type="1">\
  <li>A simple and strong recipe for transferring image-level pre-training to openvocabulary object detection.</li>\
  <li>State-of-the-art one-shot (image conditional) detection by a large margin.</li>\
  <li>A detailed scaling and ablation study to justify our design.</li>\
</ol> We believe our model will serve as a strong baseline that can be easily implemented in various frameworks, and as a flexible starting point for future \
research on tasks requiring open-vocabulary localization. We call our method \
<i>Vision Transformer for Open-World Localization</i>, or <i><b>OWL-ViT</b></i> for short.  </p>',
      unsafe_allow_html=True)
  _, b, _ = st.columns([2, 4, 2])
  with b:
    st.image(
        'paper_figures/figure1.png',
        caption='Fig. 1. Overview of our method. Left: We first pre-train an image and text encoder \
    contrastively using image-text pairs, similar to CLIP [33], ALIGN [19], and LiT [44]. Right: We then transfer the pre-trained encoders to open-vocabulary object detection \
    by removing token pooling and attaching light-weight object classification and localization heads directly to the image encoder output tokens. To achieve open-vocabulary \
    detection, query strings are embedded with the text encoder and used for classification. The model is fine-tuned on standard detection datasets. At inference time, we can use \
    text-derived embeddings for open-vocabulary detection, or image-derived embeddings for few-shot image-conditioned detection.',
        use_column_width='auto')


def related_work():
  """Returns related work section."""
  st.subheader('Related Work')
  st.markdown(
      '<p align="justify"><b>Contrastive Vision-Language Pre-Training.</b>The idea of embedding images and text into a shared space has been used to achieve “zero-shot” generalization \
for a long time [10,36,40]. Thanks to innovations in contrastive losses and better architectures, recent models can learn consistent visual and language representations from web-derived image and text \
pairs without the need for explicit human annotations. This vastly increases the available training data and has led to large improvements on zero-shot classification benchmarks [33,19,44,32]. While any of \
the recent image-text models are compatible with our approach, our model and dataset are most similar to LiT [44] and ALIGN [19].<br></br>\
<b>Closed-Vocabulary Object Detection.</b> Object detection models have been traditionally formulated for closed-vocabulary settings. Initially, “one-stage” and “two-stage” detectors, such as SSD [28] \
and Faster-RCNN [34] respectively, proliferated. More recently, DETR [6] showed that object detection can be framed as a set prediction problem, trained with bipartite matching, and achieve competitive \
results. Notably, such architectures do not require region proposal generation or non-maximum suppression. Follow-up works have proposed more efficient variants of DETR [48,41,37], including architectures \
without a “decoder-stage” [9]. Our work also simplifies DETR, in that we do not use a decoder. Compared to [9], which uses additional “detection” tokens, we further simplify the model by predicting one object \
instance directly from each image token.<br></br>\
<b>Long-Tailed and Open-Vocabulary Object Detection.</b> To go beyond a closed vocabulary, fixed classification layers can be replaced by language em-beddings to create open-vocabulary detectors [2]. \
Open-vocabulary object detection has recently seen much progress from combining contrastively trained image-text models and classic object detectors [12,20,26,45,46,42]. The main \
challenge in this task is how to transfer the image-level representations of the image-text backbone to detection despite the scarcity of localized annotations for \
rare classes. Making efficient use of the image-text pre-training is crucial since it allows for scaling without the need for expensive human annotations. Various approaches have been proposed. \
<b>ViLD</b> [12] distills embeddings obtained by applying CLIP or ALIGN to cropped image regions from a class-agnostic region proposal network (RPN). The RPN, however, limits generalization performance \
on novel objects, which is exacerbated by ViLD\'s two-step distillationtraining process. Multistage training is also used by <b>RegionCLIP</b>, which generates pseudo-labels on captioning data, followed by \
region-text contrastive pretraining, and transfer to detection. In contrast, our method fine-tunes both image and text models end-to-end on publicly available detection datasets, which simplifies training \
and improves generalization to unseen classes. <b>MDETR</b> [20] and <b>GLIP</b> [26] use a single text query for the whole image and formulate detection as the phrase grounding problem. This limits the \
number of object categories that can be processed per forward pass. Our architecture is simpler and more flexible in that it performs no image-text fusion and can handle multiple independent \
text or image-derived queries. <b>OVR-CNN</b> [42] is most similar to our approach in that it fine-tunes an image-text model to detection on a limited vocabulary and relies on image-text pre-training \
for generalization to an open vocabulary. However, we differ in all modelling and loss function choices. We use ViT [22] instead of their ResNet [15], a DETR-like model instead of their Faster-RCNN [34] \
and image-text pre-training as in LiT [44] instead of their PixelBERT [18] and visual grounding loss. Orthogonal to our approach, <b>Detic</b> [46] improves long-tail detection performance with weak supervision \
by training only the classification head on examples where only image-level annotations are available. </br><br>We note that in our definition of <i>open-vocabulary</i> detection, object categories \
may overlap between detection training and testing. When we specifically refer to detecting categories for which no localized instances were seen during training, we use the term <i>zero-shot</i>.</br><br>\
<b>Image-Conditioned Detection.</b> Related to open-vocabulary detection is the task of image-conditioned detection, which refers to the ability to detect objects matching a single <i>query image</i> which shows \
an object of the category in question [4,16,7,31]. This task is also called <i>one-shot object detection</i> because the query image is essentially a single training example. Image-based querying allows openworld \
detection when even the <i>name</i> of the object is unknown, e.g. for unique objects or specialized technical parts. Our model can perform this task without modifications by simply using image-derived instead of \
text-derived embeddings as queries. Recent prior works on this problem have focused mainly on architectural innovations, for example using sophisticated forms of cross-attention between the query and target image \
[16,7]. Our approach instead relies on a simple but large model and extensive image-text pre-training.\
</p>',
      unsafe_allow_html=True)


def method():
  """Returns method section."""
  st.subheader('Method')
  st.markdown(
      '<p align="justify">Our goal is to create a simple and scalable open-vocabulary object detector. We \
focus on standard Transformer-based models because of their scalability [22] and \
success in closed-vocabulary detection [6]. We present a two-stage recipe: <ol type="1">\
  <li>Contrastively pre-train image and text encoders on large-scale image-text data.</li>\
  <li>Add detection heads and fine-tune on medium-sized detection data.</li>\
</ol> The model can then be queried in different ways to perform open-vocabulary or few-shot detection.</p>',
      unsafe_allow_html=True)

  with st.expander('3.1. Model'):
    st.markdown(
        '<p align="justify"><b>Architecture.</b> Our model uses a standard Vision Transformer as the image \
encoder and a similar Transformer architecture as the text encoder (Figure 1). \
To adapt the image encoder for detection, we remove the token pooling and final \
projection layer, and instead linearly project each output token representation \
to obtain per-object image embeddings for classification (Figure 1, right). The \
maximum number of predicted objects is therefore equal to the number of tokens \
(sequence length) of the image encoder. This is not a bottleneck in practice since \
the sequence length of our models is at least 576 (ViT-B/32 at input size 768 × \
768), which is larger than the maximum number of instances in today’s datasets \
(e.g., 294 instances for LVIS [13]). Box coordinates are obtained by passing \
token representations through a small MLP. Our setup resembles DETR [6], but \
is simplified by removing the decoder.</br></br><b>Open-vocabulary object detection.</b> For open-vocabulary classification of \
detected objects, we follow prior work and use text embeddings, rather than \
learned class embeddings, in the output layer of the classification head [2]. The \
text embeddings, which we call queries, are obtained by passing category names \
or other textual object descriptions through the text encoder. The task of the \
model then becomes to predict, for each object, a bounding box and a probability \
with which each query applies to the object. Queries can be different for each \
image. In effect, each image therefore has its own discriminative label space, \
which is defined by a set of text strings. This approach subsumes classical closed-vocabulary \
object detection as the special case in which the complete set of object category names is used as query set for each image. \
<br>&emsp;&emsp;In contrast to several other methods [26,20], we do not combine all queries \
for an image into a single token sequence. Instead, each query consists of a separate token sequence which represents an individual object description, and is \
individually processed by the text encoder. In addition, our architecture includes \
no fusion between image and text encoders. Although early fusion seems intuitively beneficial, it dramatically reduces inference efficiency because encoding \
a query requires a forward pass through the entire image model and needs to \
be repeated for each image/query combination. In our setup, we can compute \
query embeddings independently of the image, allowing us to use thousands of \
queries per image, many more than is possible with early fusion [26].</br></br> \
<b>One or Few-Shot Transfer.</b> Our setup does not require query embeddings to \
be of textual origin. Since there is no fusion between image and text encoders, we \
can supply image- instead of text-derived embeddings as queries to the classification head without modifying the model. By using embeddings of prototypical \
object images as queries, our model can thus perform image-conditioned oneshot object detection. Using image embeddings as queries allows detection of \
objects which would be hard to describe in text. \
</p>',
        unsafe_allow_html=True)

  with st.expander('3.2. Training'):
    st.markdown(
        '<p align="justify"><b>Image-Level Contrastive Pre-Training.</b> We pre-train the image and text \
encoder contrastively using the same image-text dataset and loss as in [44] (Figure 1, left). We train both encoders from scratch with random initialization\
with a contrastive loss on the image and text representations. For the image \
representation, we use multihead attention pooling (MAP) [25,43] to aggregate \
token representation. The text representation is obtained from the final end-ofsequence (EOS) token of the text encoder. \
Alternatively, we use publicly available pre-trained CLIP models [33] (details in Appendix A1.3). \
<br>&emsp;&emsp;An advantage of our encoder-only architecture is that nearly all of the model\'s \
parameters (image and text encoder) can benefit from image-level pre-training. \
 The detection-specific heads contain at most 1.1% (depending on the model size)</br></br> \
<b>Training the Detector.</b> Fine-tuning of pre-trained models for <i>classification</i> is \
a well-studied problem. Classifiers, especially large Transformers, require carefully tuned regularization and \
data augmentation to perform well. Recipes for classifier training are now well established in the literature [39,38,3]. Here, we \
aim to provide a similar fine-tuning recipe for <i>open-vocabulary detection</i>. \
<br>&emsp;&emsp;The general detection training procedure of our model is almost identical to \
that for closed-vocabulary detectors, except that we provide the set of object \
category names as queries for each image. The classification head therefore outputs logits over the per-image label space defined by the queries, rather than a \
fixed global label space. \
<br>&emsp;&emsp;We use the bipartite matching loss introduced by DETR [6], but adapt it \
to long-tailed/open-vocabulary detection as follows. Due to the effort required \
for annotating detection datasets exhaustively, datasets with large numbers of \
classes are annotated in a federated manner [13,24]. Such datasets have nondisjoint label spaces, which means that each object can have multiple labels. We \
therefore use focal sigmoid cross-entropy [48] instead of softmax cross-entropy \
as the classification loss. Further, since not all object categories are annotated \
in every image, federated datasets provide both positive (present) and negative \
(known to be absent) annotations for each image. During training, for a given \
image, we use all its positive and negative annotations as queries. Additionally, \
we randomly sample categories in proportion to their frequency in the data and \
add them as “pseudo-negatives” to have at least 50 negatives per image [47]. \
<br>&emsp;&emsp;Even the largest federated detection datasets contain only ≈ 10<sup>6</sup> \
images, which is small in contrast to the billions of image-level weak labels which exist \
of the parameters of the model.\
</p>',
        unsafe_allow_html=True)


def experiments():
  """Returns experiments section."""
  st.subheader('Experiments')
  with st.expander('4.1. Model Details'):
    st.markdown('<p align="justify"></p>', unsafe_allow_html=True)
    st.info('WIP.', icon='⏳')

  with st.expander('4.2. Detection Data'):
    st.markdown('<p align="justify"></p>', unsafe_allow_html=True)
    st.info('WIP.', icon='⏳')

  with st.expander('4.3. Open-Vocabulary Detection Performance'):
    st.markdown('<p align="justify"></p>', unsafe_allow_html=True)
    st.info('WIP.', icon='⏳')

  with st.expander('4.4. Few-Shot Image-Conditioned Detection Performance'):
    st.markdown('<p align="justify"></p>', unsafe_allow_html=True)
    st.info('WIP.', icon='⏳')

  with st.expander('4.5. Scaling of Image-Level Pre-Training'):
    st.markdown('<p align="justify"></p>', unsafe_allow_html=True)
    st.info('WIP.', icon='⏳')

  with st.expander('4.6. How to Unlock Pre-Training Potential for Detection'):
    st.markdown('<p align="justify"></p>', unsafe_allow_html=True)
    st.info('WIP.', icon='⏳')


def conclusion():
  """Returns conclusion section."""
  st.subheader('Conclusion')
  st.markdown(
      '<p align="justify">We presented a simple recipe for transferring contrastively trained image-text \
models to detection. Our method achieves zero-shot detection results competitive with much more complex approaches on the challenging LVIS benchmark \
and outperforms existing methods on image-conditioned detection by a large \
margin. Our results suggest that pre-training on billions of image-text examples \
confers strong generalization ability that can be transferred to detection even if \
only relatively limited object-level data are available (millions of examples). In \
our analyses we disentangle the determinants of successful transfer of image-level \
representations to detection, and show that pre-training simple, scalable architectures \
on more data leads to strong zero-shot detection performance, mirroring \
previous observations for image classification tasks. We hope that our model will \
serve as a strong starting point for further research on open-world detection.<br> \
<br><b>Acknowledgements.</b> We would like to thank Sunayana Rane and Rianne van \
den Berg for help with the DETR implementation, Lucas Beyer for the data \
deduplication code, and Yi Tay for useful advice.</p>',
      unsafe_allow_html=True)


def references():
  """Returns references section."""
  st.subheader('References')
  with st.expander('Expand'):
    st.markdown(
        '<p align="justify"><ol type="1">\
<li>Arnab, A., Dehghani, M., Heigold, G., Sun, C., Luˇci´c, M., Schmid, C.: ViViT: A video vision transformer. In: ICCV. pp. 6836–6846 (October 2021)</li>\
<li>Bansal, A., Sikka, K., Sharma, G., Chellappa, R., Divakaran, A.: Zero-shot object detection. In: ECCV (September 2018)</li>\
<li>Bello, I., Fedus, W., Du, X., Cubuk, E.D., Srinivas, A., Lin, T.Y., Shlens, J., Zoph, B.: Revisiting ResNets: Improved training and scaling strategies. NeurIPS <b>34</b> (2021)</li>\
<li>Biswas, S.K., Milanfar, P.: One shot detection with laplacian object and fast matrix cosine similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence <b>38</b>(3), 546–562 (2016)</li>\
<li>Bradbury, J., Frostig, R., Hawkins, P., Johnson, M.J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., Zhang, Q.: JAX: composable transformations of Python+NumPy programs (2018), http://github.com/google/jax</li>\
<li>Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: ECCV. pp. 213–229. Springer International Publishing, Cham (2020)</li>\
<li>Chen, D.J., Hsieh, H.Y., Liu, T.L.: Adaptive image transformer for one-shot object detection. In: CVPR. pp. 12242–12251 (2021)</li>\
<li>Dehghani, M., Gritsenko, A.A., Arnab, A., Minderer, M., Tay, Y.: SCENIC: A JAX library for computer vision research and beyond. arXiv preprint arXiv:2110.11403 (2021)</li>\
<li>Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J., Liu, W.: You only look at one sequence: Rethinking transformer in vision through object detection. In: NeurIPS. vol. 34 (2021)</li>\
<li>Frome, A., Corrado, G.S., Shlens, J., Bengio, S., Dean, J., Ranzato, M., Mikolov, T.: Devise: A deep visual-semantic embedding model. In: NeurIPS. vol. 26 (2013)</li>\
<li>Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.Y., Cubuk, E.D., Le, Q.V., Zoph, B.: Simple copy-paste is a strong data augmentation method for instance segmentation. In: CVPR. pp. 2918–2928 (2021)</li>\
<li>Gu, X., Lin, T.Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921 (2021)</li>\
<li>Gupta, A., Dollar, P., Girshick, R.: LVIS: A dataset for large vocabulary instance segmentation. In: CVPR (June 2019)</li>\
<li>He, K., Gkioxari, G., Dollar, P., Girshick, R.: Mask R-CNN. In: ICCV (2017)</li>\
<li>He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (June 2016)</li>\
<li>Hsieh, T.I., Lo, Y.C., Chen, H.T., Liu, T.L.: One-shot object detection with coattention and co-excitation. In: NeurIPS. vol. 32. Curran Associates, Inc. (2019)</li>\
<li>Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: ECCV. pp. 646–661. Springer International Publishing, Cham (2016)</li>\
<li>Huang, Z., Zeng, Z., Liu, B., Fu, D., Fu, J.: Pixel-BERT: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849 (2020)</li>\
<li>Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: ICML. vol. 139, pp. 4904–4916. PMLR (2021)</li>\
<li>Kamath, A., Singh, M., LeCun, Y., Synnaeve, G., Misra, I., Carion, N.: MDETR - modulated detection for end-to-end multi-modal understanding. In: ICCV. pp. 1780–1790 (2021)</li>\
<li>Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., Houlsby, N.: Big transfer (BiT): General visual representation learning. In: ECCV. pp. 491–507. Springer International Publishing, Cham (2020)</li>\
<li>Kolesnikov, A., Dosovitskiy, A., Weissenborn, D., Heigold, G., Uszkoreit, J., Beyer, L., Minderer, M., Dehghani, M., Houlsby, N., Gelly, S., Unterthiner, T., Zhai, X.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)</li>\
<li>Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision 123(1), 32–73 (2017)</li>\
<li>Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., Duerig, T., Ferrari, V.: The Open Images Dataset V4. International Journal of Computer Vision 128(7), 1956–1981 (Mar 2020)</li>\
<li>Lee, J., Lee, Y., Kim, J., Kosiorek, A.R., Choi, S., Teh, Y.W.: Set transformer: A framework for attention-based permutation-invariant neural networks. In: ICML. Proceedings of Machine Learning Research, vol. 97, pp. 3744–3753. PMLR (2019)</li>\
<li>Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J.N., et al.: Grounded language-image pre-training. arXiv preprint arXiv:2112.03857 (2021)</li>\
<li>Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV. pp. 740–755. Springer International Publishing, Cham (2014)</li>\
<li>Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., Berg, A.C.: SSD: Single shot multibox detector. In: ECCV. pp. 21–37. Springer International Publishing, Cham (2016)</li>\
<li>Mahajan, D., Girshick, R., Ramanathan, V., He, K., Paluri, M., Li, Y., Bharambe, A., van der Maaten, L.: Exploring the limits of weakly supervised pretraining. In: ECCV. pp. 185–201. Springer International Publishing, Cham (2018)</li>\
<li>Michaelis, C., Ustyuzhaninov, I., Bethge, M., Ecker, A.S.: One-shot instance segmentation. arXiv preprint arXiv:1811.11507 (2018)</li>\
<li>Osokin, A., Sumin, D., Lomakin, V.: OS2D: One-stage one-shot object detection by matching anchor features. In: ECCV. pp. 635–652. Springer International Publishing, Cham (2020)</li>\
<li>Pham, H., Dai, Z., Ghiasi, G., Liu, H., Yu, A.W., Luong, M.T., Tan, M., Le, Q.V.: Combined scaling for zero-shot transfer learning. arXiv preprint arXiv:2111.10050 (2021)</li>\
<li>Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML. vol. 139, pp. 8748– 8763. PMLR (18–24 Jul 2021)</li>\
<li>Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks. In: NeurIPS. vol. 28. Curran Associates, Inc. (2015)</li>\
<li>Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365: A Large-Scale, High-Quality Dataset for Object Detection. In: ICCV. pp. 8429– 8438 (2019)</li>\
<li>Socher, R., Ganjoo, M., Manning, C.D., Ng, A.: Zero-shot learning through crossmodal transfer. NeurIPS 26 (2013)</li>\
<li>Song, H., Sun, D., Chun, S., Jampani, V., Han, D., Heo, B., Kim, W., Yang, M.H.: ViDT: An efficient and effective fully transformer-based object detector. In: ICLR (2022)</li>\
<li>Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., Beyer, L.: How to train your ViT? data, augmentation, and regularization in vision transformers. arXiv preprint arXiv:2106.10270 (2021)</li>\
<li>Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jegou, H.: Training data-efficient image transformers and distillation through attention. In: ICML. vol. 139, pp. 10347–10357 (July 2021)</li>\
<li>Xian, Y., Lampert, C.H., Schiele, B., Akata, Z.: Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly. IEEE transactions on pattern analysis and machine intelligence 41(9), 2251–2265 (2018)</li>\
<li>Yao, Z., Ai, J., Li, B., Zhang, C.: Efficient detr: improving end-to-end object detector with dense prior. arXiv preprint arXiv:2104.01318 (2021)</li>\
<li>Zareian, A., Rosa, K.D., Hu, D.H., Chang, S.F.: Open-vocabulary object detection using captions. In: CVPR. pp. 14393–14402 (June 2021)</li>\
<li>Zhai, X., Kolesnikov, A., Houlsby, N., Beyer, L.: Scaling vision transformers. arXiv preprint arXiv:2106.04560 (2021)</li>\
<li>Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., Beyer, L.: LiT: Zero-shot transfer with locked-image text tuning. arXiv preprint arXiv:2111.07991 (2021)</li>\
<li>Zhong, Y., Yang, J., Zhang, P., Li, C., Codella, N., Li, L.H., Zhou, L., Dai, X., Yuan, L., Li, Y., et al.: RegionCLIP: Region-based language-image pretraining. arXiv preprint arXiv:2112.09106 (2021)</li>\
<li>Zhou, X., Girdhar, R., Joulin, A., Kr¨ahenb¨uhl, P., Misra, I.: Detecting twentythousand classes using image-level supervision. In: arXiv preprint arXiv:2201.02605 (2021)</li>\
<li>Zhou, X., Koltun, V., Kr¨ahenb¨uhl, P.: Probabilistic two-stage detection. arXiv preprint arXiv:2103.07461 (2021)</li>\
<li>Zhu, X., Su, W., Lu, L., Li, B., Wang, X., Dai, J.: Deformable DETR: Deformable transformers for end-to-end object detection. In: ICLR (2021)</li>\
</p>',
        unsafe_allow_html=True)
