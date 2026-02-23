---
title: "Publications"
date: 2024-05-05
layout: "simple"
---

{{< timeline >}}

<h2>2025</h2>

{{< timelineItem icon="star" header="✨CRCE: Coreference-Retention Concept Erasure in Text-to-Image Diffusion Models" badge="BMVC 2025" subheader="Mar, 2025" >}}

<img src="CRCE.png" width="250px" style="float:left;margin-right:10px">

<b>Yuyang Xue</b>, E Moroshko, F Chen, J Sun, S McDonagh, SA Tsaftaris
<br>
<b>Abstract</b>: Existing concept erasure methods struggle with under-erasure (leaving residual traces) or over-erasure (eliminating unrelated concepts). We propose CRCE, which leverages LLMs to identify semantically related coreferential concepts to erase alongside the target and distinct concepts to preserve, enabling precise concept removal without unintended collateral damage. CRCE outperforms existing methods on object, identity, and IP erasure tasks.
<br>

<a href="https://arxiv.org/abs/2503.14232">[Paper]</a>
{{< /timelineItem >}}

{{< timelineItem icon="code" header="SWiFT: Soft-Mask Weight Fine-tuning for Bias Mitigation" badge="MELBA 2025" subheader="Aug, 2025" >}}

J Yan, F Chen, <b>Yuyang Xue</b>, Y Du, K Vilouras, SA Tsaftaris, S McDonagh
<br>
<b>Abstract</b>: SWiFT finds the relative and distinct contributions of model parameters to both bias and predictive performance, applying a two-step fine-tuning process with different gradient flows per parameter. The method consistently reduces model bias while maintaining competitive or superior diagnostic accuracy across dermatological and chest X-ray datasets, requiring only a small external dataset.
<br>

<a href="https://arxiv.org/abs/2508.18826">[Paper]</a>
{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="The State-of-the-Art in Cardiac MRI Reconstruction: Results of the CMRxRecon Challenge in MICCAI 2023" badge="Medical Image Analysis 2025" subheader="2025" >}}

J Lyu, C Qin, S Wang, F Wang, ..., <b>Yuyang Xue</b>, ..., SA Tsaftaris
<br>
The CMRxRecon challenge at MICCAI 2023 benchmarked deep learning-based cardiac MRI reconstruction. Over 285 teams participated; 22 submitted solutions. All competing methods used deep learning, with E2E-VarNet achieving top performance. This paper summarizes results, winning approaches, and future directions for accelerated cardiac MRI.
<br>

<a href="https://arxiv.org/abs/2404.01082">[Paper]</a>
<a href="https://github.com/CmrxRecon/CMRxRecon">[Code]</a>
{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Do Generative Models Learn Rare Generative Factors?" badge="Frontiers in AI 2025" subheader="2025" >}}

F Haider, E Moroshko, <b>Yuyang Xue</b>, SA Tsaftaris
<br>
An empirical investigation into whether generative models adequately capture rare factors of variation in training data, with implications for fairness and diversity in AI-generated content.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="MHAVSR: A Multi-Layer Hybrid Alignment Network for Video Super-Resolution" badge="Neurocomputing 2025" subheader="2025" >}}

X Qiu, Y Zhou, X Zhang, <b>Yuyang Xue</b>, X Lin, X Dai, H Tang, G Liu, R Yang, Z Li, et al.
<br>
A multi-layer hybrid alignment network for video super-resolution exploiting temporal correlations across frames via hybrid deformable alignment.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="A Universal Parameter-Efficient Fine-Tuning Approach for Stereo Image Super-Resolution" badge="EAAI 2025" subheader="2025" >}}

Y Zhou, <b>Yuyang Xue</b>, X Zhang, W Deng, T Wang, T Tan, Q Gao, T Tong
<br>
A parameter-efficient fine-tuning framework that adapts a pretrained single-image SR model to the stereo setting, achieving competitive performance with significantly reduced trainable parameters.
<br>

{{< /timelineItem >}}

<h2>2024</h2>

{{< timelineItem icon="star" header="✨BMFT: Achieving Fairness via Bias-based Weight Masking Fine-tuning" badge="MICCAI 2024 (Oral)" subheader="Oct, 2024" >}}

<img src="BMFT.png" width="250px" style="float:left;margin-right:10px">

<b>Yuyang Xue</b>, J Yan, R Dutt, F Haider, J Liu, S McDonagh, SA Tsaftaris
<br>
<b>Abstract</b>: We propose BMFT, a post-processing method that enhances model fairness in significantly fewer epochs without requiring original training data. BMFT produces a mask over model parameters to identify weights most responsible for biased predictions, then fine-tunes them in two phases: first updating the feature extractor, then reinitializing and fine-tuning the classification layer.
<br>

<a href="https://arxiv.org/abs/2408.06890">[Paper]</a>
<a href="https://github.com/vios-s/BMFT">[Code]</a>
{{< /timelineItem >}}

{{< timelineItem icon="star" header="✨Erase to Enhance: Data-Efficient Machine Unlearning in MRI Reconstruction" badge="MIDL 2024" subheader="May, 2024" >}}

<img src="E2E.png" width="250px" style="float:left;margin-right:10px">

<b>Yuyang Xue</b>, J Liu, S McDonagh, SA Tsaftaris
<br>
<b>Abstract</b>: Combining training data can lead to hallucinations and reduced image quality in reconstructed MRI. We use machine unlearning to remove hallucinations as a proxy for undesired data removal, showing that unlearning is achievable without full retraining. High performance is maintained even with only a subset of retain data, with implications for privacy compliance and bias mitigation.
<br>

<a href="https://arxiv.org/abs/2405.15517">[Paper]</a>
<a href="https://github.com/vios-s/ReconUnlearning">[Code]</a>
{{< /timelineItem >}}

{{< timelineItem icon="star" header="✨Inference Stage Denoising for Undersampled MRI Reconstruction" badge="ISBI 2024" subheader="Feb, 2024" >}}

<img src="isdfumr.png" width="250" style="float:left;margin-right:10px">

<b>Yuyang Xue</b>, C Qin, SA Tsaftaris
<br>
<b>Abstract</b>: We propose a conditional hyperparameter network that eliminates the need for data augmentation while maintaining robust performance under various noise levels. The model withstands various input noise levels during the test stage, and we present a hyperparameter sampling strategy that accelerates training convergence, achieving the highest accuracy and image quality in all settings compared to baselines.
<br>

<a href="https://arxiv.org/abs/2402.08692">[Paper]</a>
<a href="https://github.com/vios-s/Inference_Denoising_MRI_Recon">[Code]</a>
{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Towards Real-World Stereo Image Super-Resolution via Hybrid Degradation Model and Discriminator for Implied Stereo Image Information" badge="Expert Systems with Applications 2024" subheader="2024" >}}

Y Zhou, <b>Yuyang Xue</b>, J Bi, W He, X Zhang, J Zhang, W Deng, R Nie, J Lan, Q Gao, T Tong
<br>
A real-world stereo image super-resolution method combining a hybrid degradation pipeline with a discriminator that leverages cross-view stereo consistency information.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Two-Stage Image Colorization via Color Codebook" badge="Expert Systems with Applications 2024" subheader="2024" >}}

H Tang, Y Zhou, Y Chen, X Zhang, <b>Yuyang Xue</b>, X Lin, X Dai, X Qiu, Q Gao, T Tong
<br>
A two-stage colorization framework using a learned color codebook to produce vivid, diverse, and semantically consistent colorization results.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="A Blind Image Super-Resolution Network Guided by Kernel Estimation and Structural Prior Knowledge" badge="Scientific Reports 2024" subheader="2024" >}}

J Zhang, Y Zhou, J Bi, <b>Yuyang Xue</b>, W Deng, W He, T Zhao, K Sun, T Tong, Q Gao, et al.
<br>
A blind SR network that jointly estimates degradation kernels and exploits structural priors to handle real-world complex degradations for high-fidelity image super-resolution.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="ASteISR: Adapting Single Image Super-Resolution Pre-Trained Model for Efficient Stereo Image Super-Resolution" badge="arXiv 2024" subheader="Jul, 2024" >}}

Y Zhou, <b>Yuyang Xue</b>, W Deng, X Zhang, Q Gao, T Tong
<br>
A method to efficiently adapt powerful single-image SR pre-trained models to stereo image SR via lightweight cross-view interaction modules.
<br>

<a href="https://arxiv.org/abs/2407.03598">[Paper]</a>
{{< /timelineItem >}}

<h2>2023</h2>

{{< timelineItem icon="star" header="✨Cine Cardiac MRI Reconstruction using a Convolutional Recurrent Network with Refinement" badge="STACOM@MICCAI 2023" subheader="Sep, 2023" >}}

<img src="edipo.png" width="250" style="float:left;margin-right:10px">

<b>Yuyang Xue</b>, Y Du, G Carloni, E Pachetti, C Jordan, SA Tsaftaris
<br>
<b>Abstract</b>: We investigate a convolutional recurrent neural network (CRNN) architecture for supervised cine cardiac MRI reconstruction, combined with a single-image super-resolution refinement module. Our approach improves single-coil reconstruction by 4.4% in structural similarity and 3.9% in normalized mean square error over a plain CRNN, and applies a high-pass loss filter for greater emphasis on high-frequency details.
<br>

<a href="https://arxiv.org/abs/2309.13385">[Paper]</a>
<a href="https://github.com/vios-s/CMRxRECON_Challenge_EDIPO">[Code]</a>
{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Unveiling Fairness Biases in Deep Learning-Based Brain MRI Reconstruction" badge="CLIP@MICCAI 2023" subheader="Sep, 2023" >}}

Y Du, <b>Yuyang Xue</b>, R Dharmakumar, SA Tsaftaris
<br>
The first fairness analysis in deep learning-based brain MRI reconstruction, revealing statistically significant performance biases between gender and age subgroups. The study implements baseline ERM and rebalancing strategies to explore sources of unfairness.
<br>

<a href="https://arxiv.org/abs/2309.14392">[Paper]</a>
{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Stereo Cross Global Learnable Attention Module for Stereo Image Super-Resolution" badge="CVPR Workshop 2023" subheader="Jun, 2023" >}}

Y Zhou, <b>Yuyang Xue</b>, W Deng, R Nie, J Zhang, J Pu, Q Gao, J Lan, T Tong
<br>
A plug-and-play Stereo Cross Global Learnable Attention Module (SCGLAM) that captures long-range cross-view dependencies, outperforming prior methods on severely degraded low-resolution stereo pairs.
<br>

<a href="https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Zhou_Stereo_Cross_Global_Learnable_Attention_Module_for_Stereo_Image_Super-Resolution_CVPRW_2023_paper.html">[Paper]</a>
{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Using Less Annotation Workload to Establish a Pathological Auxiliary Diagnosis System for Gastric Cancer" badge="Cell Reports Medicine 2023" subheader="Apr, 2023" >}}

J Lan, M Chen, J Wang, M Du, Z Wu, H Zhang, <b>Yuyang Xue</b>, T Wang, L Chen, C Xu, et al.
<br>
A semi-supervised and weakly-supervised learning framework that significantly reduces annotation workload for training a gastric cancer pathological diagnosis system, achieving clinically viable performance with limited labeled data.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Prediction of Lymph Node Metastasis in Primary Gastric Cancer from Pathological Images and Clinical Data by Multimodal Multiscale Deep Learning" badge="Biomedical Signal Processing and Control 2023" subheader="2023" >}}

Z Guo, J Lan, J Wang, Z Hu, Z Wu, J Quan, Z Han, T Wang, M Du, Q Gao, ..., <b>Yuyang Xue</b>, et al.
<br>
A multimodal multiscale deep learning approach combining whole-slide pathological images with clinical data to predict lymph node metastasis in gastric cancer.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Cuss-Net: A Cascaded Unsupervised-Based Strategy and Supervised Network for Biomedical Image Diagnosis and Segmentation" badge="IEEE JBHI 2023" subheader="2023" >}}

X Zhou, Z Li, <b>Yuyang Xue</b>, S Chen, M Zheng, C Chen, Y Yu, X Nie, X Lin, L Wang, et al.
<br>
A cascaded framework combining unsupervised pretraining with supervised fine-tuning for robust biomedical image diagnosis and segmentation with limited labeled data.
<br>

{{< /timelineItem >}}

<h2>2022</h2>

{{< timelineItem icon="star" header="✨Better Performance with Transformer: CPPFormer in the Precise Prediction of Cell-Penetrating Peptides" badge="Current Medicinal Chemistry 2022" subheader="2022" >}}

<b>Yuyang Xue</b>, X Ye, L Wei, X Zhang, T Sakurai, L Wei
<br>
<b>Abstract</b>: CPPFormer applies a Transformer-based architecture to the precise prediction of cell-penetrating peptides (CPPs), leveraging self-attention to capture sequence-level dependencies. By combining the attention mechanism with a few manually engineered features, CPPFormer achieves 92.16% accuracy on the CPP924 dataset, outperforming existing CNN and RNN-based methods.
<br>

<a href="https://pubmed.ncbi.nlm.nih.gov/34544332/">[Paper]</a>
{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Enhanced Multi-Stage Network for Defocus Deblurring using Dual-Pixel Images" badge="SPIE ICSPS 2022" subheader="2022" >}}

R Li, J Xie, <b>Yuyang Xue</b>, W Zou, T Tong, M Luo, Q Gao
<br>
A multi-stage network exploiting dual-pixel sensor information to progressively restore sharp images from defocus blur.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="GLNet: Low-Light Image Enhancement via Grayscale Priors" badge="SPIE ICSPS 2022" subheader="2022" >}}

L Guo, J Xie, <b>Yuyang Xue</b>, R Li, W Zheng, T Tong, Q Gao
<br>
A grayscale-prior-guided network for low-light image enhancement that exploits structural information from the luminance channel to guide color enhancement.
<br>

{{< /timelineItem >}}

<h2>2021</h2>

{{< timelineItem icon="pencil" header="ATSE: A Peptide Toxicity Predictor by Exploiting Structural and Evolutionary Information based on Graph Neural Network and Attention Mechanism" badge="Briefings in Bioinformatics 2021" subheader="2021" >}}

L Wei, X Ye, <b>Yuyang Xue</b>, T Sakurai, L Wei
<br>
ATSE combines graph neural networks for structural modeling with attention mechanisms for evolutionary information to predict peptide toxicity, providing interpretable residue-level attention weights.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Unpaired Stain Style Transfer using Invertible Neural Networks based on Channel Attention and Long-Range Residual" badge="IEEE Access 2021" subheader="2021" >}}

J Lan, S Cai, <b>Yuyang Xue</b>, Q Gao, M Du, H Zhang, Z Wu, Y Deng, Y Huang, T Tong, et al.
<br>
An invertible neural network-based approach for unpaired histopathology stain normalization, offering exact invertibility and stable training compared to GAN-based methods, with channel attention for detail preservation.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Deep Learning Framework for Detecting Positive Lymph Nodes of Gastric Cancer on Histopathological Images" badge="ICBISP 2021" subheader="2021" >}}

Y Huang, <b>Yuyang Xue</b>, J Lan, Y Deng, G Chen, H Zhang, M Dang, T Tong
<br>
A deep learning pipeline for automatic detection of positive lymph nodes in gastric cancer histopathological whole-slide images.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Image Colorization Algorithm based on Foreground Semantic Information" badge="Journal of Computer Applications 2021" subheader="2021" >}}

L Wu, <b>Yuyang Xue</b>, T Tong, M Du, Q Gao
<br>
An automatic colorization algorithm leveraging foreground semantic segmentation to guide perceptually consistent color assignment.
<br>

{{< /timelineItem >}}

<h2>2019</h2>

{{< timelineItem icon="star" header="✨Attention Based Image Compression Post-Processing Convolutional Neural Network" badge="CVPR Workshop 2019" subheader="Jun, 2019" >}}

<b>Yuyang Xue</b>, J Su
<br>
<b>Abstract</b>: A post-processing CNN leveraging attention mechanisms to reduce compression artifacts in learned image codecs. By focusing attention on regions with the most severe distortions, the network improves perceptual quality without modifying the underlying compression algorithm.
<br>

{{< /timelineItem >}}

{{< timelineItem icon="pencil" header="Stain Style Transfer using Transitive Adversarial Networks" badge="MICCAI Workshop 2019" subheader="2019" >}}

S Cai, <b>Yuyang Xue</b>, Q Gao, M Du, G Chen, H Zhang, T Tong
<br>
A GAN-based stain style transfer method using transitive adversarial learning to handle multi-domain stain normalization without requiring direct paired data between all domain pairs.
<br>

{{< /timelineItem >}}

<h2>2017</h2>

{{< timelineItem icon="pencil" header="Image Color Correction Database for Subjective Perceptual Consistency Assessment" badge="Acta Electronica Sinica 2017" subheader="2017" >}}

H Zhang, Y Niu, <b>Yuyang Xue</b>
<br>
A curated database for benchmarking image color correction algorithms against human subjective perceptual consistency judgments.
<br>

{{< /timelineItem >}}

{{< /timeline >}}
