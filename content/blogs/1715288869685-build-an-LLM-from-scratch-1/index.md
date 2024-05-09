---
title: "Build a Large Language Model (from scratch) 1"
date: 2024-05-09
draft: false
series: ["LLM from scratch"]
series_order: 1
description: "An exercise to go through LLM training, from data processing to deploying."
tags: ["LLMs", "Hands on", "Book"]
---
 
{{< alert "heart" >}}
This is a series of blogs that detailed recorded the exploration of the book [Build a Large Language Model (From Scratch)](http://mng.bz/orYv). The corresponding code repository is [here](https://github.com/rasbt/LLMs-from-scratch). Please support the author and the book.
{{< /alert >}}

## Introduction

### What is LLM?

- An LLM, a large language model, is a neural network designed to understand, generate, and respond to human-like text.
  - The "large" in large language model refers to both the model's size in terms of parameters and the immense dataset on which it's trained.
- LLMs utilize an architecture called the *transformer*, which allows them to pay selective attention to different parts of the input when making predictions, making them especially adept at handling the nuances and complexities of human language.
- LLMs are also often referred to as a form of generative artificial intelligence (AI), often abbreviated as generative AI or GenAI.

### Applications of LLMs

- Machine translation
- Text generation
- Sentiment analysis
- Text summarisation
- and many others.

### Stages of building and using LLMs

![](pretraining.png)

- The general process of creating an LLM includes pretraining and finetuning.
  - This pretrained model then serves as a foundational resource that can be further refined through finetuning, a process where the model is specifically trained on a narrower dataset that is more specific to particular tasks or domains.
  - The two most popular categories of finetuning LLMs include *instruction finetuning* and *finetuning for classification* tasks.
    - In instruction-finetuning, the labeled dataset consists of instruction and answer pairs, such as a query to translate a text accompanied by the correctly translated text. 
    - In classification finetuning, the labeled dataset consists of texts and associated class labels, for example, emails associated with spam and non-spam labels.




