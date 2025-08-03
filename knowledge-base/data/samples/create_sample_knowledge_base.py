#!/usr/bin/env python3
"""
Create Sample Knowledge Base

This script creates a comprehensive sample knowledge base with various
document types, entities, and relationships for testing and demonstration.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


def create_ai_research_papers():
    """Create sample AI research papers"""
    papers = [
        {
            "title": "Attention Is All You Need: Understanding Transformer Architecture",
            "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N.", "Uszkoreit, J."],
            "abstract": """
The Transformer architecture has revolutionized natural language processing by 
relying entirely on attention mechanisms, dispensing with recurrence and convolutions. 
This paper introduces the groundbreaking "Attention Is All You Need" concept that 
forms the foundation of modern large language models.

The key innovation lies in the multi-head self-attention mechanism that allows 
the model to weigh the importance of different parts of the input sequence when 
processing each element. This enables parallel processing and better handling 
of long-range dependencies compared to traditional RNN and CNN architectures.

The Transformer consists of an encoder-decoder structure where both components 
are composed of stacks of identical layers. Each encoder layer has two sub-layers: 
a multi-head self-attention mechanism and a position-wise fully connected feed-forward 
network. The decoder has an additional sub-layer that performs multi-head attention 
over the encoder output.

Experimental results on machine translation tasks demonstrate that Transformer 
models achieve superior performance while being more parallelizable and requiring 
significantly less time to train. The model achieved state-of-the-art results 
on the WMT 2014 English-to-German and English-to-French translation tasks.

This architecture has since become the foundation for breakthrough models like 
BERT, GPT, T5, and many others, fundamentally changing the landscape of NLP 
and enabling the development of large language models that power modern AI applications.
            """.strip(),
            "content": """
1. Introduction

The field of natural language processing has been dominated by sequence-to-sequence 
models based on recurrent neural networks (RNNs) and convolutional neural networks (CNNs). 
These models process sequences sequentially, which limits parallelization and can 
struggle with long-range dependencies.

2. Model Architecture

The Transformer follows an encoder-decoder structure:

2.1 Encoder
The encoder is composed of N=6 identical layers. Each layer has two sub-layers:
- Multi-head self-attention mechanism
- Position-wise fully connected feed-forward network

2.2 Decoder  
The decoder is also composed of N=6 identical layers with an additional sub-layer:
- Masked multi-head self-attention
- Multi-head attention over encoder output
- Position-wise fully connected feed-forward network

2.3 Attention Mechanism
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Multi-head attention allows the model to jointly attend to information from 
different representation subspaces at different positions.

3. Training and Results

The model was trained on the WMT 2014 English-German dataset (4.5M sentence pairs) 
and WMT 2014 English-French dataset (36M sentence pairs). Training used Adam 
optimizer with β₁=0.9, β₂=0.98, and ε=10⁻⁹.

Results:
- WMT 2014 EN-DE: 28.4 BLEU score
- WMT 2014 EN-FR: 41.8 BLEU score

4. Conclusion

The Transformer architecture demonstrates that attention mechanisms alone are 
sufficient for achieving state-of-the-art results in sequence transduction tasks. 
The model's parallelizability and superior performance make it an attractive 
alternative to recurrent and convolutional models.
            """.strip(),
            "keywords": ["transformer", "attention", "neural networks", "NLP", "machine translation"],
            "publication_date": "2017-06-12",
            "venue": "NIPS 2017",
            "citations": 45000,
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Devlin, J.", "Chang, M.", "Lee, K.", "Toutanova, K."],
            "abstract": """
BERT (Bidirectional Encoder Representations from Transformers) represents a 
significant advancement in language representation learning. Unlike previous 
models that process text left-to-right or right-to-left, BERT trains 
bidirectional representations by jointly conditioning on both left and right 
context in all layers.

The pre-training approach uses two novel unsupervised tasks: Masked Language 
Modeling (MLM) and Next Sentence Prediction (NSP). These tasks enable BERT 
to learn rich bidirectional representations that can be fine-tuned for a 
wide variety of downstream tasks with minimal task-specific modifications.

BERT achieves state-of-the-art results on eleven natural language processing 
tasks, including question answering, natural language inference, and text 
classification. The model demonstrates the power of transfer learning in NLP, 
where a single pre-trained model can be adapted to numerous specific tasks.
            """.strip(),
            "content": """
1. Introduction

Language model pre-training has shown significant improvements in many NLP tasks. 
There are two existing strategies for applying pre-trained language representations 
to downstream tasks: feature-based and fine-tuning approaches.

BERT improves upon previous methods by removing the unidirectional constraint 
through a "masked language model" (MLM) pre-training objective.

2. Model Architecture

BERT uses a multi-layer bidirectional Transformer encoder based on the original 
Transformer implementation. The model architecture is characterized by:

- L: number of layers (Transformer blocks)
- H: hidden size
- A: number of self-attention heads

BERT_BASE: L=12, H=768, A=12, Total Parameters=110M
BERT_LARGE: L=24, H=1024, A=16, Total Parameters=340M

3. Pre-training Tasks

3.1 Masked Language Model (MLM)
Randomly mask 15% of input tokens and predict the masked words. This allows 
the model to learn bidirectional representations.

3.2 Next Sentence Prediction (NSP)  
Given two sentences A and B, predict whether B is the actual next sentence 
that follows A in the original document.

4. Fine-tuning Procedure

For each downstream task, we simply plug in the task-specific inputs and 
outputs into BERT and fine-tune all parameters end-to-end.

5. Experiments and Results

BERT achieves new state-of-the-art results on:
- GLUE score: 80.5% (7.7% improvement)
- MultiNLI accuracy: 86.7% (4.6% improvement)  
- SQuAD v1.1 F1: 93.2% (1.5% improvement)
- SQuAD v2.0 F1: 83.1% (5.1% improvement)

6. Conclusion

BERT demonstrates that bidirectional pre-training is crucial for language 
representations. The conceptual simplicity and empirical power of BERT has 
led to widespread adoption and numerous improvements in the field.
            """.strip(),
            "keywords": ["BERT", "bidirectional", "transformers", "pre-training", "fine-tuning", "NLP"],
            "publication_date": "2018-10-11",
            "venue": "NAACL 2019", 
            "citations": 35000,
            "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf"
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "authors": ["Brown, T.", "Mann, B.", "Ryder, N.", "Subbiah, M."],
            "abstract": """
GPT-3 demonstrates that scaling up language models leads to qualitatively 
different capabilities. With 175 billion parameters, GPT-3 shows strong 
performance on many NLP tasks without task-specific fine-tuning, using 
only a few examples or even just a task description.

The model exhibits emergent abilities in few-shot, one-shot, and zero-shot 
settings across a diverse range of tasks including translation, question 
answering, cloze tasks, unscrambling words, and novel tasks like arithmetic 
and creative writing.

This work suggests that scaling language models is a promising path toward 
general artificial intelligence, though important limitations and potential 
risks must be carefully considered.
            """.strip(),
            "content": """
1. Introduction

Recent advances in language modeling have demonstrated that increasing model 
size improves performance and sample efficiency on a wide range of downstream 
tasks. This work tests whether this trend continues by training a 175B parameter 
autoregressive language model.

2. Approach

GPT-3 uses the same architecture as GPT-2, with a few modifications:
- Alternating dense and locally banded sparse attention patterns
- 175 billion parameters
- 2048-token context length
- Trained on ~300B tokens from diverse internet text

Model configurations:
- GPT-3 Small: 125M parameters
- GPT-3 Medium: 350M parameters  
- GPT-3 Large: 760M parameters
- GPT-3 XL: 1.3B parameters
- GPT-3 2.7B: 2.7B parameters
- GPT-3 6.7B: 6.7B parameters
- GPT-3 13B: 13B parameters
- GPT-3 175B: 175B parameters

3. Training Dataset

The training dataset consists of:
- Common Crawl (filtered): 410B tokens
- WebText2: 19B tokens
- Books1: 12B tokens
- Books2: 55B tokens
- Wikipedia: 3B tokens

4. Results

GPT-3 shows strong performance across many tasks:

4.1 Language Modeling
- Penn Tree Bank: 20.50 perplexity (state-of-the-art)
- WikiText-103: 9.9 perplexity

4.2 Few-Shot Learning
- SuperGLUE: 71.8% (few-shot) vs 89.3% (fine-tuned SOTA)
- Reading comprehension: Strong performance on CoQA, QuAC
- Translation: Competitive with supervised models
- Arithmetic: Can perform 2-digit addition/subtraction

4.3 Novel Tasks
- Creative writing and storytelling
- Code generation and completion
- Logical reasoning problems

5. Limitations and Risks

- Computational cost and environmental impact
- Potential for misuse in generating harmful content
- Biases present in training data
- Lack of factual accuracy guarantees
- Difficulty in interpretability

6. Conclusion

GPT-3 demonstrates that language model scaling leads to broad improvements 
across many tasks. However, significant challenges remain in areas like 
bias, safety, and alignment with human values.
            """.strip(),
            "keywords": ["GPT-3", "large language models", "few-shot learning", "scaling", "emergent abilities"],
            "publication_date": "2020-05-28",
            "venue": "NeurIPS 2020",
            "citations": 15000,
            "pdf_url": "https://arxiv.org/pdf/2005.14165.pdf"
        }
    ]
    
    return papers


def create_ai_concepts_encyclopedia():
    """Create encyclopedia entries for AI concepts"""
    concepts = [
        {
            "title": "Machine Learning",
            "category": "Artificial Intelligence",
            "definition": """
Machine Learning (ML) is a subset of artificial intelligence that focuses on 
developing algorithms and statistical models that enable computer systems to 
improve their performance on a specific task through experience, without being 
explicitly programmed for every scenario.
            """.strip(),
            "content": """
Machine Learning Overview

Machine learning represents a paradigm shift from traditional programming, where 
instead of writing explicit instructions, we train models on data to learn patterns 
and make predictions or decisions.

Types of Machine Learning:

1. Supervised Learning
- Uses labeled training data
- Goal: Learn mapping from inputs to outputs
- Examples: Classification, regression
- Algorithms: Linear regression, decision trees, neural networks

2. Unsupervised Learning  
- Uses unlabeled data
- Goal: Discover hidden patterns or structures
- Examples: Clustering, dimensionality reduction
- Algorithms: K-means, PCA, autoencoders

3. Reinforcement Learning
- Learns through interaction with environment
- Goal: Maximize cumulative reward
- Examples: Game playing, robotics
- Algorithms: Q-learning, policy gradients

4. Semi-supervised Learning
- Uses both labeled and unlabeled data
- Goal: Leverage unlabeled data to improve performance
- Examples: Self-training, co-training

Key Concepts:

Training: Process of teaching the algorithm using historical data
Features: Individual measurable properties of observed phenomena
Model: Mathematical representation learned from data
Overfitting: When model learns training data too specifically
Cross-validation: Technique to assess model generalization

Applications:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Fraud detection
- Autonomous vehicles
- Medical diagnosis
- Financial trading

Challenges:
- Data quality and quantity requirements
- Feature selection and engineering
- Model interpretability
- Bias and fairness
- Computational resources
- Generalization to new scenarios

Future Directions:
- Automated machine learning (AutoML)
- Federated learning
- Quantum machine learning
- Neuromorphic computing
- Continual learning
            """.strip(),
            "related_concepts": ["Artificial Intelligence", "Deep Learning", "Neural Networks", "Data Science"],
            "last_updated": "2024-01-15"
        },
        {
            "title": "Neural Networks", 
            "category": "Machine Learning",
            "definition": """
Neural Networks are computing systems inspired by biological neural networks 
that constitute animal brains. They consist of interconnected nodes (neurons) 
that process information through their connections and activation functions.
            """.strip(),
            "content": """
Neural Networks: Biological Inspiration Meets Computing

Neural networks represent one of the most successful approaches in machine learning, 
drawing inspiration from the structure and function of biological brains to create 
powerful computational models.

Architecture Components:

1. Neurons (Nodes)
- Basic processing units
- Receive inputs, apply activation function
- Produce output signal
- Characterized by weights and bias

2. Layers
- Input Layer: Receives raw data
- Hidden Layers: Process information
- Output Layer: Produces final results

3. Connections (Edges)
- Links between neurons
- Carry weighted signals
- Determine information flow

4. Activation Functions
- Sigmoid: S-shaped curve, outputs between 0 and 1
- ReLU: Rectified Linear Unit, outputs max(0, x)
- Tanh: Hyperbolic tangent, outputs between -1 and 1
- Softmax: Converts outputs to probability distribution

Learning Process:

1. Forward Propagation
- Input flows through network layer by layer
- Each neuron applies weighted sum and activation
- Produces prediction at output layer

2. Loss Calculation
- Compare prediction with actual target
- Calculate error using loss function
- Common losses: Mean squared error, cross-entropy

3. Backpropagation
- Calculate gradients of loss with respect to weights
- Propagate error backwards through network
- Update weights to minimize loss

4. Optimization
- Gradient descent and variants
- Adam, RMSprop, SGD optimizers
- Learning rate scheduling

Types of Neural Networks:

1. Feedforward Networks
- Information flows in one direction
- Fully connected layers
- Good for tabular data

2. Convolutional Neural Networks (CNNs)
- Specialized for grid-like data (images)
- Convolutional and pooling layers
- Translation invariant features

3. Recurrent Neural Networks (RNNs)
- Designed for sequential data
- Memory through hidden states
- Variants: LSTM, GRU

4. Transformer Networks
- Attention-based architecture
- Parallel processing capability
- State-of-the-art for NLP tasks

Training Considerations:

Hyperparameters:
- Learning rate: Controls update step size
- Batch size: Number of samples per update
- Number of epochs: Training iterations
- Network architecture: Layers, neurons

Regularization:
- Dropout: Randomly disable neurons during training
- L1/L2 regularization: Add penalty terms to loss
- Batch normalization: Normalize layer inputs
- Early stopping: Stop training when validation improves

Common Challenges:
- Vanishing/exploding gradients
- Overfitting to training data
- Computational complexity
- Hyperparameter tuning
- Interpretability

Applications:
- Computer vision: Image classification, object detection
- Natural language processing: Translation, sentiment analysis
- Speech recognition and synthesis
- Game playing: Chess, Go, video games
- Autonomous systems: Self-driving cars, drones
- Healthcare: Medical imaging, drug discovery
- Finance: Trading, risk assessment

Recent Advances:
- Transformer architectures
- Generative adversarial networks (GANs)
- Self-supervised learning
- Neural architecture search (NAS)
- Attention mechanisms
- Graph neural networks
            """.strip(),
            "related_concepts": ["Deep Learning", "Machine Learning", "Artificial Intelligence", "Backpropagation"],
            "last_updated": "2024-01-20"
        },
        {
            "title": "Natural Language Processing",
            "category": "Artificial Intelligence", 
            "definition": """
Natural Language Processing (NLP) is a branch of artificial intelligence that 
deals with the interaction between computers and human language, enabling machines 
to understand, interpret, and generate human language in a valuable way.
            """.strip(),
            "content": """
Natural Language Processing: Bridging Human and Machine Communication

Natural Language Processing represents one of the most challenging and impactful 
areas of artificial intelligence, as it attempts to bridge the gap between human 
communication and computer understanding.

Core NLP Tasks:

1. Text Preprocessing
- Tokenization: Breaking text into words/subwords
- Normalization: Converting to standard format
- Stop word removal: Filtering common words
- Stemming/Lemmatization: Reducing to root forms

2. Syntactic Analysis
- Part-of-speech tagging: Identifying word types
- Parsing: Analyzing grammatical structure
- Dependency parsing: Finding word relationships
- Constituency parsing: Hierarchical phrase structure

3. Semantic Analysis
- Named entity recognition (NER): Identifying entities
- Word sense disambiguation: Determining word meanings
- Semantic role labeling: Identifying argument roles
- Coreference resolution: Linking referring expressions

4. Pragmatic Analysis
- Intent recognition: Understanding user goals
- Sentiment analysis: Determining emotional tone
- Discourse analysis: Understanding text structure
- Context understanding: Incorporating background knowledge

Key NLP Applications:

1. Machine Translation
- Statistical machine translation (SMT)
- Neural machine translation (NMT)
- Transformer-based models (e.g., Google Translate)
- Real-time speech translation

2. Information Extraction
- Named entity extraction
- Relationship extraction  
- Event extraction
- Knowledge graph construction

3. Text Classification
- Sentiment analysis
- Topic classification
- Spam detection
- Content moderation

4. Question Answering
- Factual question answering
- Reading comprehension
- Conversational QA
- Knowledge-based QA

5. Text Generation
- Language modeling
- Summarization
- Creative writing
- Code generation

6. Conversational AI
- Chatbots and virtual assistants
- Dialogue systems
- Voice interfaces
- Customer service automation

Technical Approaches:

1. Rule-Based Systems
- Hand-crafted linguistic rules
- Regular expressions
- Context-free grammars
- Expert system approaches

2. Statistical Methods
- N-gram language models
- Hidden Markov Models (HMMs)
- Conditional Random Fields (CRFs)
- Support Vector Machines (SVMs)

3. Neural Approaches
- Recurrent Neural Networks (RNNs)
- Convolutional Neural Networks (CNNs)
- Long Short-Term Memory (LSTM)
- Transformer architectures

4. Pre-trained Language Models
- Word2Vec and GloVe embeddings
- BERT and its variants
- GPT series models
- T5, RoBERTa, ELECTRA

Modern NLP Pipeline:

1. Data Collection and Preparation
- Text corpus gathering
- Data cleaning and filtering
- Annotation and labeling
- Dataset splitting

2. Preprocessing
- Tokenization and normalization
- Subword tokenization (BPE, WordPiece)
- Handling special characters and formatting
- Language detection

3. Feature Engineering
- Bag-of-words representations
- TF-IDF vectorization
- Word embeddings
- Contextual embeddings

4. Model Training
- Architecture selection
- Hyperparameter tuning
- Training data augmentation
- Transfer learning and fine-tuning

5. Evaluation
- Intrinsic evaluation: Perplexity, BLEU scores
- Extrinsic evaluation: Task-specific metrics
- Human evaluation: Fluency, adequacy
- Bias and fairness assessment

Challenges in NLP:

1. Ambiguity
- Lexical ambiguity: Multiple word meanings
- Syntactic ambiguity: Multiple parse trees
- Semantic ambiguity: Multiple interpretations
- Pragmatic ambiguity: Context-dependent meaning

2. Context and World Knowledge
- Long-distance dependencies
- Implicit knowledge requirements
- Common sense reasoning
- Cultural and domain-specific knowledge

3. Variability and Creativity
- Linguistic diversity across languages
- Informal language and slang
- Creative language use
- Code-switching and multilingualism

4. Evaluation Challenges
- Subjective nature of language quality
- Multiple valid outputs
- Evaluation metric limitations
- Bias in evaluation datasets

Current Research Directions:

1. Large Language Models
- Scaling model size and training data
- Emergent capabilities investigation
- Efficiency and compression techniques
- Multimodal language models

2. Few-Shot and Zero-Shot Learning
- In-context learning capabilities
- Prompt engineering techniques
- Meta-learning approaches
- Cross-lingual transfer

3. Controllable Generation
- Style and attribute control
- Factual accuracy and grounding
- Ethical and safe generation
- Personalization techniques

4. Multimodal NLP
- Vision-language models
- Speech-text integration
- Video understanding
- Embodied language understanding

Future Outlook:

The field of NLP continues to evolve rapidly with advances in:
- More efficient architectures
- Better multilingual capabilities
- Improved reasoning abilities
- Enhanced factual accuracy
- Greater interpretability
- Reduced computational requirements
- More robust evaluation methods
            """.strip(),
            "related_concepts": ["Machine Learning", "Computational Linguistics", "Text Mining", "Language Models"],
            "last_updated": "2024-01-25"
        }
    ]
    
    return concepts


def create_sample_documents():
    """Create diverse sample documents for the knowledge base"""
    documents = []
    
    # Add research papers
    papers = create_ai_research_papers()
    for paper in papers:
        doc = {
            "id": str(uuid.uuid4()),
            "title": paper["title"],
            "content": f"{paper['abstract']}\n\n{paper['content']}",
            "metadata": {
                "type": "research_paper",
                "authors": paper["authors"],
                "keywords": paper["keywords"],
                "publication_date": paper["publication_date"],
                "venue": paper["venue"],
                "citations": paper["citations"],
                "pdf_url": paper.get("pdf_url"),
                "created_at": datetime.utcnow().isoformat(),
                "source": "Academic Papers Collection"
            }
        }
        documents.append(doc)
    
    # Add encyclopedia entries
    concepts = create_ai_concepts_encyclopedia()
    for concept in concepts:
        doc = {
            "id": str(uuid.uuid4()),
            "title": concept["title"],
            "content": f"{concept['definition']}\n\n{concept['content']}",
            "metadata": {
                "type": "encyclopedia_entry",
                "category": concept["category"],
                "definition": concept["definition"],
                "related_concepts": concept["related_concepts"],
                "last_updated": concept["last_updated"],
                "created_at": datetime.utcnow().isoformat(),
                "source": "AI Concepts Encyclopedia"
            }
        }
        documents.append(doc)
    
    # Add technical tutorials
    tutorials = [
        {
            "title": "Getting Started with Python for Machine Learning",
            "content": """
Python has become the de facto language for machine learning due to its simplicity, 
extensive libraries, and strong community support. This tutorial provides a 
comprehensive introduction to using Python for ML projects.

Essential Python Libraries for ML:

1. NumPy
- Fundamental package for numerical computing
- Provides support for arrays and mathematical functions
- Foundation for other ML libraries

import numpy as np
# Create arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2], [3, 4]])

2. Pandas
- Data manipulation and analysis library
- Provides DataFrame for structured data
- Essential for data preprocessing

import pandas as pd
# Load data
df = pd.read_csv('data.csv')
# Basic operations
df.head(), df.describe(), df.info()

3. Scikit-learn
- Comprehensive ML library
- Provides algorithms for classification, regression, clustering
- Tools for model evaluation and selection

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

4. Matplotlib and Seaborn
- Data visualization libraries
- Essential for exploratory data analysis
- Create plots and charts

import matplotlib.pyplot as plt
import seaborn as sns

Basic ML Workflow in Python:

1. Data Loading and Exploration
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('dataset.csv')

# Explore data
print(df.shape)
print(df.info())
print(df.describe())
```

2. Data Preprocessing
```python
# Handle missing values
df.dropna()  # or df.fillna()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

3. Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

4. Model Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Best Practices:

1. Version Control
- Use Git for code versioning
- Track experiments and results
- Maintain reproducible environments

2. Code Organization
- Structure projects with clear directories
- Use functions and classes for reusability
- Document code with docstrings

3. Experiment Tracking
- Log hyperparameters and results
- Use tools like MLflow or Weights & Biases
- Compare model performance systematically

4. Testing and Validation
- Write unit tests for data processing
- Use cross-validation for robust evaluation
- Test models on holdout datasets

Common Pitfalls to Avoid:

1. Data Leakage
- Ensure proper train/validation/test splits
- Avoid using future information
- Be careful with feature engineering

2. Overfitting
- Use regularization techniques
- Monitor validation performance
- Apply early stopping

3. Evaluation Mistakes
- Use appropriate metrics for the problem
- Consider class imbalance
- Validate on representative data

Advanced Topics:

1. Feature Engineering
- Domain-specific feature creation
- Automated feature selection
- Dimensionality reduction techniques

2. Model Selection
- Hyperparameter tuning with GridSearch
- Cross-validation strategies
- Ensemble methods

3. Production Deployment
- Model serialization with pickle/joblib
- API development with Flask/FastAPI
- Monitoring and maintenance

Resources for Learning:

1. Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Python Machine Learning" by Sebastian Raschka
- "Introduction to Statistical Learning" by James et al.

2. Online Courses
- Coursera Machine Learning Course
- Kaggle Learn modules
- Fast.ai practical deep learning

3. Practice Platforms
- Kaggle competitions
- Google Colab for experimentation
- Jupyter notebooks for exploration

This tutorial provides a foundation for getting started with Python for machine 
learning. The key is to practice with real datasets and gradually build complexity 
in your projects.
            """.strip(),
            "metadata": {
                "type": "tutorial",
                "difficulty": "beginner",
                "duration": "2-3 hours",
                "prerequisites": ["Basic Python knowledge"],
                "topics": ["Python", "Machine Learning", "Data Science"],
                "last_updated": "2024-01-10"
            }
        }
    ]
    
    for tutorial in tutorials:
        doc = {
            "id": str(uuid.uuid4()),
            "title": tutorial["title"],
            "content": tutorial["content"],
            "metadata": {
                **tutorial["metadata"],
                "created_at": datetime.utcnow().isoformat(),
                "source": "Technical Tutorials Collection"
            }
        }
        documents.append(doc)
    
    return documents


def create_entity_relationships():
    """Create sample entity relationships for knowledge graph"""
    entities = [
        {
            "id": "entity_1",
            "text": "Artificial Intelligence",
            "type": "CONCEPT",
            "properties": {
                "definition": "The simulation of human intelligence in machines",
                "field": "Computer Science",
                "established": "1950s"
            }
        },
        {
            "id": "entity_2", 
            "text": "Machine Learning",
            "type": "CONCEPT",
            "properties": {
                "definition": "A subset of AI focused on learning from data",
                "field": "Computer Science",
                "established": "1950s-1960s"
            }
        },
        {
            "id": "entity_3",
            "text": "Deep Learning", 
            "type": "CONCEPT",
            "properties": {
                "definition": "A subset of ML using neural networks with multiple layers",
                "field": "Computer Science",
                "established": "2000s"
            }
        },
        {
            "id": "entity_4",
            "text": "Neural Networks",
            "type": "CONCEPT", 
            "properties": {
                "definition": "Computing systems inspired by biological neural networks",
                "field": "Computer Science",
                "established": "1940s"
            }
        },
        {
            "id": "entity_5",
            "text": "Google",
            "type": "ORGANIZATION",
            "properties": {
                "founded": "1998",
                "industry": "Technology",
                "headquarters": "Mountain View, CA"
            }
        },
        {
            "id": "entity_6",
            "text": "OpenAI",
            "type": "ORGANIZATION",
            "properties": {
                "founded": "2015",
                "industry": "AI Research",
                "headquarters": "San Francisco, CA"
            }
        },
        {
            "id": "entity_7",
            "text": "Transformer",
            "type": "MODEL",
            "properties": {
                "introduced": "2017",
                "architecture": "Attention-based",
                "applications": "NLP"
            }
        },
        {
            "id": "entity_8",
            "text": "BERT",
            "type": "MODEL",
            "properties": {
                "introduced": "2018",
                "architecture": "Bidirectional Transformer",
                "developed_by": "Google"
            }
        }
    ]
    
    relationships = [
        {
            "id": "rel_1",
            "subject": "entity_2",  # Machine Learning
            "predicate": "is_subset_of",
            "object": "entity_1",   # Artificial Intelligence
            "properties": {"strength": 1.0}
        },
        {
            "id": "rel_2", 
            "subject": "entity_3",  # Deep Learning
            "predicate": "is_subset_of",
            "object": "entity_2",   # Machine Learning
            "properties": {"strength": 1.0}
        },
        {
            "id": "rel_3",
            "subject": "entity_3",  # Deep Learning
            "predicate": "uses",
            "object": "entity_4",   # Neural Networks
            "properties": {"strength": 0.9}
        },
        {
            "id": "rel_4",
            "subject": "entity_8",  # BERT
            "predicate": "based_on",
            "object": "entity_7",   # Transformer
            "properties": {"strength": 1.0}
        },
        {
            "id": "rel_5",
            "subject": "entity_8",  # BERT
            "predicate": "developed_by",
            "object": "entity_5",   # Google
            "properties": {"strength": 1.0}
        },
        {
            "id": "rel_6",
            "subject": "entity_7",  # Transformer
            "predicate": "uses",
            "object": "entity_4",   # Neural Networks
            "properties": {"strength": 0.8}
        }
    ]
    
    return entities, relationships


def main():
    """Create comprehensive sample knowledge base"""
    print("Creating Sample Knowledge Base...")
    
    # Ensure directories exist
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample documents
    print("Creating sample documents...")
    documents = create_sample_documents()
    
    # Save documents as individual files
    docs_dir = samples_dir / "documents"
    docs_dir.mkdir(exist_ok=True)
    
    for doc in documents:
        # Save as text file
        filename = f"{doc['id']}.txt"
        filepath = docs_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc['title']}\n")
            f.write(f"ID: {doc['id']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(doc['content'])
        
        # Save metadata as JSON
        metadata_filename = f"{doc['id']}_metadata.json"
        metadata_filepath = docs_dir / metadata_filename
        
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(doc['metadata'], f, indent=2)
    
    # Save complete document collection
    with open(samples_dir / "documents_collection.json", 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    
    # Create knowledge graph data
    print("Creating knowledge graph data...")
    entities, relationships = create_entity_relationships()
    
    # Save entities and relationships
    with open(samples_dir / "entities.json", 'w', encoding='utf-8') as f:
        json.dump(entities, f, indent=2)
    
    with open(samples_dir / "relationships.json", 'w', encoding='utf-8') as f:
        json.dump(relationships, f, indent=2)
    
    # Create sample queries file
    print("Creating sample queries...")
    sample_queries = [
        {
            "query": "What is machine learning?",
            "type": "definitional",
            "expected_results": ["Machine Learning encyclopedia entry"]
        },
        {
            "query": "Transformer architecture attention mechanism",
            "type": "semantic",
            "expected_results": ["Attention Is All You Need paper"]
        },
        {
            "query": "BERT bidirectional pre-training",
            "type": "semantic", 
            "expected_results": ["BERT paper"]
        },
        {
            "query": "Python machine learning tutorial",
            "type": "instructional",
            "expected_results": ["Python ML tutorial"]
        },
        {
            "query": "relationships between AI and ML",
            "type": "relationship",
            "expected_results": ["Knowledge graph relationships"]
        }
    ]
    
    with open(samples_dir / "sample_queries.json", 'w', encoding='utf-8') as f:
        json.dump(sample_queries, f, indent=2)
    
    # Create summary report
    print("Creating summary report...")
    summary = {
        "created_at": datetime.utcnow().isoformat(),
        "total_documents": len(documents),
        "document_types": {
            "research_papers": len([d for d in documents if d['metadata']['type'] == 'research_paper']),
            "encyclopedia_entries": len([d for d in documents if d['metadata']['type'] == 'encyclopedia_entry']),
            "tutorials": len([d for d in documents if d['metadata']['type'] == 'tutorial'])
        },
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "sample_queries": len(sample_queries),
        "files_created": [
            "documents_collection.json",
            "entities.json", 
            "relationships.json",
            "sample_queries.json",
            f"documents/ (individual files for {len(documents)} documents)"
        ]
    }
    
    with open(samples_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Sample Knowledge Base Created ===")
    print(f"Location: {samples_dir.absolute()}")
    print(f"Documents: {len(documents)}")
    print(f"Entities: {len(entities)}")
    print(f"Relationships: {len(relationships)}")
    print(f"Sample Queries: {len(sample_queries)}")
    print("\nFiles created:")
    for file in summary["files_created"]:
        print(f"  - {file}")
    
    print("\nTo use this sample data:")
    print("1. Start the Knowledge Base System: python run_knowledge_base.py")
    print("2. Upload documents via web interface: http://localhost:8080")
    print("3. Or run sample queries: python examples/sample_queries.py")


if __name__ == "__main__":
    main()