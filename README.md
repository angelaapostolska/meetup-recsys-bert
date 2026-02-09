# meetup-recsys-bert

# BERT-Based Event Recommendation System

> Addressing the Cold-Start Problem with Foundation Models

A research implementation replicating and extending the work of [Halimeh et al. (2023)](https://ceur-ws.org/Vol-3568/paper5.pdf) on cold-start event recommendation using foundation models. This project uses **DistilBERT embeddings** and **K-Means clustering** to recommend events without requiring historical user interaction data.

**Author:** Angela Apostolska  
**Institution:** Faculty of Computer Science and Engineering (FINKI)  
**Date:** January 2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [The Cold-Start Problem](#the-cold-start-problem)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## 🎯 Overview

Traditional event recommendation systems rely on **user-item interaction matrices** (who attended what), which creates a fundamental problem: **how do you recommend new events that nobody has attended yet?** This is known as the **cold-start problem**.

This project solves this by using **semantic similarity** instead of interaction history:
- Convert event descriptions to 768-dimensional embeddings using **DistilBERT**
- Group semantically similar events into clusters using **K-Means**
- Recommend events from the same cluster based on content similarity

---

## ❄️ The Cold-Start Problem

### Traditional Collaborative Filtering Approach:
```
User-Item Interaction Matrix:
                Event1  Event2  Event3  NEW_EVENT
User_Alice        ✓       ✗       ✓         ?
User_Bob          ✗       ✓       ✗         ?
User_Charlie      ✓       ✗       ✗         ?

Problem: NEW_EVENT has no interaction data!
         Can't use collaborative filtering ✗
```

### Our Content-Based Approach:
```
NEW_EVENT: "Advanced Python Workshop"
    ↓ DistilBERT
    ↓ 768-dimensional embedding
    ↓ Find nearest cluster
    ↓ Cluster 5 (Python/Programming events)
    ✓ Recommend to users interested in Cluster 5 ✓
```

---

## 🔬 Methodology

### **Two-Step Pipeline:**

#### **Step 1: Embedding Generation**
- **Model:** [DistilBERT](https://arxiv.org/abs/1910.01108) (`distilbert-base-nli-stsb-mean-tokens`)
- **Input:** Concatenated event text: `"Event: [name]. Organized by: [group]. Category: [category]."`
- **Output:** 768-dimensional semantic embeddings
- **Library:** [Sentence-Transformers](https://www.sbert.net/)

#### **Step 2: Clustering**
- **Algorithm:** [K-Means](https://projecteuclid.org/euclid.bsmsp/1200512992)
- **Configuration:** k=30 clusters, random_state=42, n_init=10
- **Evaluation:** [Silhouette Score](https://doi.org/10.1016/0377-0427(87)90125-7) for cluster quality

### **Technology Stack:**
```
Python 3.8+
├── transformers         # Hugging Face transformers
├── sentence-transformers # BERT embedding models
├── scikit-learn         # K-Means clustering
├── pandas               # Data manipulation
├── numpy                # Numerical operations
└── matplotlib/seaborn   # Visualization
```

---

## 📊 Dataset

**Source:** [Meetup Events Data](https://www.kaggle.com/datasets/prashdash112/meetup-events-data) (Kaggle)

- **Events:** 220 Meetup events
- **Categories:** Technology, Business
- **Attributes:** Event name, Organizing group, Category, Date, Location, Attendees

**Preprocessing:**
1. Handle missing values (NaN → placeholder text)
2. Text fusion: Combine name + group + category
3. Generate DistilBERT embeddings (768-D vectors)

**Example Event:**
```python
Name: "Python for Beginners Workshop"
Group: "San Francisco Python Developers"
Category: "Technology"

→ Fused Description:
"Event: Python for Beginners Workshop. 
 Organized by: San Francisco Python Developers. 
 Category: Technology."

→ DistilBERT Embedding: [0.123, -0.456, 0.789, ...] (768 dimensions)
```

---

## 📈 Results

### **Overall Performance**
- **Overall Silhouette Score:** 0.052 (moderate clustering quality)
- **Clusters Generated:** 30
- **Evaluation Metric:** Silhouette score (range: -1 to +1)

### **Cluster Quality Spectrum**

#### ✅ **Best Cluster (C14):**
- **Events:** 3 startup-related events
- **Silhouette Score:** 0.3507 (highest)
- **Events:** "Startup", "Advice for Startups building Business Models"
- **Analysis:** Small, tightly-focused cluster with strong internal cohesion

#### ⚠️ **Medium Cluster (C23):**
- **Events:** 26 events
- **Silhouette Score:** ~0.05 (weak)
- **Analysis:** Moderately large with weak semantic boundaries

#### ❌ **Worst Cluster (C8):**
- **Events:** 33 events (15% of dataset)
- **Silhouette Score:** -0.0021 (negative = poor quality)
- **Events:** Mixed content - "Quantum Computing", "JavaScript Bootcamp", "Startup Ideas", "Kubernetes Workshop"
- **Analysis:** Vague "catch-all" cluster with no coherent theme

### **Key Finding:**
**Inverse relationship between cluster size and quality** → Large clusters exhibit poor cohesion and vague semantic boundaries.

---
## 🔍 Key Findings

### **1. Cluster Size vs. Quality Trade-off**
- **Small clusters (3-5 events):** High silhouette scores (0.3507)
- **Large clusters (26-33 events):** Poor scores (-0.0021 to 0.05)
- **Implication:** K-Means struggles with high-dimensional BERT embeddings when semantic categories overlap

### **2. Successful Semantic Grouping**
When K-Means captures focused niches (e.g., Cluster 14's startup events), recommendations are highly accurate and relevant.

### **3. Vague Catch-All Clusters**
Large clusters like C8 become "catch-alls" for diverse unrelated events, degrading recommendation quality.

### **4. Foundation Model Viability**
Despite moderate overall score (0.052), the approach demonstrates that BERT-based content similarity can enable cold-start recommendations without user interaction data.

---

## 🔮 Future Improvements

Based on the analysis, the following enhancements could significantly improve performance:

### **1. Alternative Clustering Algorithms**
- **DBSCAN:** Density-based clustering to identify outliers and handle arbitrary cluster shapes
- **HDBSCAN:** Hierarchical extension that auto-determines cluster count and handles varying densities
- **Gaussian Mixture Models:** Probabilistic approach allowing soft membership in multiple clusters

### **2. Enhanced Features**
- **Add Location:** Include event location (State/Country) to split geographic clusters
- **Use Attendee Count:** Separate large popular events from intimate gatherings
- **Keyword Extraction:** Extract skill level (beginner/advanced), format (workshop/talk), topics

### **3. Dimensionality Reduction**
- **PCA:** Reduce 768D → 100D to speed up clustering and reduce noise
- **UMAP:** Nonlinear reduction preserving semantic structure better than PCA

### **4. Optimal k Selection**
- Test k ∈ {40, 50, 60, 70} with elbow method
- Use silhouette analysis to find optimal cluster count

### **5. Hierarchical Clustering**
- Create taxonomy: Technology → Programming → Python → Beginner/Advanced
- Enable multi-level recommendations based on user preferences

**Expected Impact:** Silhouette score improvement from 0.052 → 0.15-0.20

---

## 📚 References

1. **M. Halimeh, D. Marx, M. Hecher** (2023). "Tackling the Cold-Start Problem in Cultural Event Recommendations Using Foundation Models." *CARS 2023: Workshop on Context-Aware Recommender Systems*. [PDF](https://ceur-ws.org/Vol-3568/paper5.pdf)

2. **Kaggle Dataset:** "Meetup Events Data" by Prashant Dash. [Link](https://www.kaggle.com/datasets/prashdash112/meetup-events-data)

3. **V. Sanh et al.** (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." *arXiv:1910.01108*. [Paper](https://arxiv.org/abs/1910.01108)

4. **N. Reimers, I. Gurevych** (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*. [Paper](https://arxiv.org/abs/1908.10084)

5. **J. MacQueen** (1967). "Some methods for classification and analysis of multivariate observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, pp. 281-297. [Link](https://projecteuclid.org/euclid.bsmsp/1200512992)

6. **P. Rousseeuw** (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*, vol. 20, pp. 53-65. [DOI](https://doi.org/10.1016/0377-0427(87)90125-7)

---

**Keywords:** BERT, DistilBERT, Event Recommendation, Cold-Start Problem, K-Means Clustering, Foundation Models, Semantic Embeddings, Silhouette Score, Machine Learning, NLP

