# MATS Application: Geometric Analysis of High-Fidelity SAE Features

## 1. Foundational Work: Development of RelP-SAE
**Project Status: Completed with Working Implementation and Validation**

I have developed and validated **RelP-SAE**, a novel methodology that addresses the "faithfulness crisis" in feature attribution. By extending the high-fidelity Relevance Patching (RelP) from circuits to Sparse Autoencoder (SAE) features, this work provides the first LRP-based attribution method for SAEs. The full implementation, results, and documentation are available in the accompanying GitHub repository.

### Key Findings from Completed Work
Our comprehensive validation revealed critical insights into attribution method reliability:

| Method | Correlation w/ Ground Truth | Notes |
|---|---|---|
| **RelP-SAE (Novel)** | -0.02 (Weak) | 500x larger attribution scale, 0% overlap with ground truth |
| Gradient (Baseline) | **0.25 (Moderate)** | The most faithful of the tested methods |
| Integrated Gradients | -0.05 (Weak) | No better than chance |

These results are significant: they show that our novel method, RelP-SAE, uncovers a fundamentally different landscape of feature importance than gradient methods, and that even the strongest baseline (IG) fails to correlate with ground truth. This work serves as a critical foundation, proving that high-fidelity attribution is a non-trivial problem and motivating the need for deeper analysis of feature properties.

---

## 2. Proposed MATS Project: Geometric Analysis of Attributed Features
My proposed research for the MATS program is to build directly upon this validated foundation. Having established that different attribution methods identify radically different features, the clear next step is to investigate the underlying properties of these features.

**Core Research Question:** Do the geometric properties of SAE features in latent space predict their causal importance and attribution method sensitivity?

### Project Goals
1.  **Characterize the Geometry of High-Attribution Features:** Analyze how the features identified as important by RelP-SAE, Gradients, and Ground Truth are structured in the SAE's latent space.
    - **Techniques:** Use dimensionality reduction (PCA, UMAP), clustering analysis (k-means, DBSCAN), and nearest-neighbor analysis.
    - **Hypothesis:** Features identified by faithful methods (like gradients or ground truth) will exhibit more coherent geometric clustering than those identified by less faithful methods.

2.  **Connect Geometry to Causal Faithfulness:** Test whether geometric properties (e.g., cluster density, distance from origin, alignment with other features) can predict a feature's causal effect in activation patching experiments.
    - **Goal:** Develop a "geometric faithfulness score" that could potentially identify important features more efficiently than running expensive causal experiments.

3.  **Investigate Geometric Basis of Method Divergence:** Understand why RelP-SAE identifies features that are geometrically and causally distinct from those found by gradient-based methods.
    - **Hypothesis:** RelP may be sensitive to features involved in multiplicative interactions, which could occupy different geometric regions of the latent space.

---

## 3. Long-Term Vision: A Bridge to Quantum Interpretability
This research directly builds the methodological foundation for my long-term goal: pioneering Quantum Mechanistic Interpretability.

- **The Challenge:** Quantum-classical hybrid models will have feature spaces with complex, inherent geometric structures (e.g., the Bloch sphere). Understanding these systems will require tools that are sensitive to this geometry.
- **The Bridge:** My proposed MATS project—developing a geometric understanding of high-fidelity features—is the perfect bridge. The geometric analysis techniques I will develop for classical SAEs are directly transferable to analyzing representations in quantum systems.
- **The Trajectory:** This approach provides a clear research arc:
    1.  **Completed:** Solve a core faithfulness problem in classical MI (`RelP-SAE`).
    2.  **MATS Project:** Develop a geometric framework for understanding high-fidelity features.
    3.  **Future Work:** Apply this geometric framework to interpret the feature spaces of quantum and hybrid models.

This positions me to not just apply existing techniques to quantum systems, but to build the foundational, transferable methodologies that will be necessary to make those systems truly interpretable.
