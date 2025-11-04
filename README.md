# 🩸 BLOSMIA – BLOod SMear Image Analyser

## Project

**BLOSMIA (BLOod SMear Image Analyser)** is a collaborative research initiative jointly developed by the **Department of Computer Science, Madras Christian College (MCC)** and the **Department of Transfusion Medicine and Immunohematology, Christian Medical College (CMC), Vellore**, South India.

Over the past decade, this project has focused on the **segmentation and classification of peripheral blood smear images** to identify and analyze red and white blood cells.

At present, BLOSMIA aims to **optimize accuracy and processing speed**, enhancing its reliability as a clinical diagnostic tool.

---

## Overview

Blood examination plays a crucial role in clinical diagnostics — nearly **70% of medical decisions** depend on laboratory test results.

In the **peripheral blood smear test**, stained slides are examined microscopically to study cell morphology. At **CMC Vellore**, these slides are barcoded, digitized, and analyzed by BLOSMIA’s **AI model** for automated detection and classification of blood cells.

* **Red Blood Cells (RBCs):** Transport oxygen throughout the body.
* **White Blood Cells (WBCs):** Defend against infections and are categorized into five types:

  * Neutrophils
  * Eosinophils
  * Lymphocytes
  * Monocytes
  * Basophils

BLOSMIA automates this classification process, providing **accurate cell counts**, **reducing manual workload**, and **improving diagnostic precision and efficiency** in hematology.

---

## Existing System

The existing **AI-based BLOSMIA system** automates the analysis of peripheral blood smears by:

* Detecting **RBCs** and **WBCs**
* Classifying WBCs into five distinct types
* Generating a **complete blood count (CBC) report**

### Key Features:

* **Automated Analysis:** Replaces the time-consuming manual process, minimizing human error.
* **Optimized Database:** Enables efficient data management and retrieval.
* **Retraining Module:** Supports continuous model improvement with new data.
* **Model Evaluation:** Comparative testing of CNN, Inception V3, and Inception V4 for maximum accuracy.
* **Modular API Design:** Simplifies integration with other hospital systems.
* **AWS Cloud Deployment:** Enables secure, real-time, multi-user access with scalability and reliability.

---

## Objectives

The main objectives of BLOSMIA’s current development phase are:

* 🔬 **Enhance WBC detection** by retraining the model with expanded datasets for higher accuracy.
* 🧠 **Improve preprocessing** for better noise reduction, contrast, and segmentation.
* ♻️ **Implement continuous retraining** to adapt dynamically with new patient data.
* ⚖️ **Identify the most efficient deep learning model** through comparative analysis.
* 💾 **Optimize database design** for faster and more reliable data handling.
* 🔗 **Build modular APIs** to enable smooth integration and system scalability.
* ⏱️ **Reduce manual effort** and accelerate diagnostic reporting.

---

## Proposed System

The proposed system focuses on **enhancing accuracy and efficiency** in WBC classification by improving both preprocessing and the deep learning architecture.

### 🔧 Preprocessing Enhancements:

* Improved image handling and dataset organization.
* Automatic separation of **RBC** and **WBC** crops into distinct folders.
* **Timestamp-based naming** to prevent duplication.
* **Automated clearing** of input folders and database blobs for seamless operation.

### 🧩 Model Development:

* Introduced a new **WBC-only deep learning model (`v3_wbc_only.h5`)**, trained exclusively on WBC blob images.
* Previous combined model (`v3_rbc.h5`) used both RBC and WBC data; the new model focuses on **distinct WBC morphological features** for improved accuracy.

### 📊 Model Evaluation:

A comparative analysis between the old and new models was conducted using:

* **Accuracy**
* **Precision**
* **Recall**
* **Confusion Matrix**

The new **WBC-only model** demonstrated **better cell type differentiation**, significantly improving the reliability of automated WBC classification in BLOSMIA.
