# Face Authentication System (Production-Oriented)

![Level 1](https://img.shields.io/badge/Level%201-Face%20Detection-brightgreen)
![Level 2](https://img.shields.io/badge/Level%202-Liveness%20Verification-blue)
![Level 3](https://img.shields.io/badge/Level%203-Secure%20Authentication-orange)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

---

## Overview

This project implements a **secure, production-style Face Authentication System** designed using real-world biometric security principles rather than demo-level face recognition.

The system follows a **multi-level verification pipeline** that enforces liveness checks, quality validation, encrypted biometric storage, and multi-frame authentication voting to reduce spoofing and false acceptance.

The architecture and logic are inspired by how **enterprise biometric authentication systems** are designed.

---

## Design Philosophy

This system intentionally avoids common insecure practices:

- No single-frame authentication
- No raw face image storage
- No direct embedding comparison without validation
- No bypass of liveness checks

Instead, it enforces:

- Deterministic state-machine flow
- Temporal (multi-frame) authentication
- Encrypted biometric data
- Fail-safe resets and recovery paths
- Explicit separation between detection, liveness, and authentication

---

## Authentication Levels

---

### Level 1: Face Detection

**Objective:**  
Ensure a valid human face is present before any further processing.

**Implemented Features:**
- Real-time face detection using MediaPipe
- Bounding box stabilization
- Continuous face presence monitoring
- Automatic reset when the face is lost

**Security Benefit:**  
Prevents false triggers and unnecessary processing when no face is present.

---

### Level 2: Liveness Verification

**Objective:**  
Verify that the detected face is live and not a spoof.

**Implemented Checks:**
- Blink detection
- Head movement detection
- Frame-count-based validation
- One-time liveness consumption per session

**Security Benefit:**  
Protects against photo attacks, video replays, and screen-based spoofing.

---

### Level 3: Secure Registration and Authentication

**Objective:**  
Authenticate users using robust biometric embeddings and controlled decision logic.

#### Registration Pipeline
- ArcFace embedding generation
- Time-bound capture window
- Multiple face samples per user
- Quality checks (blur, lighting, face size)
- Encrypted storage of embeddings

#### Authentication Pipeline
- Multi-frame voting mechanism
- Same-user consistency enforcement
- Majority-based decision logic
- Automatic rejection on instability

**Security Benefit:**  
Reduces false acceptance caused by noise, motion blur, or transient matches.

---

## Security Architecture (Recruiter-Focused)

This system incorporates multiple real-world biometric security practices:

- Encrypted storage of facial embeddings
- No plaintext biometric data on disk
- Temporal authentication with voting
- Explicit state-machine transitions:
  - WAIT_FACE → WAIT_LIVENESS → AUTHENTICATING → LOCKED
- Liveness reuse prevention
- Quality gating before both registration and authentication
- Automatic rollback on failure states

This architecture mirrors approaches used in high-security biometric systems to reduce FAR (False Acceptance Rate).

---

## Project Structure

```text
FACEAUTHENTICATION/
│
├── app/
│   ├── gui.py
│   ├── arcface_embedder.py
│   ├── face_matcher.py
│   ├── face_aligner.py
│   ├── face_alignment.py
│   ├── crypto_utils.py
│   ├── session_store.py
│   └── welcomescreen.py
│
├── face_detection/
│   └── face_detector.py
│
├── anti_spoofing/
│   └── predictor.py
│
├── models/
│   └── arcface_r100.onnx   (ignored via .gitignore)
│
├── run.py
├── requirements.txt
└── README.md

```
---

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/Pujakurde/Face_Authentication_System.git
cd Face_Authentication_System
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python run.py
```

---

## Runtime Flow

1. Camera initialization
2. Face detection loop
3. Liveness challenge enforcement
4. Registration or authentication process
5. Authentication decision finalization
6. System lock and welcome screen launch

---

## Future Work (Level 4 – Planned)

* Deep-learning-based anti-spoofing (CNN)
* Print and screen attack detection
* Mobile camera robustness
* Authentication audit logging
* Threat simulation and attack testing

---

## Why This Project Stands Out

* Not a tutorial-based implementation
* Designed with security-first principles
* Clear state management and flow control
* Extendable for research and production
* Suitable for system design and security interviews

---

## Author

**Puja Kurde**
B.Tech in Data Science

Focus Areas:

* Secure Systems
* Computer Vision
* Applied Machine Learning
