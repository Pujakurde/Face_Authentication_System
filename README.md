# Secure Face Authentication System

A secure, offline Face Authentication System built in Python using OpenCV and CustomTkinter.  
The system is designed with **progressive security levels**, where each level strengthens protection against spoofing and unauthorized access.

This repository documents **Levels 1 to 4** only.

---

## Project Goal

To design a **real-time, local face authentication system** that:
- Verifies identity using facial biometrics
- Ensures the presence of a live human
- Prevents basic spoofing attacks
- Avoids cloud APIs and external services

---

## Technology Stack

- Python
- OpenCV
- CustomTkinter (GUI)
- NumPy
- Encrypted local storage
- ArcFace-based face embeddings
- Modular security pipeline

---

## High-Level Flow

Camera Input  
→ Face Detection  
→ Liveness Verification  
→ Anti-Spoofing Check  
→ Face Recognition  
→ Secure Authentication Result  

---

## Security Levels Overview

| Level | Purpose |
|------|--------|
| Level 1 | Face Detection |
| Level 2 | Face Recognition |
| Level 3 | Liveness Detection |
| Level 4 | Static Photo Anti-Spoofing |

---

## Level 1 – Face Detection

**Objective:**  
Ensure a human face is present before any authentication logic runs.

**Implemented Features:**
- Real-time face detection
- Authentication blocked if:
  - No face is detected
  - More than one face is detected

**Security Benefit:**  
Prevents blind or accidental authentication attempts.

---

## Level 2 – Face Recognition

**Objective:**  
Verify that the detected face belongs to a registered user.

**Implemented Features:**
- Face embeddings generated per frame
- Multiple embeddings collected during registration
- Mean-normalized embeddings stored securely
- Sliding-window vote-based authentication

**Security Benefit:**  
Prevents unauthorized users from authenticating even if a face is present.

---

## Level 3 – Liveness Detection

**Objective:**  
Ensure the face belongs to a **live human**, not a static image.

**Implemented Features:**
- Blink detection
- Head movement detection
- Multi-frame liveness confirmation
- Liveness state locked per session

**Security Benefit:**  
Blocks basic photo attacks and printed face images.

---

## Level 4 – Anti-Spoofing (Static Photo Defense)

**Current Level**

**Objective:**  
Prevent authentication using **non-moving fake face sources**.

### Attacks Covered
- Printed photographs
- Static images shown on mobile screens
- Screenshot-based attacks

### Anti-Spoofing Strategy
A lightweight heuristic-based anti-spoofing gate is applied **after liveness and before identity matching**.

**Signals Used:**
- Blur (focus) score
- Inter-frame motion
- Face-to-frame area ratio
- Motion consistency across frames

### Key Design Rules
- Anti-spoofing runs **once per authentication session**
- Once passed, the result is locked
- Identity votes are not affected while anti-spoofing is pending

### What Level 4 Prevents

| Attack Type | Status |
|------------|--------|
| Printed photo | Blocked |
| Static phone image | Blocked |
| Screenshot attack | Blocked |
| No-motion face | Blocked |

### What Level 4 Does NOT Prevent

| Attack Type | Status |
|------------|--------|
| Video replay attacks | Not handled |
| Deepfake videos | Not handled |
| 3D mask attacks | Not handled |
| High-quality animated displays | Not handled |

---

## Data Security

- Face embeddings are stored in encrypted form
- No raw facial images are used for authentication
- All processing is local and offline

---

## Logging and Auditing

- All system activity is written to `logs/app.log`
- No debug output is printed to the terminal
- Logs include:
  - Registration events
  - Authentication attempts
  - Liveness results
  - Anti-spoofing decisions

---

## Current Status

- Levels 1 to 4 implemented and functional
- Static photo spoofing successfully blocked
- System ready for next security upgrade

---

## Disclaimer

This project is intended for **educational and research purposes**.  
Additional protections are required before production use.

---

## Next Step

**Level 5 – Advanced Spoofing Defense**  
(Video replay detection, temporal analysis, and stronger attack resistance)
