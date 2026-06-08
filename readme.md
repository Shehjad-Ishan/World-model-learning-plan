# 4-Week Research Blueprint: From Transformers to World Models

**Target Audience:** ML Engineer with experience in PyTorch, Computer Vision, and Transformers.
**Cadence:** 1 to 2 Hours Daily (~42 Hours total).
**Core Objective:** Build a functional, action-conditioned latent world model from scratch.

---

## WEEK 1: THE FOUNDATION & DATA ENGINE
*Goal: Understand the macro-architecture and build a robust data collection pipeline to generate offline training data.*

### 📅 Day 1: The Blueprint
*   **Tasks:** Read the seminal paper: *World Models* (Ha & Schmidhuber, 2018). Focus deeply on the Abstract, Introduction, and Section 3.
*   **Focus:** Understand how the three core components ($V$: Perception, $M$: Memory, $C$: Controller) pass tensors to one another. 
*   **Deliverable:** A conceptual block diagram sketched out on paper or a tablet showing the tensor layout.

### 📅 Day 2: Environment Setup
*   **Tasks:** Initialize a clean git repository. Create a virtual environment and install dependencies.
*   **Dependencies:** `gymnasium[box2d]`, `torch`, `torchvision`, `numpy`, `matplotlib`.
*   **Deliverable:** Run a short python script that initializes the `CarRacing-v2` environment, passes random actions, and renders the window without crashing.

### 📅 Day 3: The Rollout Script
*   **Tasks:** Write a clean Python routine (`rollout.py`) designed to interact with the environment programmatically.
*   **Focus:** Implement a simple heuristic policy (e.g., constant acceleration with small random steering adjustments) so the car actually traverses different parts of the track rather than just spinning out immediately.
*   **Deliverable:** A working execution loop that can run 1,000 steps continuously.

### 📅 Day 4: Data Serialization
*   **Tasks:** Update `rollout.py` to buffer and save state transitions.
*   **Focus:** At each step $t$, collect: `obs_t` (RGB array), `action_t` (continuous array), `reward_t` (float), and `obs_{t+1}`. Compress these frames to $64 \times 64 \times 3$ or $96 \times 96 \times 3$ to save space.
*   **Deliverable:** Save these matrices to disk efficiently using `np.savez_compressed` or `safetensors`.

### 📅 Day 5: Pipeline & DataLoader Testing
*   **Tasks:** Run your rollout script in the background to accumulate between 10,000 and 20,000 unique frame transitions. 
*   **Focus:** Write a custom PyTorch `Dataset` and `DataLoader` class that reads these compressed files, applies minimal tensor normalization, and streams them in batches.
*   **Deliverable:** Run a test script confirming a batch size of 64 yields shapes `[64, 3, 64, 64]` for image states and `[64, 3]` for actions under 5ms.

### 📅 Days 6–7: The Dreamer Transition
*   **Tasks:** Read the foundational sections of *DreamerV3* (Hafner et al., 2023). 
*   **Focus:** Focus strictly on how modern world models transition state structures. Understand the difference between deterministic hidden states and stochastic categorical latents.
*   **Deliverable:** Note down the core loss terms used to keep a world model stable.

---

## WEEK 2: THE PERCEPTION LAYER (V)
*Goal: Compress high-dimensional pixel inputs into compact, low-dimensional latent vectors.*

### 📅 Day 8: VAE Architecture Definition
*   **Tasks:** Write the PyTorch class definition for a Convolutional Variational Autoencoder (`models/vae.py`).
*   **Architecture:** 
    *   Encoder: 4 `Conv2d` layers with stride 2, followed by two parallel linear layers predicting `mu` and `logvar`.
    *   Decoder: A linear layer mapping the latent code back, followed by 4 `ConvTranspose2d` layers.
*   **Deliverable:** Initialize the model and pass a dummy tensor of `[1, 3, 64, 64]` to verify the spatial math lines up. Target a latent size ($z$) of 32 or 64 dimensions.

### 📅 Day 9: The Reparameterization Trick & Loss
*   **Tasks:** Implement the forward pass reparameterization function: $z = \mu + \epsilon \cdot \sigma$.
*   **Focus:** Write the comprehensive loss function combining Mean Squared Error (MSE) reconstruction loss with the Kullback-Leibler (KL) Divergence penalty.
*   **Deliverable:** A verified, compilation-free loss module ready for a training loop.

### 📅 Days 10–11: Training the VAE
*   **Tasks:** Write and execute the training loop (`train_vae.py`) utilizing the data collected in Week 1.
*   **Focus:** Use an Adam optimizer. Carefully tune the $\beta$ weight scale applied to the KL loss term—if it's too high, your reconstructions will be blurry; if it's too low, the latent space won't form a smooth Gaussian distribution.
*   **Deliverable:** Train for 20-30 epochs until the validation loss curve plateaus cleanly.

### 📅 Day 12: Reconstructive Debugging
*   **Tasks:** Write a evaluation script to assess your perception layer.
*   **Focus:** Load a batch of unseen frames from a validation set, extract their latent representations, pass them through the decoder, and save a combined image plotting the original vs. reconstructed frames side-by-side.
*   **Deliverable:** Visual proof that your VAE preserves critical structural details like track boundaries and vehicle headings.

### 📅 Days 13–14: Abstract Feature Paradigms
*   **Tasks:** Read Yann LeCun’s position paper *A Path Towards Autonomous Machine Intelligence* (2022).
*   **Focus:** Internalize the core philosophy of Joint Embedding Predictive Architectures (JEPA). Contrast why predicting raw pixels can be computationally wasteful compared to predicting abstract latent properties.
*   **Deliverable:** Write a 1-page summary comparing generative world models (Dreamer) to non-generative energy-based world models (I-JEPA).

---

## WEEK 3: THE DYNAMICS ENGINE (M)
*Goal: Build the internal physics simulator that predicts state transitions through time.*

### 📅 Day 15: Sequence Model Architecture
*   **Tasks:** Design the backbone of your transition model (`models/dynamics.py`). 
*   **Options:** You can use an LSTM layer or a casual self-attention Transformer block. For this specific task, an LSTM is often faster to debug.
*   **Deliverable:** Set up the module to accept a concatenated vector consisting of the current encoded latent state $z_t$ and the action vector $a_t$.

### 📅 Day 16: The Causal Forward Pass
*   **Tasks:** Implement the forward autoregressive execution logic.
*   **Focus:** Ensure that when given an historical sequence length of $T$ steps, the model outputs a prediction vector representing the anticipated next latent space state $\hat{z}_{t+1}$.
*   **Deliverable:** Verify tensor shape mapping: `[Batch, Sequence_Length, Latent_Dim + Action_Dim]` $\rightarrow$ `[Batch, Sequence_Length, Latent_Dim]`.

### 📅 Days 17–18: Training the Transition Physics
*   **Tasks:** Write the training orchestration script (`train_dynamics.py`).
*   **Focus:** Freeze your trained VAE encoder. Process your trajectory sequences through the encoder to convert video frames into static arrays of latent vectors. Train your sequence model to minimize the MSE loss between its prediction $\hat{z}_{t+1}$ and the actual subsequent encoded vector $z_{t+1}$.
*   **Deliverable:** Train for 30 epochs, monitoring the drop in latent state prediction errors over time.

### 📅 Days 19–20: Simulating Inside the "Dream Space"
*   **Tasks:** Write a validation routine to test if your model can accurately simulate the environment autonomously.
*   **Focus:** Take a single true starting frame from a real sequence, encode it to $z_0$, and then lock out the real game engine. Autoregressively feed the model's own predictions back into itself for 50 sequential steps while passing a fixed action (e.g., "hard left turn").
*   **Deliverable:** Pass those 50 hallucinated latents through your VAE decoder and save them as a video. If you see the track curving smoothly in the generated frames, your world model has successfully learned the latent physics.

### 📅 Day 21: High-Scale Frontiers
*   **Tasks:** Read NVIDIA’s research paper on *Cosmos 3* (2026).
*   **Focus:** Study how the simple sequential state loop you just coded scales up into a multi-modal transformer foundation architecture handling massive video streams and physical simulations.
*   **Deliverable:** Identify how unified tokenization strategies alter performance at scale.

---

## WEEK 4: THE CONTROLLER & APPLICATION PORTFOLIO (C)
*Goal: Train an operational control policy entirely inside your simulated environment and document your findings.*

### 📅 Day 22: The Controller Network
*   **Tasks:** Write the final architectural component (`models/controller.py`).
*   **Focus:** Keep this network highly lightweight—a single linear layer or a minimal 2-layer Multi-Layer Perceptron (MLP) is sufficient. Its only job is to map an incoming latent vector $z_t$ directly to an action vector $a_t$.
*   **Deliverable:** Script verification that the controller outputs valid action spaces (Steering, Gas, Braking) bounded appropriately.

### 📅 Days 23–24: Training Within the Imagination
*   **Tasks:** Implement the dream-space training loop (`train_controller.py`).
*   **Focus:** The controller must learn to drive *entirely inside the hallucinated latents* generated by your frozen Week 3 dynamics engine. Use an evolution strategy like CMA-ES or a simple policy gradient loop (REINFORCE) to optimize the controller against the rewards predicted by the world model.
*   **Deliverable:** A fully optimized controller weight-set that has never once seen a raw pixel or interacted with the true Gymnasium simulator.

### 📅 Day 25: Closing the Loop (The Real-World Deployment)
*   **Tasks:** Write the evaluation script (`run_agent.py`) to connect all three pieces in the real environment.
*   **Loop:** Real Gym Environment Frame $\rightarrow$ Frozen VAE Encoder $\rightarrow$ Controller Choice $\rightarrow$ Step Environment.
*   **Deliverable:** Watch the agent navigate the track in real-time. Document its performance metrics (average track completion score over 10 episodes).

### 📅 Day 26: Codebase Refactoring
*   **Tasks:** Clean up your repository layout.
*   **Focus:** Abstract hardcoded hyperparameters into a unified configuration file (`config.yaml`). Ensure your modules are properly commented and comply with standard style guidelines.
*   **Deliverable:** A highly readable, production-grade PyTorch codebase.

### 📅 Days 27–28: The Application Portfolio Deliverable
*   **Tasks:** Write a comprehensive `README.md` for your GitHub profile.
*   **Inclusions:**
    1.  An architecture block diagram explaining your specific pipeline.
    2.  An animated GIF or video clip showing your agent driving successfully.
    3.  A research abstract or future work section outlining how this precise spatial-temporal predictive loop can be applied to the **MILK10k medical dataset** (e.g., using macro clinical views as state inputs and dermoscopic translations as the action-conditioned latent goal targets).
*   **Deliverable:** A completed, highly competitive proof-of-work project ready to anchor your graduate school applications.
