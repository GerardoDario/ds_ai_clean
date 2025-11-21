# Reinforcement Learning - Aprendizaje por Refuerzo

Este directorio contiene recursos y trabajos relacionados con Reinforcement Learning.

## 📚 Temas Principales

### 1. Fundamentos
- **Conceptos Básicos**
  - Agent, Environment, State, Action, Reward
  - Policy (π): estrategia del agente
  - Value Function: V(s) y Q(s,a)
  - Model-based vs Model-free
  - On-policy vs Off-policy

- **Markov Decision Process (MDP)**
  - Estados, acciones, transiciones
  - Recompensas y retornos
  - Factor de descuento (γ)
  - Bellman Equations

### 2. Métodos Tabulares
- **Dynamic Programming**
  - Policy Iteration
  - Value Iteration
  - Policy Evaluation

- **Monte Carlo Methods**
  - First-visit MC
  - Every-visit MC
  - MC Control
  - Exploring Starts

- **Temporal-Difference (TD) Learning**
  - TD(0)
  - SARSA (State-Action-Reward-State-Action)
  - Q-Learning
  - Expected SARSA

- **n-step Bootstrapping**
  - n-step TD
  - n-step SARSA
  - n-step Q-Learning

### 3. Value-Based Deep RL
- **Deep Q-Networks (DQN)**
  - Experience Replay
  - Target Networks
  - Variantes:
    - Double DQN
    - Dueling DQN
    - Prioritized Experience Replay
    - Rainbow DQN
    - Noisy DQN

### 4. Policy-Based Methods
- **Policy Gradient**
  - REINFORCE
  - Baseline methods
  - Actor-Critic
  - Advantage function

- **Advanced Policy Gradient**
  - **A3C** (Asynchronous Advantage Actor-Critic)
  - **A2C** (Advantage Actor-Critic)
  - **PPO** (Proximal Policy Optimization)
  - **TRPO** (Trust Region Policy Optimization)

### 5. Actor-Critic Methods
- **Deterministic Policy Gradient**
  - **DDPG** (Deep Deterministic Policy Gradient)
  - **TD3** (Twin Delayed DDPG)

- **Soft Actor-Critic (SAC)**
  - Maximum entropy RL
  - Off-policy learning

### 6. Model-Based RL
- **World Models**
  - Learning environment dynamics
  - Planning in learned models

- **Dyna-Q**
  - Planning and learning
  - Simulated experience

- **AlphaZero/MuZero**
  - Monte Carlo Tree Search (MCTS)
  - Self-play

### 7. Multi-Agent RL
- Cooperative agents
- Competitive agents
- Mixed scenarios
- QMIX, MADDPG

### 8. Inverse RL y Imitation Learning
- Learning from demonstrations
- Behavioral cloning
- Inverse Reinforcement Learning
- GAIL (Generative Adversarial Imitation Learning)

### 9. Hierarchical RL
- Options framework
- Feudal Networks
- Goal-conditioned RL

## 🔧 Herramientas y Bibliotecas

### Entornos de Simulación
- **OpenAI Gym**: Estándar de facto para RL
- **Gymnasium**: Fork mantenido de Gym
- **Unity ML-Agents**: Entornos 3D
- **MuJoCo**: Física para robótica
- **PyBullet**: Física de código abierto
- **Atari**: Juegos clásicos de Atari
- **PettingZoo**: Multi-agent environments

### Bibliotecas de RL
- **Stable-Baselines3**: Implementaciones de algoritmos populares (PyTorch)
- **RLlib (Ray)**: RL escalable y distribuido
- **TF-Agents**: TensorFlow para RL
- **Dopamine**: Framework de Google para investigación
- **CleanRL**: Implementaciones simples y limpias
- **Spinning Up (OpenAI)**: Recursos educativos

### Herramientas de Visualización
- **TensorBoard**: Monitoreo de entrenamiento
- **Weights & Biases**: Tracking de experimentos
- **Gym Monitor**: Grabación de episodios

## 📖 Recursos Recomendados

### Cursos
- [CS285 - Deep Reinforcement Learning (UC Berkeley)](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [David Silver's RL Course (DeepMind)](https://www.davidsilver.uk/teaching/)
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [Coursera - Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning)

### Libros
- **"Reinforcement Learning: An Introduction" - Sutton & Barto** (La biblia del RL)
- "Deep Reinforcement Learning Hands-On" - Maxim Lapan
- "Foundations of Deep Reinforcement Learning" - Laura Graesser, Wah Loon Keng
- "Algorithms for Reinforcement Learning" - Csaba Szepesvári

### Papers Fundamentales
- "Playing Atari with Deep Reinforcement Learning" (DQN, 2013)
- "Human-level control through deep reinforcement learning" (Nature DQN, 2015)
- "Continuous control with deep reinforcement learning" (DDPG, 2015)
- "Asynchronous Methods for Deep Reinforcement Learning" (A3C, 2016)
- "Proximal Policy Optimization Algorithms" (PPO, 2017)
- "Mastering the game of Go with deep neural networks" (AlphaGo, 2016)
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2019)

### Blogs y Recursos
- [OpenAI Blog - RL Section](https://openai.com/blog/tags/reinforcement-learning/)
- [DeepMind Blog](https://deepmind.com/blog)
- [Lil'Log - RL Posts](https://lilianweng.github.io/lil-log/)
- [Distill.pub - RL Articles](https://distill.pub/)

## 🚀 Proyectos Sugeridos

### Principiantes
1. **CartPole**: Balance de péndulo invertido
2. **Mountain Car**: Carro en una colina
3. **Frozen Lake**: Grid world simple
4. **Taxi**: Navegación y pickup/dropoff

### Intermedios
5. **LunarLander**: Control continuo
6. **Atari Games**: Breakout, Pong, Space Invaders
7. **BipedalWalker**: Locomoción
8. **Custom Grid Worlds**: Crear tus propios entornos

### Avanzados
9. **Multi-Agent Soccer**: Coordinación de equipos
10. **Robotic Manipulation**: Brazo robótico con MuJoCo
11. **Trading Bot**: RL para trading algorítmico
12. **Autonomous Driving**: Simulación de conducción
13. **Dota 2/StarCraft II**: RL en juegos complejos

## 📊 Entornos Populares

### Clásicos
- CartPole-v1
- MountainCar-v0
- Acrobot-v1
- Pendulum-v1

### Atari
- Breakout
- Pong
- Space Invaders
- Pac-Man

### Control Continuo
- LunarLander
- BipedalWalker
- HalfCheetah (MuJoCo)
- Ant (MuJoCo)
- Humanoid (MuJoCo)

### Robótica
- FetchReach
- HandManipulate
- ShadowHand

## 💡 Best Practices

1. **Empieza simple**: Usa CartPole antes de Atari
2. **Hyperparameter tuning**: RL es muy sensible a hiperparámetros
3. **Semilla aleatoria**: Usa múltiples seeds para evaluación
4. **Monitoreo**: Tracking de recompensas, loss, entropía
5. **Normalización**: Normaliza observaciones y recompensas
6. **Debugging**: RL es difícil de debuggear, empieza con implementaciones probadas
7. **Evaluation**: Separa entrenamiento de evaluación
8. **Paciencia**: RL requiere mucho tiempo de entrenamiento
9. **Baseline**: Compara con algoritmos establecidos
10. **Reproducibilidad**: Documenta seeds, hiperparámetros, código

## 🎯 Aplicaciones del Mundo Real

### Robótica
- Manipulación de objetos
- Locomoción
- Navegación autónoma
- Assembly tasks

### Juegos
- AlphaGo, AlphaZero
- Dota 2 (OpenAI Five)
- StarCraft II (AlphaStar)

### Finanzas
- Trading algorítmico
- Portfolio management
- Option pricing

### Sistemas de Recomendación
- Personalización de contenido
- Ad placement
- News recommendation

### Recursos y Energía
- Control de HVAC
- Grid optimization
- Resource allocation

### Salud
- Treatment planning
- Drug discovery
- Personalized medicine

## 🔬 Tendencias Actuales

- **Offline RL**: Aprender de datos históricos sin interacción
- **Meta-RL**: Aprender a aprender
- **World Models**: Modelos generativos del entorno
- **Sim-to-Real**: Transferencia de simulación a mundo real
- **Safe RL**: RL con restricciones de seguridad
- **Multi-task RL**: Un agente, múltiples tareas
- **Explainable RL**: Interpretabilidad de políticas
- **Sample Efficiency**: Reducir muestras necesarias

## ⚠️ Desafíos Comunes y Soluciones

1. **Sample Inefficiency**: Millones de pasos de entrenamiento
   - *Soluciones*: Model-based RL, curriculum learning, transfer learning, usar algoritmos off-policy (SAC, TD3)

2. **Inestabilidad**: Colapso del entrenamiento
   - *Soluciones*: Target networks, gradient clipping, normalización de rewards, usar PPO en lugar de TRPO

3. **Sensibilidad a hiperparámetros**: Pequeños cambios, grandes efectos
   - *Soluciones*: Grid search, Bayesian optimization, usar implementaciones validadas (Stable-Baselines3), documentar todos los hiperparámetros

4. **Reproducibilidad**: Resultados variables entre runs
   - *Soluciones*: Fijar random seeds, múltiples runs con estadísticas, documentar versiones de bibliotecas

5. **Credit Assignment**: Asignar crédito a acciones pasadas
   - *Soluciones*: Usar métodos TD con n-step returns, GAE (Generalized Advantage Estimation), hindsight experience replay

6. **Exploration vs Exploitation**: Balance difícil
   - *Soluciones*: ε-greedy scheduling, entropy regularization, curiosity-driven exploration, count-based bonuses

7. **Sparse Rewards**: Señales de aprendizaje infrecuentes
   - *Soluciones*: Reward shaping, curriculum learning, hindsight experience replay, imitation learning

8. **Transferencia**: Modelos específicos a entornos
   - *Soluciones*: Domain randomization, sim-to-real techniques, meta-RL, progressive neural networks
