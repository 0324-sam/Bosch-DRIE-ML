Deep reactive ion etching (DRIE) is widely used in semiconductor manufacturing to fabricate high-aspect-ratio silicon struc-
tures. The Bosch process achieves anisotropic etching through alternating etching and passivation steps, resulting in progressively
evolving trench profiles and characteristic sidewall scalloping. Accurately predicting this geometric evolution is essential for
process optimization and dimensional control. However, conventional physics-based simulations can be computationally expensive,
particularly when exploring different process conditions and multiple etching cycles. This study investigates a geometric machine
learning-based surrogate model for predicting silicon trench profile evolution during the DRIE process. A dataset of simulated
trench profiles will be generated across different opening widths and process conditions. Each etch front will be represented as a
graph, with nodes describing local geometric features, including spatial position, surface normal, curvature and etch depth. A graph
neural network (GNN) will employ message passing between neighboring nodes to learn the local geometric changes occurring
during each etching cycle. By iteratively applying the learned model, trench profiles can be predicted over successive cycles
without repeatedly performing computationally expensive simulations. The graph representation will incorporate relative spatial
relationships and accommodate changes in graph size as trenches deepen, thus enabling the model to handle evolving profiles with
varying geometries. The proposed surrogate model will be evaluated against simulations calibrated on actual experimental data
by comparing predicted etch depths, trench widths, overall profile geometries and computational runtimes. Ultimately, this study
aims to develop an efficient and geometrically accurate surrogate modeling framework for accelerating trench profile prediction
and supporting process optimization in Bosch DRIE
