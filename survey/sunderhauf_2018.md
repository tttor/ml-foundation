# The Limits and Potentials of Deep Learning for Robotic
* Niko Sünderhauf et al

## intro
How much trust can we put
in the predictions of a deep learning system when misclassi-
fications can have catastrophic consequences? How can we
estimate the uncertainty in a deep network’s predictions and
how can we fuse these predictions with prior knowledge
and other sensors in a probabilistic framework? How well
does deep learning perform in realistic unconstrained open-
set scenarios where objects of unknown class and appearance
are regularly encountered?
How can we generate enough high-quality training data?
Do we rely on data solely collected on robots in real-
world scenarios or do we require data augmentation through
simulation? How can we ensure the learned policies transfer
well to different situations, from simulation to reality, or
between different robots?

## challenges for deep learning in robotic vision

* learning challenges
  * Uncertainty Estimation:
    reliably estimate the uncertainty in their predictions
  * Identify Unknowns:
    must not assign high-confidence scores to unknown objects
    or falsely recognize them as one of the known classes
  * Incremental Learning:
    to learn from new training samples of known classes during deployment
    and adopt its internal representations accordingly
  * Class-Incremental Learning:
    to extend its knowledge and efficiently learn new classes without
    forgetting the previously learned representations
  * Active Learning:
    to select the most informative samples for incremental learning
    techniques on its own.

* Embodiment Challenges
  * Temporal Embodiment:
    accumulate evidence over time to improve its predictions
  * Spatial Embodiment:
    Observing an object from different viewpoints can
    help to disambiguate its semantic properties, improve depth
    perception, or segregate an object from other objects or
    the background in cluttered scenes.
  * Active Vision:
  * Manipulation for Perception:

* Reasoning Challenges
  * Reasoning About Object and Scene Semantics:
  * Reasoning About Object and Scene Geometry:
  * Joint Reasoning about Semantics and Geometry:

## the role of simulation for pixel -to -action robotics
calculate that learning this task, which trains to convergence
in 24 hours using a CPU compute cluster, would take 53
days on the real robot even with continuous training for 24
hours a day.

transfer learning methods to bridge the reality gap that
separates simulation from real world domains.
