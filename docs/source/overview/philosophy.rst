==================
Design Philosophy
==================

Scikit-Robot is designed with the vision of making robot development accessible, flexible, and efficient. This page explains the core design principles that guide the development of scikit-robot.

Pure Python Implementation
===========================

Scikit-robot is implemented entirely in pure Python, following the philosophy of popular scientific Python libraries like NumPy and Scikit-learn. This approach offers several advantages:

**Accessibility**: Python's straightforward syntax and extensive ecosystem lower the barrier to entry for robotics development. Researchers and engineers can quickly prototype ideas without dealing with complex build systems or compilation issues.

**Integration**: Pure Python implementation enables seamless integration with the rich Python ecosystem, including data analysis tools (NumPy, Pandas), machine learning frameworks (TensorFlow, PyTorch), and visualization libraries (Matplotlib, Plotly).

**Cross-platform**: The library works across Linux, macOS, and Windows (via WSL2) without platform-specific compilation requirements.

Modular and Reconfigurable Robotics
====================================

One of the key motivations behind scikit-robot is to support **modular reconfigurable robotics** - robots that can dynamically change their physical morphology by adding, removing, or reconfiguring modules.

Traditional robot description formats like URDF were designed for static, fixed-morphology robots. Scikit-robot addresses this limitation through:

Integrated URDF Toolchain
--------------------------

The framework provides a complete toolchain for URDF manipulation:

- **modularize-urdf**: Converts monolithic URDF files into reusable xacro macros
- **change-urdf-root**: Dynamically reconfigures kinematic hierarchies by changing the root link
- **convert-urdf-mesh**: Optimizes 3D meshes for efficient simulation
- **visualize-urdf**: Provides immediate visual feedback for model transformations

Dynamic Root Transformation
----------------------------

Scikit-robot introduces a dynamic root transformation algorithm that enables non-root link connections. This removes constraints that have traditionally limited modular robot design, allowing flexible connection topologies.

For example, if you design a gripper module with its root link at the base, but need to connect it at the gripper tip, scikit-robot can dynamically reconfigure the URDF hierarchy to make this connection natural and physically meaningful.

Hash-based Model Management
============================

To address the challenges of model versioning and distribution in reconfigurable robotics, scikit-robot implements a comprehensive hash-based model management system:

**Content Hashing**: Rather than relying on filenames or metadata, scikit-robot computes comprehensive hashes that include:

- URDF file content (XML structure and all attributes)
- All referenced 3D meshes
- All texture files

**Automatic Distribution**: Hash values serve as unique identifiers, enabling automatic downloading and local caching of models and associated assets. This provides a pip-like experience for robot models.

**Model Identity**: The hash system ensures that reconfigured models maintain their identity and enables reliable duplicate detection.

Inherited Wisdom from EusLisp
==============================

Scikit-robot inherits core architectural principles from EusLisp, a robotics programming language with decades of research and practical experience. Key concepts include:

CascadedCoords: Efficient Coordinate Management
------------------------------------------------

The CascadedCoords class provides hierarchical coordinate transformation with several advantages:

**Incremental Updates**: When a parent coordinate changes, only affected branches are recalculated, not the entire tree.

**Lazy Evaluation**: Transformations are computed only when needed, reducing unnecessary calculations.

**Memory Efficiency**: Coordinate objects are shared within the hierarchy.

This approach contrasts with conventional methods that recalculate all transformations from scratch, providing significant performance benefits for complex robot models.

Design-to-Deployment Workflow
==============================

Scikit-robot establishes an integrated workflow that spans the entire robot development lifecycle:

1. **Design**: Start with CAD software (Fusion 360, SolidWorks, Onshape)
2. **Export**: Convert CAD models to URDF using standard exporters
3. **Process**: Use scikit-robot's toolchain to modularize, optimize, and configure
4. **Simulate**: Validate in simulators (PyBullet, etc.)
5. **Deploy**: Transfer the exact same model to real hardware

This seamless transition from design to deployment reduces the traditional barriers between simulation-based development and real-world implementation.

Visualization as a First-Class Feature
=======================================

Scikit-robot treats visualization not as an afterthought but as a core component of the development workflow:

**Immediate Feedback**: Every model transformation provides immediate visual feedback
**Multiple Backends**: Support for Trimesh, Pyrender, and Jupyter notebooks
**Interactive Exploration**: Enable developers to understand kinematic structures intuitively

The enhanced visual semantics compensate for URDF's limited semantic expressiveness, allowing developers to comprehend complex reconfiguration operations through immediate graphical feedback.

Community and Open Development
===============================

Scikit-robot is developed as an open-source project with:

**Public Development**: All code is available on GitHub
**Continuous Integration**: Daily automated testing ensures reliability
**Community Contribution**: Bug reports and feature requests are welcome in both English and Japanese

The goal is to provide a stable, actively maintained foundation for the robotics research community.

Future Directions
=================

Looking ahead, key development directions include:

**Novel User Interfaces**: Developing interfaces that overlay virtual robot models onto real-world counterparts, enabling intuitive design and automatic reconfiguration triggering.

**Enhanced Semantics**: Expanding semantic annotations and standardized module interface descriptions.

**Optimization Strategies**: Exploring machine learning-based optimization of reconfiguration strategies.

By following these design principles, scikit-robot aims to democratize robot development and accelerate research in adaptive and reconfigurable robotic systems.
