# VLSI EDA Algorithms
Electronic Design Automation (EDA) industry is ripe with softwares suited to automate development of VLSI designs (referred to as CAD tools, or Computer Aided Design tools) - which are the heartbeat of modern electronic devices. 

The logic of a design (a boolean function of various input variables), can represent simple functions such as 2-bit adders, or complex ones that involve designing for optimizing specific workloads such as ML training. Such a design is represented as an acyclic graph of gates, so that it can be further mapped to various device architectures, the gates placed on a device, connected to each other using physical wires, all the while keeping in check the timing constraints (how fast the design is expected to run) and congestion constraints (overloading of gates at a particular physical location).

Conversion of design logic to layout is called design flow, and it has four main steps:
  - Synthesis
  - Technology mapping
  - Placement
  - Routing

I focus on placement in this project, which is an NP-hard optimzation problem. Most of the solutions to this problem are heuristics-based. The industry relies primarily on heuristic algorithms (that may use ML/AI for smaller subproblems) that can solve placement and routing. 

## Placement
### Wirelength Estimation
#### Problem statement
Placement needs to be optimized for wirelength amongst other constraints. Lower wirelength also leads to lower delay. 

#### Solution
In general, the closer the gates are placed, the lower the wirelength. But the gates are also required to be connected to input/output ports on the edges of the device, that constraints certain gates to be closer to the edges as well. Moreover, since the physical space required by a gate isn't void, overcrowding of gates at a physical location needs to be avoided. Industry processors deploy thousands to millions of gates depending on the design, that need to be optimized for placement. The algorithm for the same also needs to be fast and efficient in order to have a low compile runtime and memory.

Quadratic Placement is a technique that aids the same. It reduces gates to points and minimizes the **total quadratic distance**, calculated by summing the gate distances pair-wise. By setting the partial differentials of the total quadratic distance to zero, we get the same no. of equations as the no. of gates, thereby giving us an analytical solution for the positions of the gates.

This technique is often coupled with recursive partitioning, that recursively divides the device into halfs and reassigns the gates proportionally to the two halfs. This helps reduce overcrowding of gates, that is a consequence of size zero for gates by the quadratic placement algorithm. **Reference**: [PROUD: a sea-of-gates placement algorithm](https://doi.org/10.1109/54.9271)

Although recursive partitioning helps, it doesn't completely avoid gate overlaps. Legalization is the final process in placement that reassigns gates by spreading them to nearby locations to completely remove gate overlaps.

qp_partitioner.py implements quadratic placement with recursive partitioning. 
