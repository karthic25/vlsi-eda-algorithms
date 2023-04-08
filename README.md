# VLSI CAD
Electronic Design Automation (EDA) industry is ripe with softwares suited to automate development of VLSI designs (referred to as CAD tools, or Computer Aided Design tools) - which are the heartbeat of modern electronic devices. 

The logic of a design (a boolean function of various input variables), can represent simple functions such as 2-bit adders, or complex ones that involve designing for optimizing specific workloads such as ML training. Such a design is represented as an acyclic graph of gates, so that it can be further mapped to various device architectures, the gates placed on a device, connected to each other using physical wires, all the while keeping in check the timing constraints (how fast the design is expected to run) and congestion constraints (overloading of gates at a particular physical location).

Conversion of design logic to layout is called design flow, and it has four main steps:
  - Synthesis
  - Technology mapping
  - Placement
  - Routing

I focus on placement and routing in this project, each of which are NP-hard optimzation problems. Most of the solutions to these problems are heuristics-based. Although Google achieved a recent breakthrough on using reinforcement learning trained on edge-based graph neural networks to automate floorplanning (https://www.nature.com/articles/s41586-021-03544-w), not all industries that develop processors have been able to achieve this feat. They still rely on engineers to come up with heuristic algorithms (that may use ML/AI for smaller subproblems) that can solve placement and routing. 

## Placement
### Wirelength Estimation
#### Problem statement
The purpose of this wirelength estimation is to provide an optimal placement for a design (an acyclic graph of interconnected nodes, each of which needs to be placed in a rectangular device, under certain constraints), by estimating and minimizing total wirelength of a design. With varying placements of the design gates, the total wirelength will vary. By minimizing the wirelength, we also expect to minimize the delay caused due to large wirelengths.

#### Solution
In general, the closer the gates are placed, the lower the wirelength. But the gates are also required to be connected to input/output ports on the edges of the device, that constraints certain gates to be closer to the edges as well. Moreover, since the physical space required by a gate isn't void, overcrowding of gates at a physical location needs to be avoided. Industry processors deploy thousands to millions of gates depending on the design, that need to be optimized for placement. The algorithm for the same also needs to be fast and efficient in order to have a low compile runtime and memory.

Quadratic Placement is a technique that aids the same. It Assumes size zero for each of the gates, and minimizes the total quadratic distance, calculated by summing the gate distances pair-wise. By setting the partial differentials of the total quadratic distance to zero, we get the same no. of equations as the no. of gates, thereby giving us an analytical solution for the positions of the gates.

This technique is often coupled with recursive partitioning, that recursively divides the device into halfs and reassigns the gates proportionally to the two halfs. This helps reduce overcrowding of gates, that is a consequence of size zero for gates by the quadratic placement algorithm.

Although recursive partitioning helps, it doesn't completely avoid gate overlaps. Legalization is a process that reassigns gates by spreading them to nearby locations to completely remove gate overlaps.
