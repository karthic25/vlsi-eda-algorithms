# VLSI CAD
Electronic Design Automation (EDA) industry is ripe with softwares suited to automate development of VLSI designs (VLSI CAD) - which are the heartbeat of modern electronic devices. 

The logic of a design, a boolean function of various input variables, can represent simple functions such as 2-bit adders, or complex ones that involve designing for optimizing specific workloads such as ML training. Such a design is represented as an acyclic graph of gates, so that it can be further mapped to various device architectures, the gates be placed on a device, and connected to each other using physical wires, all the while keeping in check the timing constraints (how fast the design is expected to run) and congestion constraints (overloading of gates at a particular physical location).

Conversion of design logic to layout is called design flow, and it has four main steps:
  - Synthesis
  - Technology mapping
  - Placement
  - Routing

I focus on placement and routing in this repository, each of which are optimzation problems that are NP-hard. Most of the solutions to these problems are based on heuristics. Although Google achieved a recent breakthrough on using reinforcement learning trained on edge-based graph neural networks to automate floorplanning (https://www.nature.com/articles/s41586-021-03544-w), not all industries that develop processors have achieved this feat. They still rely on engineers to come up with heuristic algorithms (that may use ML/AI for smaller subproblems) that can solve placement and routing. 

Wirelength estimation happens early in the placement flow, thereby directing global placement and local placement steps that follow. The placement achieved thereafter directs routing of the design as well. It is therefore critical to achieve a high accuracy at this step.

## Wirelength Estimation
The purpose of this step is to provide an optimal placement for the design, by estimating and optimizing total wirelength of a design. The algorithm for the same also needs to be fast and efficient in order to have a low compile runtime and memory.
