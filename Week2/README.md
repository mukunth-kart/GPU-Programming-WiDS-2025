# **Week 2 — Parallel Thinking & CUDA Programming Basics**

Read the assignemtn_2.pdf in this folder to learn more, only basic things here.

This week marks the transition from high-level GPU intuition to **writing your first CUDA code**.
Mapped data-parallel problems onto threads, blocks, and grids. Further optimization in week-3.

Three kernels were written in cuda language:
1) Element wise multiply and scale by a given value (multiply_scale.cu)
2) ReLU activateion function (relu.cu)
3) Vector Addition (vector_add.cu)

These kernels were not timed except the vector addition(vecAddEvent.cu) using cudaEvent library.

Note:
To execute them compile using nvcc and then execute the created output file(usually a.out).
