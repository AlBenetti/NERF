# NERF

![Unit Tests](https://github.com/AlBenetti/NERF/actions/workflows/test.yml/badge.svg)

This repository contains the code for the paper [**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**](https://arxiv.org/abs/2003.08934) by Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng. The attempt is to learn a continuous volumetric representation of a scene from a set of images and use it to render novel views. The model is trained end-to-end to minimize a reconstruction loss and a smoothness loss. The model is trained on the pytorch lightning framework.
