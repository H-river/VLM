## High-level plan

At a high level, I would treat this as a **simulator-first data-generation project** for a simple optical digital twin, and only later as an LLM problem.

- **Define the optical abstraction of the left-side setup.**  
  Start by reducing the diagram to the three optical elements that matter most for v1: **laser source → lens → camera plane**. Treat the arm as a **positioning mechanism** for the camera, not as part of the optical model, unless robot pose itself becomes one of your learning targets.

- **Build the first simulator around free-space propagation plus a thin lens.**  
  Use `waveprop` as the main propagation engine for the first prototype. Its official repository says it supports multiple scalar diffraction models, including **Fraunhofer, Fresnel, angular spectrum, and direct integration**, plus **off-axis propagation/rescaling**, **arbitrary amplitude or phase masks**, and **PyTorch support**; its examples also include focal-plane style runs with an `f_lens` argument. citeturn1view0turn5view1

- **Create a strict structured input schema before generating any data.**  
  Do not start from open-ended text prompts. Start from a machine-readable config with fields like:
  - source parameters: wavelength, initial beam waist, beam mode assumption, power normalization  
  - lens parameters: focal length, clear aperture, diameter  
  - sensor parameters: pixel pitch, sensor size, image resolution  
  - geometry: laser-to-lens distance, lens-to-camera distance  
  - alignment variables: x/y offset, tilt, focus error, clipping  
  This gives you a clean bridge from future LLM outputs to deterministic simulations.

- **Generate the dataset in two stages.**  
  First, run **controlled sweeps** where you vary one parameter at a time, so you can understand cause and effect. After that, move to **broader randomized combinations** or DOE-style sampling for model training. This keeps the early dataset interpretable and the later dataset diverse.

- **Store both raw optical outputs and analysis-friendly labels.**  
  For every run, save:
  - the raw intensity field  
  - a rendered beam-profile image  
  - metadata for all instrument settings and distances  
  - derived beam measurements  
  A strong analysis layer for the image side is `laserbeamsize`, whose documentation says it implements the **ISO 11146 variance method** and can estimate **beam center, major/minor axes, and rotation** from beam images. citeturn3view0turn3view1

- **Make beam width the primary label, and M² a later-stage label.**  
  For the first data collection stage, I would focus on single-plane observables such as **centroid**, **major/minor width**, **ellipticity**, **rotation**, and optionally **FWHM**. For **M²**, the `laserbeamsize` documentation notes that estimation uses beam diameters measured at **multiple propagation distances**, and the ISO 11146 family is specifically about measuring **beam widths, divergence angles, and beam-propagation ratios**. So M² should be treated as a **multi-plane measurement task**, not a single-image label. citeturn3view0turn0search13

- **Validate the physics before scaling the dataset.**  
  Before generating a large corpus, validate a small aligned Gaussian baseline. Thin-lens Gaussian-beam theory predicts a transformed beam with a new waist position and size after the lens, so your simulator should reproduce that kind of focusing behavior before you introduce misalignment, clipping, aberration, or noise. citeturn2search0turn2search10

- **Add the LLM only after the simulation pipeline is stable.**  
  The LLM should sit on top of the dataset, not replace it. The natural sequence is:
  1. parse experiment descriptions into structured configs  
  2. map diagrams or notes into candidate optical layouts  
  3. compare simulated outputs against expected beam behavior  
  4. recommend next alignment changes or measurement points  

In one sentence: **build the optical core first, standardize the measurement labels second, and only then train or prompt an LLM to reason over the setup**.