> The purpose of this working directory is to build a code repository for MolPallete, a variant of MolPLA tailored for Flavor Compound Optimization and its preprocessing pipeline.

### What is MolPallete?
1. MolPallete is a variant of MolPLA that features an auxiliary pretraining objective related to Lead Optimization (Core Decoration)
2. MolPallete is specifically trained on flavor compounds.
3. The condition vector for MolPallete should either protein pocket context or neutral (ligand-only).
4. MolPallete also shares aspects with MolDAM's anchored paradigm.

### List of Instructions

- READ: `/home/mogan/papers/molpla.pdf`
- READ: `/home/mogan/github/MolPLA/*`
	- Focus on the README.md
	- Memorize the core model architecture 
	- Infer its data preprocessing pipeline 

- READ: `/home/mogan/github/MolDAM_prep` 
	- Absorb its code-base structure and implementation style.
	- Apply it by building a new code-base in `/home/mogan/github/MolPallete/molpallete_prep/*`
	- Devise a plan for building pretraining data instances from `~/mogan/datasets/flavordb` and `~/mogan/datasets/coconut`.

- READ: `/home/mogan/github/MolDAM/*`
	- Absorb its code-base structure and implementation style
	- Apply it by building a new code-base in `/home/mogan/github/MolPallete/molpallete/*`
		- Only inherit the primary traits of MolPLA, not MoLDAM.
	- Check whether it is feasible to import MolDAM's assembly head and tokenization head.

- TASK:
	- Build code repositories for `/home/mogan/github/MolPallete/`
	- Build a pretraining dataset out of the FlavorDB and Coconut datasets.
	- Utilize various decomposition algorithms: `naveja`, `macfrag`, `synton`.
	- The decomposition scheme should be efficient as MoLDAM while the data instance construction should follow MolPLA.
	- Use an agentic approach to perform these complex tasks.