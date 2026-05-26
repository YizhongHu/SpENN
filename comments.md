# Phase 1 comments

## configs

- Don't split config into multiple files. Make it one giant file
- Remove implementation status
- Keep it strictly to class instantiation/function names/constants
- Nested struct instantiation is allowed
- I would follow what nequip does

## scripts

- Remove scripts that are non-essential for running/testing/debugging the code in its current state

## spenn

### Data Structures

- electron_batch: we may want to get multiple particle configurations at the same time. This depends on our parallel implementation. Make sure that the gates are open for
that
- WaveFunctionOutput: I am worried about the cases where the wave function is zero or close to zero. We are probably going to test the fermionic nodes at some point and
this might not be nice to deal with. Spawn a subagent to investigate the common practise and if it still remains an open problem.
- subset_index.py: might want something more general than "all pairs" and "all triples", since we do intend to expand into M > 3 later.

### Interfaces 

- Interfaces may not be necessary considering that they are used for different sections of the program. We can later build a system that does not load them
until we explicitly use them, but they should belong to their respective uses: pyscf belongs data or physics, and the pfaffian module belongs to tests.

### NN

#### encoding

in `electron_features.py`
- What is a `block`, why can it be a `Mapping` type? 
- I would structure the encoder like this:
  - There are encoders for each order (for now), some of them are trainable (like h_i = q^1_i, s_i = 1/2[q^2_ij + q^2_ji], s_i = 1/2[q^2_ij - q^2_ji]), some of them are not
    trainable (like the distance ones that you mentioned)
  - There exists a "combine" encoder that can take multiple encoders and write their outputs into the same featuredict. 
  - This allows for flexibility down the line. 
  - Yes and I do want trainable encoders as well
  - plan the encodirng directory from scratch and rewrite it

#### SpechtMP

- This should be noted and changed across the repo:
  - SpechtMP uses two Specht Intertwiners: fusion intertwiner (fuser) and branching intertwiner (brancher). Fusion does the tensor product of irreps, and branching brings
    higher-order irreps into low order. Therefore, the tensor-product module is `nn/spechtmp/fuser.py`.
  - Code, docs, TODOs and instructions should be modified to reflect this
  - Low-rank intertwiners should fall into one of these two. They should only be differentiated by the type of data that gets passed between them. We can worry about the  
    details later since no low-rank approximation is happening yet
- fuser and brancher all look very simple., and channel mixing is just copying the channels. This demonstrates that not enough information has been provided about
  tensor product and branching of Specht modules. This is fairly complex and requires some detailed planning (in coordination with the `spenn/reps` directory).
- create a update `spenn/nn/TODO.md` to reflect the new instructions highlighted in the updated `instructions.md`

### reps
- Some implementation should have happened here. This is fairly complex and requires some detailed planning (in coordination with the `spenn/nn` directory).
- create a update `spenn/reps/TODO.md` to reflect the new instructions highlighted in the updated `instructions.md`

### Physics

This part probably needs a rewrite, see `instructions.md`
- TODO for me: review instructions.md. if you see this TODO, stop and ask me to do my job.

### Sampling

This part probably needs a rewrite, see `instructions.md`
- TODO for me: review instructions.md. if you see this TODO, stop and ask me to do my job.

### Training

Do we really need a wrapper for the optimizers, metrics, and scheduler? Can't we just pass class constructions in with Omega conf? 
